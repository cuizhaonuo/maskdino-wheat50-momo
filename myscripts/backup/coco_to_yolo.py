#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
from tqdm import tqdm

# 可选：只有在 --task seg 并且 COCO segmentation 为 RLE 时需要
try:
    from pycocotools import mask as maskUtils
except Exception:
    maskUtils = None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_coco_json(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 有些 json 不包含 info/licenses，做兼容
    data.setdefault("images", [])
    data.setdefault("annotations", [])
    data.setdefault("categories", [])
    return data


def build_cat_mappings(categories: List[Dict]) -> Tuple[Dict[int, int], List[str]]:
    """
    将 COCO 的类别 id → 连续 [0..K-1]，并返回 names 列表（按 id 排序稳定）
    """
    # 用 category id 的排序来保证稳定性
    categories_sorted = sorted(categories, key=lambda c: c["id"])
    id2idx = {}
    names = []
    for idx, c in enumerate(categories_sorted):
        id2idx[c["id"]] = idx
        names.append(c.get("name", f"cls{idx}"))
    return id2idx, names


def xywh_to_yolo(x, y, w, h, img_w, img_h):
    """
    COCO bbox: [x_min, y_min, w, h] → YOLO: [x_center/img_w, y_center/img_h, w/img_w, h/img_h]
    并裁剪到图像范围内
    """
    # 裁剪边界
    x = max(0.0, min(x, img_w))
    y = max(0.0, min(y, img_h))
    w = max(0.0, min(w, img_w - x))
    h = max(0.0, min(h, img_h - y))

    if img_w <= 0 or img_h <= 0:
        return None
    if w <= 1e-6 or h <= 1e-6:
        return None

    xc = x + w / 2.0
    yc = y + h / 2.0
    return [xc / img_w, yc / img_h, w / img_w, h / img_h]


def norm_polygon(poly: Iterable[float], img_w: int, img_h: int) -> Optional[List[float]]:
    """
    将 COCO polygon（x1,y1,x2,y2,...）归一化到 [0,1]，并做最小长度检查
    """
    poly = list(poly)
    if len(poly) < 6 or len(poly) % 2 == 1 or img_w <= 0 or img_h <= 0:
        return None
    norm = []
    for i, v in enumerate(poly):
        if i % 2 == 0:  # x
            nv = max(0.0, min(float(v), img_w)) / img_w
        else:           # y
            nv = max(0.0, min(float(v), img_h)) / img_h
        norm.append(nv)
    # 有些极端情况下所有点相同或面积趋近 0，可在此进一步过滤（可选）
    return norm


def rle_to_polygons(segm, img_h, img_w) -> List[List[float]]:
    """
    将 RLE segmentation 转为 polygons（需要 pycocotools）。
    返回归一化后的 polygon 列表（每个 polygon: [x1,y1,x2,y2,...] 归一化到 [0,1]）
    """
    if maskUtils is None:
        raise RuntimeError("需要 pycocotools 才能从 RLE 转 polygon，请先安装 pycocotools。")

    if isinstance(segm, dict) and "counts" in segm:
        rles = [segm]
    elif isinstance(segm, list) and len(segm) > 0 and isinstance(segm[0], dict):
        rles = segm
    else:
        return []

    m = maskUtils.decode(rles)
    if m.ndim == 3:
        m = np.any(m, axis=2)
    m = m.astype(np.uint8)

    # 提取外轮廓（这一步你也可以用 cv2.findContours；这里我们使用 pycocotools 的 toBbox 不提供多边形，故简单走 cv2 会更直接）
    try:
        import cv2
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            flat = cnt.reshape(-1, 2).astype(float)
            poly = []
            for x, y in flat:
                poly.extend([x, y])
            nrm = norm_polygon(poly, img_w, img_h)
            if nrm:
                polys.append(nrm)
        return polys
    except ImportError:
        # 没有 cv2：退化为单个 bbox 外接四边形（不推荐，但可用作兜底）
        bbox = maskUtils.toBbox(maskUtils.encode(np.asfortranarray(m)))
        x, y, w, h = [float(t) for t in bbox]
        # 造一个矩形多边形
        poly = [x, y, x+w, y, x+w, y+h, x, y+h]
        nrm = norm_polygon(poly, img_w, img_h)
        return [nrm] if nrm else []


def convert_split(
    coco_dir: Path,
    yolo_dir: Path,
    split: str,
    task: str,
    skip_crowd: bool = True,
    copy_images: bool = True
) -> Tuple[int, int]:
    """
    将单个 split（如 train2017）从 COCO 转到 YOLO

    返回: (写入的图片数, 写入的标注条目数)
    """
    ann_path = coco_dir / "annotations" / f"instances_{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"找不到标注文件: {ann_path}")

    coco = load_coco_json(ann_path)
    id2idx, names = build_cat_mappings(coco.get("categories", []))

    # 索引 images 与 annotations
    images = {img["id"]: img for img in coco["images"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # 准备输出目录
    split_map = {"train2017": "train", "val2017": "val", "test2017": "test"}
    yolo_split = split_map.get(split, split)  # 如果不是这三个，就原样保留

    img_in_dir = coco_dir / split  # 默认图片目录名与 split 相同
    img_out_dir = yolo_dir / "images" / yolo_split
    lbl_out_dir = yolo_dir / "labels" / yolo_split
    ensure_dir(img_out_dir)
    ensure_dir(lbl_out_dir)

    n_imgs, n_lbls = 0, 0

    for img_id, img in tqdm(images.items(), desc=f"Converting {split}"):
        file_name = img["file_name"]
        img_w = int(img["width"])
        img_h = int(img["height"])

        # 标签文件路径
        stem = Path(file_name).stem
        label_path = lbl_out_dir / f"{stem}.txt"

        lines = []
        for ann in anns_by_img.get(img_id, []):
            if skip_crowd and int(ann.get("iscrowd", 0)) == 1:
                continue
            cat_id = ann["category_id"]
            if cat_id not in id2idx:
                continue
            cls = id2idx[cat_id]

            if task == "det":
                # bbox 转 YOLO (xc,yc,w,h)
                bbox = ann.get("bbox", None)
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = [float(v) for v in bbox]
                yolo_box = xywh_to_yolo(x, y, w, h, img_w, img_h)
                if yolo_box is None:
                    continue
                line = f"{cls} " + " ".join(f"{v:.6f}" for v in yolo_box)
                lines.append(line)
                n_lbls += 1

            elif task == "seg":
                seg = ann.get("segmentation", None)
                # segmentation 可能是多段 polygon 或 RLE
                if not seg:
                    # 回退到 bbox（可选：也可跳过）
                    bbox = ann.get("bbox", None)
                    if bbox:
                        x, y, w, h = [float(v) for v in bbox]
                        # 用矩形生成一个简化 polygon
                        poly = [x, y, x+w, y, x+w, y+h, x, y+h]
                        nrm = norm_polygon(poly, img_w, img_h)
                        if nrm:
                            line = f"{cls} " + " ".join(f"{v:.6f}" for v in nrm)
                            lines.append(line)
                            n_lbls += 1
                    continue

                if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                    # COCO polygon：list[list[float]]
                    # YOLOv8 seg 一行一个实例；如有多段，可**拼接到同一行**（Ultralytics 支持多个片段）
                    merged = []
                    for poly in seg:
                        nrm = norm_polygon(poly, img_w, img_h)
                        if nrm:
                            merged.extend(nrm)
                    if len(merged) >= 6 and len(merged) % 2 == 0:
                        line = f"{cls} " + " ".join(f"{v:.6f}" for v in merged)
                        lines.append(line)
                        n_lbls += 1

                elif isinstance(seg, dict) and "counts" in seg:
                    # RLE
                    polys = rle_to_polygons(seg, img_h, img_w)
                    if len(polys) == 0:
                        continue
                    merged = []
                    for p in polys:
                        merged.extend(p)
                    if len(merged) >= 6 and len(merged) % 2 == 0:
                        line = f"{cls} " + " ".join(f"{v:.6f}" for v in merged)
                        lines.append(line)
                        n_lbls += 1
                else:
                    # 未知的 segmentation 结构，略过
                    continue

            else:
                raise ValueError(f"未知任务类型: {task}")

        # 写标签
        if lines:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        # 复制图片（如果存在）
        src_img = img_in_dir / file_name
        if copy_images:
            dst_img = img_out_dir / Path(file_name).name
            ensure_dir(dst_img.parent)
            if not src_img.exists():
                # 有些 COCO 数据集图片在子目录里；尝试相对路径查找
                alt = coco_dir / Path(file_name)
                if alt.exists():
                    src_img = alt
            if src_img.exists():
                shutil.copy2(src_img, dst_img)

        n_imgs += 1

    return n_imgs, n_lbls


def write_data_yaml(
    yolo_dir: Path,
    names: List[str],
    splits: List[str],
    task: str
):
    """
    写 data.yaml，用于 Ultralytics YOLO 训练
    """
    # 相对/绝对路径都可以；这里写绝对路径更稳妥
    lines = [
        f"path: {str(yolo_dir)}",
        f"task: {task}",               # 'detect' 或 'segment'
        f"train: {str((yolo_dir / 'images' / splits[0]).resolve()) if splits else ''}",
    ]
    if len(splits) > 1:
        lines.append(f"val: {str((yolo_dir / 'images' / splits[1]).resolve())}")
    else:
        # 没有显式 val 时，train=val
        lines.append(f"val: {str((yolo_dir / 'images' / splits[0]).resolve())}")

    # names 列表
    lines.append("names:")
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")

    with open(yolo_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO dataset to YOLO format (det/seg)."
    )
    parser.add_argument("--coco_dir", type=str, required=True,
                        help="COCO 根目录（包含 annotations/ 和 各 split 图片目录）")
    parser.add_argument("--yolo_dir", type=str, required=True,
                        help="YOLO 输出根目录（将创建 images/ 与 labels/ 子目录）")
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"],
                        help="需要转换的 split 列表（需与 annotations/instances_*.json 对应）")
    parser.add_argument("--task", type=str, default="det", choices=["det", "seg"],
                        help="det: 目标检测；seg: 实例分割（YOLOv8 polygon 标签）")
    parser.add_argument("--copy_images", action="store_true",
                        help="是否复制图片到输出目录（默认不复制，仅生成 labels）")
    parser.add_argument("--keep_crowd", action="store_true",
                        help="保留 iscrowd=1 的标注（默认跳过）")
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    yolo_dir = Path(args.yolo_dir)
    ensure_dir(yolo_dir)

    # 用第一个 split 的 categories 生成 names；若不同 split 类别不一致，请自行合并/校验
    first_ann = coco_dir / "annotations" / f"instances_{args.splits[0]}.json"
    coco_first = load_coco_json(first_ann)
    id2idx, names = build_cat_mappings(coco_first.get("categories", []))

    total_imgs, total_lbls = 0, 0
    for sp in args.splits:
        n_i, n_l = convert_split(
            coco_dir=coco_dir,
            yolo_dir=yolo_dir,
            split=sp,
            task=args.task,
            skip_crowd=not args.keep_crowd,
            copy_images=args.copy_images,
        )
        total_imgs += n_i
        total_lbls += n_l

    # 写 data.yaml
    write_data_yaml(
        yolo_dir=yolo_dir,
        names=names,
        splits=args.splits,
        task="detect" if args.task == "det" else "segment"
    )

    print(f"[完成] 共处理图片: {total_imgs}, 写入标注条目: {total_lbls}")
    print(f"[提示] data.yaml 已写入: {yolo_dir / 'data.yaml'}")
    print(f"[结构] {yolo_dir}/images/{{split}} 与 {yolo_dir}/labels/{{split}}")


if __name__ == "__main__":
    main()
