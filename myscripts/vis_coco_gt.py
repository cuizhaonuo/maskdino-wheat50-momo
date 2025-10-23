#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import argparse
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as mask_util


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_coco(ann_json: str) -> COCO:
    assert osp.isfile(ann_json), f"Not found: {ann_json}"
    return COCO(ann_json)


def build_category_maps(coco: COCO) -> Tuple[Dict[int, str], Dict[int, Tuple[int,int,int]]]:
    """
    返回:
      id2name: {cat_id -> cat_name}
      id2color: {cat_id -> BGR color tuple}
    """
    cats = coco.loadCats(coco.getCatIds())
    id2name = {c["id"]: c["name"] for c in cats}
    # 为类别生成稳定颜色（BGR）
    id2color = {}
    for c in cats:
        cid = c["id"]
        # 固定随机色（用类 id 做种子）
        rng = np.random.default_rng(cid)
        color = rng.integers(0, 255, size=3, dtype=np.uint8).tolist()  # RGB
        id2color[cid] = (int(color[2]), int(color[1]), int(color[0]))  # BGR
    # 若没有类别（极端情况），给一个默认 wheat 颜色
    if not id2name:
        id2name = {1: "wheat"}
        id2color = {1: (0, 0, 255)}
    return id2name, id2color


def draw_detection(img: np.ndarray, bbox: List[float], color: Tuple[int,int,int], txt: str = "", tl: int = 2):
    x, y, w, h = bbox
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, p1, p2, color, thickness=tl)
    if txt:
        tf = max(tl - 1, 1)
        ((tw, th), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=tf)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, txt, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), tf, cv2.LINE_AA)


def polygon_to_mask(polys: List[List[float]], h: int, w: int) -> np.ndarray:
    """
    polys: [ [x1,y1,x2,y2,...], ... ]
    return: HxW uint8 mask
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not polys:
        return mask
    cv2_polys = []
    for p in polys:
        if len(p) < 6:
            continue
        pts = np.asarray(p, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2_polys.append(pts)
    if cv2_polys:
        cv2.fillPoly(mask, cv2_polys, 1)
    return mask


def rle_to_mask(rle_obj: Dict[str, Any]) -> np.ndarray:
    m = mask_util.decode(rle_obj)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int,int,int], alpha: float = 0.5):
    """
    在 img 上按 alpha 叠加彩色 mask（mask=0/1）
    """
    if mask is None or mask.size == 0:
        return
    c = np.array(color_bgr, dtype=np.uint8).reshape(1,1,3)
    color_layer = np.ones_like(img, dtype=np.uint8) * c
    m3 = np.repeat(mask[:, :, None], 3, axis=2)
    # alpha blend: img = (1-alpha)*img + alpha*color
    img[m3.astype(bool)] = (img[m3.astype(bool)] * (1 - alpha) + color_layer[m3.astype(bool)] * alpha).astype(np.uint8)


def visualize_split(
    img_root: str,
    ann_json: str,
    out_det_dir: str,
    out_seg_dir: str,
    alpha: float = 0.5,
):
    coco = load_coco(ann_json)
    id2name, id2color = build_category_maps(coco)

    imgid_list = coco.getImgIds()
    imgs = coco.loadImgs(imgid_list)

    ensure_dir(out_det_dir)
    ensure_dir(out_seg_dir)

    for img_info in tqdm(imgs, desc=f"Vis {osp.basename(ann_json)}"):
        file_name = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]
        img_path = osp.join(img_root, file_name)
        # if not osp.isfile(img_path):
        #     # 有些 COCO 标注可能 file_name 不含子目录，这里兜底再拼一次
        #     alt = osp.join(img_root, osp.basename(file_name))
        #     if osp.isfile(alt):
        #         img_path = alt
        #     else:
        #         print(f"[WARN] missing image: {img_path}")
        #         continue

        img_det = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_seg = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_det is None or img_seg is None:
            print(f"[WARN] failed to read: {img_path}")
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_info["id"]], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # ---- draw per instance ----
        for ann in anns:
            cid = ann.get("category_id", 1)
            cname = id2name.get(cid, str(cid))
            color = id2color.get(cid, (0, 0, 255))

            # detection
            if "bbox" in ann:
                draw_detection(img_det, ann["bbox"], color, txt=cname, tl=2)

            # segmentation
            seg = ann.get("segmentation", None)
            mask = None
            if isinstance(seg, list):
                # polygon
                mask = polygon_to_mask(seg, h, w)
            elif isinstance(seg, dict) and "size" in seg and "counts" in seg:
                # RLE
                # mask = rle_to_mask(seg)
                from pycocotools import mask as maskUtils
                def decode_coco_rle_to_mask(coco_rle):
                    """
                    将 COCO RLE 解码回二值掩码 (HxW)
                    """
                    rle = coco_rle.copy()
                    # 确保 counts 是 bytes
                    if isinstance(rle.get('counts'), str):
                        rle['counts'] = rle['counts'].encode('ascii')
                    return maskUtils.decode(rle).astype(np.uint8)
                mask = decode_coco_rle_to_mask(seg)
            if mask is not None:
                # overlay_mask(img_seg, mask, color, alpha=alpha)
                def overlay_mask_on_image(img_bgr, mask, color_bgr=None, alpha=0.5):
                    """
                    在 BGR 图上叠加单个二值掩码（0/1），返回新图。
                    """
                    import random
                    if color_bgr is None:
                        color_bgr = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    overlay = img_bgr.copy()
                    overlay[mask.astype(bool)] = (
                            overlay[mask.astype(bool)] * (1 - alpha) + np.array(color_bgr, dtype=np.float32) * alpha
                    ).astype(np.uint8)
                    return overlay
                img_seg = overlay_mask_on_image(img_seg, mask, color_bgr=color, alpha=alpha)

        # ---- save ----
        out_det_path = osp.join(out_det_dir, osp.basename(file_name))
        out_seg_path = osp.join(out_seg_dir, osp.basename(file_name))
        cv2.imwrite(out_det_path, img_det)
        cv2.imwrite(out_seg_path, img_seg)


def main():
    ap = argparse.ArgumentParser(description="COCO GT visualization: detection (bbox) and segmentation (masks)")
    ap.add_argument("--img-root", required=True,
                    help="coco 根目录，包含 train2017/ val2017/ test2017/")
    ap.add_argument("--ann-root", required=True,
                    help="annotations 根目录，包含 instances_train2017.json / val2017.json / test2017.json")
    ap.add_argument("--out-root", required=True, default="./vis_maskdino",
                    help="输出根目录，将创建 detection/segmentation/{train,val,test}")
    ap.add_argument("--alpha", type=float, default=0.5, help="segmentation 叠加透明度 0~1")
    args = ap.parse_args()

    # 组织路径
    splits = {
        "train": ("train2017", "instances_train2017.json"),
        "val":   ("val2017",   "instances_val2017.json"),
        "test":  ("test2017",  "instances_test2017.json"),
    }

    for split, (img_dirname, ann_filename) in splits.items():
        img_dir = osp.join(args.img_root, img_dirname)
        ann_json = osp.join(args.ann_root, ann_filename)

        det_dir = osp.join(args.out_root, "detection", split)
        seg_dir = osp.join(args.out_root, "segmentation", split)
        ensure_dir(det_dir)
        ensure_dir(seg_dir)

        if not osp.isfile(ann_json):
            print(f"[INFO] skip {split}: missing {ann_json}")
            continue
        visualize_split(
            img_root=img_dir,
            ann_json=ann_json,
            out_det_dir=det_dir,
            out_seg_dir=seg_dir,
            alpha=args.alpha,
        )

    print(f"Done. Results saved under: {args.out_root}")


if __name__ == "__main__":
    main()
