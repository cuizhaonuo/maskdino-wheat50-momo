#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
from pycocotools import mask as mask_util


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def rle_to_binary_mask(segm: Dict[str, Any]) -> np.ndarray:
    """
    segm: {"size":[H,W], "counts": <rle str or list>}
    return: HxW uint8 (0/1)
    """
    rle = segm
    if isinstance(segm.get("counts", None), list):
        # uncompressed RLE
        h, w = int(segm["size"][0]), int(segm["size"][1])
        rle = mask_util.frPyObjects(segm, h, w)
    m = mask_util.decode(rle)  # HxW or HxWx1
    if m.ndim == 3:
        m = m[..., 0]
    return (m.astype(np.uint8) > 0).astype(np.uint8)


def clip_poly_to_image(poly: List[float], w: int, h: int) -> List[float]:
    """
    clip polygon coordinates to image boundary.
    """
    if not poly or len(poly) < 6:
        return []
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    p = np.round(p, 2)  # 保留两位小数，避免超长小数
    p = p.reshape(-1).tolist()
    # 必须是偶数且 >= 6
    if len(p) < 6 or (len(p) % 2) != 0:
        return []
    return p


def binary_mask_to_polygons(
    mask: np.ndarray,
    approx_eps: float,
    min_area: float,
    keep_holes: bool,
    img_w: int,
    img_h: int,
) -> List[List[float]]:
    """
    Convert a binary mask (uint8 0/1) to COCO polygon list(s).
    Return: [ [x1,y1, x2,y2, ...], ... ]  (each list length >= 6 & even)
    """
    mask255 = (mask > 0).astype(np.uint8) * 255
    mode = cv2.RETR_CCOMP if keep_holes else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(mask255, mode, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[List[float]] = []
    if not contours:
        return polys

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < float(min_area):
            continue
        if approx_eps > 0.0:
            cnt = cv2.approxPolyDP(cnt, approx_eps, True)  # (N,1,2)
        if cnt.shape[0] < 3:
            continue
        cnt = cnt.reshape(-1, 2).astype(np.float32)
        # OpenCV 是 (x,y)
        poly = cnt.reshape(-1).tolist()
        poly = clip_poly_to_image(poly, img_w, img_h)
        if poly:
            polys.append(poly)
    return polys


def mask_from_polygons(polys: List[List[float]], h: int, w: int) -> np.ndarray:
    """
    Build binary mask from polygons for robust bbox/area recompute.
    """
    m = np.zeros((h, w), dtype=np.uint8)
    if not polys:
        return m
    cv_polys = []
    for p in polys:
        if not p or len(p) < 6:
            continue
        pts = np.asarray(p, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv_polys.append(pts)
    if cv_polys:
        cv2.fillPoly(m, cv_polys, 1)
    return m


def recompute_from_mask(mask: np.ndarray) -> Tuple[List[float], float]:
    """
    bbox (x,y,w,h) and area from a binary mask.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
    area = float(mask.sum())
    return bbox, area


def normalize_polygon_list(segm: Any, w: int, h: int) -> List[List[float]]:
    """
    Ensure 'segmentation' is a proper list[list[float]] clipped to image.
    """
    if not isinstance(segm, list):
        return []
    out: List[List[float]] = []
    for poly in segm:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        poly = clip_poly_to_image(poly, w, h)
        if poly:
            out.append(poly)
    return out


def convert_annotations_file(
    in_json: str,
    out_json: str,
    approx_eps: float = 1.0,
    min_area: float = 10.0,
    keep_holes: bool = False,
    drop_empty: bool = False,
    keep_by_rect: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Convert one COCO annotations json.
    Return: (num_total_anns, num_converted_from_rle, num_fixed_by_rect, num_dropped)
    """
    with open(in_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 建 index: image_id -> (w,h)
    id2size: Dict[int, Tuple[int, int]] = {}
    for im in coco.get("images", []):
        w = int(im.get("width", 0))
        h = int(im.get("height", 0))
        id2size[int(im["id"])] = (w, h)

    anns = coco.get("annotations", [])
    total = len(anns)
    converted = 0
    fixed_by_rect = 0
    dropped = 0
    new_anns: List[Dict[str, Any]] = []

    for ann in anns:
        img_id = int(ann["image_id"])
        w, h = id2size.get(img_id, (None, None))
        if w is None or h is None or w <= 0 or h <= 0:
            # 没有图片尺寸信息，直接跳过该实例（极少见）
            dropped += 1
            continue

        segm = ann.get("segmentation", None)
        ok = False

        # ===== 1) RLE → polygon =====
        if isinstance(segm, dict) and "counts" in segm and "size" in segm:
            mask = rle_to_binary_mask(segm)
            # size 若与 image 不一致，不缩放，直接按 decode 尺寸做 contours；
            # bbox/area 稳妥起见一律从 mask 重算。
            polys = binary_mask_to_polygons(
                mask, approx_eps=approx_eps, min_area=min_area, keep_holes=keep_holes, img_w=w, img_h=h
            )
            if polys:
                m = mask_from_polygons(polys, h, w)
                bbox, area = recompute_from_mask(m)
                ann["segmentation"] = polys
                ann["bbox"] = bbox
                ann["area"] = area
                ann["iscrowd"] = 0
                ok = True
                converted += 1
            else:
                # 走缺省处理
                segm = None  # 标记为没有有效分割

        # ===== 2) 已是 polygon → 规范化（clip + 过滤）=====
        elif isinstance(segm, list):
            polys = normalize_polygon_list(segm, w, h)
            if polys:
                m = mask_from_polygons(polys, h, w)
                bbox, area = recompute_from_mask(m)
                ann["segmentation"] = polys
                ann["bbox"] = bbox
                ann["area"] = area
                ann["iscrowd"] = 0
                ok = True
            else:
                segm = None  # 标记为没有有效分割

        # ===== 3) 其它/缺失 → segm = None 统一处理 =====
        else:
            segm = None

        # ===== 4) 对缺失/空分割的处理策略 =====
        if not ok:
            if keep_by_rect and ("bbox" in ann) and ann["bbox"] and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
                # 用 bbox 生成矩形 polygon，保证 segm 存在，避免训练侧 KeyError
                rect_poly = [
                    ann["bbox"][0], ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3],
                    ann["bbox"][0], ann["bbox"][1] + ann["bbox"][3],
                ]
                rect_poly = clip_poly_to_image(rect_poly, w, h)
                if rect_poly:
                    polys = [rect_poly]
                    m = mask_from_polygons(polys, h, w)
                    bbox, area = recompute_from_mask(m)
                    ann["segmentation"] = polys
                    ann["bbox"] = bbox
                    ann["area"] = area
                    ann["iscrowd"] = 0
                    ok = True
                    fixed_by_rect += 1

        if ok:
            new_anns.append(ann)
        else:
            if drop_empty:
                dropped += 1
            else:
                # 明确保留，但必须保证有 segmentation 字段，避免训练侧 KeyError
                ann["segmentation"] = []
                ann["bbox"] = [0.0, 0.0, 0.0, 0.0]
                ann["area"] = 0.0
                ann["iscrowd"] = 0
                new_anns.append(ann)

    # 覆盖并保存
    coco["annotations"] = new_anns
    ensure_dir(os.path.dirname(out_json))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    return total, converted, fixed_by_rect, dropped


def main():
    parser = argparse.ArgumentParser(description="Convert COCO RLE/bitmask segmentation to robust polygons and sanitize.")
    parser.add_argument("--ann-root", required=True,
                        help="输入 annotations 目录，包含 instances_train2017.json / val2017.json / test2017.json")
    parser.add_argument("--out-root", required=True,
                        help="输出目录，将写入清洗后的 JSON（建议：annotations_polygon/）")
    parser.add_argument("--approx-eps", type=float, default=1.0,
                        help="多边形近似 epsilon（像素），越大越简化；0 表示不过度简化")
    parser.add_argument("--min-area", type=float, default=10.0,
                        help="过滤面积小于该阈值的轮廓（像素）")
    parser.add_argument("--keep-holes", action="store_true",
                        help="是否保留孔洞（会作为独立 polygon 导出；COCO 不支持带孔的单 polygon）")
    # 缺失/空分割的处理策略（二选一）：默认保留并用 bbox→矩形 polygon 修复；或选择直接丢弃
    parser.add_argument("--drop-missing", action="store_true",
                        help="缺失/空 segmentation 的实例直接丢弃（默认会用 bbox 生成矩形 polygon 保留）")
    args = parser.parse_args()

    in_root = args.ann_root
    out_root = args.out_root
    ensure_dir(out_root)

    files = [
        "instances_train2017.json",
        "instances_val2017.json",
        "instances_test2017.json",
    ]
    summary: List[Tuple[str, int, int, int, int]] = []

    for fn in files:
        in_json = os.path.join(in_root, fn)
        if not os.path.isfile(in_json):
            print(f"[INFO] skip missing file: {in_json}")
            continue
        out_json = os.path.join(out_root, fn)
        total, converted, fixed_by_rect, dropped = convert_annotations_file(
            in_json,
            out_json,
            approx_eps=args.approx_eps,
            min_area=args.min_area,
            keep_holes=args.keep_holes,
            drop_empty=args.drop_missing,
            keep_by_rect=not args.drop_missing,
        )
        summary.append((fn, total, converted, fixed_by_rect, dropped))
        print(f"[DONE] {fn}: total={total}, converted_from_rle={converted}, "
              f"fixed_by_rect={fixed_by_rect}, dropped={dropped}, out={out_json}")

    print("\n=== Summary ===")
    for fn, total, converted, fixed_by_rect, dropped in summary:
        print(f"{fn}: total={total}, converted_from_rle={converted}, fixed_by_rect={fixed_by_rect}, dropped={dropped}")
    print(f"Output written to: {out_root}")


if __name__ == "__main__":
    main()
