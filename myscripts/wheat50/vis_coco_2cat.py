#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接执行：可视化 coco_2cat 下 train2017 / val2017 / test2017 的 COCO 实例标注，
把分割多边形按类别上色画到原图上，输出到 coco_2cat_vis/<同名子目录>。

颜色（BGR）：
  wheat   (category_id=1) -> 绿色，细线
  wheat_2 (category_id=2) -> 红色，粗线（少数类，突出显示）

用法：
  python tools/wheat50/vis_coco_2cat.py                 # 三个 split 全跑
  python tools/wheat50/vis_coco_2cat.py test2017        # 只跑指定 split
"""
import os
import sys
import json
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

COCO_ROOT = "/public/home/zncui/datasets/wheat50/coco_2cat"
VIS_ROOT  = "/public/home/zncui/datasets/wheat50/coco_2cat_vis"
SPLITS    = ["train2017", "val2017", "test2017"]

# category_id -> (BGR color, line thickness)
CAT_STYLE = {
    1: ((0, 0, 255), 1),    # wheat   -> 红
    2: ((255, 0, 0), 2),    # wheat_2 -> 蓝（加粗）
}
DRAW_BBOX = False  # 如需同时画检测框可改 True


def polys_to_contours(seg):
    """COCO polygon segmentation -> list of cv2 contours (int32 Nx1x2)."""
    contours = []
    if isinstance(seg, list):
        for poly in seg:
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                contours.append(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
    return contours


def vis_split(split):
    ann_path = os.path.join(COCO_ROOT, "annotations", f"instances_{split}.json")
    img_dir = os.path.join(COCO_ROOT, split)
    out_dir = os.path.join(VIS_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    with open(ann_path, "r") as f:
        data = json.load(f)

    id2img = {im["id"]: im for im in data["images"]}
    anns_by_img = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    n_ok, n_miss = 0, 0
    for img_id, im in tqdm(id2img.items(), desc=split, total=len(id2img)):
        src = os.path.join(img_dir, im["file_name"])
        canvas = cv2.imread(src)
        if canvas is None:
            n_miss += 1
            continue
        for a in anns_by_img.get(img_id, []):
            color, thick = CAT_STYLE.get(a["category_id"], ((255, 255, 255), 1))
            contours = polys_to_contours(a.get("segmentation", []))
            if contours:
                cv2.drawContours(canvas, contours, -1, color, thick)
            if DRAW_BBOX and "bbox" in a:
                x, y, w, h = [int(round(v)) for v in a["bbox"]]
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        cv2.imwrite(os.path.join(out_dir, im["file_name"]), canvas)
        n_ok += 1

    print(f"[{split}] saved {n_ok} images -> {out_dir} (missing src: {n_miss})")


def main():
    splits = sys.argv[1:] if len(sys.argv) > 1 else SPLITS
    for split in splits:
        vis_split(split)
    print(f"[done] visualized splits: {splits}")


if __name__ == "__main__":
    main()
