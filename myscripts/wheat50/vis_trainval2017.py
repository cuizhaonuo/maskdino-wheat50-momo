#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化 coco_2cat_with_points 的 train2017 / val2017 标注：
- wheat_2 实例：绿色轮廓
- wheat   实例：蓝色轮廓
- 小穗点：红色实心圆点
输出到 <dataset>/vis/{train2017,val2017}/，文件名与原图一致。

用法：
  /public/home/zncui/anaconda3/envs/MaskDINO/bin/python myscripts/wheat50/vis_trainval2017.py [train2017 val2017]
"""
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

DATASET = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points"

COLOR_WHEAT = (255, 0, 0)       # BGR 蓝
COLOR_WHEAT2 = (0, 255, 0)      # BGR 绿
COLOR_POINT = (0, 0, 255)       # BGR 红

CAT_WHEAT2 = 2


def vis_split(split):
    coco_json = os.path.join(DATASET, "annotations", f"instances_{split}.json")
    img_dir = os.path.join(DATASET, split)
    out_dir = os.path.join(DATASET, "vis", split)
    os.makedirs(out_dir, exist_ok=True)

    coco = json.load(open(coco_json))
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    for img in tqdm(coco["images"], desc=f"vis {split}"):
        im = cv2.imread(os.path.join(img_dir, img["file_name"]))
        if im is None:
            print(f"[skip] {img['file_name']}")
            continue
        for a in anns_by_img.get(img["id"], []):
            color = COLOR_WHEAT2 if a["category_id"] == CAT_WHEAT2 else COLOR_WHEAT
            for poly in a["segmentation"]:
                pts = np.array(poly, np.int32).reshape(-1, 1, 2)
                if len(pts) < 3:
                    continue
                cv2.polylines(im, [pts], True, color, 2, cv2.LINE_AA)
            for x, y in a.get("points", []):
                cv2.circle(im, (int(round(x)), int(round(y))), 3, COLOR_POINT, -1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, img["file_name"]), im)

    print(f"[done] {split}: {len(coco['images'])} images -> {out_dir}")


def main():
    splits = sys.argv[1:] or ["train2017", "val2017"]
    for s in splits:
        vis_split(s)


if __name__ == "__main__":
    main()
