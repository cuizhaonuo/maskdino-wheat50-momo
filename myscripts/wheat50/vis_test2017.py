#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化 coco_2cat_with_points/test2017 的（伪）标注：
- wheat   实例：蓝色轮廓（不填充，方便观察原图）
- wheat_2 实例：红色轮廓，小穗点画黄色圆点
输出到 <dataset>/vis/test2017/，文件名与 test2017 一致。

用法：
  /public/home/zncui/anaconda3/envs/MaskDINO/bin/python myscripts/wheat50/vis_test2017.py
"""
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

DATASET = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points"
COCO_JSON = os.path.join(DATASET, "annotations", "instances_test2017.json")
IMG_DIR = os.path.join(DATASET, "test2017")
OUT_DIR = os.path.join(DATASET, "vis", "test2017")

COLOR_WHEAT = (255, 60, 0)      # BGR 蓝
COLOR_WHEAT2 = (0, 40, 255)     # BGR 红
COLOR_POINT = (0, 255, 255)     # BGR 黄


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coco = json.load(open(COCO_JSON))
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    for img in tqdm(coco["images"], desc="vis test2017"):
        im = cv2.imread(os.path.join(IMG_DIR, img["file_name"]))
        if im is None:
            print(f"[skip] {img['file_name']}")
            continue
        for a in anns_by_img.get(img["id"], []):
            color = COLOR_WHEAT2 if a["category_id"] == 2 else COLOR_WHEAT
            for poly in a["segmentation"]:
                pts = np.array(poly, np.int32).reshape(-1, 1, 2)
                if len(pts) < 3:
                    continue
                cv2.polylines(im, [pts], True, color, 2, cv2.LINE_AA)
            for x, y in a.get("points", []):
                cv2.circle(im, (int(round(x)), int(round(y))), 3, COLOR_POINT, -1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT_DIR, img["file_name"]), im)

    print(f"[done] {len(coco['images'])} images -> {OUT_DIR}")


if __name__ == "__main__":
    main()
