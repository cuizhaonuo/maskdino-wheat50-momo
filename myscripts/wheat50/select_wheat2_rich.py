#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 train2017 中挑选 wheat_2 实例多、且所有 wheat_2 mask 都不贴边界的图片。

筛选规则：
  1. 该图所有 wheat_2 实例的 polygon 坐标都距图像四边 > MARGIN 像素（不贴边界）；
  2. 按 wheat_2 实例数降序取前 N（默认 20），并列时按文件名排序保证可复现。

输出：
  <dataset>/vis/selected/<原文件名>   （带轮廓+点标注的可视化图）
  <dataset>/vis/selected/selected.json （选中清单：文件名 / image_id / wheat_2 数）
"""
import json
import os
import shutil
from collections import defaultdict

DATASET = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points"
SPLIT = "train2017"
CAT_WHEAT2 = 2
MARGIN = 2
TOPN = 20

ANN = os.path.join(DATASET, "annotations", f"instances_{SPLIT}.json")
VIS_DIR = os.path.join(DATASET, "vis", SPLIT)
OUT_DIR = os.path.join(DATASET, "vis", "selected")


def poly_extent(ann):
    xs, ys = [], []
    for poly in ann["segmentation"]:
        xs += poly[0::2]
        ys += poly[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    coco = json.load(open(ANN))
    imgs = {i["id"]: i for i in coco["images"]}
    w2 = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == CAT_WHEAT2:
            w2[a["image_id"]].append(a)

    cands = []
    for iid, anns in w2.items():
        im = imgs[iid]
        W, H = im["width"], im["height"]
        if any(
            poly_extent(a)[0] <= MARGIN
            or poly_extent(a)[1] <= MARGIN
            or poly_extent(a)[2] >= W - MARGIN
            or poly_extent(a)[3] >= H - MARGIN
            for a in anns
        ):
            continue
        cands.append((len(anns), im["file_name"], iid))

    cands.sort(key=lambda t: (-t[0], t[1]))
    picked = cands[:TOPN]

    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for n, fn, iid in picked:
        src = os.path.join(VIS_DIR, fn)
        if not os.path.exists(src):
            print(f"[skip] {fn} (no vis image)")
            continue
        shutil.copy2(src, os.path.join(OUT_DIR, fn))
        records.append({"file_name": fn, "image_id": iid, "n_wheat2": n})

    with open(os.path.join(OUT_DIR, "selected.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(f"[done] {len(records)} images -> {OUT_DIR}")
    for r in records:
        print(f"  wheat_2={r['n_wheat2']:2d}  {r['file_name']}")


if __name__ == "__main__":
    main()
