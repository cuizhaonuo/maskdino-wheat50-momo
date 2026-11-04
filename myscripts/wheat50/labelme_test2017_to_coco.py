#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把人工修订后的 labelme_test2017-completed 回写为 COCO instances_test2017.json。

分组规则（与 coco_test2017_to_labelme.py 的导出约定互逆）：
  - 有 group_id：同组 polygon 合成一个实例(多段),组内 spikelet 点归该实例；
  - group_id=None 的 polygon：人工新增的实例，各自独立；
  - group_id=None 的 spikelet 点：人工新增的点，按点包含关系归入 wheat_2 实例
    (被多个包含时取面积最小者；无包含者则落入最近 wheat_2 质心 150px 内，再否则丢弃并计数)；
  - 只含 spikelet 点而无 polygon 的组（polygon 被删但点留着）：丢弃并计数；
  - wheat 实例组内的 spikelet 点：schema 规定 wheat 不带点，丢弃并计数。

用法：
  /public/home/zncui/anaconda3/envs/MaskDINO/bin/python myscripts/wheat50/labelme_test2017_to_coco.py
"""
import json
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np

DATASET = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points"
LABELME_DIR = os.path.join(DATASET, "labelme_test2017-completed", "labelme_test2017")
CUR_JSON = os.path.join(DATASET, "annotations", "instances_test2017.json")
BACKUP_JSON = os.path.join(DATASET, "annotations", "instances_test2017_pseudo.json")

CATEGORIES = [
    {"id": 1, "name": "wheat", "supercategory": "none"},
    {"id": 2, "name": "wheat_2", "supercategory": "none"},
]
LABEL_TO_CATID = {"wheat": 1, "wheat_2": 2}
NEAREST_MAX_DIST = 150.0  # 无包含关系时,新增点归入最近 wheat_2 质心的最大距离(px)


def poly_area(points):
    p = np.asarray(points, dtype=np.float64)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


def main():
    old = json.load(open(CUR_JSON))
    shutil.copy2(CUR_JSON, BACKUP_JSON)
    print(f"[backup] pseudo-label json -> {BACKUP_JSON}")

    # 沿用旧的 image 列表与 id(file_name 对齐),保证下游引用稳定
    images = old["images"]
    stats = defaultdict(int)
    annotations = []
    ann_id = 1

    for img in images:
        lm_path = os.path.join(LABELME_DIR, os.path.splitext(img["file_name"])[0] + ".json")
        lm = json.load(open(lm_path))
        groups = defaultdict(lambda: {"polys": [], "pts": []})
        for s in lm["shapes"]:
            gid = s.get("group_id")
            key = gid if gid is not None else ("__none__", id(s))
            if s["shape_type"] == "polygon":
                groups[key]["polys"].append(s)
            elif s["shape_type"] == "point":
                groups[key]["pts"].append(s)

        # 第一遍:polygon 组成实例
        instances = []  # {cat_id, segmentation, pts_raw(list of [x,y])}
        orphan_pt_groups = []  # 只有点的组(带 group_id)
        loose_pts = []         # group_id=None 的新增点
        for key, g in groups.items():
            if g["polys"]:
                labels = [p["label"] for p in g["polys"]]
                if len(set(labels)) > 1:
                    stats["mixed_label_group"] += 1
                cat = LABEL_TO_CATID[labels[0]]
                seg = []
                for p in g["polys"]:
                    flat = [float(v) for xy in p["points"] for v in xy]
                    if len(flat) >= 6:
                        seg.append(flat)
                if not seg:
                    stats["empty_poly_group"] += 1
                    continue
                instances.append({"cat_id": cat, "segmentation": seg,
                                  "pts_raw": [[float(x), float(y)] for q in g["pts"] for x, y in q["points"]]})
            else:
                if isinstance(key, tuple):      # group_id=None 的新增点
                    loose_pts.extend([float(x), float(y)] for q in g["pts"] for x, y in q["points"])
                else:                            # 有组但 polygon 被删
                    orphan_pt_groups.append(g)

        # 第二遍:新增点归入 wheat_2 实例
        w2 = [inst for inst in instances if inst["cat_id"] == 2]
        w2_contours = [[np.array(p["points"], dtype=np.float32) for p in g["polys"]]
                       for key, g in groups.items() if g["polys"]
                       and LABEL_TO_CATID[g["polys"][0]["label"]] == 2]
        w2_cent = []
        for inst in w2:
            xs = [v for poly in inst["segmentation"] for v in poly[0::2]]
            ys = [v for poly in inst["segmentation"] for v in poly[1::2]]
            w2_cent.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        for x, y in loose_pts:
            best, best_area = None, None
            for inst, contours in zip(w2, w2_contours):
                if any(cv2.pointPolygonTest(c, (x, y), False) >= 0 for c in contours):
                    area = sum(poly_area(np.array(inst["segmentation"][k], dtype=np.float64).reshape(-1, 2))
                               for k in range(len(inst["segmentation"])))
                    if best is None or area < best_area:
                        best, best_area = inst, area
            if best is None and w2_cent:
                d2 = [((cx - x) ** 2 + (cy - y) ** 2, i) for i, (cx, cy) in enumerate(w2_cent)]
                d2min, imin = min(d2)
                if d2min ** 0.5 <= NEAREST_MAX_DIST:
                    best = w2[imin]
                    stats["loose_pt_by_nearest"] += 1
            if best is None:
                stats["loose_pt_dropped"] += 1
            else:
                best["pts_raw"].append([x, y])
                stats["loose_pt_attached"] += 1

        for g in orphan_pt_groups:
            stats["orphan_pts_dropped"] += len(g["pts"])

        # 第三遍:写 COCO anns
        h, w = img["height"], img["width"]
        for inst in instances:
            mask = np.zeros((h, w), np.uint8)
            for poly in inst["segmentation"]:
                pts = np.array(poly, np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], 1)
            xs = [v for poly in inst["segmentation"] for v in poly[0::2]]
            ys = [v for poly in inst["segmentation"] for v in poly[1::2]]
            if inst["cat_id"] == 1 and inst["pts_raw"]:
                stats["wheat_pts_dropped"] += len(inst["pts_raw"])
            annotations.append({
                "id": ann_id, "image_id": img["id"], "category_id": inst["cat_id"],
                "segmentation": inst["segmentation"],
                "area": float(mask.sum()),
                "bbox": [float(min(xs)), float(min(ys)),
                         float(max(xs) - min(xs)), float(max(ys) - min(ys))],
                "iscrowd": 0,
                "points": inst["pts_raw"] if inst["cat_id"] == 2 else [],
            })
            ann_id += 1

    coco = {"info": {"description": "test2017 manually corrected from p30 pseudo labels "
                                    "(labelme round-trip, 2026-10-09)"},
            "licenses": [], "images": images,
            "annotations": annotations, "categories": CATEGORIES}
    with open(CUR_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    from collections import Counter
    cats = Counter(a["category_id"] for a in annotations)
    n_pts = sum(len(a["points"]) for a in annotations)
    print(f"[done] {len(images)} images, {len(annotations)} anns "
          f"(wheat={cats[1]} wheat_2={cats[2]}), spikelet points={n_pts}")
    print(f"[stats] {dict(stats)}")


if __name__ == "__main__":
    main()
