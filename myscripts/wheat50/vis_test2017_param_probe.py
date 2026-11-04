#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 test2017 已选出的 300 张 tile 上，用指定的 score_thr / nms_thr 重新推理并画轮廓图，
用于与 build_test2017_p30.py 的正式结果（0.4 / 0.372）逐张对比。不改写 test2017 本身。

用法：
  CUDA_VISIBLE_DEVICES=1 /public/home/zncui/anaconda3/envs/MaskDINO/bin/python \
      myscripts/wheat50/vis_test2017_param_probe.py --score_thr 0.3 --nms_iou_thr 0.35
"""
import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

PROJ = "/public/home/zncui/pycharm_project/maskdino-wheat50"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "myscripts"))
sys.path.insert(0, os.path.join(PROJ, "myscripts", "wheat50"))

import build_test2017_p30 as B  # noqa: E402
from myscripts.infer.seg_infer import (  # noqa: E402
    DatasetConfig, ModelConfig, SlidingWindowConfig, SearchSpace, OutputOptions,
)

DATASET = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points"
IMG_DIR = os.path.join(DATASET, "test2017")

COLOR_WHEAT = (255, 60, 0)      # BGR 蓝
COLOR_WHEAT2 = (0, 40, 255)     # BGR 红
COLOR_POINT = (0, 255, 255)     # BGR 黄


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_thr", type=float, required=True)
    ap.add_argument("--nms_iou_thr", type=float, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(
        DATASET, "vis", f"test2017_s{args.score_thr}_nms{args.nms_iou_thr}")
    os.makedirs(out_dir, exist_ok=True)

    runner = B.P30SegInference(
        DatasetConfig(yolo_root=""),
        ModelConfig(project_root=PROJ, config_rel=B.CONFIG, checkpoint_rel=B.WEIGHTS,
                    device=B.DEVICE, imports=("maskdino",)),
        SlidingWindowConfig(tile=B.TILE, overlap=B.OVERLAP),
        SearchSpace(base_keep_score=B.BASE_KEEP),
        OutputOptions(vis=False, excel=False, draw_scatter=False),
    )
    print("==> Loading model ...")
    runner.load_model()

    names = sorted(os.listdir(IMG_DIR))
    tot = defaultdict(int)
    for name in tqdm(names, desc="infer"):
        im = cv2.imread(os.path.join(IMG_DIR, name))
        if im is None:
            continue
        props = B.tile_proposals(runner, im)
        dets = [p for p in props if p.score >= args.score_thr]
        dets = B.mask_nms_merge_points(dets, args.nms_iou_thr, B.IOU_TYPE) if dets else []

        tot["wheat"] += sum(1 for d in dets if d.label == 0)
        tot["wheat_2"] += sum(1 for d in dets if d.label == 1)
        tot["points"] += sum(len(d.points) for d in dets)

        for d in dets:
            color = COLOR_WHEAT2 if d.label == 1 else COLOR_WHEAT
            m8 = d.mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if len(c) < 3:
                    continue
                pts = c + np.array([[[d.left, d.top]]], dtype=c.dtype)
                cv2.polylines(im, [pts], True, color, 2, cv2.LINE_AA)
            for x, y in d.points:
                cv2.circle(im, (int(round(x)), int(round(y))), 3, COLOR_POINT, -1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, name), im)

    print(f"[stats] score_thr={args.score_thr} nms={args.nms_iou_thr}: "
          f"wheat={tot['wheat']} wheat_2={tot['wheat_2']} points={tot['points']} "
          f"(正式版 0.4/0.372: wheat=16074 wheat_2=1635 points=30366)")
    print(f"[done] {len(names)} images -> {out_dir}")


if __name__ == "__main__":
    main()
