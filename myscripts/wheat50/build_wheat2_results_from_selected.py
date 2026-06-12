#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接执行脚本：从 selected 大图导出「含完整 wheat_2」的小图 + labelme 标注。

流程：
  1) 把 random_50_from_C/selected 下的大图按 1024x1024 滑窗（overlap=128）切成小图；
  2) 对每个小图用 wheat50_r50_cat2_default 权重推理（MaskDINO，2 类：wheat / wheat_2）；
  3) 保留条件：该小图含 wheat_2（contiguous label=1），且其 wheat_2 全部完整
     （没有任何一个 wheat_2 贴 window 边界=被截断）；任一 wheat_2 被截断则丢弃整张小图；
  4) 对保留的小图写出：
       selected/results/<name>.jpg   小图原图
       selected/results/<name>.json  labelme 格式推理结果（含全部实例：wheat + wheat_2）
       selected/viz/<name>.jpg       实例轮廓可视化（wheat=红，wheat_2=蓝）
     一图一 json。

小图命名与 coco_2cat/test2017 一致：<原图stem>__x<X>_y<Y>_w1024_h1024

用法：
  cd <PROJ> && PYTHONPATH=<PROJ> python myscripts/wheat50/build_wheat2_results_from_selected.py
"""
import os
import sys
import json

import cv2
import numpy as np
from tqdm import tqdm

PROJ = "/public/home/zncui/pycharm_project/maskdino-wheat50"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "myscripts"))

from myscripts.seg_infer import (  # noqa: E402
    Detectron2SegInference, DatasetConfig, ModelConfig,
    SlidingWindowConfig, SearchSpace, OutputOptions, IOUType,
    list_images_in_dir,
)

# ---------------- 路径配置 ----------------
SELECTED_DIR = "/public/home/zncui/datasets/wheat50/total_images/random_50_from_C/selected"
RESULTS_DIR  = os.path.join(SELECTED_DIR, "results")
VIZ_DIR      = os.path.join(SELECTED_DIR, "viz")

CONFIG  = os.path.join(PROJ, "output_v2/wheat50_r50_cat2_default/config.yaml")
WEIGHTS = os.path.join(PROJ, "output_v2/wheat50_r50_cat2_default/model_final.pth")
DEVICE  = "cuda:0"

# ---------------- 切分 / 推理参数 ----------------
TILE          = 1024
OVERLAP       = 128
SCORE_THR     = 0.4
NMS_IOU_THR   = 0.372
IOU_TYPE      = IOUType.IOU_Min
BASE_KEEP     = 0.05          # 粗筛阈值（收集原始 proposal）
BORDER_MARGIN = 2            # 贴边容差（px），用于截断判定

# ---------------- 类别参数 ----------------
# MaskDINO 输出 contiguous label：0 -> wheat，1 -> wheat_2
SELECT_LABEL  = 1            # wheat_2
LABEL_TO_NAME = {0: "wheat", 1: "wheat_2"}

# 可视化配色（BGR）：wheat=红(细)，wheat_2=蓝(粗)
VIS_STYLE = {0: ((0, 0, 255), 1), 1: ((255, 0, 0), 2)}

LABELME_VERSION = "5.2.1"


def det_contours(det):
    """det.mask（裁到 bbox 的局部 mask）-> 还原到 tile 局部坐标的轮廓点列表。"""
    m8 = (det.mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        pts = [[float(det.left + p[0][0]), float(det.top + p[0][1])] for p in c]
        polys.append(pts)
    return polys


def dets_to_labelme(dets, image_filename, tile, label_to_name):
    """把 tile 局部 dets 转成标准 labelme 5.x 字典（含全部类别实例）。"""
    shapes = []
    for d in dets:
        name = label_to_name.get(int(d.label), str(int(d.label)))
        for pts in det_contours(d):
            shapes.append({
                "label": name,
                "points": pts,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": tile,
        "imageWidth": tile,
    }


def draw_vis(crop, dets):
    """在 crop 副本上按类别画实例轮廓（只描边）。"""
    vis = crop.copy()
    for d in dets:
        color, thick = VIS_STYLE.get(int(d.label), ((255, 255, 255), 1))
        m8 = (d.mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask 是裁到 bbox 的局部块，画到 vis 的对应 ROI 上
        roi = vis[d.top:d.top + d.mask.shape[0], d.left:d.left + d.mask.shape[1]]
        if roi.shape[0] == d.mask.shape[0] and roi.shape[1] == d.mask.shape[1]:
            cv2.drawContours(roi, contours, -1, color, thick)
    return vis


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    images = list_images_in_dir(SELECTED_DIR)  # 只取顶层 .JPG（results/viz 子目录不会被回扫）
    print(f"[info] source big images: {len(images)}")

    dataset = DatasetConfig(yolo_root="")  # 本流程不依赖 YOLO 标注
    model = ModelConfig(
        project_root=PROJ, config_rel=CONFIG, checkpoint_rel=WEIGHTS,
        device=DEVICE, imports=("maskdino",),
    )
    sw = SlidingWindowConfig(tile=TILE, overlap=OVERLAP, pad_value=(114, 114, 114))
    search = SearchSpace(base_keep_score=BASE_KEEP)
    outputs = OutputOptions(vis=False, excel=False, draw_scatter=False, out_dir=SELECTED_DIR)
    runner = Detectron2SegInference(dataset, model, sw, search, outputs)

    print("==> Loading model ...")
    runner.load_model()

    n_tiles_total = 0
    n_saved = 0
    for img_path in tqdm(images, desc="Big images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[skip] cannot read {img_path}")
            continue
        H, W = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ys, xs = runner._tile_origins(H, W, TILE, OVERLAP)
        for y0 in ys:
            for x0 in xs:
                n_tiles_total += 1
                crop = img[y0:y0 + TILE, x0:x0 + TILE]
                if crop.shape[0] != TILE or crop.shape[1] != TILE:
                    padded = np.full((TILE, TILE, 3), sw.pad_value, dtype=np.uint8)
                    padded[:crop.shape[0], :crop.shape[1]] = crop
                    crop = padded

                props = runner._tile_proposals(crop)
                dets = runner._apply_thresh_nms(props, SCORE_THR, NMS_IOU_THR, IOU_TYPE)

                wheat2 = [d for d in dets if int(d.label) == SELECT_LABEL]
                if not wheat2:
                    continue
                # 任一 wheat_2 贴边（被截断）则丢弃整张小图
                if any(runner._touches_border(d, TILE, BORDER_MARGIN) for d in wheat2):
                    continue

                name = f"{stem}__x{x0}_y{y0}_w{TILE}_h{TILE}"
                img_name = name + ".jpg"
                cv2.imwrite(os.path.join(RESULTS_DIR, img_name), crop)
                with open(os.path.join(RESULTS_DIR, name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(dets_to_labelme(dets, img_name, TILE, LABEL_TO_NAME),
                              f, ensure_ascii=False, indent=2)
                cv2.imwrite(os.path.join(VIZ_DIR, img_name), draw_vis(crop, dets))
                n_saved += 1

    print(f"[done] scanned {n_tiles_total} tiles, saved {n_saved} tiles "
          f"(含完整 wheat_2) -> {RESULTS_DIR} & {VIZ_DIR}")


if __name__ == "__main__":
    main()
