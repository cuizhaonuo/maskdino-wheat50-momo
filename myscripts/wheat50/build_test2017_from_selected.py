#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接执行脚本：从 selected 目录的大图构建 coco_2cat/test2017。

流程：
  1) 把 random_50_from_C/selected 下的大图按 1024x1024 滑窗（overlap=128）切成小图；
  2) 对每个小图用 wheat50_r50_cat2_default 权重推理（MaskDINO，2 类：wheat / wheat_2）；
  3) 选出“含有 wheat_2（contiguous label=1）”的小图，按目标实例数从多到少取 100 张；
  4) 写出 test2017 图片 + COCO 标注 instances_test2017.json（以模型预测作为伪标注）。

用法：直接运行即可
  python tools/wheat50/build_test2017_from_selected.py
"""
import os
import sys

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
COCO_ROOT    = "/public/home/zncui/datasets/wheat50/coco_2cat"
TEST_IMG_DIR = os.path.join(COCO_ROOT, "test2017")
TEST_JSON    = os.path.join(COCO_ROOT, "annotations", "instances_test2017.json")

CONFIG  = os.path.join(PROJ, "output_v2/wheat50_r50_cat2_default/config.yaml")
WEIGHTS = os.path.join(PROJ, "output_v2/wheat50_r50_cat2_default/model_final.pth")
DEVICE  = "cuda:0"

# ---------------- 切分 / 推理参数 ----------------
TILE         = 1024
OVERLAP      = 128
SCORE_THR    = 0.4          # 作为伪标注保留的实例置信度阈值
NMS_IOU_THR  = 0.372
IOU_TYPE     = IOUType.IOU_Min
BASE_KEEP    = 0.05         # 粗筛阈值（收集原始 proposal）

# ---------------- 类别 / 选择参数 ----------------
# MaskDINO 输出的是 contiguous label：0 -> wheat，1 -> wheat_2
SELECT_LABEL = 1            # 选含有 wheat_2 的小图
NUM_SELECT   = 100
# contiguous label -> COCO category_id（与 data_register 的 {1:0, 2:1} 一致）
LABEL_TO_CATID = {0: 1, 1: 2}
CATEGORIES = [
    {"id": 1, "name": "wheat",   "supercategory": "none"},
    {"id": 2, "name": "wheat_2", "supercategory": "none"},
]


def main():
    images = list_images_in_dir(SELECTED_DIR)
    print(f"[info] source big images: {len(images)}")
    for p in images:
        print("   -", os.path.basename(p))

    dataset = DatasetConfig(yolo_root="")  # 本流程不依赖 YOLO 标注
    model = ModelConfig(
        project_root=PROJ,
        config_rel=CONFIG,
        checkpoint_rel=WEIGHTS,
        device=DEVICE,
        imports=("maskdino",),
    )
    sw = SlidingWindowConfig(tile=TILE, overlap=OVERLAP, pad_value=(114, 114, 114))
    search = SearchSpace(base_keep_score=BASE_KEEP)
    outputs = OutputOptions(vis=False, excel=False, draw_scatter=False, out_dir=COCO_ROOT)

    runner = Detectron2SegInference(dataset, model, sw, search, outputs)
    n_img, n_ann = runner.export_coco_tiles(
        image_paths=images,
        tile=TILE, overlap=OVERLAP,
        score_thr=SCORE_THR, nms_iou_thr=NMS_IOU_THR, iou_type=IOU_TYPE,
        select_label=SELECT_LABEL, num_select=NUM_SELECT,
        out_images_dir=TEST_IMG_DIR, out_json=TEST_JSON,
        categories=CATEGORIES, label_to_catid=LABEL_TO_CATID,
    )
    print(f"[done] test2017: {n_img} images, {n_ann} annotations")


if __name__ == "__main__":
    main()
