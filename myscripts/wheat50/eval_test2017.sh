#!/bin/bash
# 直接执行：在 wheat_cat2_coco_test（coco_2cat/test2017）上评估 wheat50_r50_cat2_default 模型。
set -e

PROJ=/public/home/zncui/pycharm_project/maskdino-wheat50
EXP_DIR=${PROJ}/output_v2/wheat50_r50_cat2_default
CONFIG_FILE=${EXP_DIR}/config.yaml
MODEL_FILE=${EXP_DIR}/model_final.pth
OUTPUT_DIR=${PROJ}/output_v2/eval_test2017

cd ${PROJ}
export PYTHONPATH=${PROJ}:${PYTHONPATH}

python train_net.py --eval_only \
    --config-file ${CONFIG_FILE} \
    MODEL.WEIGHTS ${MODEL_FILE} \
    OUTPUT_DIR ${OUTPUT_DIR} \
    DATASETS.TEST "('wheat_cat2_coco_test',)"
