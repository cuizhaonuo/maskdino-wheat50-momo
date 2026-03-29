#!/bin/bash

# 根目录（模型输出目录）
BASE_DIR=/public/home/zncui/pycharm_project/MaskDINO/output/experiment-longsteps

# 输出目录根路径
SAVE_ROOT=/public/home/zncui/pycharm_project/MaskDINO/myscripts/scripts1/experiment-longsteps-eval-gwhd2021_test

# 进入项目目录（根据你实际路径调整）
cd /public/home/zncui/pycharm_project/MaskDINO

# 遍历所有实验子目录
for EXP_DIR in ${BASE_DIR}/*; do
    if [ -d "$EXP_DIR" ]; then

        CONFIG_FILE=${EXP_DIR}/config.yaml
        MODEL_FILE=${EXP_DIR}/model_final.pth

        # 获取实验名称（文件夹名）
        EXP_NAME=$(basename ${EXP_DIR})

        # 设置输出路径
        OUTPUT_DIR=${SAVE_ROOT}/${EXP_NAME}

        # 判断文件是否存在
        if [ -f "$CONFIG_FILE" ] && [ -f "$MODEL_FILE" ]; then
            echo "Running eval for: ${EXP_NAME}"

            python train_net.py --eval_only \
                --config-file ${CONFIG_FILE} \
                MODEL.WEIGHTS ${MODEL_FILE} \
                OUTPUT_DIR ${OUTPUT_DIR} \
                DATASETS.TEST "('gwhd_test',)"

        else
            echo "Skip ${EXP_NAME}, missing config or model."
        fi
    fi
done