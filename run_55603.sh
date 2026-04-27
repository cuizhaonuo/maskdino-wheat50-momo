#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python main.py \
  -i "/public/home/qwang/CommonAlgorithm/aikenong_dataset/2024年通辽农场玉米数据收集/无人机数据/7.23" \
  -o "/public/home/zncui/datasets/aikenong/2025_tongliao_maize_7.23" \
  -r "/public/home/zncui/pycharm_project/MaskDINO/error_info.xlsx"

CUDA_VISIBLE_DEVICES=0 python main.py \
  -i "/public/home/qwang/CommonAlgorithm/aikenong_dataset/2024年通辽农场玉米数据收集/无人机数据/7.25" \
  -o "/public/home/zncui/datasets/aikenong/2025_tongliao_maize_7.25" \
  -r "/public/home/zncui/pycharm_project/MaskDINO/error_info.xlsx"

CUDA_VISIBLE_DEVICES=0 python main.py \
  -i "/public/home/qwang/CommonAlgorithm/aikenong_dataset/2024年通辽农场玉米数据收集/无人机数据/7.27" \
  -o "/public/home/zncui/datasets/aikenong/2025_tongliao_maize_7.27" \
  -r "/public/home/zncui/pycharm_project/MaskDINO/error_info.xlsx"

CUDA_VISIBLE_DEVICES=0 python main.py \
  -i "/public/home/qwang/CommonAlgorithm/aikenong_dataset/2024年通辽农场玉米数据收集/无人机数据/7.29" \
  -o "/public/home/zncui/datasets/aikenong/2025_tongliao_maize_7.29" \
  -r "/public/home/zncui/pycharm_project/MaskDINO/error_info.xlsx"

