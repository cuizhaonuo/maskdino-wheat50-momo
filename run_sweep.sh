#!/usr/bin/env bash
set -eu pipefail

GPU_ID=2

CONFIG="/public/home/zncui/pycharm_project/MaskDINO/output/maskdino_swinl-time-10-19-20-00/config_v1_patch_weight.yaml"
ROOT_OUT="/public/home/zncui/pycharm_project/MaskDINO/output/experiment1"

NUM_QUERIES_DN=(100 200)
NUM_QUERIES_O2O=(300 600 900)
NUM_QUERIES_O2M=(450 900 1350)
NUM_TARGETS_K=(6)
NUM_DDQ_IOU_THRESHOLDS=(0.8)

for dn in "${NUM_QUERIES_DN[@]}"; do
  for o2o in "${NUM_QUERIES_O2O[@]}"; do
    for o2m in "${NUM_QUERIES_O2M[@]}"; do
      for k in "${NUM_TARGETS_K[@]}"; do
        for iou in "${NUM_DDQ_IOU_THRESHOLDS[@]}"; do

          tag="dn${dn}_o2o${o2o}_o2m${o2m}_k${k}_iou${iou}"
          OUTPUT_DIR="${ROOT_OUT}/${tag}"

          mkdir -p "${ROOT_OUT}"

          echo "Running: ${tag}"

          CUDA_VISIBLE_DEVICES="${GPU_ID}" python train_net.py --config-file "${CONFIG}" \
            OUTPUT_DIR "${OUTPUT_DIR}" \
            MODEL.MaskDINO.DN_NUM "${dn}" \
            MODEL.MaskDINO.NUM_OBJECT_QUERIES "${o2o}" \
            MODEL.MaskDINO.NUM_OBJECT_QUERIES_O2M "${o2m}" \
            MODEL.MaskDINO.NUM_TARGETS_K "${k}" \
            MODEL.MaskDINO.DDQ_IOU_THRESHOLD "${iou}"

        done
      done
    done
  done
done
