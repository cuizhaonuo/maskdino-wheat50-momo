#!/bin/bash

GPU_ID=1
PROJECT_ROOT=/public/home/zncui/pycharm_project/MaskDINO
CONFIG_DIR=${PROJECT_ROOT}/configs/coco/instance-segmentation/ablation_final_noweight_maskdino
OUTPUT_ROOT=${PROJECT_ROOT}/output/ablation

CONFIGS=(
    final_noweight_maskdino_ablation_1_o2m_q1350_k3.yaml
    final_noweight_maskdino_ablation_1_o2m_q1350_k3_sep-branch.yaml
    final_noweight_maskdino_ablation_2_ddq_iou0.8_matcher-nms.yaml
    final_noweight_maskdino_ablation_3_layer-wise-iou_align-loss.yaml
)

echo "========================================"
echo "Running ablation_final_noweight_maskdino"
echo "========================================"

for cfg in "${CONFIGS[@]}"
do

    exp_name="${cfg%.yaml}"
    config_path="${CONFIG_DIR}/${cfg}"

    output_dir="${OUTPUT_ROOT}/${exp_name}"
    log_path="${OUTPUT_ROOT}/${exp_name}.log"

    mkdir -p "${output_dir}"

    echo ""
    echo "----------------------------------------"
    echo "Starting experiment: ${exp_name}"
    echo "Config: ${config_path}"
    echo "----------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PROJECT_ROOT}/train_net.py \
        --config-file "${config_path}" \
        OUTPUT_DIR "${output_dir}" \
        > "${log_path}" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ Finished: ${exp_name}"
    else
        echo "❌ Failed: ${exp_name}"
        echo "Skipping and continuing..."
    fi

done

echo ""
echo "All experiments in ablation_final_noweight_maskdino finished."