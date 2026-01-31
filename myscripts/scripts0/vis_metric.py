import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

mode = "ablation" # "experiment" or "ablation"

if mode == "ablation":
    output_base_dir = "/public/home/zncui/pycharm_project/MaskDINO/output"
    output_dirs = [
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-one-to-one-300",
        # # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-ddq-initial-layer-wise"
        # # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-matching-prob-iou-cls-bias",
        # # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-matching-prob-iou-cls-bias-v2"
        # # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many",
        # # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many-k6",
        # # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many-v2-dac",
        # # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-dn-two-stage-q300/",
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-dn-two-stage-q900/",
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many-v2-dac-align-no-rank-weight-dn/",
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many-v2-dac-align-no-rank-weight-dn-x4/",
        # "experiment1/dn100_o2o600_o2m450_k3_iou0.8",
        # "experiment1/dn100_o2o900_o2m1350_k3_iou0.8",
        "maskdino_swinl-time-10-19-20-00",
        "experiment-longsteps/bs1_dn100_o2o900_o2m1350_k3_iou0.8",
        "experiment-longsteps/bs1_dn100_o2o900_o2m1350_k3_iou0.8_copypaste_pro0.6_max8_overlap0.85/",
    ]
    labels = [
        "base",
        # "base-dn-two-stage",
        # "v1",
        # "v1-sep",
        # "e1",
        # "e2",
        "v1-longsteps",
        "v2-longsteps",
    ]

if mode == "experiment":
    output_base_dir = "/public/home/zncui/pycharm_project/MaskDINO/output/experiment0"
    output_dirs = os.listdir(output_base_dir)
    labels = output_dirs

bbox_segm_metrics = []
total_losses_list = []
for output_dir, label in zip(output_dirs, labels):
    json_file = os.path.join(output_base_dir, output_dir, "metrics.json")
    with open(json_file, "r") as f:
        metrics = []
        total_losses = []
        for line in f:
            metric = json.loads(line)
            if "bbox/AP" in metric and "segm/AP" in metric:
                metrics.append(metric)
            if "total_loss" in metric:
                total_losses.append(metric["total_loss"])
        bbox_AP = [m["bbox/AP"] for m in metrics]
        segm_AP = [m["segm/AP"] for m in metrics]
        bbox_segm_metrics.append((bbox_AP, segm_AP, label))
        total_losses_list.append(total_losses)

max_bbox_AP = -np.inf
max_bbox_label = ''
max_segm_AP = -np.inf
max_segm_label = ''
TOPK = 4
for (bbox_AP, segm_AP, label), total_losses in zip(bbox_segm_metrics, total_losses_list):
    bbox_AP = np.asarray(bbox_AP)
    segm_AP = np.asarray(segm_AP)
    k_bbox_AP_index = np.argsort(bbox_AP)[-TOPK:]
    k_bbox_AP = bbox_AP[k_bbox_AP_index].sum()
    if k_bbox_AP > max_bbox_AP:
        max_bbox_AP = k_bbox_AP
        max_bbox_label = label
    k_segm_AP_index = np.argsort(segm_AP)[-TOPK:]
    k_segm_AP = segm_AP[k_segm_AP_index].sum()
    if k_segm_AP > max_segm_AP:
        max_segm_AP = k_segm_AP
        max_segm_label = label

print(f"Max BBox AP(sum top{TOPK}): {max_bbox_AP} by {max_bbox_label}")
print(f"Max Segm AP(sum top{TOPK}): {max_segm_AP} by {max_segm_label}")

def smooth(y, k=5):
    return pd.Series(y).rolling(k, center=False).mean()

if mode == "ablation":
    rows = 1
    cols = 3
    fig, ax = plt.subplots(rows, cols, figsize=(10, 10))
    for (bbox_AP, segm_AP, label), total_losses in zip(bbox_segm_metrics, total_losses_list):
        ax[0].plot(bbox_AP[:-3], label=label)
        ax[1].plot(segm_AP[:-3], label=label)
        ax[2].plot(total_losses[:-20], label=label)
    # 为每个图例绘制虚线，虚线高度为最后一个数值
    for idx, ((bbox_AP, segm_AP, label), total_losses) in enumerate(zip(bbox_segm_metrics, total_losses_list)):
        if len(bbox_AP) > 0:
            y_val = bbox_AP[-4]
            ax[0].axhline(y=y_val, color='r', linestyle='--', alpha=0.5)
            ax[0].text(len(bbox_AP) - 3, y_val, f'{y_val:.2f}', color='r', fontsize=12, va='bottom')
        if len(segm_AP) > 0:
            y_val = segm_AP[-4]
            ax[1].axhline(y=y_val, color='r', linestyle='--', alpha=0.5)
            ax[1].text(len(segm_AP) - 3, y_val, f'{y_val:.2f}', color='r', fontsize=12, va='bottom')
        # if len(total_losses) > 0:
        #     y_val = total_losses[-21]
        #     ax[2].axhline(y=y_val, linestyle='--', alpha=0.5)
        #     ax[2].text(len(total_losses) - 20, y_val, f'{y_val:.2f}', fontsize=8, va='bottom')ne(y=total_losses[-21], linestyle='--', alpha=0.5)

    ax[0].set_ylabel("BBox AP")
    ax[1].set_ylabel("Segm AP")
    ax[2].set_ylabel("Total Loss")
    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.show()

if mode == "experiment":
    fig, ax = plt.subplots(1, 2, figsize=(16, 12))

    for (bbox_AP, segm_AP, label), total_losses in zip(bbox_segm_metrics, total_losses_list):
        ax[0].plot(bbox_AP, label=label)
        ax[1].plot(segm_AP, label=label)
        # ax[2].plot(total_losses, label=label)

    ax[0].set_ylabel("BBox AP")
    ax[1].set_ylabel("Segm AP")
    # ax[2].set_ylabel("Total Loss")
    # ax[2].set_xlim(0, 125)
    # ax[2].set_ylim(0, 500)

    # ⭐ 关键：统一 legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=1,  # 根据 label 数量调整
        fontsize=9,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])  # 给 legend 留空间
    plt.show()


