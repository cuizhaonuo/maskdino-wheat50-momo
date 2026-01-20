import os
import json
import pandas as pd
from matplotlib import pyplot as plt

mode = "ablation" # "experiment" or "ablation"

if mode == "ablation":
    output_base_dir = "/public/home/zncui/pycharm_project/MaskDINO/output"
    output_dirs = [
        "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-one-to-one-300",
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-ddq-initial-layer-wise"
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-matching-prob-iou-cls-bias",
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-matching-prob-iou-cls-bias-v2"
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many",
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many-k6",
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many-v2-dac",
        # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-dn-two-stage-q300/",
        "maskdino_swinl-time-10-19-20-00-v1-patch-2000-dn-two-stage-q900/",
        "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many-v2-dac-align-no-rank-weight-dn/",
        "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-two-stage-all-layer-iou-cls-1.0-iou-5.0-no-gated-attn-ddq-one2many-v2-dac-align-no-rank-weight-dn-x4/",
        "experiment0/two-stage-cls-1.0-ddq-0.8-align-layer-wise-iou-25.0-matcher-nms-no-sep-branch-dn-100-o2m-1350-k-6-q300",
    ]
    labels = [
        "base",
        "base-dn-two-stage",
        "v1",
        "v1-sep",
        "v1-align-matcher"
    ]

if mode == "experiment":
    output_base_dir = "/public/home/zncui/pycharm_project/MaskDINO/output/experiment1"
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

def smooth(y, k=5):
    return pd.Series(y).rolling(k, center=False).mean()

if mode == "ablation":
    rows = 1
    cols = 3
    fig, ax = plt.subplots(rows, cols, figsize=(10, 10))
    for (bbox_AP, segm_AP, label), total_losses in zip(bbox_segm_metrics, total_losses_list):
        ax[0].plot(bbox_AP, label=label)
        ax[1].plot(segm_AP, label=label)
        ax[2].plot(total_losses, label=label)
    ax[0].set_ylabel("BBox AP")
    ax[1].set_ylabel("Segm AP")
    ax[2].set_ylabel("Total Loss")
    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.show()

if mode == "experiment":
    fig, ax = plt.subplots(1, 3, figsize=(16, 12))

    for (bbox_AP, segm_AP, label), total_losses in zip(bbox_segm_metrics, total_losses_list):
        ax[0].plot(bbox_AP, label=label)
        ax[1].plot(segm_AP, label=label)
        ax[2].plot(total_losses, label=label)

    ax[0].set_ylabel("BBox AP")
    ax[1].set_ylabel("Segm AP")
    ax[2].set_ylabel("Total Loss")

    # ⭐ 关键：统一 legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,  # 根据 label 数量调整
        fontsize=9,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])  # 给 legend 留空间
    plt.show()


