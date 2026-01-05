import os
import re
import json
import pandas as pd
from matplotlib import pyplot as plt

output_base_dir = "/public/home/zncui/pycharm_project/MaskDINO/output"
output_dirs = [
    "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-one-to-one-300",
    "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0",
    "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-matching-prob-iou",
    # "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-matching-prob-iou-cls-bias-v2"
]
labels = [
    "baseline",
    "p1",
    "p2",
    # "v2"
]

# prefix_dir = "maskdino_swinl-time-10-19-20-00-v1-patch-2000-no-dn-learn-tgt-all-layer-iou-cls-1.0-iou-5.0-matching-prob-iou"
# output_dirs = [out_dir for out_dir in os.listdir(output_base_dir) if prefix_dir in out_dir]
# labels = []
# for out_dir in output_dirs:
#     match = re.search(f"{prefix_dir}-(.+)$", out_dir)
#     if match:
#         label = match.group(1)
#         labels.append(label)
#     else:
#         labels.append("p")

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

rows = 1
cols = 3
fig, ax = plt.subplots(rows, cols)
# eval_period = 267
# eval_steps = list(range(eval_period, 2000, eval_period)) + [2000]
# train_steps = list(range(20, 2000, 20)) + [2000]
for (bbox_AP, segm_AP, label), total_losses in zip(bbox_segm_metrics, total_losses_list):
    ax[0].plot(bbox_AP, label=label)
    ax[0].set_ylabel("BBox AP")
    ax[1].plot(segm_AP, label=label)
    ax[1].set_ylabel("Segm AP")
    # train_steps = list(range(0, 2000, len(total_losses)))
    ax[2].plot(total_losses, label=label)
    ax[2].set_ylabel("Total Loss")
for a in ax:
    a.legend()
plt.tight_layout()
plt.show()

