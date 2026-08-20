import json
import math
import matplotlib.pyplot as plt


metrics_file = "/public/home/zncui/pycharm_project/MaskDINO/output_v2/wheat50_r50_cat2_default/metrics.json"


def read_metrics(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print("跳过无法解析的行：", line[:100])

    return records


def collect(records, key):
    xs, ys = [], []

    for r in records:
        if "iteration" in r and key in r:
            xs.append(r["iteration"])
            ys.append(r[key])

    return xs, ys


def plot_metrics_group(records, keys, title, ylabel, ncols=3, marker=False):
    """一张图中包含多个子图，按类别分组展示"""
    valid_keys = []
    for key in keys:
        x, y = collect(records, key)
        if len(x) > 0:
            valid_keys.append((key, x, y))
        else:
            print(f"跳过 {key}：没有找到数据")

    if not valid_keys:
        print(f"分组 {title} 没有可用的数据")
        return

    n = len(valid_keys)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    # 统一 axes 为一维数组方便遍历
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (key, x, y) in enumerate(valid_keys):
        ax = axes[i]
        if marker:
            ax.plot(x, y, marker="o", markersize=2, linewidth=1)
        else:
            ax.plot(x, y, linewidth=1)

        ax.set_title(key, fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.5)

    # 隐藏多余的空白子图
    for j in range(len(valid_keys), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


records = read_metrics(metrics_file)

print(f"读取到 {len(records)} 条日志")


# =====================
# 1. 训练 loss（6 项，2×3 布局）
# =====================

loss_keys = [
    "total_loss",
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "loss_mask",
    "loss_dice",
]

plot_metrics_group(
    records,
    loss_keys,
    title="Training Loss",
    ylabel="Loss",
    ncols=3,
    marker=False,
)


# =====================
# 2. bbox 验证指标（8 项，3×3 布局）
# =====================

bbox_keys = [
    "bbox/AP",
    "bbox/AP50",
    "bbox/AP75",
    "bbox/APs",
    "bbox/APm",
    "bbox/APl",
    "bbox/AP-wheat",
    "bbox/AP-wheat_2",
]

plot_metrics_group(
    records,
    bbox_keys,
    title="Validation BBox Metrics",
    ylabel="AP",
    ncols=3,
    marker=True,
)


# =====================
# 3. segm 验证指标（8 项，3×3 布局）
# =====================

segm_keys = [
    "segm/AP",
    "segm/AP50",
    "segm/AP75",
    "segm/APs",
    "segm/APm",
    "segm/APl",
    "segm/AP-wheat",
    "segm/AP-wheat_2",
]

plot_metrics_group(
    records,
    segm_keys,
    title="Validation Segm Metrics",
    ylabel="AP",
    ncols=3,
    marker=True,
)


# =====================
# 4. 学习率（单独一张小图）
# =====================

x, y = collect(records, "lr")
if len(x) > 0:
    plt.figure(figsize=(5, 3))
    plt.plot(x, y, linewidth=1)
    plt.title("Learning Rate", fontsize=10)
    plt.xlabel("Iteration", fontsize=9)
    plt.ylabel("LR", fontsize=9)
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.show()