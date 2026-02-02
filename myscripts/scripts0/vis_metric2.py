import matplotlib.pyplot as plt
import numpy as np

models = ["baseline", "v1", "v2"]
colors = ["tab:blue", "tab:orange", "tab:green"]

metrics = {
    "AP@[0.50:0.95] (all)": [0.550, 0.584, 0.584],
    # "AP@0.50 (all)":        [0.926, 0.923, 0.925],
    # "AP@0.75 (all)":        [0.607, 0.673, 0.669],
    "AP@[0.50:0.95] (small)":[0.377, 0.401, 0.401],
    "AP@[0.50:0.95] (medium)":[0.611, 0.648, 0.647],

    # "AR@1 (all)":           [0.019, 0.020, 0.020],
    # "AR@10 (all)":          [0.178, 0.184, 0.184],
    "AR@[0.50:0.95] (all)":         [0.611, 0.642, 0.640],
    "AR@[0.50:0.95] (small)":       [0.466, 0.504, 0.501],
    "AR@[0.50:0.95] (medium)":      [0.669, 0.698, 0.696],
}

# 2x5 子图（正好 10 个指标）
rows, cols = 2, 3
fig, axes = plt.subplots(rows, cols, figsize=(7, 6))
axes = axes.flatten()

x = np.arange(len(models)) * 0.5
bar_width = 0.3

def set_tight_ylim(ax, vals):
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin

    # 差距很小时也要有可视空间；span 越小，padding 越小但不为 0
    pad = max(0.0005, span * 0.6)  # 你也可以把 0.6 调大来更“放大”

    lower = vmin - pad
    upper = vmax + pad

    # 防止出现负的 y 轴（像 AR@1 这种很小）
    if lower < 0:
        lower = 0
    ax.set_ylim(lower, upper)

def annotate_bars(ax, rects, vals):
    # AR@1 用 3~4 位更合适；其它一般 3 位够用
    span = max(vals) - min(vals)
    fmt = "{:.4f}" if span < 0.01 else "{:.3f}"

    y0, y1 = ax.get_ylim()
    yspan = y1 - y0
    offset = yspan * 0.03  # 标注离柱子顶的距离

    for r, v in zip(rects, vals):
        ax.text(
            r.get_x() + r.get_width() / 2,
            r.get_height() + offset,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=9
        )

for ax, (metric_name, vals) in zip(axes, metrics.items()):
    rects = ax.bar(x, vals, color=colors, width=bar_width)

    ax.set_title(metric_name, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # 先设置 tight y 轴，再标注（标注需要知道 y 轴范围）
    set_tight_ylim(ax, vals)
    annotate_bars(ax, rects, vals)

# 全局图例
fig.legend(models, loc="upper center", ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
