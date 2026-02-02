import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

base_dir = "/public/home/zncui/pycharm_project/MaskDINO/myscripts"

input_dirs = [
    "baseline-search",
    "baseline-x-search",
    "baseline-xx-search",
]
row_labels = [
    "baseline",
    "v1",
    "v2"
]

source_dir = "/public/home/zncui/datasets/zhaoxian/wheats_rle_yolo"
split = "val"

# ----------------------------
# 1) total_num
# ----------------------------
total_num = 0
label_dir = os.path.join(source_dir, "labels", split)
for f in os.listdir(label_dir):
    if f.endswith(".txt"):
        with open(os.path.join(label_dir, f)) as fh:
            total_num += len(fh.readlines())

# ----------------------------
# 2) 读数据 & 收集 iou_type
# ----------------------------
dfs = []
all_iou_types = set()

for d in input_dirs:
    df = pd.read_csv(os.path.join(base_dir, d, "grid_search_results.csv"))
    df["score_thr"] = df["score_thr"].round(4)
    df["nms_thr"] = df["nms_thr"].round(4)
    df["error"] = df["sum_diff"] / total_num
    dfs.append(df)
    all_iou_types.update(df["iou_type"].unique())

iou_types = sorted(all_iou_types)
n_rows = len(dfs)
n_cols = len(iou_types)

# ----------------------------
# 3) 统一 color range + size range（推荐：便于对比）
# ----------------------------
all_errors = np.concatenate([df["error"].to_numpy() for df in dfs])
vmin, vmax = float(np.nanmin(all_errors)), float(np.nanmax(all_errors))

norm = Normalize(vmin=vmin, vmax=vmax)

# 圆点大小范围（可调）
s_min, s_max = 30, 700  # 单位是 scatter 的面积 points^2

def size_map(val):
    """把 error 映射到圆点面积：小值小点，大值大点（线性映射）"""
    if vmax == vmin:
        return (s_min + s_max) / 2.0
    t = (val - vmin) / (vmax - vmin)  # 0~1
    return s_min + t * (s_max - s_min)

# ----------------------------
# 4) 画圆圈方格图（bubble grid）
# ----------------------------
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5.5 * n_cols, 4.8 * n_rows),
    constrained_layout=True
)
axes = np.atleast_2d(axes)

last_sc = None

for r, (df, row_label) in enumerate(zip(dfs, row_labels)):
    for c, iou_type in enumerate(iou_types):
        ax = axes[r, c]
        group = df[df["iou_type"] == iou_type].copy()

        if group.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        # 用 pivot_table 处理重复组合
        pivot = group.pivot_table(
            index="nms_thr",
            columns="score_thr",
            values="error",
            aggfunc="mean"
        ).sort_index(axis=0).sort_index(axis=1)

        x_vals = pivot.columns.to_numpy()
        y_vals = pivot.index.to_numpy()
        Z = pivot.values  # shape: (len(y), len(x))

        # 将网格展开成散点
        X, Y = np.meshgrid(x_vals, y_vals)
        x_flat = X.ravel()
        y_flat = Y.ravel()
        z_flat = Z.ravel()

        # 去掉 nan
        mask = np.isfinite(z_flat)
        x_flat, y_flat, z_flat = x_flat[mask], y_flat[mask], z_flat[mask]

        sizes = np.array([size_map(v) for v in z_flat])

        # 画圆点：颜色深浅 + 大小共同表达 error
        sc = ax.scatter(
            x_flat, y_flat,
            s=sizes,
            c=z_flat,
            cmap="viridis",
            norm=norm,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.9
        )
        last_sc = sc

        # 轴刻度（显示真实阈值）
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{v:.4f}" for v in x_vals], rotation=45, ha="right")
        ax.set_yticks(y_vals)
        ax.set_yticklabels([f"{v:.4f}" for v in y_vals])

        ax.set_title(iou_type)
        ax.set_xlabel("score_thr")
        ax.set_ylabel("nms_thr")

        # 网格线让“方格”更明显（可选）
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.25)

        # 行标签
        if c == 0:
            ax.text(
                -0.35, 0.5, row_label,
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=12
            )

# ----------------------------
# 5) 共享 colorbar
# ----------------------------
cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist())
cbar.set_label("error")

# ----------------------------
# 6) (可选) 加一个“大小图例”
# ----------------------------
# 选几个代表值做 size legend
legend_vals = np.linspace(vmin, vmax, 3)
handles = [
    plt.Line2D([], [], linestyle="",
               marker="o",
               markersize=np.sqrt(size_map(v)),  # markersize 用半径近似（sqrt面积）
               markerfacecolor=plt.cm.viridis(norm(v)),
               markeredgecolor="k",
               label=f"{v:.4f}")
    for v in legend_vals
]
# 把图例放在右上角（你也可以改位置）
axes[0, -1].legend(handles=handles, title="error (size/color)", loc="upper right", frameon=True)

plt.show()
