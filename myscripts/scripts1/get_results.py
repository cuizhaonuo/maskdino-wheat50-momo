import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 路径配置
# =========================
MASKDINO_ROOT = Path("/public/home/zncui/pycharm_project/MaskDINO/output/normal")
MMDET_ROOT = Path("/public/home/zncui/pycharm_project/mmdetection/work_dirs/zhaoxian")
OUT_DIR = Path("./metric_curve_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 工具函数
# =========================
def safe_float(x):
    if x is None:
        return np.nan
    try:
        v = float(x)
        if math.isnan(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def normalize_percent(v):
    """
    将 0~1 的 mAP 转成 0~100
    -1 和 NaN 视为缺失值
    """
    v = safe_float(v)
    if np.isnan(v):
        return np.nan
    if v == -1.0:
        return np.nan
    if 0.0 <= v <= 1.0:
        return v * 100.0
    return v


def try_read_json_lines(file_path: Path):
    """
    逐行读取 JSON Lines 文件
    """
    records = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    records.append(d)
                except Exception:
                    print(f"[WARN] 非法 JSON，已跳过: {file_path}, line={lineno}")
    except Exception as e:
        print(f"[WARN] 文件读取失败: {file_path}, error={e}")
    return records


def is_maskdino_eval_record(d):
    return ("bbox/AP" in d) or ("segm/AP" in d)


def is_mmdet_eval_record(d):
    return ("coco/bbox_mAP" in d) or ("coco/segm_mAP" in d)


# =========================
# 读取 MaskDINO
# =========================
def collect_maskdino_eval_history(root: Path):
    rows = []

    if not root.exists():
        print(f"[WARN] 路径不存在: {root}")
        return rows

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        metric_file = subdir / "metrics.json"
        if not metric_file.exists():
            continue

        records = try_read_json_lines(metric_file)
        if not records:
            continue

        for d in records:
            if not is_maskdino_eval_record(d):
                continue
            print(d.get("bbox/AP"), ' -- ', d.get('iteration'))
            rows.append({
                "source": "MaskDINO",
                "experiment": subdir.name,
                "file": str(metric_file),

                "bbox/AP": safe_float(d.get("bbox/AP")),
                "bbox/AP50": safe_float(d.get("bbox/AP50")),
                "bbox/AP75": safe_float(d.get("bbox/AP75")),
                "bbox/APs": safe_float(d.get("bbox/APs")),
                "bbox/APm": safe_float(d.get("bbox/APm")),
                "bbox/APl": safe_float(d.get("bbox/APl")),

                "segm/AP": safe_float(d.get("segm/AP")),
                "segm/AP50": safe_float(d.get("segm/AP50")),
                "segm/AP75": safe_float(d.get("segm/AP75")),
                "segm/APs": safe_float(d.get("segm/APs")),
                "segm/APm": safe_float(d.get("segm/APm")),
                "segm/APl": safe_float(d.get("segm/APl")),
            })

    return rows


# =========================
# 读取 MMDetection
# =========================
def collect_mmdet_eval_history(root: Path):
    rows = []

    if not root.exists():
        print(f"[WARN] 路径不存在: {root}")
        return rows

    for scalars_file in root.rglob("scalars.json"):
        if scalars_file.parent.name != "vis_data":
            continue

        records = try_read_json_lines(scalars_file)
        if not records:
            continue

        rel_parts = scalars_file.relative_to(root).parts
        if len(rel_parts) >= 4:
            exp_name = rel_parts[0]
            run_name = rel_parts[1]
            experiment = f"{exp_name}/{run_name}"
        else:
            experiment = str(scalars_file.relative_to(root))

        for d in records:
            if not is_mmdet_eval_record(d):
                continue

            rows.append({
                "source": "MMDetection",
                "experiment": experiment,
                "file": str(scalars_file),

                "bbox/AP": normalize_percent(d.get("coco/bbox_mAP")),
                "bbox/AP50": normalize_percent(d.get("coco/bbox_mAP_50")),
                "bbox/AP75": normalize_percent(d.get("coco/bbox_mAP_75")),
                "bbox/APs": normalize_percent(d.get("coco/bbox_mAP_s")),
                "bbox/APm": normalize_percent(d.get("coco/bbox_mAP_m")),
                "bbox/APl": normalize_percent(d.get("coco/bbox_mAP_l")),

                "segm/AP": normalize_percent(d.get("coco/segm_mAP")),
                "segm/AP50": normalize_percent(d.get("coco/segm_mAP_50")),
                "segm/AP75": normalize_percent(d.get("coco/segm_mAP_75")),
                "segm/APs": normalize_percent(d.get("coco/segm_mAP_s")),
                "segm/APm": normalize_percent(d.get("coco/segm_mAP_m")),
                "segm/APl": normalize_percent(d.get("coco/segm_mAP_l")),
            })

    return rows


# =========================
# 绘图函数
# =========================
def plot_metric_group_curves(df, metric_cols, title, out_file):
    """
    2x3 子图，每个子图一个指标
    横轴为每个模型的第几次评估：1, 2, 3, ...
    """
    if df.empty:
        print(f"[WARN] 没有数据可画图: {title}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    exp_list = sorted(df["experiment_full"].unique())

    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]

        for exp in exp_list:
            sub = df[df["experiment_full"] == exp].copy()
            sub = sub.sort_values("eval_id")

            x = sub["eval_id"].values
            y = sub[metric].values

            valid = ~np.isnan(y)
            x = x[valid]
            y = y[valid]

            if len(x) == 0:
                continue

            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=exp)

        ax.set_title(metric)
        ax.set_xlabel("Evaluation Index")
        ax.set_ylabel("Metric (%)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 保存图片: {out_file}")


def plot_single_metric_curves(df, metric, title, out_file):
    """
    单独画一个指标，横轴为第几次评估
    """
    if df.empty:
        print(f"[WARN] 没有数据可画图: {title}")
        return

    plt.figure(figsize=(12, 7))

    exp_list = sorted(df["experiment_full"].unique())

    for exp in exp_list:
        sub = df[df["experiment_full"] == exp].copy()
        sub = sub.sort_values("eval_id")

        x = sub["eval_id"].values
        y = sub[metric].values

        valid = ~np.isnan(y)
        x = x[valid]
        y = y[valid]

        if len(x) == 0:
            continue

        plt.plot(x, y, marker="o", linewidth=2, markersize=4, label=exp)

    plt.xlabel("Evaluation Index")
    plt.ylabel(f"{metric} (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"[INFO] 保存图片: {out_file}")


# =========================
# 主函数
# =========================
def main():
    maskdino_rows = collect_maskdino_eval_history(MASKDINO_ROOT)
    mmdet_rows = collect_mmdet_eval_history(MMDET_ROOT)

    all_rows = maskdino_rows + mmdet_rows
    if not all_rows:
        print("[ERROR] 没有找到任何评估指标")
        return

    df = pd.DataFrame(all_rows)

    # 完整实验名，避免重名
    df["experiment_full"] = df["source"] + "/" + df["experiment"]

    # 保证原始顺序下，每个实验的评估编号为 1,2,3,...
    df["eval_id"] = df.groupby("experiment_full").cumcount() + 1

    bbox_cols = ["bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/APs", "bbox/APm", "bbox/APl"]
    segm_cols = ["segm/AP", "segm/AP50", "segm/AP75", "segm/APs", "segm/APm", "segm/APl"]

    # 保存总表
    df.to_csv(OUT_DIR / "all_eval_history.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] 保存 CSV: {OUT_DIR / 'all_eval_history.csv'}")

    # 分组大图
    plot_metric_group_curves(
        df=df,
        metric_cols=bbox_cols,
        title="BBox Metrics Curves",
        out_file=OUT_DIR / "bbox_curves.png"
    )

    plot_metric_group_curves(
        df=df,
        metric_cols=segm_cols,
        title="Segm Metrics Curves",
        out_file=OUT_DIR / "segm_curves.png"
    )

    # 核心指标单图
    plot_single_metric_curves(
        df=df,
        metric="bbox/AP",
        title="BBox AP Curve",
        out_file=OUT_DIR / "bbox_AP_curve.png"
    )

    plot_single_metric_curves(
        df=df,
        metric="bbox/AP50",
        title="BBox AP50 Curve",
        out_file=OUT_DIR / "bbox_AP50_curve.png"
    )

    plot_single_metric_curves(
        df=df,
        metric="segm/AP",
        title="Segm AP Curve",
        out_file=OUT_DIR / "segm_AP_curve.png"
    )

    plot_single_metric_curves(
        df=df,
        metric="segm/AP50",
        title="Segm AP50 Curve",
        out_file=OUT_DIR / "segm_AP50_curve.png"
    )

    print(f"[INFO] 全部完成，输出目录: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()