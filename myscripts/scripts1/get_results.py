import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 路径配置
# =========================
MASKDINO_ROOT = Path("/public/home/zncui/pycharm_project/MaskDINO/output/experiment-longsteps")
MMDET_ROOT = Path("/public/home/zncui/pycharm_project/mmdetection/work_dirs/zhaoxian")
OUT_DIR = Path("./metric_curve_plots3")
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


def round_metric_columns(df, decimals=4):
    metric_cols = [
        "bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/APs", "bbox/APm", "bbox/APl",
        "segm/AP", "segm/AP50", "segm/AP75", "segm/APs", "segm/APm", "segm/APl",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = df[col].round(decimals)
    return df


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

        eval_idx = 0
        for d in records:
            if not is_maskdino_eval_record(d):
                continue

            eval_idx += 1

            rows.append({
                "source": "MaskDINO",
                "experiment": subdir.name,
                "file": str(metric_file),
                "iteration": safe_float(d.get("iteration")),
                "epoch": safe_float(d.get("epoch")),
                "eval_order_in_file": eval_idx,

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

        eval_idx = 0
        for d in records:
            if not is_mmdet_eval_record(d):
                continue

            eval_idx += 1

            rows.append({
                "source": "MMDetection",
                "experiment": experiment,
                "file": str(scalars_file),
                "step": safe_float(d.get("step")),
                "epoch": safe_float(d.get("epoch")),
                "iter": safe_float(d.get("iter")),
                "eval_order_in_file": eval_idx,

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
# 提取每个模型最后一次评估结果
# =========================
def build_final_eval_table(df):
    """
    从全部评估历史中，提取每个 experiment_full 的最后一次评估结果
    """
    if df.empty:
        return pd.DataFrame()

    work_df = df.copy()
    work_df = work_df.sort_values(["experiment_full", "eval_id"])

    final_df = work_df.groupby("experiment_full", as_index=False).tail(1).copy()

    preferred_cols = [
        "source",
        "experiment",
        "experiment_full",
        "eval_id",
        "epoch",
        "iteration",
        "iter",
        "step",
        "bbox/AP",
        "bbox/AP50",
        "bbox/AP75",
        "bbox/APs",
        "bbox/APm",
        "bbox/APl",
        "segm/AP",
        "segm/AP50",
        "segm/AP75",
        "segm/APs",
        "segm/APm",
        "segm/APl",
        "file",
    ]

    final_cols = [c for c in preferred_cols if c in final_df.columns] + [
        c for c in final_df.columns if c not in preferred_cols
    ]
    final_df = final_df[final_cols]

    sort_cols = []
    ascending = []
    if "segm/AP" in final_df.columns:
        sort_cols.append("segm/AP")
        ascending.append(False)
    if "bbox/AP" in final_df.columns:
        sort_cols.append("bbox/AP")
        ascending.append(False)

    if sort_cols:
        final_df = final_df.sort_values(sort_cols, ascending=ascending)

    final_df = round_metric_columns(final_df, decimals=4)
    final_df.reset_index(drop=True, inplace=True)
    return final_df


# =========================
# 提取每个模型最佳 segm/AP
# 规则：
# 1. 每个模型内部找 segm/AP 最大
# 2. 若 segm/AP 相同，取 eval_id 最大
# 3. 每个模型只保留 1 条
# =========================
def build_best_segm_ap_table(df):
    """
    对每个模型（experiment_full）分别找出 segm/AP 最高的一次评估结果。
    如果同一模型内部 segm/AP 并列最高，则选择 eval_id 最大的那一条。
    最终每个模型只保留 1 条记录。
    """
    if df.empty or "segm/AP" not in df.columns:
        return pd.DataFrame()

    work_df = df.copy()
    work_df = work_df[~work_df["segm/AP"].isna()].copy()

    if work_df.empty:
        return pd.DataFrame()

    result_rows = []

    for exp_name, sub_df in work_df.groupby("experiment_full"):
        sub_df = sub_df.copy()

        best_segm_ap = sub_df["segm/AP"].max()
        candidate_df = sub_df[sub_df["segm/AP"] == best_segm_ap].copy()

        best_row = candidate_df.sort_values("eval_id", ascending=True).tail(1)
        result_rows.append(best_row)

    best_df = pd.concat(result_rows, axis=0, ignore_index=True)

    preferred_cols = [
        "source",
        "experiment",
        "experiment_full",
        "eval_id",
        "epoch",
        "iteration",
        "iter",
        "step",
        "bbox/AP",
        "bbox/AP50",
        "bbox/AP75",
        "bbox/APs",
        "bbox/APm",
        "bbox/APl",
        "segm/AP",
        "segm/AP50",
        "segm/AP75",
        "segm/APs",
        "segm/APm",
        "segm/APl",
        "file",
    ]

    final_cols = [c for c in preferred_cols if c in best_df.columns] + [
        c for c in best_df.columns if c not in preferred_cols
    ]
    best_df = best_df[final_cols]

    best_df = best_df.sort_values(
        by=["segm/AP", "bbox/AP", "experiment_full"],
        ascending=[False, False, True]
    )

    best_df = round_metric_columns(best_df, decimals=4)
    best_df.reset_index(drop=True, inplace=True)
    return best_df


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

    # 保证每个实验评估编号为 1,2,3,...
    df["eval_id"] = df.groupby("experiment_full").cumcount() + 1

    bbox_cols = ["bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/APs", "bbox/APm", "bbox/APl"]
    segm_cols = ["segm/AP", "segm/AP50", "segm/AP75", "segm/APs", "segm/APm", "segm/APl"]

    df = round_metric_columns(df, decimals=4)

    # 保存总历史表
    history_csv = OUT_DIR / "all_eval_history.csv"
    history_xlsx = OUT_DIR / "all_eval_history.xlsx"
    df.to_csv(history_csv, index=False, encoding="utf-8-sig")
    df.to_excel(history_xlsx, index=False)
    print(f"[INFO] 保存 CSV: {history_csv}")
    print(f"[INFO] 保存 Excel: {history_xlsx}")

    # 每个模型最后一次评估结果
    final_df = build_final_eval_table(df)

    final_csv = OUT_DIR / "final_eval_results.csv"
    final_xlsx = OUT_DIR / "final_eval_results.xlsx"
    final_df.to_csv(final_csv, index=False, encoding="utf-8-sig")
    final_df.to_excel(final_xlsx, index=False)

    print(f"[INFO] 保存最后一次评估结果 CSV: {final_csv}")
    print(f"[INFO] 保存最后一次评估结果 Excel: {final_xlsx}")

    print("\n========== 每个模型最后一次评估结果 ==========")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(final_df)

    # 每个模型最佳 segm/AP，若 segm/AP 相同则取 eval_id 最大
    best_segm_df = build_best_segm_ap_table(df)

    if not best_segm_df.empty:
        best_csv = OUT_DIR / "best_segm_ap_result.csv"
        best_xlsx = OUT_DIR / "best_segm_ap_result.xlsx"

        best_segm_df.to_csv(best_csv, index=False, encoding="utf-8-sig")
        best_segm_df.to_excel(best_xlsx, index=False)

        print(f"[INFO] 保存每个模型最佳 segm/AP 结果 CSV: {best_csv}")
        print(f"[INFO] 保存每个模型最佳 segm/AP 结果 Excel: {best_xlsx}")

        print("\n========== 每个模型最佳 segm/AP 结果（若并列取 eval_id 最大）==========")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(best_segm_df)
    else:
        print("[WARN] 没有找到有效的 segm/AP 结果，未生成 best_segm_ap_result 文件")

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