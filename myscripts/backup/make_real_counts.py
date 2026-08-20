#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

def main():
    # 输入路径
    pred_csv = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/swinl_xinxiang_0.6_0.01/test_pred_vs_gt.csv"
    blue_csv = "/public/home/zncui/datasets/xinxiang/34images/blue_out/counts.csv"

    # 输出路径
    out_dir = "/public/home/zncui/datasets/xinxiang"
    out_csv = os.path.join(out_dir, "real.csv")
    os.makedirs(out_dir, exist_ok=True)

    # 读取 CSV
    pred_df = pd.read_csv(pred_csv)
    blue_df = pd.read_csv(blue_csv)
    blue_df['filename'] = blue_df['filename'].str.replace(r'.jpg$', '.JPG', regex=True)

    # 只保留所需列并规范列名
    needed_pred_cols = ["filename", "ins_counts", "pred_counts"]
    needed_blue_cols = ["filename", "blue_counts"]

    missing_pred = [c for c in needed_pred_cols if c not in pred_df.columns]
    missing_blue = [c for c in needed_blue_cols if c not in blue_df.columns]
    if missing_pred:
        raise ValueError(f"预测文件缺少列: {missing_pred}")
    if missing_blue:
        raise ValueError(f"蓝点文件缺少列: {missing_blue}")

    pred_df = pred_df[needed_pred_cols].copy()
    blue_df = blue_df[needed_blue_cols].copy()

    # 将计数列转为数值（遇到非法值设为 NaN）
    pred_df["pred_counts"] = pd.to_numeric(pred_df["pred_counts"], errors="coerce")
    blue_df["blue_counts"] = pd.to_numeric(blue_df["blue_counts"], errors="coerce")

    # 以第二个文件(blue_df)为主，去第一个文件(pred_df)里找相同 file_name
    merged = blue_df.merge(pred_df[["filename", "pred_counts"]],
                           on="filename", how="left")

    # 只保留两个文件都出现的文件名（如果想保留没有匹配的行，可去掉这行）
    merged = merged.dropna(subset=["pred_counts"])

    # 缺失或非法转为 0（一般不该出现，防御性处理）
    merged["pred_counts"] = merged["pred_counts"].fillna(0)
    merged["blue_counts"] = merged["blue_counts"].fillna(0)

    # 计算 real_counts = pred_counts + blue_counts
    merged["real_counts"] = merged["pred_counts"] + merged["blue_counts"]

    # 输出所需列并转为整数（如需保留小数可去掉 astype(int)）
    out_df = merged[["filename", "pred_counts", "blue_counts", "real_counts"]].copy()
    # 若计数应为整数：
    for c in ["pred_counts", "blue_counts", "real_counts"]:
        # 先填充再转 int，防止浮点误差
        out_df[c] = out_df[c].fillna(0).round().astype(int)

    # 保存
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    # 打印一些信息
    print(f"匹配到的文件数量: {len(out_df)}")
    print(f"已保存到: {out_csv}")

if __name__ == "__main__":
    main()
