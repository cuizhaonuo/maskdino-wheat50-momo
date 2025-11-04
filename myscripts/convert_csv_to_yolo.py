#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random
from pathlib import Path

import pandas as pd  # 若环境没有，可用内置 csv 替代

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_list_evenly(items, seed=2025):
    """将列表随机打乱后按 1:1 均分；若为奇数，多出的一个分到 test。"""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    mid = len(items) // 2
    val = items[:mid]
    test = items[mid:]
    return val, test

def random_box_norm(rng: random.Random):
    """
    生成一条 YOLO 归一化框 (x_center, y_center, w, h)，均在 [0,1]。
    控制 w,h 在 [0.02, 0.25] 之间，避免过小/过大。
    """
    w = rng.uniform(0.02, 0.25)
    h = rng.uniform(0.02, 0.25)
    x = rng.uniform(w / 2.0, 1.0 - w / 2.0)
    y = rng.uniform(h / 2.0, 1.0 - h / 2.0)
    return x, y, w, h

def write_yolo_label(file_path: Path, n_objects: int, seed_base: int = 2025):
    """
    写入 YOLO txt 标签文件。每一行格式: "0 x y w h"
    若 n_objects <= 0，则写入空文件（合法）。
    """
    rng = random.Random(seed_base + hash(file_path.stem) % (10**6))
    lines = []
    for _ in range(max(0, int(n_objects))):
        x, y, w, h = random_box_norm(rng)
        lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    file_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    # 路径配置
    ROOT = Path("/public/home/zncui/datasets/xinxiang")
    CSV_PATH = ROOT / "real.csv"
    SRC_IMG_DIR = ROOT / "150Areas"   # 源图所在目录

    # 目标 YOLO 目录（在 xinxiang 下直接创建 images/ 与 labels/）
    IMAGES_DIR = ROOT / 'yolo' / "images"
    LABELS_DIR = ROOT / 'yolo' / "labels"

    # 创建目标子目录
    for sub in ["val", "test"]:
        ensure_dir(IMAGES_DIR / sub)
        ensure_dir(LABELS_DIR / sub)

    # 读取 CSV
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 列检查
    required_cols = {"filename", "real_counts"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必须列: {missing}，当前列有: {list(df.columns)}")

    # 仅保留 filename 与 real_counts
    df = df[["filename", "real_counts"]].copy()

    # 组合出绝对图像路径，过滤不存在的文件并给出警告
    abs_paths = []
    missing_files = []
    for fn in df["filename"]:
        p = SRC_IMG_DIR / str(fn)
        if p.exists():
            abs_paths.append(p)
        else:
            missing_files.append(str(p))

    if missing_files:
        print(f"[警告] 下列图片在源目录中不存在（将被跳过，共 {len(missing_files)} 张）：")
        for s in missing_files[:20]:
            print("  -", s)
        if len(missing_files) > 20:
            print(f"  ... 以及 {len(missing_files) - 20} 个更多缺失文件")

    # 过滤掉缺失的记录
    df = df[df["filename"].apply(lambda x: (SRC_IMG_DIR / str(x)).exists())].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("可用的图片记录为空，请检查 CSV 与 150Areas 目录。")

    # 划分 val / test (1:1)
    filenames = df["filename"].tolist()
    val_list, test_list = split_list_evenly(filenames, seed=2025)

    # 方便快速查询 real_counts
    rc_map = dict(zip(df["filename"], df["real_counts"]))

    # 复制图片 + 生成标签
    def process_split(split_name, name_list):
        img_dst_dir = IMAGES_DIR / split_name
        lbl_dst_dir = LABELS_DIR / split_name
        copied, labeled = 0, 0

        for fn in name_list:
            src = SRC_IMG_DIR / str(fn)
            if not src.exists():
                # 理论上之前已过滤，这里再兜底
                print(f"[跳过] 源图不存在：{src}")
                continue

            # 复制图片
            dst_img = img_dst_dir / src.name
            try:
                shutil.copy2(src, dst_img)
                copied += 1
            except Exception as e:
                print(f"[错误] 复制失败 {src} -> {dst_img}，原因：{e}")
                continue

            # 写入标签
            dst_lbl = lbl_dst_dir / (src.stem + ".txt")
            n_obj = int(rc_map.get(str(fn), 0))
            try:
                write_yolo_label(dst_lbl, n_obj)
                labeled += 1
            except Exception as e:
                print(f"[错误] 写标签失败 {dst_lbl}，原因：{e}")

        print(f"[{split_name}] 复制图片: {copied}，生成标签: {labeled}")

    process_split("val", val_list)
    process_split("test", test_list)

    # 汇总信息
    print("\n=== 完成 ===")
    print(f"目标 images 目录: {IMAGES_DIR}")
    print(f"目标 labels 目录: {LABELS_DIR}")
    print(f"val 数量: {len(val_list)}")
    print(f"test 数量: {len(test_list)}")

if __name__ == "__main__":
    main()
