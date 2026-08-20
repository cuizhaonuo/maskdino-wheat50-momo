#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import csv
import glob
import argparse
import numpy as np
from typing import Tuple, List


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_blue_mask_hsv(img_bgr: np.ndarray,
                        hsv_low: Tuple[int, int, int] = (100, 80, 50),
                        hsv_high: Tuple[int, int, int] = (140, 255, 255)) -> np.ndarray:
    """
    基于 HSV 的蓝色阈值，得到二值掩码（包含网格与蓝点）
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    return mask


def remove_largest_component(blue_mask: np.ndarray) -> np.ndarray:
    """
    去掉面积最大的连通域（认为是整张图的蓝色网格）
    返回：去网格后的掩码
    """
    # connected components：0 是背景
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)

    if num_labels <= 1:
        return blue_mask.copy()

    # 找到面积最大的非背景标签
    areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
    max_idx = np.argmax(areas) + 1        # 补上被排除的背景索引
    out = blue_mask.copy()
    out[labels == max_idx] = 0
    return out


def filter_small_round_blobs(mask: np.ndarray,
                             min_area: int = 8,
                             max_area: int = 400,
                             min_circularity: float = 0.45) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    可选的稳健过滤：仅保留面积合适、近圆的连通域（避免偶发噪声）
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    refined = np.zeros_like(mask)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < min_circularity:
            continue
        kept.append(c)
        cv2.drawContours(refined, [c], -1, 255, thickness=-1)

    return refined, kept


def paint_yellow(img_bgr: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
    """
    将通过筛选的蓝点区域替换为黄色 (BGR: 0,255,255)
    """
    out = img_bgr.copy()
    if not contours:
        return out
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    out[mask > 0] = (0, 255, 255)
    return out


def process_image(img_path: str,
                  hsv_low=(100, 80, 50),
                  hsv_high=(140, 255, 255),
                  min_area: int = 8,
                  max_area: int = 400,
                  min_circularity: float = 0.45,
                  debug_dir: str = "") -> Tuple[int, np.ndarray]:
    """
    核心流程：蓝色掩码 -> 移除最大连通域(网格) -> 过滤点 -> 计数与改色
    返回：(蓝点数量, 改色后的图像)
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0, None

    blue_mask = build_blue_mask_hsv(img, hsv_low, hsv_high)
    no_grid_mask = remove_largest_component(blue_mask)
    dots_mask, contours = filter_small_round_blobs(
        no_grid_mask, min_area=min_area, max_area=max_area, min_circularity=min_circularity
    )
    count = len(contours)
    painted = paint_yellow(img, contours)

    if debug_dir:
        ensure_dir(debug_dir)
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(debug_dir, f"{base}_blue_mask.png"), blue_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_no_grid.png"), no_grid_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_dots_mask.png"), dots_mask)

    return count, painted


def main():
    parser = argparse.ArgumentParser(description="Extract blue dots by removing largest connected component (grid).")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出改色图片目录")
    parser.add_argument("--csv", type=str, required=True, help="计数 CSV 路径")
    parser.add_argument("--debug_dir", type=str, default="", help="可选：导出中间掩码")

    # 阈值与过滤参数，可按需微调
    parser.add_argument("--h_low", type=int, default=95)
    parser.add_argument("--s_low", type=int, default=55)
    parser.add_argument("--v_low", type=int, default=50)
    parser.add_argument("--h_high", type=int, default=150)
    parser.add_argument("--s_high", type=int, default=255)
    parser.add_argument("--v_high", type=int, default=255)
    parser.add_argument("--min_area", type=int, default=8)
    parser.add_argument("--max_area", type=int, default=400)
    parser.add_argument("--min_circularity", type=float, default=0.45)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    if args.debug_dir:
        ensure_dir(args.debug_dir)
    ensure_dir(os.path.dirname(args.csv))

    hsv_low = (args.h_low, args.s_low, args.v_low)
    hsv_high = (args.h_high, args.s_high, args.v_high)

    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG")
    img_paths = []
    for ext in image_exts:
        img_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"[WARN] 在 {args.input_dir} 未找到图片。")
        return

    rows = [["filename", "blue_counts"]]
    total = 0

    for p in img_paths:
        cnt, painted = process_image(
            p, hsv_low=hsv_low, hsv_high=hsv_high,
            min_area=args.min_area, max_area=args.max_area,
            min_circularity=args.min_circularity,
            debug_dir=args.debug_dir
        )
        rows.append([os.path.basename(p), cnt])
        total += cnt
        if painted is not None:
            cv2.imwrite(os.path.join(args.output_dir, os.path.basename(p)), painted)
        print(f"{os.path.basename(p)} -> 蓝点数量: {cnt}")

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"完成。共处理 {len(img_paths)} 张图片，蓝点总数 = {total}")
    print(f"结果图片目录: {args.output_dir}")
    print(f"计数 CSV: {args.csv}")
    if args.debug_dir:
        print(f"调试掩码目录: {args.debug_dir}")


if __name__ == "__main__":
    main()
