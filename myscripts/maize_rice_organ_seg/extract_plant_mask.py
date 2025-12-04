#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

# ================== 配置 ==================
ROOT_DIR = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset"

# strategy:
#   - "nonwhite": 非白即前景（适合 maize 纯白背景）
#   - "gray": 灰度+阈值（适合 rice 复杂背景）
DATA_CONFIGS = [
    # {
    #     "name": "maize",
    #     "img_dir": os.path.join(ROOT_DIR, "maize"),
    #     "label_dir": os.path.join(ROOT_DIR, "maize_yolo_labels"),
    #     "mask_dir": os.path.join(ROOT_DIR, "maize_mask"),
    #     "overlay_dir": os.path.join(ROOT_DIR, "maize_overlay"),
    #     "strategy": "nonwhite",
    # },
    {
        "name": "rice",
        "img_dir": os.path.join(ROOT_DIR, "rice"),
        "label_dir": os.path.join(ROOT_DIR, "rice_yolo_labels"),
        "mask_dir": os.path.join(ROOT_DIR, "rice_mask"),
        "overlay_dir": os.path.join(ROOT_DIR, "rice_overlay"),
        "strategy": "gray",
    },
]


# ================== 工具函数 ==================
def load_yolo_box(txt_path, img_w, img_h):
    """从 YOLO txt 中读取第一行框 (class cx cy w h)，返回像素级 (x1,y1,x2,y2)"""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"找不到标注文件: {txt_path}")
    with open(txt_path, "r") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"{txt_path} 中没有标注行")
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"{txt_path} 格式错误，期望: class cx cy w h")

    _, cx, cy, bw, bh = map(float, parts[:5])
    cx *= img_w; cy *= img_h
    bw *= img_w; bh *= img_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"YOLO 框计算结果异常，请检查标注: {txt_path}")
    return x1, y1, x2, y2


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def list_images(img_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [n for n in sorted(os.listdir(img_dir)) if os.path.splitext(n)[1].lower() in exts]
    return files


def make_overlay(image_bgr, mask):
    """mask=255 的区域涂成红色"""
    over = image_bgr.copy()
    over[mask == 255] = (0, 0, 255)
    return over


# ========== maize：非白即前景 ==========
def extract_plant_mask_nonwhite(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bg = (s < 15) & (v > 240)      # 白/亮背景
    mask = (~bg).astype(np.uint8) * 255
    return mask


# ========== rice：灰度 + 阈值 ==========
def connected_components_filter(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def extract_plant_mask_gray(roi_bgr):
    """
    只用灰度阈值：CLAHE增强 -> 高斯平滑 -> Otsu 与 反相 Otsu 二选一
    -> 形态学清理 -> 连通域去噪
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 对复杂背景做轻度增强，拉开前景/背景对比
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 手动阈值分割
    _, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY) # 230

    # mask 进行翻转
    mask = cv2.bitwise_not(mask)

    # # 平滑减少噪声，便于 Otsu
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #
    # # 两种阈值都试一下，按前景比例选择更合理的
    # _, m_bin  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    # _, m_inv  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #
    # # 目标通常占比不应过小/过大，这里用 5%~80% 的启发式选择
    # def score(mask):
    #     ratio = mask.mean() / 255.0
    #     target = 0.30  # 期望前景面积比例（经验值）
    #     # 距离越小越好；若极端(过大/过小)再轻度惩罚
    #     penalty = 0.0
    #     if ratio < 0.02 or ratio > 0.90:
    #         penalty = 0.5
    #     return abs(ratio - target) + penalty
    #
    # mask = m_inv if score(m_inv) < score(m_bin) else m_bin
    #
    # # 形态学：小开去噪 + 闭合断裂
    # k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    #
    # # 连通域去小噪点
    # H, W = mask.shape
    # min_area = max(30, int(0.0005 * H * W))
    # mask = connected_components_filter(mask, min_area)
    #
    # # 可选：轻微膨胀补边
    # mask = cv2.dilate(mask, k3, iterations=1)

    return mask


def get_mask_by_strategy(roi_bgr, strategy: str):
    if strategy == "nonwhite":
        return extract_plant_mask_nonwhite(roi_bgr)
    elif strategy == "gray":
        return extract_plant_mask_gray(roi_bgr)
    else:
        raise ValueError(f"未知分割策略: {strategy}")


# ================== 主流程 ==================
def process_split(cfg):
    name = cfg["name"]
    img_dir = cfg["img_dir"]
    label_dir = cfg["label_dir"]
    mask_dir = cfg["mask_dir"]
    overlay_dir = cfg["overlay_dir"]
    strategy = cfg.get("strategy", "nonwhite")

    ensure_dir(mask_dir); ensure_dir(overlay_dir)

    if not os.path.isdir(img_dir):
        print(f"[{name}] 图像目录不存在，跳过: {img_dir}"); return
    if not os.path.isdir(label_dir):
        print(f"[{name}] 标签目录不存在，跳过: {label_dir}"); return

    img_files = list_images(img_dir)
    if not img_files:
        print(f"[{name}] 没有找到图片，目录: {img_dir}"); return

    print(f"\n[{name}] 开始处理（策略: {strategy}），共 {len(img_files)} 张")
    processed = skipped = 0

    for i, img_name in enumerate(img_files, 1):
        img_path = os.path.join(img_dir, img_name)
        base = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, base + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[{name}] 无法读取图片: {img_path}"); skipped += 1; continue
        H, W = img.shape[:2]

        try:
            x1, y1, x2, y2 = load_yolo_box(label_path, W, H)
        except Exception as e:
            print(f"[{name}] 跳过 {img_name}: {e}"); skipped += 1; continue

        roi = img[y1:y2, x1:x2]
        roi_mask = get_mask_by_strategy(roi, strategy)

        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = roi_mask

        cv2.imwrite(os.path.join(mask_dir,   f"{base}_mask.png"), full_mask)
        cv2.imwrite(os.path.join(overlay_dir,f"{base}_overlay.png"), make_overlay(img, full_mask))

        processed += 1
        if i % 50 == 0 or i == len(img_files):
            print(f"[{name}] 进度: {i}/{len(img_files)}  成功 {processed}，跳过 {skipped}")

    print(f"[{name}] 完成。成功: {processed}，跳过: {skipped}")
    print(f"[{name}] Mask: {mask_dir}")
    print(f"[{name}] Overlay: {overlay_dir}")


def main():
    for cfg in DATA_CONFIGS:
        process_split(cfg)


if __name__ == "__main__":
    main()
