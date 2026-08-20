#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import csv
import glob
import math
import argparse
from typing import List, Tuple

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images_in_dir(d: str) -> List[str]:
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff",
            "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(d, e)))
    files.sort()
    return files

def compute_sliding_positions(H: int, W: int, tile: int, overlap: int) -> Tuple[List[int], List[int]]:
    stride = max(1, tile - overlap)
    y_list = list(range(0, max(1, H - tile + 1), stride))
    if y_list[-1] != max(0, H - tile):
        y_list.append(max(0, H - tile))
    x_list = list(range(0, max(1, W - tile + 1), stride))
    if x_list[-1] != max(0, W - tile):
        x_list.append(max(0, W - tile))
    return y_list, x_list

def crop_and_save(img_path: str,
                  save_root: str,
                  tile: int,
                  overlap: int,
                  pad_to_tile: bool,
                  pad_value: Tuple[int, int, int]):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Failed to read: {img_path}")
        return []

    H, W = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(save_root, base)
    ensure_dir(out_dir)

    ys, xs = compute_sliding_positions(H, W, tile, overlap)
    records = []  # for manifest

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            crop = img[y0:y1, x0:x1]

            if pad_to_tile:
                # 固定大小：tile x tile，用 pad_value 填充
                canvas = (pad_value if isinstance(pad_value, tuple) else (114,114,114))
                tile_img = (cv2.merge([ ]))  # just to hint types to readers
                pad = cv2.copyMakeBorder(
                    crop,
                    top=0, bottom=tile - crop.shape[0],
                    left=0, right=tile - crop.shape[1],
                    borderType=cv2.BORDER_CONSTANT,
                    value=pad_value
                )
                out_img = pad
            else:
                # 保持原始边缘尺寸（不填充）
                out_img = crop

            out_name = f"{base}_y{y0}_x{x0}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, out_img)

            records.append({
                "orig_image": os.path.basename(img_path),
                "crop_image": os.path.relpath(out_path, save_root),
                "x": x0, "y": y0,
                "h": out_img.shape[0],  # 高
                "w": out_img.shape[1],  # 宽
                "x_end": x0 + min(tile, W - x0),
                "y_end": y0 + min(tile, H - y0),
            })

    print(f"[OK] {img_path} -> {len(records)} tiles at {out_dir}")
    return records

def main():
    parser = argparse.ArgumentParser(description="Crop large images into overlapping tiles.")
    parser.add_argument("--target_images_dir", type=str,
                        default="/public/home/zncui/datasets/xinxiang/51images/ori",
                        help="输入的大图目录")
    parser.add_argument("--save_images_dir", type=str,
                        default="/public/home/zncui/datasets/xinxiang/cropped_51images",
                        help="输出的小图根目录（按大图名分子目录）")
    parser.add_argument("--tile", type=int, default=1024, help="滑窗边长")
    parser.add_argument("--overlap", type=int, default=256, help="滑窗重叠")
    parser.add_argument("--pad_to_tile", action="store_true",
                        help="是否把边缘裁块填充到固定 tile 大小")
    parser.add_argument("--pad_b", type=int, default=114, help="填充色 B 通道")
    parser.add_argument("--pad_g", type=int, default=114, help="填充色 G 通道")
    parser.add_argument("--pad_r", type=int, default=114, help="填充色 R 通道")
    args = parser.parse_args()

    ensure_dir(args.save_images_dir)
    images = list_images_in_dir(args.target_images_dir)
    if not images:
        print(f"[INFO] No images found in {args.target_images_dir}")
        return

    all_rows = []
    for p in images:
        rows = crop_and_save(
            img_path=p,
            save_root=args.save_images_dir,
            tile=args.tile,
            overlap=args.overlap,
            pad_to_tile=args.pad_to_tile,
            pad_value=(args.pad_b, args.pad_g, args.pad_r),
        )
        all_rows.extend(rows)

    # 写清单
    manifest_path = os.path.join(args.save_images_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["orig_image", "crop_image", "x", "y", "h", "w", "x_end", "y_end"])
        for r in all_rows:
            writer.writerow([r["orig_image"], r["crop_image"], r["x"], r["y"],
                             r["h"], r["w"], r["x_end"], r["y_end"]])
    print(f"[Saved] manifest -> {manifest_path}")

if __name__ == "__main__":
    main()
