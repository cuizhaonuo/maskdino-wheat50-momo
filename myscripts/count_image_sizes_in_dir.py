#!/usr/bin/env python3
"""Count how many images share the same width x height in a directory."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count image size distribution in a directory."
    )
    parser.add_argument(
        "--input_dir",
        default="/public/home/zncui/datasets/maize_rice_dataset/all_imgs",
        help="Directory containing images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan sub-directories.",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
        help="Allowed image extensions (case-insensitive).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Only print top-k sizes by count (0 means print all).",
    )
    return parser.parse_args()


def iter_image_files(root: Path, recursive: bool, exts: set[str]) -> Iterable[Path]:
    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.glob("*")

    for p in iterator:
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Invalid input directory: {input_dir}")
        return 1

    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.exts}
    size_counter: Counter[str] = Counter()

    num_files = 0
    num_failed = 0

    for img_path in iter_image_files(input_dir, args.recursive, exts):
        num_files += 1
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except (OSError, UnidentifiedImageError):
            num_failed += 1
            continue

        size_counter[f"{width}x{height}"] += 1

    sorted_items = sorted(size_counter.items(), key=lambda x: (-x[1], x[0]))
    if args.topk > 0:
        sorted_items = sorted_items[: args.topk]

    print("size,count")
    for size, count in sorted_items:
        print(f"{size},{count}")

    print(f"\n[INFO] scanned_files={num_files}")
    print(f"[INFO] valid_images={sum(size_counter.values())}")
    print(f"[INFO] unique_sizes={len(size_counter)}")
    print(f"[INFO] failed_images={num_failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

