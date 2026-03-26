#!/usr/bin/env python3
"""Batch resize overlay images to a fixed square size."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


DEFAULT_INPUT_DIR = "/public/home/zncui/datasets/maize_rice_dataset/all_imgs_out2/overlay"
DEFAULT_OUTPUT_DIR = "/public/home/zncui/datasets/maize_rice_dataset/all_imgs_out2_overlay_resized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize all images in overlay directory to a fixed size."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Input directory that contains overlay images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for resized images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=800,
        help="Target width and height (square).",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize_image(src_path: Path, dst_path: Path, size: int) -> bool:
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    h, w = img.shape[:2]
    interpolation = cv2.INTER_AREA if (w > size or h > size) else cv2.INTER_LINEAR
    resized = cv2.resize(img, (size, size), interpolation=interpolation)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst_path), resized))


def show_progress(current: int, total: int, width: int = 30) -> None:
    if total <= 0:
        return
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r[PROGRESS] |{bar}| {current}/{total} ({ratio * 100:6.2f}%)"
    sys.stdout.write(msg)
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = args.size

    if target_size <= 0:
        print(f"[ERROR] --size must be positive, got: {target_size}")
        return 1

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return 1

    image_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and is_image_file(p)])
    if not image_paths:
        print(f"[WARN] No images found in: {input_dir}")
        return 0

    ok = 0
    failed = []

    total = len(image_paths)
    for idx, src_path in enumerate(image_paths, start=1):
        dst_path = output_dir / src_path.name
        if resize_image(src_path, dst_path, target_size):
            ok += 1
        else:
            failed.append(src_path.name)
        show_progress(idx, total)

    print()

    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] target_size={target_size}x{target_size}")
    print(f"[INFO] total={total}, success={ok}, failed={len(failed)}")

    if failed:
        print("[WARN] Failed files:")
        for name in failed:
            print(f"  - {name}")

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

