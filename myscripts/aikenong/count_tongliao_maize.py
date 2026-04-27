#!/usr/bin/env python3
"""统计通辽玉米数据目录中的图片数量，并按数量尽量均衡地分组。

功能:
1) 统计 root 下每个一级子目录（递归）中 .jpg/.jpeg/.png 图片数量（大小写不敏感）
2) 按图片数量将这些一级子目录划分为 N 组（默认 8），尽量平衡各组总量
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DirCount:
    name: str
    path: Path
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归统计每个一级子目录中的图片数量，并按N组均衡划分。"
    )
    parser.add_argument(
        "root",
        type=Path,
        help="根目录（其下每个一级子目录会被单独统计）",
    )
    parser.add_argument(
        "-N",
        "--num-groups",
        type=int,
        default=8,
        help="划分组数，默认 8",
    )
    parser.add_argument(
        "--include-root-files",
        action="store_true",
        help="额外统计根目录本身（不在子目录内）的图片数量",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="可选：将结果写入 JSON 文件",
    )
    parser.add_argument(
        "--only-top-level",
        nargs="*",
        default=None,
        help=(
            "可选：仅统计指定一级子目录名。支持空格分隔和逗号分隔，例如: "
            "--only-top-level 7.23 7.25 或 --only-top-level 7.23,7.25"
        ),
    )
    parser.add_argument(
        "--size-progress",
        dest="size_progress",
        action="store_true",
        default=True,
        help="统计尺寸时显示目录级进度条（默认开启）",
    )
    parser.add_argument(
        "--no-size-progress",
        dest="size_progress",
        action="store_false",
        help="统计尺寸时关闭目录级进度条",
    )
    return parser.parse_args()


def count_images_recursive(directory: Path, image_exts: Iterable[str]) -> int:
    return sum(1 for _ in iter_image_files(directory, image_exts))


def iter_image_files(directory: Path, image_exts: Iterable[str]) -> Iterator[Path]:
    exts = {e.lower() for e in image_exts}
    for current_root, _, filenames in os.walk(directory, followlinks=False):
        root_path = Path(current_root)
        for filename in filenames:
            if Path(filename).suffix.lower() in exts:
                yield root_path / filename


def get_image_size(path: Path) -> Tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            head = f.read(24)

            if len(head) >= 24 and head.startswith(b"\x89PNG\r\n\x1a\n") and head[12:16] == b"IHDR":
                width, height = struct.unpack(">II", head[16:24])
                return int(width), int(height)

            if len(head) >= 2 and head[:2] == b"\xff\xd8":
                f.seek(2)
                return _parse_jpeg_size(f)

    except (OSError, struct.error):
        return None

    return None


def _parse_jpeg_size(file_obj) -> Tuple[int, int] | None:
    while True:
        byte = file_obj.read(1)
        if not byte:
            return None
        if byte != b"\xff":
            continue

        marker = file_obj.read(1)
        if not marker:
            return None

        while marker == b"\xff":
            marker = file_obj.read(1)
            if not marker:
                return None

        marker_code = marker[0]
        if marker_code in (0xD8, 0xD9):
            continue

        seg_len_bytes = file_obj.read(2)
        if len(seg_len_bytes) != 2:
            return None
        seg_len = struct.unpack(">H", seg_len_bytes)[0]
        if seg_len < 2:
            return None

        # SOF0~SOF15 (except DHT/DAC) contain image height/width.
        if 0xC0 <= marker_code <= 0xCF and marker_code not in (0xC4, 0xC8, 0xCC):
            sof_payload = file_obj.read(5)
            if len(sof_payload) != 5:
                return None
            height, width = struct.unpack(">HH", sof_payload[1:5])
            return int(width), int(height)

        file_obj.seek(seg_len - 2, os.SEEK_CUR)


def count_image_sizes_recursive(
    directory: Path,
    image_exts: Iterable[str],
    *,
    total_images_hint: int | None = None,
    progress_label: str | None = None,
    show_progress: bool = False,
) -> tuple[Counter[Tuple[int, int]], int]:
    size_counts: Counter[Tuple[int, int]] = Counter()
    unreadable = 0
    processed = 0

    total = total_images_hint
    if show_progress and total is None:
        total = count_images_recursive(directory, image_exts)

    last_update = 0.0
    label = progress_label or directory.name

    for image_path in iter_image_files(directory, image_exts):
        size = get_image_size(image_path)
        if size is None:
            unreadable += 1
        else:
            size_counts[size] += 1

        processed += 1
        if show_progress and total and total > 0:
            now = time.monotonic()
            if processed == total or (processed == 1) or (now - last_update >= 0.12):
                _print_progress_line(label, processed, total)
                last_update = now

    if show_progress and total == 0:
        _print_progress_line(label, 0, 0)

    return size_counts, unreadable


def _print_progress_line(label: str, current: int, total: int) -> None:
    bar_width = 24
    safe_total = max(total, 1)
    ratio = min(max(current / safe_total, 0.0), 1.0)
    filled = int(round(ratio * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    line = f"\r[尺寸统计] {label:<16} [{bar}] {current:>6}/{total:<6} {ratio * 100:>5.1f}%"
    print(line, end="", file=sys.stderr, flush=True)
    if current >= total:
        print(file=sys.stderr, flush=True)


def normalize_dir_filters(raw_values: Sequence[str] | None) -> List[str]:
    if not raw_values:
        return []

    names: List[str] = []
    for value in raw_values:
        for token in value.split(","):
            token = token.strip()
            if token:
                names.append(token)
    return names


def collect_top_level_dir_counts(root: Path, only_names: Sequence[str] | None = None) -> List[DirCount]:
    requested = set(only_names or [])

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if requested:
        subdirs = [p for p in subdirs if p.name in requested]
    subdirs = sorted(subdirs, key=lambda p: p.name)

    results: List[DirCount] = []
    for subdir in subdirs:
        results.append(
            DirCount(
                name=subdir.name,
                path=subdir,
                count=count_images_recursive(subdir, IMAGE_EXTS),
            )
        )
    return results


def greedy_balance_partition(dir_counts: Sequence[DirCount], num_groups: int) -> List[List[DirCount]]:
    if num_groups <= 0:
        raise ValueError("num_groups 必须是正整数")

    groups: List[List[DirCount]] = [[] for _ in range(num_groups)]
    group_sums = [0] * num_groups

    # LPT 贪心：先放大目录，再始终放入当前总量最小的组。
    for item in sorted(dir_counts, key=lambda x: x.count, reverse=True):
        min_idx = min(range(num_groups), key=lambda i: (group_sums[i], i))
        groups[min_idx].append(item)
        group_sums[min_idx] += item.count

    return groups


def summarize_groups(groups: Sequence[Sequence[DirCount]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for idx, group in enumerate(groups, start=1):
        total = sum(x.count for x in group)
        summary.append(
            {
                "group_index": idx,
                "total_images": total,
                "directories": [
                    {
                        "name": x.name,
                        "path": str(x.path),
                        "images": x.count,
                    }
                    for x in group
                ],
            }
        )
    return summary


def print_counts(dir_counts: Sequence[DirCount]) -> None:
    if not dir_counts:
        print("未发现一级子目录。")
        return

    name_width = max(len("目录名"), *(len(d.name) for d in dir_counts))
    cnt_width = max(len("图片数"), *(len(str(d.count)) for d in dir_counts))

    print("\n[每个一级子目录图片数量]")
    print(f"{'目录名':<{name_width}}  {'图片数':>{cnt_width}}")
    print(f"{'-' * name_width}  {'-' * cnt_width}")
    for d in dir_counts:
        print(f"{d.name:<{name_width}}  {d.count:>{cnt_width}}")
    print(f"{'合计':<{name_width}}  {sum(d.count for d in dir_counts):>{cnt_width}}")


def print_groups(groups: Sequence[Sequence[DirCount]]) -> None:
    print("\n[均衡分组建议]")
    for idx, group in enumerate(groups, start=1):
        total = sum(x.count for x in group)
        members = ", ".join(f"{x.name}({x.count})" for x in group) if group else "(空组)"
        print(f"组{idx:02d} | 总数 {total:>6} | {members}")


def print_size_distribution(size_counts: Counter[Tuple[int, int]], unreadable: int = 0) -> None:
    print("\n[图片尺寸分布]")
    if not size_counts:
        print("未发现可解析尺寸的图片。")
        if unreadable:
            print(f"无法解析尺寸的图片数: {unreadable}")
        return

    rows = sorted(size_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    size_strs = [f"{w}x{h}" for (w, h), _ in rows]
    size_width = max(len("尺寸"), *(len(s) for s in size_strs))
    cnt_width = max(len("数量"), *(len(str(c)) for _, c in rows))

    print(f"{'尺寸':<{size_width}}  {'数量':>{cnt_width}}")
    print(f"{'-' * size_width}  {'-' * cnt_width}")
    for (w, h), c in rows:
        print(f"{f'{w}x{h}':<{size_width}}  {c:>{cnt_width}}")
    if unreadable:
        print(f"无法解析尺寸的图片数: {unreadable}")



def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"错误: 根目录不存在或不是目录: {root}", file=sys.stderr)
        return 2

    if args.num_groups <= 0:
        print("错误: --num-groups 必须为正整数", file=sys.stderr)
        return 2

    only_top_level = normalize_dir_filters(args.only_top_level)
    requested_set = set(only_top_level)

    if requested_set:
        existing_top_level = {p.name for p in root.iterdir() if p.is_dir()}
        missing = sorted(requested_set - existing_top_level)
        if missing:
            print(f"警告: 以下指定一级子目录不存在，将被忽略: {missing}", file=sys.stderr)

    try:
        dir_counts = collect_top_level_dir_counts(root, only_names=only_top_level)
    except PermissionError as exc:
        print(f"错误: 无权限访问目录: {exc}", file=sys.stderr)
        return 3

    root_files_count = 0
    if args.include_root_files:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                root_files_count += 1

    size_counts: Counter[Tuple[int, int]] = Counter()
    unreadable_size_files = 0
    for d in dir_counts:
        subdir_size_counts, subdir_unreadable = count_image_sizes_recursive(
            d.path,
            IMAGE_EXTS,
            total_images_hint=d.count,
            progress_label=d.name,
            show_progress=bool(args.size_progress),
        )
        size_counts.update(subdir_size_counts)
        unreadable_size_files += subdir_unreadable

    if args.include_root_files:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                size = get_image_size(p)
                if size is None:
                    unreadable_size_files += 1
                else:
                    size_counts[size] += 1

    print(f"根目录: {root}")
    if requested_set:
        print(f"仅统计指定一级子目录: {sorted(requested_set)}")
    print_counts(dir_counts)

    groups = greedy_balance_partition(dir_counts, args.num_groups)
    print_groups(groups)
    print_size_distribution(size_counts, unreadable_size_files)

    totals = [sum(x.count for x in group) for group in groups]
    non_empty = [t for t in totals if t > 0]
    min_non_empty = min(non_empty) if non_empty else 0

    print("\n[统计信息]")
    print(f"一级子目录数量: {len(dir_counts)}")
    print(f"总图片数(子目录累计): {sum(d.count for d in dir_counts)}")
    if args.include_root_files:
        print(f"根目录文件图片数: {root_files_count}")
        print(f"总图片数(含根目录文件): {sum(d.count for d in dir_counts) + root_files_count}")
    print(f"分组数: {args.num_groups}")
    print(f"各组总数: {totals}")
    print(f"组间差值(最大-最小非空): {max(totals) - min_non_empty if totals else 0}")

    if args.json_out is not None:
        payload = {
            "root": str(root),
            "image_exts": sorted(IMAGE_EXTS),
            "num_groups": args.num_groups,
            "include_root_files": bool(args.include_root_files),
            "root_files_count": root_files_count,
            "directory_counts": [
                {"name": d.name, "path": str(d.path), "images": d.count}
                for d in dir_counts
            ],
            "groups": summarize_groups(groups),
            "size_distribution": [
                {"width": w, "height": h, "images": c}
                for (w, h), c in sorted(size_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
            ],
            "unreadable_size_files": unreadable_size_files,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n已写入 JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

