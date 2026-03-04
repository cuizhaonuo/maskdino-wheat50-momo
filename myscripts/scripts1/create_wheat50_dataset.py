#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 防止大图触发 PIL 的 DecompressionBomb 警告误判


def list_jpgs(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in exts])


def sample_images(jpgs: List[Path], k: int, rng: random.Random) -> List[Path]:
    if len(jpgs) <= k:
        return jpgs
    return rng.sample(jpgs, k)


def tile_image(
    img_path: Path,
    out_dir: Path,
    tile_size: int = 1024,
    drop_remainder: bool = True,
    quality: int = 95,
) -> List[Path]:
    """
    将单张图片切成 tile_size x tile_size 的块。
    - drop_remainder=True: 丢弃不足 tile_size 的边缘残块，保证输出块尺寸一致。
    返回生成的 tile 文件路径列表。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = []
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size

        max_x = (w // tile_size) * tile_size if drop_remainder else w
        max_y = (h // tile_size) * tile_size if drop_remainder else h

        # 仅生成完整 tile（如果 drop_remainder=False，你可以自行改为 padding 或保存残块）
        for top in range(0, max_y, tile_size):
            for left in range(0, max_x, tile_size):
                if drop_remainder and (left + tile_size > w or top + tile_size > h):
                    continue
                crop = im.crop((left, top, left + tile_size, top + tile_size))
                if crop.size != (tile_size, tile_size):
                    # 保险：确保输出尺寸一致
                    continue

                base = img_path.stem
                tile_name = f"{base}__x{left}_y{top}_w{tile_size}_h{tile_size}.jpg"
                out_path = out_dir / tile_name
                crop.save(out_path, "JPEG", quality=quality)
                tiles.append(out_path)

    return tiles


def equal_quota_split(total: int, groups: int) -> List[int]:
    """
    将 total 平均分给 groups 组。余数从前往后分配（保证“比例一致”= 每组尽量相同）。
    """
    base = total // groups
    rem = total % groups
    quotas = [base] * groups
    for i in range(rem):
        quotas[i] += 1
    return quotas


def main(
    src_root: str,
    dir_names: List[str],
    out_root: str,
    per_dir_pick: int = 10,
    tile_size: int = 1024,
    results_n: int = 1000,
    seed: int = 20240516,
    drop_remainder: bool = True,
):
    rng = random.Random(seed)

    src_root = Path(src_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) 每个目录随机选 per_dir_pick 张，切分并保存到 out_root/<dir_name>/
    all_tiles_by_dir: List[Tuple[str, List[Path]]] = []

    for dname in dir_names:
        src_dir = src_root / dname
        if not src_dir.exists():
            raise FileNotFoundError(f"Source dir not found: {src_dir}")

        jpgs = list_jpgs(src_dir)
        chosen = sample_images(jpgs, per_dir_pick, rng)

        dst_dir = out_root / dname
        dst_dir.mkdir(parents=True, exist_ok=True)

        tiles = []
        for img_path in chosen:
            tiles.extend(
                tile_image(
                    img_path=img_path,
                    out_dir=dst_dir,
                    tile_size=tile_size,
                    drop_remainder=drop_remainder,
                )
            )

        print(f"[{dname}] JPGs={len(jpgs)}, chosen={len(chosen)}, tiles={len(tiles)}")
        all_tiles_by_dir.append((dname, tiles))

    # 2) 在 out_root 下创建 results，按“各目录比例一致”抽取 results_n 张
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    groups = len(all_tiles_by_dir)
    quotas = equal_quota_split(results_n, groups)

    picked_total = 0
    for (dname, tiles), quota in zip(all_tiles_by_dir, quotas):
        if len(tiles) == 0:
            print(f"[WARN] {dname} has 0 tiles, skip.")
            continue

        # 如果 tile 数量不足 quota，就全取（并提示）
        if len(tiles) < quota:
            pick = tiles[:]  # all
            print(f"[WARN] {dname} tiles ({len(tiles)}) < quota ({quota}), pick all.")
        else:
            pick = rng.sample(tiles, quota)

        for tpath in pick:
            # results 命名带目录名，确保可回溯
            dst_name = f"{dname}__{tpath.name}"
            dst_path = results_dir / dst_name
            # 复制（如想省空间可改成 os.symlink）
            shutil.copy2(tpath, dst_path)

        picked_total += len(pick)

    print(f"[DONE] results picked: {picked_total} -> {results_dir}")


if __name__ == "__main__":
    # ====== 按你的描述默认配置 ======
    SRC_ROOT = "/public/lab_share/nfs_field/raw_data/DJI/P1/zhaoxian2023/wheat/20240516DJI/DCIM/"
    DIRS = [
        "DJI_202405161402_007_2023D6-250cm-A50",
        "DJI_202405161402_008_2023D6-250cm-A50",
        "DJI_202405161548_009_2023D6-250cm-A50",
        "DJI_202405161548_011_2023D8-250cm-A50",
        "DJI_202405161548_012_2023D8-250cm-A50",
    ]
    OUT_ROOT = "/public/home/zncui/datasets/wheat50/total_images"

    # 你可以按需修改这些默认值
    main(
        src_root=SRC_ROOT,
        dir_names=DIRS,
        out_root=OUT_ROOT,
        per_dir_pick=10,
        tile_size=1024,
        results_n=1000,
        seed=20240516,
        drop_remainder=True,  # 保证所有输出 tile 都是 1024x1024
    )