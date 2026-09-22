#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线构建 wheat_2 实例库，供训练期 copy-paste 增强使用。

把 COCO 标注里每一个 wheat_2 实例抠成「RGB 补丁 + 二值掩码 + 局部多边形 + 局部 points」
存进一个 npz。训练时 maskdino/data/copy_paste.py 从中随机取样贴到别的图上。

三条质量闸门（缺一不可，理由见 experiments-bank/ideas/copy-paste-wheat2.md）：

  1. 源实例不得贴到源图边界（--border-margin，默认 2px）。
     贴边的 wheat_2 是被裁切过的「半截穗」，其标注仍是 wheat_2 —— 把它当完整穗贴出去
     等于重新注入 2026-09-02 刚修掉的那类标签噪声（INPUT.DEMOTE_BORDER_TRUNCATED）。
  2. 带点数据集要求 len(points) >= --min-points（默认 1）。
     实测 train 里存在 0 点的 wheat_2，贴出去会教模型「有穗但无小穗」。
  3. 掩码面积 >= --min-area（默认 500 px²），丢掉退化实例。

用法：
  # Stage2（带点）
  PYTHONPATH=. python myscripts/backup/build_wheat2_bank.py \
    --ann /public/home/zncui/datasets/wheat50/coco_2cat_with_points/annotations/instances_train2017.json \
    --img-dir /public/home/zncui/datasets/wheat50/coco_2cat_with_points/train2017 \
    --out /public/home/zncui/datasets/wheat50/banks/wheat2_pts_train.npz

  # Stage1（无点）
  PYTHONPATH=. python myscripts/backup/build_wheat2_bank.py \
    --ann /public/home/zncui/datasets/wheat50/coco_2cat/annotations/instances_train2017.json \
    --img-dir /public/home/zncui/datasets/wheat50/coco_2cat/train2017 \
    --out /public/home/zncui/datasets/wheat50/banks/wheat2_nop_train.npz --min-points 0

  # 自检：随机抽若干条画出来核对对齐
  ... --selfcheck-dir output_v2/bank_selfcheck --selfcheck-n 20
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from myscripts.backup.coco_rle_to_polygon import (
    mask_from_polygons,
    normalize_polygon_list,
)

WHEAT2_CATEGORY_ID = 2  # COCO 原始 id（contiguous id 为 1，见 data_register/wheat_data_cat2.py）


def touches_border(mask, margin):
    """掩码是否贴住图像边界（margin 像素容差）。与 mapper 的降级判据同口径。"""
    if margin <= 0:
        return False
    m = mask > 0
    return bool(
        m[:margin, :].any() or m[-margin:, :].any()
        or m[:, :margin].any() or m[:, -margin:].any()
    )


def build(args):
    with open(args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann["category_id"]) == args.category_id:
            by_img[ann["image_id"]].append(ann)

    patches, masks, polys_l, points_l, srcs = [], [], [], [], []
    n_total = 0
    drop = {"border": 0, "points": 0, "area": 0, "poly": 0, "read": 0}

    for k, (img_id, anns) in enumerate(sorted(by_img.items())):
        im_rec = images[img_id]
        H, W = int(im_rec["height"]), int(im_rec["width"])
        path = os.path.join(args.img_dir, im_rec["file_name"])
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            drop["read"] += len(anns)
            continue
        img = img[:, :, ::-1]  # -> RGB，与 mapper 的 image_format 一致（见下方 --image-format 说明）

        for ann in anns:
            n_total += 1
            polys = normalize_polygon_list(ann["segmentation"], W, H)
            if not polys:
                drop["poly"] += 1
                continue
            m = mask_from_polygons(polys, H, W)
            if int(m.sum()) < args.min_area:
                drop["area"] += 1
                continue
            if touches_border(m, args.border_margin):
                drop["border"] += 1
                continue
            pts = ann.get("points", []) or []
            if len(pts) < args.min_points:
                drop["points"] += 1
                continue

            ys, xs = np.where(m > 0)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1

            patches.append(np.ascontiguousarray(img[y0:y1, x0:x1]))
            masks.append(np.ascontiguousarray(m[y0:y1, x0:x1].astype(np.uint8)))
            polys_l.append([
                (np.asarray(p, dtype=np.float32).reshape(-1, 2) - [x0, y0]).reshape(-1).tolist()
                for p in polys
            ])
            points_l.append(
                (np.asarray(pts, dtype=np.float32).reshape(-1, 2) - [x0, y0])
                if len(pts) else np.zeros((0, 2), dtype=np.float32)
            )
            srcs.append(f'{im_rec["file_name"]}#{ann.get("id", -1)}')

        if args.limit and k + 1 >= args.limit:
            break

    n_keep = len(patches)
    print(f"扫描 wheat_2 {n_total} 个 -> 入库 {n_keep} 个（{n_keep/max(n_total,1):.1%}）")
    for key, label in (("border", f"贴边(margin={args.border_margin})"),
                       ("points", f"点数 < {args.min_points}"),
                       ("area", f"面积 < {args.min_area}"),
                       ("poly", "多边形为空"), ("read", "图读失败")):
        if drop[key]:
            print(f"  刷掉 {drop[key]:5d}  {label}")
    if n_keep == 0:
        raise SystemExit("入库 0 条，检查闸门参数")

    hw = np.array([[p.shape[0], p.shape[1]] for p in patches])
    npt = np.array([len(p) for p in points_l])
    print(f"  补丁尺寸 h 中位 {np.median(hw[:,0]):.0f} / w 中位 {np.median(hw[:,1]):.0f}"
          f"  掩码面积中位 {np.median([int(m.sum()) for m in masks]):.0f}")
    print(f"  points 数 中位 {np.median(npt):.0f}  范围 {npt.min()}~{npt.max()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        patches=np.array(patches, dtype=object),
        masks=np.array(masks, dtype=object),
        polys=np.array(polys_l, dtype=object),
        points=np.array(points_l, dtype=object),
        sources=np.array(srcs),
        meta=np.array([json.dumps({
            "ann": args.ann, "img_dir": args.img_dir, "n_total": n_total, "n_keep": n_keep,
            "border_margin": args.border_margin, "min_points": args.min_points,
            "min_area": args.min_area, "color": "RGB",
        }, ensure_ascii=False)]),
    )
    print(f"已写入 {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    if args.selfcheck_dir:
        selfcheck(patches, masks, polys_l, points_l, srcs, args)


def selfcheck(patches, masks, polys_l, points_l, srcs, args):
    """随机抽若干条，把补丁 / 掩码轮廓 / points 画在一起，肉眼核对对齐。"""
    os.makedirs(args.selfcheck_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(patches), size=min(args.selfcheck_n, len(patches)), replace=False)
    for i in idx:
        vis = patches[i].copy()[:, :, ::-1]  # -> BGR 供 imwrite
        vis = np.ascontiguousarray(vis)
        cnts, _ = cv2.findContours((masks[i] > 0).astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
        for p in polys_l[i]:
            pts = np.asarray(p, np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 1)
        for x, y in np.asarray(points_l[i]).reshape(-1, 2):
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (255, 0, 0), -1)
        out = os.path.join(args.selfcheck_dir, f"{i:05d}_{len(points_l[i])}pts.png")
        cv2.imwrite(out, vis)
    print(f"自检图 {len(idx)} 张 -> {args.selfcheck_dir}"
          f"  （红=掩码轮廓 绿=多边形 蓝=小穗点，三者应重合/落在穗上）")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, help="COCO 标注 json")
    ap.add_argument("--img-dir", required=True, help="图片目录")
    ap.add_argument("--out", required=True, help="输出 npz")
    ap.add_argument("--category-id", type=int, default=WHEAT2_CATEGORY_ID)
    ap.add_argument("--border-margin", type=int, default=2,
                    help="闸门1：源实例贴住源图边界这么多像素以内则丢弃（0 关闭）")
    ap.add_argument("--min-points", type=int, default=1,
                    help="闸门2：小穗点数下限；无点数据集传 0")
    ap.add_argument("--min-area", type=int, default=500, help="闸门3：掩码面积下限（px²）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张图（调试用）")
    ap.add_argument("--selfcheck-dir", default="", help="非空则输出自检可视化")
    ap.add_argument("--selfcheck-n", type=int, default=20)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
