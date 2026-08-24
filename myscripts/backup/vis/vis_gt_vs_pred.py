#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""左 GT / 右 Pred 的逐图对比图。

参数不自己定义一套，而是直接吃 seg_infer.py 的 run-config（--config 指向同一份 YAML），
数据源、模型、滑窗、fix_* 阈值、gt_cat_ids / pred_labels 全部复用，保证画出来的
Pred 与 seg_infer 那次跑出的 CSV 计数逐图一致。

用法（仓库根目录）：
    PYTHONPATH=. python myscripts/backup/vis/vis_gt_vs_pred.py \
        --config myscripts/infer/configs/coco_val2017_10imgs_wheat2.yaml \
        --out_dir output_v2/eval_10imgs_wheat2/compare_gt_pred

只支持 --source coco（GT 轮廓要从 instances_*.json 的 polygon 里取）。
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "myscripts", "infer"))

from seg_infer import (  # noqa: E402
    DatasetConfig, ModelConfig, SlidingWindowConfig, SearchSpace, OutputOptions,
    Detectron2SegInference, IOUType, build_argparser, load_run_config, ensure_dir,
)

GT_COLOR = (0, 200, 0)        # BGR 绿
PRED_COLOR = (0, 0, 255)      # BGR 红
BANNER_H = 56
GAP = 12                      # 两幅图之间的留白


def build_runner(a) -> Detectron2SegInference:
    runner = Detectron2SegInference(
        DatasetConfig(yolo_root=a.yolo_root, source=a.source, data_root=a.data_root,
                      coco_ann=a.coco_ann, gt_cat_ids=tuple(a.gt_cat_ids),
                      pred_labels=tuple(a.pred_labels)),
        ModelConfig(project_root=a.project_root, config_rel=a.model_config_rel,
                    checkpoint_rel=a.checkpoint_rel, device=a.device,
                    imports=tuple(a.imports), opts=tuple(str(o) for o in a.opts)),
        SlidingWindowConfig(tile=a.tile, overlap=a.overlap, pad_value=(114, 114, 114)),
        SearchSpace(iou_types=tuple(a.iou_types),
                    score_thr_range=(a.score_start, a.score_end, a.score_num),
                    nms_iou_thr_range=(a.nms_start, a.nms_end, a.nms_num),
                    base_keep_score=a.base_keep_score),
        OutputOptions(vis=False, excel=False, draw_scatter=False, out_dir=a.out_dir),
    )
    return runner


def load_gt_polys(ann_path: str, cat_ids):
    """basename -> [实例, ...]，每个实例是 [polygon(np.int32 Nx2), ...]。

    按实例分组而不是把多边形拍平：一个实例的 segmentation 可能有多段轮廓，
    拍平后 len() 就不等于实例数，GT 计数会虚高。
    只取 cat_ids（空则全取）的非 crowd 实例。
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2name = {im["id"]: os.path.basename(im["file_name"]) for im in data.get("images", [])}
    out = {name: [] for name in id2name.values()}
    keep = set(cat_ids) if cat_ids else None
    for a in data.get("annotations", []):
        if int(a.get("iscrowd", 0)) == 1:
            continue
        if keep is not None and int(a.get("category_id", -1)) not in keep:
            continue
        name = id2name.get(a.get("image_id"))
        if name is None:
            continue
        seg = a.get("segmentation")
        if not isinstance(seg, list):     # RLE 不在本脚本范围内
            continue
        polys = [np.asarray(p, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                 for p in seg if len(p) >= 6]
        if polys:
            out[name].append(polys)
    return out


def draw_gt(img, instances):
    vis = img.copy()
    for polys in instances:
        cv2.drawContours(vis, polys, -1, GT_COLOR, 2)
    return vis


def draw_pred(img, dets):
    """与 seg_infer._visualize_and_save 同样的裁剪逻辑，只换颜色。"""
    vis = img.copy()
    H, W = img.shape[:2]
    for d in dets:
        y1o, x1o = int(d.top), int(d.left)
        y2o, x2o = y1o + d.mask.shape[0], x1o + d.mask.shape[1]
        y1, y2 = max(0, min(H, y1o)), max(0, min(H, y2o))
        x1, x2 = max(0, min(W, x1o)), max(0, min(W, x2o))
        if y2 <= y1 or x2 <= x1:
            continue
        m = d.mask[y1 - y1o:(y1 - y1o) + (y2 - y1), x1 - x1o:(x1 - x1o) + (x2 - x1)].astype(np.uint8) * 255
        roi = vis[y1:y2, x1:x2]
        m = m[:roi.shape[0], :roi.shape[1]]
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours, -1, PRED_COLOR, 2)
    return vis


def banner(width, text, color):
    bar = np.full((BANNER_H, width, 3), 255, dtype=np.uint8)
    cv2.putText(bar, text, (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    return bar


def compose(gt_img, pred_img, n_gt, n_pred):
    W = gt_img.shape[1]
    left = np.vstack([banner(W, f"GT: {n_gt}", GT_COLOR), gt_img])
    right = np.vstack([banner(W, f"Pred: {n_pred}  (diff {n_pred - n_gt:+d})", PRED_COLOR), pred_img])
    gap = np.full((left.shape[0], GAP, 3), 255, dtype=np.uint8)
    return np.hstack([left, gap, right])


def main():
    p = argparse.ArgumentParser(description="逐图绘制 左GT / 右Pred 对比图")
    p.add_argument("--config", required=True, help="seg_infer.py 的 run-config YAML")
    p.add_argument("--out_dir", default="", help="输出目录（默认写到 config 的 out_dir 同级的 compare_gt_pred/）")
    args = p.parse_args()

    a = build_argparser().parse_args([])          # 取全部默认值
    cfg = load_run_config(args.config)
    for k, v in cfg.items():
        setattr(a, k, v)
    if a.source != "coco":
        raise SystemExit(f"本脚本只支持 --source coco（当前 {a.source}），GT 轮廓要从 COCO polygon 取")

    out_dir = args.out_dir or os.path.join(os.path.dirname(a.out_dir.rstrip("/")), "compare_gt_pred")
    ensure_dir(out_dir)

    split = a.split or a.infer_split
    ann_path = a.coco_ann or os.path.join(a.data_root, "annotations", f"instances_{split}.json")
    gt_polys = load_gt_polys(ann_path, a.gt_cat_ids)
    print(f"[gt] {ann_path} -> {sum(len(v) for v in gt_polys.values())} 个实例"
          + (f"，只取 category_id={sorted(a.gt_cat_ids)}" if a.gt_cat_ids else ""))

    runner = build_runner(a)
    print("==> Loading model ...")
    runner.load_model()

    images = sorted(
        os.path.join(a.data_root, split, f)
        for f in os.listdir(os.path.join(a.data_root, split))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )
    iou_type = IOUType(a.fix_iou_type)
    print(f"==> {len(images)} images | score_thr={a.fix_score_thr} "
          f"nms_iou_thr={a.fix_nms_iou_thr} iou_type={iou_type.value}")

    rows = []
    for path in images:
        name = os.path.basename(path)
        img = cv2.imread(path)
        dets = runner._apply_thresh_nms(runner._proposals_for_image(path),
                                        a.fix_score_thr, a.fix_nms_iou_thr, iou_type)
        polys = gt_polys.get(name, [])
        canvas = compose(draw_gt(img, polys), draw_pred(img, dets), len(polys), len(dets))
        cv2.imwrite(os.path.join(out_dir, name), canvas)
        rows.append((name, len(polys), len(dets)))
        print(f"  {name}  GT={len(polys):>3}  Pred={len(dets):>3}  diff={len(dets)-len(polys):+d}")

    tg, tp = sum(r[1] for r in rows), sum(r[2] for r in rows)
    ae = sum(abs(r[2] - r[1]) for r in rows)
    print(f"\nGT={tg}  Pred={tp}  净偏差={tp-tg:+d}  sum|GT-Pred|={ae}"
          f"  ({ae/tg*100:.1f}%)  MAE={ae/len(rows):.2f}/图")
    print(f"[Saved] {len(rows)} 张对比图 -> {out_dir}")


if __name__ == "__main__":
    main()
