#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
小穗点回归（端到端）推理脚本
============================

配套模型见 `小穗点回归方案.md`：在 MaskDINO 每个实例 query 上增加逐点预测头，
对每个 `wheat_2` 实例同时输出实例掩码 + 内部小穗点（坐标与数量），`wheat` 实例不输出点。

本脚本与 `seg_infer.py` 的区别
------------------------------
- `seg_infer.py` 面向大图，做滑窗 + 全局 Mask-NMS 的“麦穗实例计数”。
- 本脚本面向已切好的 1024×1024 小图（`coco_2cat_with_points`），**整图直接前向**，
  逐实例取出 `pred_spikelet_points / pred_spikelet_counts`，可视化并（可选）与 GT 小穗点对比。

推理产出
--------
1. 每张图的可视化：实例掩码轮廓（按类别着色）+ 小穗点（绿色实心点）+ 每穗小穗计数文本。
2. CSV：每图的 wheat_2 实例数、小穗总数（预测 vs GT，若提供 GT json）。
3. 若提供 GT json：打印/保存小穗总数的 MAE / RMSE。

用法
----
# 直接用训练好的点回归模型在 val 集推理 + 与 GT 对比
python myscripts/infer/seg_infer_reg.py \
  --config_file output_v2/wheat305_r50_cat2_point_reg_coord5.0_cls2.0/config.yaml \
  --weights     output_v2/wheat305_r50_cat2_point_reg_coord5.0_cls2.0/model_final.pth \
  --images      /public/home/zncui/datasets/wheat50/coco_2cat_with_points/val2017 \
  --gt_json     /public/home/zncui/datasets/wheat50/coco_2cat_with_points/annotations/instances_val2017.json \
  --out_dir     ./vis_spikelet_reg_val

# 仅推理一个文件夹（无 GT）
python myscripts/infer/seg_infer_reg.py --images /path/to/imgs --out_dir ./vis_out
"""

import os
import sys
import glob
import json
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple

# 脚本位于 myscripts/infer/，需把仓库根目录加入 sys.path 才能 import maskdino / 本地 detectron2
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch

try:
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images_in_dir(d: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
            "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(d, e)))
    files.sort()
    return files


# -----------------------------
# GT points (COCO json) loader
# -----------------------------
def _parse_points_field(points) -> np.ndarray:
    """anno['points'] 兼容 [[x,y],...] 嵌套 与 [x,y,x,y,...] 扁平两种形式 -> (N,2) float32。"""
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr.reshape(-1, 2)


def load_gt_by_filename(gt_json: str, spikelet_cat_id: int = 2) -> Dict[str, Dict[str, Any]]:
    """
    返回 {file_name: {"spike_count": int(category==spikelet 的实例数),
                       "spikelet_total": int(所有点数之和),
                       "points": List[(N,2)]}}
    """
    with open(gt_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2name = {im["id"]: im["file_name"] for im in data["images"]}
    out: Dict[str, Dict[str, Any]] = {
        im["file_name"]: {"spike_count": 0, "spikelet_total": 0, "points": []}
        for im in data["images"]
    }
    for a in data["annotations"]:
        if int(a.get("category_id", -1)) != spikelet_cat_id:
            continue
        fn = id2name.get(a["image_id"])
        if fn is None:
            continue
        pts = _parse_points_field(a.get("points"))
        rec = out[fn]
        rec["spike_count"] += 1
        rec["spikelet_total"] += int(pts.shape[0])
        rec["points"].append(pts)
    return out


# -----------------------------
# Model wrapper
# -----------------------------
def build_predictor(config_file: str, weights: str, device: str,
                    point_threshold: Optional[float],
                    imports: Tuple[str, ...]) -> DefaultPredictor:
    # 注册自定义 meta-arch / heads（必须在 build cfg 前 import maskdino）
    for m in imports:
        try:
            __import__(m)
            print(f"[imports] imported module: {m}")
        except Exception as e:
            print(f"[imports] WARNING: failed to import '{m}': {e}")

    cfg = get_cfg()
    from detectron2.projects.deeplab import add_deeplab_config
    add_deeplab_config(cfg)
    from maskdino import add_maskdino_config
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)

    if weights:
        cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    # 实例分数过滤在 infer_image 里按 score_thr 自行完成（模型本身只按 topk 返回固定数量实例）。
    if point_threshold is not None:
        cfg.MODEL.MaskDINO.TEST.POINT_THRESHOLD = float(point_threshold)

    print(f"[cfg]     -> {config_file}")
    print(f"[weights] -> {cfg.MODEL.WEIGHTS}")
    print(f"[point_threshold] -> {cfg.MODEL.MaskDINO.TEST.POINT_THRESHOLD}")
    return DefaultPredictor(cfg)


@torch.no_grad()
def infer_image(predictor: DefaultPredictor, img_bgr: np.ndarray,
                score_thr: float, spikelet_class_id: int) -> List[Dict[str, Any]]:
    """
    整图前向，返回每个保留实例的 dict：
        {bbox:[x1,y1,x2,y2], score, label, mask(HxW bool),
         spikelet_points:(K,2) float, spikelet_count:int}
    小穗点坐标已经是原图像素坐标（模型内部用预测框把框相对坐标还原过）。
    """
    outputs = predictor(img_bgr)
    if "instances" not in outputs:
        return []
    inst = outputs["instances"].to("cpu")
    n = len(inst)
    if n == 0:
        return []

    boxes = inst.pred_boxes.tensor.numpy() if inst.has("pred_boxes") else None
    scores = inst.scores.numpy() if inst.has("scores") else np.zeros((n,), np.float32)
    labels = inst.pred_classes.numpy() if inst.has("pred_classes") else np.zeros((n,), np.int32)
    masks = inst.pred_masks.numpy().astype(bool) if inst.has("pred_masks") else None

    has_points = inst.has("pred_spikelet_points")
    sp_points = inst.pred_spikelet_points if has_points else None         # list[Tensor[Ki,2]]
    sp_counts = inst.pred_spikelet_counts.numpy() if inst.has("pred_spikelet_counts") else None

    results: List[Dict[str, Any]] = []
    for i in range(n):
        if float(scores[i]) < score_thr:
            continue
        lab = int(labels[i])
        if has_points and lab == spikelet_class_id:
            pts = sp_points[i]
            pts = pts.cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
            pts = pts.reshape(-1, 2).astype(np.float32)
        else:
            pts = np.zeros((0, 2), dtype=np.float32)
        results.append({
            "bbox": boxes[i].tolist() if boxes is not None else [0, 0, 0, 0],
            "score": float(scores[i]),
            "label": lab,
            "mask": masks[i] if masks is not None else None,
            "spikelet_points": pts,
            "spikelet_count": int(sp_counts[i]) if sp_counts is not None else int(pts.shape[0]),
        })
    return results


# -----------------------------
# Visualization
# -----------------------------
_CLASS_COLORS = [
    (255, 0, 0),    # 0: wheat   -> 蓝 (BGR)
    (0, 0, 255),    # 1: wheat_2 -> 红 (BGR)
    (0, 255, 0),
    (255, 255, 0),
]


def _class_color(label: int) -> Tuple[int, int, int]:
    return _CLASS_COLORS[label % len(_CLASS_COLORS)]


def visualize(img_bgr: np.ndarray, dets: List[Dict[str, Any]],
              gt_points: Optional[List[np.ndarray]] = None,
              draw_box: bool = False) -> np.ndarray:
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    # GT 小穗点（黄色空心圈）便于和预测对比
    if gt_points is not None:
        for pts in gt_points:
            for (x, y) in pts:
                cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 255, 255), 1, cv2.LINE_AA)

    for d in dets:
        color = _class_color(d["label"])
        mask = d["mask"]
        # 掩码轮廓
        if mask is not None and mask.any():
            m8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)

        if draw_box:
            x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

        # 小穗点（绿色实心）+ 每穗计数
        pts = d["spikelet_points"]
        if pts is not None and len(pts) > 0:
            for (x, y) in pts:
                cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1, cv2.LINE_AA)
            x1, y1 = int(round(d["bbox"][0])), int(round(d["bbox"][1]))
            tx, ty = max(0, min(W - 1, x1)), max(12, min(H - 1, y1 - 4))
            cv2.putText(vis, str(d["spikelet_count"]), (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


# -----------------------------
# Main loop
# -----------------------------
def run(args):
    ensure_dir(args.out_dir)
    predictor = build_predictor(
        args.config_file, args.weights, args.device,
        args.point_threshold, tuple(args.imports),
    )

    gt_map = None
    if args.gt_json and os.path.isfile(args.gt_json):
        gt_map = load_gt_by_filename(args.gt_json, args.spikelet_cat_id)
        print(f"[gt] loaded {len(gt_map)} images from {args.gt_json}")

    images = list_images_in_dir(args.images)
    if not images:
        print(f"[warn] no images found under {args.images}")
        return
    print(f"[run] {len(images)} images")

    rows = [("filename", "pred_spike", "pred_spikelet", "gt_spike", "gt_spikelet")]
    pred_totals, gt_totals = [], []

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[skip] unreadable: {img_path}")
            continue
        dets = infer_image(predictor, img, args.score_thr, args.spikelet_class_id)

        spike_dets = [d for d in dets if d["label"] == args.spikelet_class_id]
        pred_spike = len(spike_dets)
        pred_spikelet = int(sum(d["spikelet_count"] for d in spike_dets))

        fn = os.path.basename(img_path)
        gt_rec = gt_map.get(fn) if gt_map else None
        gt_spike = gt_rec["spike_count"] if gt_rec else -1
        gt_spikelet = gt_rec["spikelet_total"] if gt_rec else -1
        gt_points = gt_rec["points"] if (gt_rec and args.draw_gt) else None

        rows.append((fn, pred_spike, pred_spikelet, gt_spike, gt_spikelet))
        if gt_rec:
            pred_totals.append(pred_spikelet)
            gt_totals.append(gt_spikelet)

        if args.vis:
            vis = visualize(img, dets, gt_points=gt_points, draw_box=args.draw_box)
            cv2.imwrite(os.path.join(args.out_dir, fn), vis)

    # CSV
    csv_path = os.path.join(args.out_dir, "spikelet_pred_vs_gt.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"[saved] CSV -> {csv_path}")

    # 指标（仅在有 GT 时）
    if pred_totals:
        p = np.asarray(pred_totals, dtype=np.float64)
        g = np.asarray(gt_totals, dtype=np.float64)
        mae = float(np.mean(np.abs(p - g)))
        rmse = float(np.sqrt(np.mean((p - g) ** 2)))
        denom = float(g.sum()) if g.sum() > 0 else 1.0
        rel = float(np.sum(np.abs(p - g)) / denom * 100.0)
        summary = (f"[metrics] images={len(p)}  spikelet count "
                   f"MAE={mae:.3f}  RMSE={rmse:.3f}  "
                   f"sum|GT-Pred|/sumGT={rel:.2f}%  "
                   f"(GT_total={int(g.sum())}, Pred_total={int(p.sum())})")
        print(summary)
        with open(os.path.join(args.out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(summary + "\n")

    print("Done.")


def build_argparser():
    p = argparse.ArgumentParser(description="MaskDINO 小穗点回归模型推理 / 可视化 / 计数对比")
    p.add_argument("--config_file", type=str,
                   default="output_v2/wheat305_r50_cat2_point_reg_coord5.0_cls2.0/config.yaml")
    p.add_argument("--weights", type=str,
                   default="output_v2/wheat305_r50_cat2_point_reg_coord5.0_cls2.0/model_final.pth")
    p.add_argument("--images", type=str,
                   default="/public/home/zncui/datasets/wheat50/coco_2cat_with_points/val2017",
                   help="待推理图片文件夹")
    p.add_argument("--gt_json", type=str,
                   default="/public/home/zncui/datasets/wheat50/coco_2cat_with_points/annotations/instances_val2017.json",
                   help="可选：COCO GT json，用于小穗计数对比；不存在则跳过")
    p.add_argument("--out_dir", type=str, default="./vis_spikelet_reg")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--imports", type=str, nargs="*", default=["maskdino"])

    p.add_argument("--score_thr", type=float, default=0.3,
                   help="实例分数阈值（保留 score>=该值 的实例）")
    p.add_argument("--point_threshold", type=float, default=None,
                   help="小穗点存在性阈值；默认沿用 config 中 MaskDINO.TEST.POINT_THRESHOLD")
    p.add_argument("--spikelet_cat_id", type=int, default=2,
                   help="GT json 中带小穗点的类别 id（wheat_2 = 2）")
    p.add_argument("--spikelet_class_id", type=int, default=1,
                   help="模型输出里 wheat_2 的连续 label（1 -> 见 thing_dataset_id_to_contiguous_id 2:1）")

    p.add_argument("--no_vis", dest="vis", action="store_false", help="不保存可视化图")
    p.add_argument("--draw_gt", action="store_true", help="在可视化中叠加 GT 小穗点（黄色空心圈）")
    p.add_argument("--draw_box", action="store_true", help="可视化中绘制实例框")
    p.set_defaults(vis=True)
    return p


def main():
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
