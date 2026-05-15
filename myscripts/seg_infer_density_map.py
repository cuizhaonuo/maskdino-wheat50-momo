#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskDINO 密度图推理脚本。

输入 COCO 格式数据集，使用训练好的 MaskDINO 模型进行推理，
对每张图中预测为 wheat_2（label==1）的实例绘制 N×4 可视化网格：
  col1: 原图 + 预测轮廓
  col2: GT points 散点图
  col3: GT density（由 points 经高斯核生成）
  col4: 模型预测的 pred_density

Usage:
  python seg_infer_density_map.py \
      --coco_root /path/to/coco \
      --split val2017 \
      --model_config_rel /path/to/config.yaml \
      --checkpoint_rel /path/to/model_final.pth \
      --fix_iou_type IOU \
      --fix_score_thr 0.3 \
      --fix_nms_iou_thr 0.4 \
      --out_dir /path/to/output \
      --spikelet_grid
"""

import argparse
import os
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
except:
    pass

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import generate_density_map

from pycocotools.coco import COCO


# ═══════════════════════════════════════════════════════════
# COCO helpers
# ═══════════════════════════════════════════════════════════

def load_coco(coco_root: str, split: str) -> COCO:
    json_path = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"COCO annotation not found: {json_path}")
    return COCO(json_path)


def coco_list_images(coco_root: str, split: str) -> List[str]:
    coco = load_coco(coco_root, split)
    img_dir = os.path.join(coco_root, split)
    images = []
    for img_id in sorted(coco.imgs.keys()):
        img_info = coco.imgs[img_id]
        images.append(os.path.join(img_dir, img_info["file_name"]))
    return images


def coco_get_gt_instances(coco: COCO, img_path: str) -> List[Dict]:
    """返回每个 GT 实例的 points, mask, bbox_xywh."""
    basename = os.path.basename(img_path)
    img_id = None
    for iid, info in coco.imgs.items():
        if os.path.basename(info["file_name"]) == basename:
            img_id = iid
            break
    if img_id is None:
        return []

    img_info = coco.imgs[img_id]
    H, W = img_info["height"], img_info["width"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    instances = []
    for ann in anns:
        # ---- mask from polygon segmentation ----
        seg = ann.get("segmentation", [])
        mask = np.zeros((H, W), dtype=bool)
        if isinstance(seg, list) and len(seg) > 0:
            mask_uint8 = np.zeros((H, W), dtype=np.uint8)
            for poly in seg:
                if len(poly) < 6:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(mask_uint8, [pts], 1)
            mask = mask_uint8.astype(bool)
        elif isinstance(seg, dict) and "counts" in seg:
            from pycocotools import mask as coco_mask_util
            decoded = coco_mask_util.decode(seg)
            if decoded.ndim == 3:
                decoded = decoded[..., 0]
            mask = decoded.astype(bool)

        # ---- points ----
        pts = ann.get("points", [])
        if pts and len(pts) > 0:
            points = np.array(pts, dtype=np.float32).reshape(-1, 2)
        else:
            points = np.zeros((0, 2), dtype=np.float32)

        instances.append({
            "points": points,
            "mask": mask,
            "bbox_xywh": np.array(ann.get("bbox", [0, 0, 0, 0]), dtype=np.float32),
            "category_id": ann.get("category_id", 0),
        })
    return instances


# ═══════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════

def build_predictor(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda:0",
    score_thr_test: float = 0.01,
) -> DefaultPredictor:
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr_test
    cfg.freeze()
    return DefaultPredictor(cfg)


# ═══════════════════════════════════════════════════════════
# Matching predictions to GT
# ═══════════════════════════════════════════════════════════

def match_predictions_to_gt(
    pred_masks: np.ndarray,          # (N_pred, H, W) bool
    gt_instances: List[Dict],
) -> Dict[int, int]:
    """Match each prediction to best GT by mask IoU. Returns {pred_idx: gt_idx}."""
    match = {}
    if not gt_instances:
        return match

    H_gt, W_gt = gt_instances[0]["mask"].shape
    N_pred = pred_masks.shape[0]

    for p_idx in range(N_pred):
        pmask = pred_masks[p_idx]
        # Derive tight bbox from pred mask for efficient IoU
        ys, xs = np.where(pmask)
        if len(xs) == 0:
            continue
        x1, x2 = max(0, xs.min()), min(W_gt, xs.max() + 1)
        y1, y2 = max(0, ys.min()), min(H_gt, ys.max() + 1)
        if x2 <= x1 or y2 <= y1:
            continue

        pred_crop = pmask[y1:y2, x1:x2]
        best_gt, best_iou = -1, 0.0
        for g_idx, gt in enumerate(gt_instances):
            gt_crop = gt["mask"][y1:y2, x1:x2]
            inter = np.logical_and(gt_crop, pred_crop).sum()
            union = np.logical_or(gt_crop, pred_crop).sum()
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_gt = iou, g_idx
        if best_gt >= 0 and best_iou >= 0.05:
            match[p_idx] = best_gt
    return match


# ═══════════════════════════════════════════════════════════
# Spikelet grid visualization
# ═══════════════════════════════════════════════════════════

def visualize_spikelet_grid(
    img_path: str,
    pred_masks: np.ndarray,          # (N_pred, H, W) bool
    pred_scores: np.ndarray,         # (N_pred,)
    pred_labels: np.ndarray,         # (N_pred,)
    pred_density_maps: np.ndarray,   # (N_pred, H, W) float32
    gt_instances: List[Dict],
    match: Dict[int, int],
    out_dir: str,
):
    """Generate N×4 grid: [original, GT points, GT density, pred density]."""
    # Filter to label==1 (wheat_2) predictions
    wheat2_indices = [i for i, lbl in enumerate(pred_labels) if lbl == 1]
    N = len(wheat2_indices)
    if N == 0:
        print(f"  [spikelet_grid] No wheat_2 predictions for {os.path.basename(img_path)}, skip.")
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(N, 4, figsize=(4 * 3.5, N * 3.5))
    if N == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Original + Pred Outline", "GT Points", "GT Density (Gaussian)", "Pred Density (Model)"]

    for row, p_idx in enumerate(wheat2_indices):
        gt_idx = match.get(p_idx, -1)
        mask = pred_masks[p_idx]
        score = pred_scores[p_idx]
        density = pred_density_maps[p_idx]

        # Pred mask contour (global coords)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Common axis setup for all 4 columns
        for col in range(4):
            ax = axes[row, col]
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img_rgb, extent=[0, W, H, 0], zorder=0)

        # ---- Col 0: Original image + pred mask outline ----
        ax0 = axes[row, 0]
        for cnt in contours:
            ax0.plot(cnt[:, 0, 0], cnt[:, 0, 1], color="lime", linewidth=1.5, zorder=2)
        if gt_idx >= 0:
            gt_mask = gt_instances[gt_idx]["mask"]
            gt_contours, _ = cv2.findContours(
                (gt_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in gt_contours:
                ax0.plot(cnt[:, 0, 0], cnt[:, 0, 1], color="red", linewidth=1.0, linestyle="--", zorder=2)

        # ---- Col 1: GT points scatter ----
        ax1 = axes[row, 1]
        if gt_idx >= 0:
            gt_pts = gt_instances[gt_idx]["points"]
            if len(gt_pts) > 0:
                ax1.scatter(gt_pts[:, 0], gt_pts[:, 1], c="red", s=5, marker="o", zorder=3)
        for cnt in contours:
            ax1.plot(cnt[:, 0, 0], cnt[:, 0, 1], color="lime", linewidth=1.5, zorder=2)

        # ---- Col 2: GT density (Gaussian from points) ----
        ax2 = axes[row, 2]
        if gt_idx >= 0:
            gt_pts = gt_instances[gt_idx]["points"]
            if len(gt_pts) > 0:
                gt_density = generate_density_map(gt_pts, (H, W), sigma=3.0)
                gt_density_norm = gt_density / (np.max(gt_density) + 1e-6)
                ax2.imshow(gt_density_norm, cmap="jet", alpha=0.55, extent=[0, W, H, 0], zorder=1)
        for cnt in contours:
            ax2.plot(cnt[:, 0, 0], cnt[:, 0, 1], color="lime", linewidth=1.5, zorder=2)

        # ---- Col 3: Pred density map ----
        ax3 = axes[row, 3]
        if density is not None and density.size > 0 and np.max(density) > 0:
            # density is already at full image resolution
            density_norm = density / (np.max(density) + 1e-6)
            ax3.imshow(density_norm, cmap="jet", alpha=0.55, extent=[0, W, H, 0], zorder=1)
        for cnt in contours:
            ax3.plot(cnt[:, 0, 0], cnt[:, 0, 1], color="lime", linewidth=1.5, zorder=2)

        # ---- Row label ----
        gt_npts = len(gt_instances[gt_idx]["points"]) if gt_idx >= 0 else 0
        pred_sum = float(density.sum()) if density is not None else 0
        axes[row, 0].set_ylabel(
            f"R{row + 1}  s={score:.2f}\nGT:{gt_npts}  pr_sum:{pred_sum:.1f}",
            fontsize=7, rotation=0, labelpad=35, va="center",
        )

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9)

    fig.suptitle(os.path.basename(img_path), fontsize=10, y=1.01)
    fig.tight_layout()

    out_name = os.path.splitext(os.path.basename(img_path))[0] + "_spikelet_grid.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [spikelet_grid] Saved -> {out_path}")


# ═══════════════════════════════════════════════════════════
# Main inference loop
# ═══════════════════════════════════════════════════════════

def run_inference(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load COCO ----
    print(f"Loading COCO: {args.coco_root} / {args.split}")
    coco = load_coco(args.coco_root, args.split)
    image_paths = coco_list_images(args.coco_root, args.split)
    print(f"  Found {len(image_paths)} images in split '{args.split}'")

    # ---- Build predictor ----
    print(f"Loading model: {args.model_config_rel}")
    print(f"  Checkpoint: {args.checkpoint_rel}")
    predictor = build_predictor(
        config_path=args.model_config_rel,
        checkpoint_path=args.checkpoint_rel,
        device=args.device,
    )

    # ---- Inference loop ----
    print(f"Inference on {len(image_paths)} images ...")
    csv_rows = [("filename", "gt_count", "pred_count_wheat1", "pred_count_wheat2")]

    for img_path in tqdm(image_paths, desc=args.split):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Cannot read: {img_path}")
            continue

        # Run model
        outputs = predictor(img)

        # Load GT for counts
        gt_instances = coco_get_gt_instances(coco, img_path)
        gt_count = len(gt_instances)

        if "instances" not in outputs or len(outputs["instances"]) == 0:
            csv_rows.append((os.path.basename(img_path), gt_count, 0, 0))
            continue

        instances = outputs["instances"].to("cpu")

        # ---- Filter by score threshold ----
        keep = instances.scores >= args.fix_score_thr
        instances = instances[keep]

        if len(instances) == 0:
            csv_rows.append((os.path.basename(img_path), gt_count, 0, 0))
            continue

        # Extract predictions
        pred_masks = instances.pred_masks.numpy().astype(bool)       # (N, H, W)
        pred_boxes = instances.pred_boxes.tensor.numpy()              # (N, 4) xyxy
        pred_scores = instances.scores.numpy()                        # (N,)
        pred_labels = instances.pred_classes.numpy()                  # (N,)

        has_density = instances.has("pred_density_maps")
        if has_density:
            pred_density_maps = instances.pred_density_maps.numpy().astype(np.float32)
        else:
            H, W = img.shape[:2]
            pred_density_maps = np.zeros((len(instances), H, W), dtype=np.float32)

        # Count per category
        n_wheat1 = int((pred_labels == 0).sum())
        n_wheat2 = int((pred_labels == 1).sum())
        csv_rows.append((os.path.basename(img_path), gt_count, n_wheat1, n_wheat2))

        # Match predictions to GT (for visualization pairing)
        match = match_predictions_to_gt(pred_masks, gt_instances)

        # Generate N×4 spikelet grid
        if args.spikelet_grid:
            visualize_spikelet_grid(
                img_path=img_path,
                pred_masks=pred_masks,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                pred_density_maps=pred_density_maps,
                gt_instances=gt_instances,
                match=match,
                out_dir=args.out_dir,
            )

    # ---- Save CSV ----
    csv_path = os.path.join(args.out_dir, f"{args.split}_counts.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for r in csv_rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    print(f"[Saved] CSV -> {csv_path}")

    print(f"\nDone. Output saved to: {args.out_dir}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def build_argparser():
    p = argparse.ArgumentParser(
        description="MaskDINO Density Map Inference with N×4 Spikelet Grid Visualization"
    )

    # Dataset
    p.add_argument("--coco_root", type=str, required=True,
                   help="COCO format dataset root directory.")
    p.add_argument("--split", type=str, default="val2017",
                   help="Dataset split name (default: val2017).")

    # Model
    p.add_argument("--model_config_rel", type=str, required=True,
                   help="Path to model config.yaml.")
    p.add_argument("--checkpoint_rel", type=str, required=True,
                   help="Path to model checkpoint (model_final.pth).")
    p.add_argument("--device", type=str, default="cuda:0")

    # Inference params
    p.add_argument("--fix_iou_type", type=str, default="IOU",
                   choices=["IOU", "IOU_Min"],
                   help="IoU type (default: IOU).")
    p.add_argument("--fix_score_thr", type=float, default=0.3,
                   help="Score threshold (default: 0.3).")
    p.add_argument("--fix_nms_iou_thr", type=float, default=0.4,
                   help="NMS IoU threshold (default: 0.4).")

    # Output
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for visualizations.")
    p.add_argument("--spikelet_grid", action="store_true",
                   help="Generate N×4 spikelet grid visualization.")

    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run_inference(args)
