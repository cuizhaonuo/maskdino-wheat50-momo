#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wheat instance segmentation on Detectron2 with sliding-window + global Mask-NMS.
Modes:
- search: grid search (iou_type x score_thr x nms_iou_thr x nms_mode x merge_score_mode) on val then test with best
- infer : skip search, directly use fixed params (including nms_mode) on a split (default: test)

NEW (Resume + streaming CSV):
- 推理每张图片后立刻把结果追加写入 CSV（append + flush + fsync）
- 重新运行时自动读取已存在的 CSV，跳过已处理图片（resume）
- 为避免不同参数组合互相污染输出，按参数组合创建独立 run_dir（含 CSV/散点图/可视化）

Notes
-----
1) 支持 Detectron2 Model Zoo 别名（如 COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml）。
2) 若 cfg/模型依赖第三方项目（如 MaskDINO），请通过 --imports maskdino 等先导入模块。
3) ROI_HEADS.SCORE_THRESH_TEST 会被设置为 base_keep_score（默认 0.01）仅粗筛；真正阈值与 NMS 由网格搜索/固定参数控制。
"""

import os
import math
import glob
import argparse
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import torch  # noqa: E402

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
except Exception:
    pass

# ====== Detectron2 ======
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class DatasetConfig:
    yolo_root: str

@dataclass
class ModelConfig:
    project_root: str
    config_rel: str
    checkpoint_rel: str
    device: str = "cuda:0"
    imports: Tuple[str, ...] = tuple()

@dataclass
class SlidingWindowConfig:
    tile: int = 800
    overlap: int = 256
    pad_value: Tuple[int, int, int] = (114, 114, 114)  # BGR

@dataclass
class SearchSpace:
    iou_types: Tuple[str, ...] = ("IOU", "IOU_Min")
    score_thr_range: Tuple[float, float, int] = (0.1, 0.8, 30)
    nms_iou_thr_range: Tuple[float, float, int] = (0.1, 0.8, 30)

    # NEW: include NMS method in search
    nms_modes: Tuple[str, ...] = ("plain", "merge")          # plain=mask_nms, merge=mask_nms_with_merge
    merge_score_modes: Tuple[str, ...] = ("keep_max",)       # or ("keep_max","area_weighted")

    base_keep_score: float = 0.01

@dataclass
class OutputOptions:
    vis: bool = True
    excel: bool = True
    draw_scatter: bool = True
    out_dir: str = "./vis_wheat_seg"


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images_in_dir(d: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
            "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(d, e)))
    files.sort()
    return files

def yolo_split_paths(yolo_root: str, split: str) -> Tuple[str, str]:
    return os.path.join(yolo_root, "images", split), os.path.join(yolo_root, "labels", split)

def yolo_label_path_for_image(img_path: str, yolo_root: str) -> str:
    rel = os.path.relpath(img_path, os.path.join(yolo_root, "images"))
    split = rel.split(os.sep)[0]
    name = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(yolo_root, "labels", split, f"{name}.txt")

def yolo_count_instances(label_path: str) -> int:
    if not os.path.exists(label_path):
        return 0
    c = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            c += 1
    return c

def linspace_inclusive(start: float, end: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]

# NEW: resume helpers
def read_done_set(csv_path: str) -> set:
    """
    读取已存在的 csv，返回已完成的 filename 集合。
    兼容第一行是表头或不是表头。
    """
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("filename"):
                continue
            parts = s.split(",")
            if len(parts) >= 1:
                done.add(parts[0].strip())
    return done

def append_csv_row(csv_path: str, filename: str, gt: int, pred: int, write_header_if_new: bool = True):
    """
    追加一行到 csv。若文件不存在且 write_header_if_new=True，则先写表头。
    每次写完 flush + fsync，利于断点续跑。
    """
    is_new = not os.path.exists(csv_path)
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "a", encoding="utf-8") as f:
        if is_new and write_header_if_new:
            f.write("filename,ins_counts,pred_counts\n")
        f.write(f"{filename},{gt},{pred}\n")
        f.flush()
        os.fsync(f.fileno())


# -----------------------------
# Proposal & IoU / NMS
# -----------------------------
@dataclass
class DetProposal:
    bbox: np.ndarray
    score: float
    label: int
    mask: np.ndarray    # cropped to bbox (h,w) bool
    top: int            # global y of bbox top
    left: int           # global x of bbox left
    area: int           # mask.sum()

class IOUType(Enum):
    IOU = "IOU"
    IOU_Min = "IOU_Min"

def mask_iou_between(a: DetProposal, b: DetProposal, iou_type: IOUType) -> float:
    xA = max(a.left, b.left)
    yA = max(a.top, b.top)
    xB = min(int(round(a.left + a.mask.shape[1])), int(round(b.left + b.mask.shape[1])))
    yB = min(int(round(a.top + a.mask.shape[0])), int(round(b.top + b.mask.shape[0])))
    if xB <= xA or yB <= yA:
        return 0.0
    a_roi = a.mask[yA - a.top:yB - a.top, xA - a.left:xB - a.left]
    b_roi = b.mask[yA - b.top:yB - b.top, xA - b.left:xB - b.left]
    inter = np.logical_and(a_roi, b_roi).sum()
    if inter <= 0:
        return 0.0
    if iou_type == IOUType.IOU:
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0
    else:
        mn = min(a.area, b.area)
        return inter / mn if mn > 0 else 0.0

def mask_nms(proposals: List[DetProposal], iou_thr: float, iou_type: IOUType) -> List[DetProposal]:
    if not proposals:
        return []
    order = sorted(range(len(proposals)), key=lambda i: proposals[i].score, reverse=True)
    keep: List[int] = []
    suppressed = np.zeros(len(proposals), dtype=bool)
    for i_idx in order:
        if suppressed[i_idx]:
            continue
        keep.append(i_idx)
        for j_idx in order:
            if suppressed[j_idx] or j_idx == i_idx:
                continue
            if mask_iou_between(proposals[i_idx], proposals[j_idx], iou_type) >= iou_thr:
                suppressed[j_idx] = True
    return [proposals[i] for i in keep]

def _merge_two_proposals(a: DetProposal, b: DetProposal, score_mode: str = "keep_max") -> DetProposal:
    ax1, ay1 = a.left, a.top
    ax2, ay2 = a.left + a.mask.shape[1], a.top + a.mask.shape[0]
    bx1, by1 = b.left, b.top
    bx2, by2 = b.left + b.mask.shape[1], b.top + b.mask.shape[0]

    ux1, uy1 = min(ax1, bx1), min(ay1, by1)
    ux2, uy2 = max(ax2, bx2), max(ay2, by2)
    U_w, U_h = int(ux2 - ux1), int(uy2 - uy1)

    canvas = np.zeros((U_h, U_w), dtype=bool)

    ay_off, ax_off = a.top - uy1, a.left - ux1
    canvas[ay_off:ay_off + a.mask.shape[0], ax_off:ax_off + a.mask.shape[1]] |= a.mask

    by_off, bx_off = b.top - uy1, b.left - ux1
    canvas[by_off:by_off + b.mask.shape[0], bx_off:bx_off + b.mask.shape[1]] |= b.mask

    if not canvas.any():
        new_mask = canvas
        tight_x1 = tight_y1 = 0
    else:
        ys, xs = np.where(canvas)
        tight_x1, tight_x2 = xs.min(), xs.max() + 1
        tight_y1, tight_y2 = ys.min(), ys.max() + 1
        new_mask = canvas[tight_y1:tight_y2, tight_x1:tight_x2]

    new_left = int(ux1 + tight_x1)
    new_top = int(uy1 + tight_y1)
    new_h, new_w = new_mask.shape[0], new_mask.shape[1]
    new_bbox = np.array([new_left, new_top, new_left + new_w, new_top + new_h], dtype=float)
    new_area = int(new_mask.sum())

    if score_mode == "area_weighted":
        denom = max(1, a.area + b.area)
        new_score = float((a.score * a.area + b.score * b.area) / denom)
    else:
        new_score = float(max(a.score, b.score))

    return DetProposal(
        bbox=new_bbox,
        score=new_score,
        label=a.label,
        mask=new_mask.astype(bool),
        top=int(new_top),
        left=int(new_left),
        area=int(new_area),
    )

def mask_nms_with_merge(proposals: List[DetProposal], iou_thr: float, iou_type: IOUType,
                        score_mode: str = "keep_max") -> List[DetProposal]:
    if not proposals:
        return []

    order = sorted(range(len(proposals)), key=lambda i: proposals[i].score, reverse=True)
    used = np.zeros(len(proposals), dtype=bool)
    merged_results: List[DetProposal] = []

    for idx in order:
        if used[idx]:
            continue

        base = proposals[idx]
        used[idx] = True

        changed = True
        while changed:
            changed = False
            for j in order:
                if used[j]:
                    continue
                iou = mask_iou_between(base, proposals[j], iou_type)
                if iou >= iou_thr:
                    base = _merge_two_proposals(base, proposals[j], score_mode=score_mode)
                    used[j] = True
                    changed = True

        merged_results.append(base)

    return merged_results


# -----------------------------
# Template base
# -----------------------------
class SegInferenceTemplate:
    def __init__(self, dataset: DatasetConfig, model: ModelConfig, sw: SlidingWindowConfig,
                 search: SearchSpace, outputs: OutputOptions):
        self.dataset = dataset
        self.model_cfg = model
        self.sw = sw
        self.search = search
        self.outputs = outputs
        self.model = None  # backend-specific predictor

    # hooks
    def load_model(self): raise NotImplementedError
    def infer_tile(self, tile_img: np.ndarray) -> List[Dict[str, Any]]: raise NotImplementedError

    def run_search(self):
        """Grid-search on val, then test with best params."""
        ensure_dir(self.outputs.out_dir)
        random.seed(0); np.random.seed(0)
        print("==> Loading model ..."); self.load_model()

        print("==> Collecting raw proposals on VAL ...")
        val_imgs_dir, _ = yolo_split_paths(self.dataset.yolo_root, "val")
        val_images = list_images_in_dir(val_imgs_dir)
        val_cache, val_gt_counts = {}, {}
        for img_path in tqdm(val_images, desc="VAL images"):
            val_cache[img_path] = self._proposals_for_image(img_path)
            val_gt_counts[img_path] = yolo_count_instances(yolo_label_path_for_image(img_path, self.dataset.yolo_root))

        print("==> Grid searching best parameters on VAL ...")
        best_cfg, best_sum_diff = self._grid_search(val_cache, val_gt_counts)

        print(f"[Best on VAL] iou_type={best_cfg['iou_type'].value}, "
              f"score_thr={best_cfg['score_thr']:.4f}, nms_iou_thr={best_cfg['nms_iou_thr']:.4f}, "
              f"nms_mode={best_cfg['nms_mode']}, merge_score_mode={best_cfg['merge_score_mode']}, "
              f"sum|GT-Pred|={best_sum_diff}")

        with open(os.path.join(self.outputs.out_dir, "best_params_on_val.txt"), "w", encoding="utf-8") as f:
            f.write(f"[Best on VAL] iou_type={best_cfg['iou_type'].value}, "
                    f"score_thr={best_cfg['score_thr']:.4f}, nms_iou_thr={best_cfg['nms_iou_thr']:.4f}, "
                    f"nms_mode={best_cfg['nms_mode']}, merge_score_mode={best_cfg['merge_score_mode']}, "
                    f"sum|GT-Pred|={best_sum_diff}\n")

        print("==> Running TEST with best parameters (resume enabled) ...")
        self._run_on_split(
            "test",
            best_cfg["iou_type"],
            best_cfg["score_thr"],
            best_cfg["nms_iou_thr"],
            best_cfg["nms_mode"],
            best_cfg["merge_score_mode"],
        )

    def run_infer_only(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float,
                       nms_mode: str, merge_score_mode: str):
        """Direct inference with fixed params on a chosen split (test/val/train)."""
        ensure_dir(self.outputs.out_dir)
        random.seed(0); np.random.seed(0)
        print("==> Loading model ..."); self.load_model()
        print(f"==> Running {split.upper()} with fixed params (resume enabled): "
              f"iou_type={iou_type.value}, score_thr={score_thr}, nms_iou_thr={nms_iou_thr}, "
              f"nms_mode={nms_mode}, merge_score_mode={merge_score_mode}")
        self._run_on_split(split, iou_type, score_thr, nms_iou_thr, nms_mode, merge_score_mode)

    # ====== shared helpers ======
    def _make_run_tag(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float,
                      nms_mode: str, merge_score_mode: str) -> str:
        # 目录名尽量短但唯一，避免小数点太长
        return (f"{split}_iou-{iou_type.value}"
                f"_s-{score_thr:.4f}_n-{nms_iou_thr:.4f}"
                f"_m-{nms_mode}_ms-{merge_score_mode}")

    def _run_on_split(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float,
                      nms_mode: str, merge_score_mode: str):
        imgs_dir, _ = yolo_split_paths(self.dataset.yolo_root, split)
        images = list_images_in_dir(imgs_dir)

        # NEW: 每个参数组合一个独立 run_dir，避免输出互相污染
        run_tag = self._make_run_tag(split, iou_type, score_thr, nms_iou_thr, nms_mode, merge_score_mode)
        run_dir = os.path.join(self.outputs.out_dir, run_tag)
        ensure_dir(run_dir)
        vis_dir = os.path.join(run_dir, "vis")
        ensure_dir(vis_dir)

        # CSV：边跑边写 + resume
        csv_path = os.path.join(run_dir, f"{split}_pred_vs_gt.csv")
        done = read_done_set(csv_path)

        skipped = 0
        newly_processed = 0

        for img_path in tqdm(images, desc=f"{split.upper()} images", total=len(images)):
            fname = os.path.basename(img_path)
            if fname in done:
                skipped += 1
                continue

            proposals = self._proposals_for_image(img_path)
            final_dets = self._apply_thresh_nms(
                proposals, score_thr, nms_iou_thr, iou_type,
                nms_mode=nms_mode,
                merge_score_mode=merge_score_mode
            )
            pred_count = len(final_dets)
            gt_count = yolo_count_instances(yolo_label_path_for_image(img_path, self.dataset.yolo_root))

            # 推理完一张就写一行（append + flush + fsync）
            if self.outputs.excel:
                append_csv_row(csv_path, fname, gt_count, pred_count, write_header_if_new=True)

            newly_processed += 1

            # 可视化：同样每张写一次；resume 时如果图已存在则跳过（避免重复）
            if self.outputs.vis:
                out_img_path = os.path.join(vis_dir, fname)
                if not os.path.exists(out_img_path):
                    self._visualize_and_save(img_path, final_dets, vis_dir, draw_contour=True)

        print(f"[Resume] skipped={skipped}, newly_processed={newly_processed}")
        if self.outputs.excel:
            print(f"[Saved] CSV (append) -> {csv_path}")

        # Scatter：从 csv 回读全量点，保证 resume 后仍是完整散点图
        if self.outputs.draw_scatter and self.outputs.excel and os.path.exists(csv_path):
            xs, ys = [], []
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.lower().startswith("filename"):
                        continue
                    parts = s.split(",")
                    if len(parts) >= 3:
                        try:
                            xs.append(int(parts[1]))
                            ys.append(int(parts[2]))
                        except Exception:
                            pass
            if xs:
                scatter_path = os.path.join(run_dir, f"scatter_{split}.png")
                self._draw_scatter(xs, ys, scatter_path)
                print(f"[Saved] Scatter -> {scatter_path}")

        print(f"{split.upper()} finished. Outputs in: {run_dir}")

    def _proposals_for_image(self, img_path: str) -> List[DetProposal]:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        H, W = img.shape[:2]
        tile, overlap = self.sw.tile, self.sw.overlap
        stride = max(1, tile - overlap)
        padv = self.sw.pad_value

        proposals: List[DetProposal] = []

        y_list = list(range(0, max(1, H - tile + 1), stride))
        if y_list[-1] != max(0, H - tile):
            y_list.append(max(0, H - tile))
        x_list = list(range(0, max(1, W - tile + 1), stride))
        if x_list[-1] != max(0, W - tile):
            x_list.append(max(0, W - tile))

        for y0 in y_list:
            for x0 in x_list:
                tile_img = np.full((tile, tile, 3), padv, dtype=np.uint8)
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                h, w = y1 - y0, x1 - x0
                tile_img[:h, :w] = img[y0:y1, x0:x1]
                valid_mask = np.zeros((tile, tile), dtype=bool)
                valid_mask[:h, :w] = True

                insts = self.infer_tile(tile_img)
                for ins in insts:
                    score = float(ins["score"])
                    if score < self.search.base_keep_score:
                        continue

                    bbox = np.array(ins["bbox"], dtype=float)
                    mask = np.logical_and(ins["mask"].astype(bool), valid_mask)

                    x1t = max(0, min(tile, int(math.floor(bbox[0]))))
                    y1t = max(0, min(tile, int(math.floor(bbox[1]))))
                    x2t = max(0, min(tile, int(math.ceil(bbox[2]))))
                    y2t = max(0, min(tile, int(math.ceil(bbox[3]))))
                    if x2t <= x1t or y2t <= y1t:
                        continue

                    mask_crop = mask[y1t:y2t, x1t:x2t]
                    area = int(mask_crop.sum())
                    if area <= 0:
                        continue

                    gx1, gy1 = x0 + x1t, y0 + y1t
                    gx2, gy2 = x0 + x2t, y0 + y2t
                    gx1 = max(0, min(W, gx1)); gx2 = max(0, min(W, gx2))
                    gy1 = max(0, min(H, gy1)); gy2 = max(0, min(H, gy2))
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue

                    proposals.append(DetProposal(
                        bbox=np.array([gx1, gy1, gx2, gy2], dtype=float),
                        score=score,
                        label=int(ins.get("label", 0)),
                        mask=mask_crop,
                        top=int(gy1),
                        left=int(gx1),
                        area=area
                    ))
        return proposals

    def _grid_search(self, val_cache: Dict[str, List[DetProposal]],
                     val_gt_counts: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
        score_list = linspace_inclusive(*self.search.score_thr_range)
        nms_list = linspace_inclusive(*self.search.nms_iou_thr_range)
        iou_types = [IOUType(t) for t in self.search.iou_types]
        nms_modes = list(self.search.nms_modes)
        merge_score_modes = list(self.search.merge_score_modes)

        best_cfg, best_sum = None, 1 << 62

        total = len(score_list) * len(nms_list) * len(iou_types) * len(nms_modes) * len(merge_score_modes)
        pbar = tqdm(total=total, desc="Grid search")

        results = []

        for iout in iou_types:
            for nms_mode in nms_modes:
                for merge_score_mode in merge_score_modes:
                    # plain NMS doesn't need multiple merge_score_mode values; skip redundant combos
                    if nms_mode == "plain" and merge_score_mode != merge_score_modes[0]:
                        continue

                    for s_thr in score_list:
                        for n_thr in nms_list:
                            sum_diff = 0
                            for img_path, props in val_cache.items():
                                pred = len(self._apply_thresh_nms(
                                    props, s_thr, n_thr, iout,
                                    nms_mode=nms_mode,
                                    merge_score_mode=merge_score_mode
                                ))
                                sum_diff += abs(val_gt_counts[img_path] - pred)

                            results.append({
                                "iou_type": iout.value,
                                "score_thr": s_thr,
                                "nms_thr": n_thr,
                                "nms_mode": nms_mode,
                                "merge_score_mode": merge_score_mode,
                                "sum_diff": sum_diff
                            })

                            if sum_diff < best_sum:
                                best_sum = sum_diff
                                best_cfg = {
                                    "iou_type": iout,
                                    "score_thr": s_thr,
                                    "nms_iou_thr": n_thr,
                                    "nms_mode": nms_mode,
                                    "merge_score_mode": merge_score_mode
                                }
                            pbar.update(1)

        pbar.close()

        csv_path = os.path.join(self.outputs.out_dir, "grid_search_results.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("iou_type,score_thr,nms_thr,nms_mode,merge_score_mode,sum_diff\n")
            for r in results:
                f.write(f"{r['iou_type']},{r['score_thr']:.4f},{r['nms_thr']:.4f},"
                        f"{r['nms_mode']},{r['merge_score_mode']},{r['sum_diff']}\n")
        print(f"[Saved] Grid search results -> {csv_path}")

        return best_cfg, best_sum

    def _apply_thresh_nms(self, proposals: List[DetProposal],
                          score_thr: float, nms_thr: float, iou_type: IOUType,
                          nms_mode: str = "merge",
                          merge_score_mode: str = "keep_max") -> List[DetProposal]:
        fil = [p for p in proposals if p.score >= score_thr]
        if not fil:
            return []

        if nms_mode == "plain":
            return mask_nms(fil, nms_thr, iou_type)
        elif nms_mode == "merge":
            return mask_nms_with_merge(fil, nms_thr, iou_type, score_mode=merge_score_mode)
        else:
            raise ValueError(f"Unknown nms_mode: {nms_mode}")

    def _visualize_and_save(self, img_path: str, dets: List[DetProposal],
                            out_dir: str, alpha: float = 0.8, draw_contour: bool = False,
                            draw_center: bool = False):
        img = cv2.imread(img_path)
        if img is None:
            return
        ensure_dir(out_dir)

        overlay = img.copy()
        rng = np.random.default_rng(12345)

        for d in dets:
            color = rng.integers(64, 256, size=3, dtype=np.uint8).tolist()  # [B, G, R]

            H, W = img.shape[:2]
            y1o, x1o = int(d.top), int(d.left)
            y2o = int(d.top + d.mask.shape[0])
            x2o = int(d.left + d.mask.shape[1])

            y1 = max(0, min(H, y1o))
            y2 = max(0, min(H, y2o))
            x1 = max(0, min(W, x1o))
            x2 = max(0, min(W, x2o))
            if y2 <= y1 or x2 <= x1:
                continue

            roi = overlay[y1:y2, x1:x2]

            my1 = y1 - y1o
            my2 = my1 + (y2 - y1)
            mx1 = x1 - x1o
            mx2 = mx1 + (x2 - x1)

            m = d.mask[my1:my2, mx1:mx2].astype(bool)
            if m.shape[0] != roi.shape[0] or m.shape[1] != roi.shape[1]:
                m = m[:roi.shape[0], :roi.shape[1]]

            roi[m] = (alpha * np.array(color, dtype=np.float32) +
                      (1 - alpha) * roi[m].astype(np.float32)).astype(np.uint8)

            if draw_contour:
                m8 = (m.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay[y1:y2, x1:x2], contours, -1, color, 2)

            if draw_center:
                ys, xs = np.where(m > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int(x1 + np.mean(xs))
                    cy = int(y1 + np.mean(ys))
                    cv2.circle(overlay, (cx, cy), radius=6, color=tuple(color), thickness=-1)

        vis = cv2.addWeighted(overlay, 1.0, img, 0.0, 0)
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, vis)

    def _draw_scatter(self, xs: List[int], ys: List[int], out_path: str):
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, s=10)
        mn, mx = min(xs + ys), max(xs + ys)
        plt.plot([mn, mx], [mn, mx], linestyle="--")
        plt.xlabel("Ground Truth Count")
        plt.ylabel("Predicted Count")
        plt.title("GT vs Pred")
        plt.tight_layout()
        ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=160)
        plt.close()


# -----------------------------
# Detectron2 impl
# -----------------------------
def _resolve_cfg_and_ckpt(project_root: str,
                          config_rel: str,
                          checkpoint_rel: Optional[str]) -> Tuple[str, Optional[str], bool]:
    """
    Return (cfg_path_or_alias, weights_path_or_url, use_model_zoo_cfg)
    - If local cfg exists: use it; else treat config_rel as model_zoo alias.
    - If weights local path exists: use it; elif startswith http/detectron2:// : pass through;
      elif None: try model_zoo checkpoint for the alias (if alias valid).
    """
    local_cfg = os.path.join(project_root, config_rel)
    use_mz = False
    if os.path.isfile(local_cfg):
        cfg_path = local_cfg
    else:
        cfg_path = model_zoo.get_config_file(config_rel)
        use_mz = True

    weights = None
    if checkpoint_rel:
        local_w = os.path.join(project_root, checkpoint_rel)
        if os.path.isfile(local_w):
            weights = local_w
        elif checkpoint_rel.startswith("http") or checkpoint_rel.startswith("detectron2://"):
            weights = checkpoint_rel
        else:
            try:
                weights = model_zoo.get_checkpoint_url(config_rel)
            except Exception:
                weights = None
    else:
        if use_mz:
            try:
                weights = model_zoo.get_checkpoint_url(config_rel)
            except Exception:
                weights = None
    return cfg_path, weights, use_mz


class Detectron2SegInference(SegInferenceTemplate):
    def load_model(self):
        for m in self.model_cfg.imports:
            try:
                __import__(m)
                print(f"[imports] imported module: {m}")
            except Exception as e:
                print(f"[imports] WARNING: failed to import '{m}': {e}")

        cfg_path, weights, used_mz = _resolve_cfg_and_ckpt(
            self.model_cfg.project_root,
            self.model_cfg.config_rel,
            self.model_cfg.checkpoint_rel
        )
        print(f"[cfg]     -> {cfg_path} ({'model_zoo' if used_mz else 'local'})")
        print(f"[weights] -> {weights}")

        cfg = get_cfg()

        # Some configs depend on deeplab
        from detectron2.projects.deeplab import add_deeplab_config
        add_deeplab_config(cfg)

        # If you use MaskDINO configs, keep this. Otherwise wrap with try/except.
        from maskdino import add_maskdino_config
        add_maskdino_config(cfg)

        cfg.merge_from_file(cfg_path)
        if weights is not None:
            cfg.MODEL.WEIGHTS = weights

        cfg.MODEL.DEVICE = self.model_cfg.device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(max(0.0, min(self.search.base_keep_score, 0.9)))

        self.model = DefaultPredictor(cfg)

    @torch.no_grad()
    def infer_tile(self, tile_img: np.ndarray) -> List[Dict[str, Any]]:
        outputs = self.model(tile_img)
        if "instances" not in outputs:
            return []
        inst = outputs["instances"]
        if len(inst) == 0:
            return []

        bboxes = inst.pred_boxes.tensor.detach().cpu().numpy() if inst.has("pred_boxes") else None
        scores = inst.scores.detach().cpu().numpy() if inst.has("scores") else None
        labels = inst.pred_classes.detach().cpu().numpy() if inst.has("pred_classes") else np.zeros((len(inst),), dtype=np.int32)

        if inst.has("pred_masks"):
            masks = inst.pred_masks.detach().cpu().numpy().astype(bool)
        else:
            return []

        out = []
        N = bboxes.shape[0] if bboxes is not None else masks.shape[0]
        for i in range(N):
            x1, y1, x2, y2 = bboxes[i].tolist()
            out.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(scores[i]) if scores is not None else 0.0,
                "label": int(labels[i]) if labels is not None else 0,
                "mask": masks[i]
            })
        return out


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Wheat Segmentation on Detectron2 with search/infer + resume CSV")

    # Modes
    p.add_argument("--mode", choices=["search", "infer"], default="search",
                   help="search: grid search on val then test; infer: direct inference with fixed params")

    # Dataset
    p.add_argument("--yolo_root", type=str,
                   default="/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image")

    # Model
    p.add_argument("--project_root", type=str, default=".")
    p.add_argument("--model_config_rel", type=str,
                   default="/public/home/zncui/pycharm_project/MaskDINO/output/maskdino_swinl-time-10-12-10-30/config.yaml")
    p.add_argument("--checkpoint_rel", type=str,
                   default="/public/home/zncui/pycharm_project/MaskDINO/output/maskdino_swinl-time-10-12-10-30/model_final.pth")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--imports", type=str, nargs="*", default=['maskdino'],
                   help="Optional modules to import before building cfg (e.g., maskdino)")

    # Sliding window
    p.add_argument("--tile", type=int, default=800)
    p.add_argument("--overlap", type=int, default=256)

    # Search space (used only in --mode search)
    p.add_argument("--score_start", type=float, default=0.1)
    p.add_argument("--score_end", type=float, default=0.6)
    p.add_argument("--score_num", type=int, default=1)
    p.add_argument("--nms_start", type=float, default=0.1)
    p.add_argument("--nms_end", type=float, default=0.6)
    p.add_argument("--nms_num", type=int, default=1)
    p.add_argument("--iou_types", type=str, nargs="+", default=["IOU", "IOU_Min"])
    p.add_argument("--base_keep_score", type=float, default=0.01)

    # NEW: search over NMS method and merge score mode
    p.add_argument("--nms_modes", type=str, nargs="+", default=["plain", "merge"],
                   choices=["plain", "merge"])
    p.add_argument("--merge_score_modes", type=str, nargs="+", default=["keep_max"],
                   choices=["keep_max", "area_weighted"])

    # Infer-only fixed params
    p.add_argument("--infer_split", choices=["test", "val", "train"], default="test")
    p.add_argument("--fix_iou_type", choices=["IOU", "IOU_Min"], default="IOU_Min")
    p.add_argument("--fix_score_thr", type=float, default=0.4)
    p.add_argument("--fix_nms_iou_thr", type=float, default=0.372)

    # NEW: infer-only NMS choices
    p.add_argument("--fix_nms_mode", choices=["plain", "merge"], default="merge")
    p.add_argument("--fix_merge_score_mode", choices=["keep_max", "area_weighted"], default="keep_max")

    # Outputs
    p.add_argument("--out_dir", type=str, default="./swinl_guo_big_images")
    p.add_argument("--no_vis", action="store_true")
    p.add_argument("--no_excel", action="store_true")
    p.add_argument("--no_scatter", action="store_true")

    return p


def main():
    args = build_argparser().parse_args()

    dataset = DatasetConfig(yolo_root=args.yolo_root)
    model = ModelConfig(
        project_root=args.project_root,
        config_rel=args.model_config_rel,
        checkpoint_rel=args.checkpoint_rel if args.checkpoint_rel else "",
        device=args.device,
        imports=tuple(args.imports)
    )
    sw = SlidingWindowConfig(tile=args.tile, overlap=args.overlap, pad_value=(114, 114, 114))
    search = SearchSpace(
        iou_types=tuple(args.iou_types),
        score_thr_range=(args.score_start, args.score_end, args.score_num),
        nms_iou_thr_range=(args.nms_start, args.nms_end, args.nms_num),
        nms_modes=tuple(args.nms_modes),
        merge_score_modes=tuple(args.merge_score_modes),
        base_keep_score=args.base_keep_score
    )
    outputs = OutputOptions(
        vis=not args.no_vis,
        excel=not args.no_excel,
        draw_scatter=not args.no_scatter,
        out_dir=args.out_dir
    )

    runner = Detectron2SegInference(dataset, model, sw, search, outputs)

    if args.mode == "search":
        runner.run_search()
    else:
        runner.run_infer_only(
            split=args.infer_split,
            iou_type=IOUType(args.fix_iou_type),
            score_thr=args.fix_score_thr,
            nms_iou_thr=args.fix_nms_iou_thr,
            nms_mode=args.fix_nms_mode,
            merge_score_mode=args.fix_merge_score_mode
        )


if __name__ == "__main__":
    main()
