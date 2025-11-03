#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wheat instance segmentation on Detectron2 with sliding-window + global Mask-NMS.
Modes:
- search: grid search (iou_type x score_thr x nms_iou_thr) on val then test with best
- infer : skip search, directly use (iou_type, score_thr, nms_iou_thr) on a split (default: test)

Template Method pattern:
- SegInferenceTemplate: defines the workflow skeleton (run_search / run_infer_only)
- Detectron2SegInference: concrete implementation using Detectron2

Notes
-----
1) 支持 Detectron2 Model Zoo 别名（如 COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml）。
   若 --model_config_rel 指向本地 yaml，也可直接使用；权重同理（本地/URL/detectron2://）。
2) 若你的 cfg/模型依赖第三方项目（如 MaskDINO），请通过 --imports maskdino 等先导入模块，以便注册 MetaArch/Heads 等。
3) ROI_HEADS.SCORE_THRESH_TEST 会被设置为 base_keep_score（默认 0.01），仅作为粗筛；真正阈值与 NMS 仍由网格搜索/固定参数控制。
"""

import os
import sys
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
except:
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
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF")
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
    # compute intersection region in global coords using cropped masks
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
        return inter / min(a.area, b.area) if min(a.area, b.area) > 0 else 0.0

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

    # ====== Modes ======
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
              f"sum|GT-Pred|={best_sum_diff}")

        # 将 print 输出保存到文件
        with open(os.path.join(self.outputs.out_dir, "best_params_on_val.txt"), "w", encoding="utf-8") as f:
            f.write(f"[Best on VAL] iou_type={best_cfg['iou_type'].value}, "
                    f"score_thr={best_cfg['score_thr']:.4f}, nms_iou_thr={best_cfg['nms_iou_thr']:.4f}, "
                    f"sum|GT-Pred|={best_sum_diff}\n")

        print("==> Running TEST with best parameters ...")
        self._run_on_split("test", best_cfg['iou_type'], best_cfg['score_thr'], best_cfg['nms_iou_thr'])

    def run_infer_only(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float):
        """Direct inference with fixed params on a chosen split (test/val/train)."""
        ensure_dir(self.outputs.out_dir)
        random.seed(0); np.random.seed(0)
        print("==> Loading model ..."); self.load_model()
        print(f"==> Running {split.upper()} with fixed params: "
              f"iou_type={iou_type.value}, score_thr={score_thr}, nms_iou_thr={nms_iou_thr}")
        self._run_on_split(split, iou_type, score_thr, nms_iou_thr)

    # ====== shared helpers ======
    def _run_on_split(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float):
        imgs_dir, _ = yolo_split_paths(self.dataset.yolo_root, split)
        images = list_images_in_dir(imgs_dir)

        rows = [("filename", "ins_counts", "pred_counts")]
        xs, ys = [], []
        for img_path in tqdm(images, desc=f"{split.upper()} images"):
            proposals = self._proposals_for_image(img_path)
            final_dets = self._apply_thresh_nms(proposals, score_thr, nms_iou_thr, iou_type)
            pred_count = len(final_dets)
            gt_count = yolo_count_instances(yolo_label_path_for_image(img_path, self.dataset.yolo_root))
            rows.append((os.path.basename(img_path), gt_count, pred_count))
            xs.append(gt_count); ys.append(pred_count)
            if self.outputs.vis:
                self._visualize_and_save(img_path, final_dets, self.outputs.out_dir)

        if self.outputs.excel:
            csv_path = os.path.join(self.outputs.out_dir, f"{split}_pred_vs_gt.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                for r in rows: f.write(f"{r[0]},{r[1]},{r[2]}\n")
            print(f"[Saved] CSV -> {csv_path}")
            # def diff_count(csv_path, out_dir):
            #     import pandas as pd
            #     df = pd.read_csv(csv_path)
            #     df["abs_diff"] = (df["ins_counts"] - df["pred_counts"]).abs()
            #     total_abs_diff = df["abs_diff"].sum()
            #     total_ins = df["ins_counts"].sum()
            #     percentage = (total_abs_diff / total_ins) * 100
            #     print("总绝对差值:", total_abs_diff)
            #     print("ins_counts 总数:", total_ins)
            #     print(f"百分比: {percentage:.2f}%")
            #     # 将 print 输出保存到文件
            #     with open(os.path.join(out_dir, f"{split}_diff_count.txt"), "w", encoding="utf-8") as f:
            #         f.write(f"总绝对差值: {total_abs_diff}\n")
            #         f.write(f"ins_counts 总数: {total_ins}\n")
            #         f.write(f"百分比: {percentage:.2f}%\n")
            # diff_count(csv_path, self.outputs.out_dir)

        if self.outputs.draw_scatter and xs:
            self._draw_scatter(xs, ys, os.path.join(self.outputs.out_dir, f"scatter_{split}.png"))
            print(f"[Saved] Scatter -> {os.path.join(self.outputs.out_dir, f'scatter_{split}.png')}")

        print(f"{split.upper()} finished.")

    def _proposals_for_image(self, img_path: str) -> List[DetProposal]:
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(img_path)
        H, W = img.shape[:2]
        tile, overlap = self.sw.tile, self.sw.overlap
        stride = max(1, tile - overlap); padv = self.sw.pad_value

        proposals: List[DetProposal] = []

        y_list = list(range(0, max(1, H - tile + 1), stride))
        if y_list[-1] != max(0, H - tile): y_list.append(max(0, H - tile))
        x_list = list(range(0, max(1, W - tile + 1), stride))
        if x_list[-1] != max(0, W - tile): x_list.append(max(0, W - tile))

        score_deleted = 0
        for y0 in y_list:
            for x0 in x_list:
                tile_img = np.full((tile, tile, 3), padv, dtype=np.uint8)
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                h, w = y1 - y0, x1 - x0
                tile_img[:h, :w] = img[y0:y1, x0:x1]
                valid_mask = np.zeros((tile, tile), dtype=bool); valid_mask[:h, :w] = True

                insts = self.infer_tile(tile_img)
                # print(f"[Debug] Tile ({x0},{y0}) - got {len(insts)} instances")
                for ins in insts:
                    score = float(ins["score"])
                    if score < self.search.base_keep_score:
                        score_deleted += 1
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
        # print(f"[Debug] Total proposals before NMS: {len(proposals)}, deleted by base score: {score_deleted}")
        return proposals

    def _grid_search(self, val_cache: Dict[str, List[DetProposal]],
                         val_gt_counts: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
            score_list = linspace_inclusive(*self.search.score_thr_range)
            nms_list = linspace_inclusive(*self.search.nms_iou_thr_range)
            iou_types = [IOUType(t) for t in self.search.iou_types]
            best_cfg, best_sum = None, 1 << 62
            total = len(score_list) * len(nms_list) * len(iou_types)
            pbar = tqdm(total=total, desc="Grid search")

            # 保存所有搜索结果
            results = []

            for iout in iou_types:
                for s_thr in score_list:
                    for n_thr in nms_list:
                        sum_diff = 0
                        for img_path, props in val_cache.items():
                            pred = len(self._apply_thresh_nms(props, s_thr, n_thr, iout))
                            sum_diff += abs(val_gt_counts[img_path] - pred)

                        results.append({
                            "iou_type": iout.value,
                            "score_thr": s_thr,
                            "nms_thr": n_thr,
                            "sum_diff": sum_diff
                        })

                        if sum_diff < best_sum:
                            best_sum = sum_diff
                            best_cfg = {"iou_type": iout, "score_thr": s_thr, "nms_iou_thr": n_thr}
                        pbar.update(1)
            pbar.close()

            # 保存到 CSV
            csv_path = os.path.join(self.outputs.out_dir, "grid_search_results.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("iou_type,score_thr,nms_thr,sum_diff\n")
                for r in results:
                    f.write(f"{r['iou_type']},{r['score_thr']:.4f},{r['nms_thr']:.4f},{r['sum_diff']}\n")
            print(f"[Saved] Grid search results -> {csv_path}")

            return best_cfg, best_sum

    def _apply_thresh_nms(self, proposals: List[DetProposal],
                          score_thr: float, nms_thr: float, iou_type: IOUType) -> List[DetProposal]:
        fil = [p for p in proposals if p.score >= score_thr]
        return mask_nms(fil, nms_thr, iou_type) if fil else []

    def _visualize_and_save(self, img_path: str, dets: List[DetProposal],
                            out_dir: str, alpha: float = 0.8, draw_contour: bool = False):
        img = cv2.imread(img_path)
        if img is None: return
        overlay = img.copy()
        rng = np.random.default_rng(12345)
        for d in dets:
            color = rng.integers(30, 255, size=3, dtype=np.uint8).tolist()
            y1, y2 = d.top, int(d.top + d.mask.shape[0])
            x1, x2 = d.left, int(d.left + d.mask.shape[1])
            # roi = overlay[y1:y2, x1:x2]
            m = d.mask
            # roi[m] = (0.2 * roi[m] + 0.8 * np.array(color, dtype=np.uint8)).astype(np.uint8)
            if draw_contour:
                m8 = (m.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img[y1:y2, x1:x2], contours, -1, color, 2)

            draw_center = True
            if draw_center:
                ys, xs = np.where(m > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int(x1 + np.mean(xs))
                    cy = int(y1 + np.mean(ys))
                    cv2.circle(img, (cx, cy), radius=6, color=(0, 0, 255), thickness=-1)  # 红色点

        # vis = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # 在 img 上绘制 5x3 的网格
        h, w = img.shape[:2]
        # 可配置的网格与样式
        grid_rows = 5
        grid_cols = 3
        line_color = (255, 0, 0)  # BGR 中的蓝色
        thickness = 2
        for i in range(1, grid_rows):
            y = i * h // grid_rows
            cv2.line(img, (0, y), (w, y), color=line_color, thickness=thickness)
        for j in range(1, grid_cols):
            x = j * w // grid_cols
            cv2.line(img, (x, 0), (x, h), color=line_color, thickness=thickness)

        vis = img
        ensure_dir(out_dir)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), vis)

    def _draw_scatter(self, xs: List[int], ys: List[int], out_path: str):
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, s=10)
        mn, mx = min(xs + ys), max(xs + ys)
        plt.plot([mn, mx], [mn, mx], linestyle="--")
        plt.xlabel("Ground Truth Count"); plt.ylabel("Predicted Count")
        plt.title("GT vs Pred")
        plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


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
        # assume a model zoo alias like "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
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
            # fallback: try model zoo url from alias if possible
            try:
                weights = model_zoo.get_checkpoint_url(config_rel)
            except Exception:
                weights = None
    else:
        # no checkpoint specified -> try model zoo default for alias
        if use_mz:
            try:
                weights = model_zoo.get_checkpoint_url(config_rel)
            except Exception:
                weights = None
    return cfg_path, weights, use_mz


class Detectron2SegInference(SegInferenceTemplate):
    def load_model(self):
        # 0) optional imports for custom projects (e.g., maskdino)
        for m in self.model_cfg.imports:
            try:
                __import__(m)
                print(f"[imports] imported module: {m}")
            except Exception as e:
                print(f"[imports] WARNING: failed to import '{m}': {e}")

        # 1) resolve cfg & weights
        cfg_path, weights, used_mz = _resolve_cfg_and_ckpt(
            self.model_cfg.project_root,
            self.model_cfg.config_rel,
            self.model_cfg.checkpoint_rel
        )
        print(f"[cfg]     -> {cfg_path} ({'model_zoo' if used_mz else 'local'})")
        print(f"[weights] -> {weights}")

        # 2) build cfg
        cfg = get_cfg()
        from detectron2.projects.deeplab import add_deeplab_config
        add_deeplab_config(cfg)  # 有些 MaskDINO 配置依赖它
        from maskdino import add_maskdino_config
        add_maskdino_config(cfg)  # MaskDINO 自己的配置键
        cfg.merge_from_file(cfg_path)
        if weights is not None:
            cfg.MODEL.WEIGHTS = weights

        # device & score threshold
        cfg.MODEL.DEVICE = self.model_cfg.device
        # keep low thresh to collect more raw proposals; we filter again later
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(max(0.0, min(self.search.base_keep_score, 0.9)))

        # 3) predictor
        self.model = DefaultPredictor(cfg)

    @torch.no_grad()
    def infer_tile(self, tile_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Return List[{'bbox': [x1,y1,x2,y2], 'score': float, 'label': int, 'mask': HxW bool}]
        """
        outputs = self.model(tile_img)
        if "instances" not in outputs:
            return []
        inst = outputs["instances"]
        if len(inst) == 0:
            return []

        # to cpu numpy
        bboxes = inst.pred_boxes.tensor.detach().cpu().numpy() if inst.has("pred_boxes") else None
        scores = inst.scores.detach().cpu().numpy() if inst.has("scores") else None
        labels = inst.pred_classes.detach().cpu().numpy() if inst.has("pred_classes") else np.zeros((len(inst),), dtype=np.int32)

        if inst.has("pred_masks"):
            # pred_masks: (N, H, W) bool tensor
            masks = inst.pred_masks.detach().cpu().numpy().astype(bool)
        else:
            # 没有分割掩码的模型无法用于本脚本
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
    p = argparse.ArgumentParser(description="Wheat Segmentation (Template Method) on Detectron2 with search/infer modes")

    # Modes
    p.add_argument("--mode", choices=["search", "infer"], default="search",
                   help="search: grid search on val then test; infer: direct inference with fixed params")

    # Dataset
    p.add_argument("--yolo_root", type=str,
                   default="/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image")

    # Model (兼容：本地相对路径 / 模型动物园别名 / detectron2:// / http URL)
    p.add_argument("--project_root", type=str,
                   default=".")
    p.add_argument("--model_config_rel", type=str,
                   default="/public/home/zncui/pycharm_project/MaskDINO/output/maskdino_swinl-time-10-12-10-30/config.yaml")
    p.add_argument("--checkpoint_rel", type=str,
                   default="/public/home/zncui/pycharm_project/MaskDINO/output/maskdino_swinl-time-10-12-10-30/model_final.pth", help="local path / detectron2://... / http... (empty -> use model zoo default if alias)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--imports", type=str, nargs="*", default=['maskdino'],
                   help="Optional modules to import before building cfg (e.g., maskdino)")

    # Sliding window
    p.add_argument("--tile", type=int, default=800)
    p.add_argument("--overlap", type=int, default=256)

    # Search space (used only in --mode search)
    p.add_argument("--score_start", type=float, default=0.1)
    p.add_argument("--score_end", type=float, default=0.6)
    p.add_argument("--score_num", type=int, default=6)
    p.add_argument("--nms_start", type=float, default=0.1)
    p.add_argument("--nms_end", type=float, default=0.6)
    p.add_argument("--nms_num", type=int, default=12)
    p.add_argument("--iou_types", type=str, nargs="+", default=["IOU", "IOU_Min"])
    p.add_argument("--base_keep_score", type=float, default=0.01)

    # Infer-only fixed params
    p.add_argument("--infer_split", choices=["test", "val", "train"], default="test")
    p.add_argument("--fix_iou_type", choices=["IOU", "IOU_Min"], default="IOU_Min")
    p.add_argument("--fix_score_thr", type=float, default=0.5)
    p.add_argument("--fix_nms_iou_thr", type=float, default=0.6)

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
            nms_iou_thr=args.fix_nms_iou_thr
        )

if __name__ == "__main__":
    # 延迟导入 torch（仅在需要时），避免无 GPU 环境卡顿

    main()
