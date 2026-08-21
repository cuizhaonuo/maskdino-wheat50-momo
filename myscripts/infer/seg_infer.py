#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wheat instance segmentation on Detectron2 with sliding-window + global Mask-NMS.
Modes:
- search: grid search (iou_type x score_thr x nms_iou_thr) on val then test with best
- infer : skip search, directly use (iou_type, score_thr, nms_iou_thr) on a split (default: test)

数据源（--source）：
- yolo : <data_root>/images/<split>/ + <data_root>/labels/<split>/<stem>.txt   需要 --split
- coco : <data_root>/<split>/ + <data_root>/annotations/instances_<split>.json  需要 --split（传子目录名，如 val2017）
- dir  : <data_root>/ 下直接列图，无 GT，只支持 --mode infer                     不需要 --split

配置文件（--config）
--------------------
参数多，推荐把常用组合写进 YAML，键名与 --xxx 一一对应（去掉 -- 前缀）。
优先级：argparse 默认值 < 配置文件 < 命令行显式传入。不传 --config 时行为与纯命令行完全一致。

  python myscripts/infer/seg_infer.py --config myscripts/infer/configs/coco_val2017.yaml
  # 临时改两个参数，其余仍取自配置文件：
  python myscripts/infer/seg_infer.py --config myscripts/infer/configs/coco_val2017.yaml \
      --split train2017 --fix_score_thr 0.5

现成配置见 myscripts/infer/configs/（coco_val2017 / yolo_guo_search / plain_dir）。
每次运行会把合并后的最终配置写到 <out_dir>/effective_config.yaml，可直接拿它 --config 复现。

用法（纯命令行）
----------------
# YOLO 大图：在 val 上网格搜参数，再用最优参数跑 test
python myscripts/infer/seg_infer.py --source yolo \
  --data_root /public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image \
  --mode search --search_split val --eval_split test \
  --model_config_rel <cfg.yaml> --checkpoint_rel <model_final.pth> --out_dir ./vis_yolo

# COCO 数据集：固定参数直接推理 val2017（GT 数从 instances_val2017.json 统计）
python myscripts/infer/seg_infer.py --source coco \
  --data_root /public/home/zncui/datasets/wheat50/coco_2cat --split val2017 --mode infer \
  --model_config_rel <cfg.yaml> --checkpoint_rel <model_final.pth> --out_dir ./vis_coco_val

# 普通图片文件夹（无标注）：只跑推理，不出散点图
python myscripts/infer/seg_infer.py --source dir \
  --data_root /path/to/images --mode infer \
  --model_config_rel <cfg.yaml> --checkpoint_rel <model_final.pth> --out_dir ./vis_dir

Template Method pattern:
- SegInferenceTemplate: defines the workflow skeleton (run_search / run_infer_only)
- Detectron2SegInference: concrete implementation using Detectron2
- DataSource: 数据组织形式（yolo / coco / 普通目录）的策略，负责「图片列表」与「GT 实例数」

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
import json
import argparse
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Iterable
import torch  # noqa: E402

import cv2
import yaml
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
    # 兼容旧调用（如 coco_add_pseudo_labels.py 的 DatasetConfig(yolo_root="")）：
    # data_root 为空时自动回落到 yolo_root。
    yolo_root: str = ""
    source: str = "yolo"                     # yolo | coco | dir
    data_root: str = ""                      # yolo/coco 根目录，或 dir 模式的图片目录
    coco_ann: str = ""                       # 可选：显式指定标注 json，覆盖 annotations/instances_<split>.json
    gt_cat_ids: Tuple[int, ...] = ()         # GT 计数的类别过滤，空 = 全算

    @property
    def root(self) -> str:
        return self.data_root or self.yolo_root

@dataclass
class ModelConfig:
    project_root: str
    config_rel: str
    checkpoint_rel: str
    device: str = "cuda:0"
    imports: Tuple[str, ...] = tuple()
    # 直接透传给 detectron2 cfg 的覆盖项（KEY VALUE KEY VALUE ...），在所有其它设置之后生效
    opts: Tuple[str, ...] = tuple()

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

def yolo_count_instances(label_path: str, cls_ids: Optional[Iterable[int]] = None) -> int:
    """统计 YOLO txt 的实例数。cls_ids 非空时只统计首列类别在其中的行（默认 None = 全算）。"""
    if not os.path.exists(label_path):
        return 0
    keep = set(cls_ids) if cls_ids else None
    c = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if keep is not None:
                try:
                    if int(float(s.split()[0])) not in keep:
                        continue
                except (ValueError, IndexError):
                    continue
            c += 1
    return c

def linspace_inclusive(start: float, end: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]


# -----------------------------
# Data sources (yolo / coco / plain dir)
# -----------------------------
class DataSource:
    """数据组织形式的策略：只负责「有哪些图」和「每张图的 GT 实例数」。

    gt_count 返回 None 表示该数据源没有 GT —— 这是流程判断「无标注」的唯一信号
    （CSV 少一列、跳过散点图、不允许 search）。
    """
    supports_search: bool = True

    def images(self, split: str) -> List[str]:
        raise NotImplementedError

    def gt_count(self, img_path: str, split: str) -> Optional[int]:
        raise NotImplementedError

    def tag(self, split: str) -> str:
        """输出文件名用的短标识。"""
        return split

    def finish(self):
        """一个 split 跑完后的收尾（默认什么都不做）。"""
        pass


class YoloSource(DataSource):
    """<root>/images/<split>/*.jpg + <root>/labels/<split>/<stem>.txt"""

    def __init__(self, root: str, gt_cat_ids: Tuple[int, ...] = ()):
        self.root = root
        self.gt_cat_ids = tuple(gt_cat_ids)

    def images(self, split: str) -> List[str]:
        imgs_dir, _ = yolo_split_paths(self.root, split)
        if not os.path.isdir(imgs_dir):
            raise FileNotFoundError(f"[yolo] 图片目录不存在: {imgs_dir}")
        return list_images_in_dir(imgs_dir)

    def gt_count(self, img_path: str, split: str) -> Optional[int]:
        return yolo_count_instances(yolo_label_path_for_image(img_path, self.root),
                                    cls_ids=self.gt_cat_ids or None)


class CocoSource(DataSource):
    """<root>/<split>/*.jpg + <root>/annotations/instances_<split>.json

    split 直接传子目录名（如 val2017），标注默认推导为 annotations/instances_<split>.json，
    可用 coco_ann 显式覆盖（兼容 split_coco_points_todo.py 产出的 images/ + annotations/instances.json）。
    """

    def __init__(self, root: str, coco_ann: str = "", gt_cat_ids: Tuple[int, ...] = ()):
        self.root = root
        self.coco_ann = coco_ann
        self.gt_cat_ids = set(gt_cat_ids)
        self._counts: Dict[str, Dict[str, int]] = {}   # split -> {basename: count}
        self._missing = 0

    def _ann_path(self, split: str) -> str:
        if self.coco_ann:
            return self.coco_ann
        return os.path.join(self.root, "annotations", f"instances_{split}.json")

    def _counts_for(self, split: str) -> Dict[str, int]:
        """懒加载并缓存：一个 split 的 json 只解析一次。"""
        if split in self._counts:
            return self._counts[split]

        ann_path = self._ann_path(split)
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(
                f"[coco] 标注文件不存在: {ann_path}\n"
                f"       split 需要传子目录名（如 val2017）；布局不标准时用 --coco_ann 显式指定。")
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # json 里 file_name 可能带子路径，统一用 basename 匹配
        id2name = {im["id"]: os.path.basename(im["file_name"]) for im in data.get("images", [])}
        counts = {name: 0 for name in id2name.values()}
        for a in data.get("annotations", []):
            if int(a.get("iscrowd", 0)) == 1:
                continue
            if self.gt_cat_ids and int(a.get("category_id", -1)) not in self.gt_cat_ids:
                continue
            name = id2name.get(a.get("image_id"))
            if name is None:
                continue
            counts[name] += 1

        cat_note = f", 只统计 category_id={sorted(self.gt_cat_ids)}" if self.gt_cat_ids else ""
        print(f"[coco] {ann_path} -> {len(counts)} images, {sum(counts.values())} instances{cat_note}")
        self._counts[split] = counts
        return counts

    def images(self, split: str) -> List[str]:
        imgs_dir = os.path.join(self.root, split)
        if not os.path.isdir(imgs_dir):
            raise FileNotFoundError(f"[coco] 图片目录不存在: {imgs_dir}")
        return list_images_in_dir(imgs_dir)

    def gt_count(self, img_path: str, split: str) -> Optional[int]:
        counts = self._counts_for(split)
        name = os.path.basename(img_path)
        if name not in counts:
            self._missing += 1
            return 0
        return counts[name]

    def finish(self):
        if self._missing:
            print(f"[coco] WARNING: {self._missing} 张图在标注 json 中查不到，GT 计为 0")
            self._missing = 0


class PlainDirSource(DataSource):
    """普通图片目录：只有图，没有 GT，因此不支持 search。"""
    supports_search = False

    def __init__(self, images_dir: str):
        self.images_dir = images_dir

    def images(self, split: str) -> List[str]:      # split 被忽略
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"[dir] 图片目录不存在: {self.images_dir}")
        return list_images_in_dir(self.images_dir)

    def gt_count(self, img_path: str, split: str) -> Optional[int]:
        return None

    def tag(self, split: str) -> str:
        return os.path.basename(self.images_dir.rstrip(os.sep)) or "images"


def build_data_source(cfg: DatasetConfig) -> DataSource:
    root = cfg.root
    if not root:
        raise ValueError("数据源根目录为空：请通过 --data_root（或旧的 --yolo_root）指定")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"数据源根目录不存在: {root}")

    if cfg.source == "yolo":
        return YoloSource(root, cfg.gt_cat_ids)
    if cfg.source == "coco":
        return CocoSource(root, cfg.coco_ann, cfg.gt_cat_ids)
    if cfg.source == "dir":
        return PlainDirSource(root)
    raise ValueError(f"未知的 --source: {cfg.source}（可选 yolo / coco / dir）")


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

def _merge_two_proposals(a: DetProposal, b: DetProposal, score_mode: str = "keep_max") -> DetProposal:
    """
    将 proposal b 合并进 proposal a，返回新的 proposal
    - 掩码：做布尔并集（OR）
    - 位置：在全局坐标构建并集包围框，拼接两者 mask，再裁剪到 mask 的紧致外接框
    - bbox/top/left/area：按合并后掩码重算
    - score：'keep_max' 保留较大分数；'area_weighted' 用面积加权平均
    - label：沿用 a 的 label（若需要也可改为面积/分数多数表决）
    """
    # 1) 计算全局并集包围框
    ax1, ay1 = a.left, a.top
    ax2, ay2 = a.left + a.mask.shape[1], a.top + a.mask.shape[0]
    bx1, by1 = b.left, b.top
    bx2, by2 = b.left + b.mask.shape[1], b.top + b.mask.shape[0]

    ux1, uy1 = min(ax1, bx1), min(ay1, by1)
    ux2, uy2 = max(ax2, bx2), max(ay2, by2)
    U_w, U_h = int(ux2 - ux1), int(uy2 - uy1)

    # 2) 将两个局部 mask 放到并集坐标的大画布上
    canvas = np.zeros((U_h, U_w), dtype=bool)
    # a 的偏移
    ay_off, ax_off = a.top - uy1, a.left - ux1
    canvas[ay_off:ay_off + a.mask.shape[0], ax_off:ax_off + a.mask.shape[1]] |= a.mask
    # b 的偏移
    by_off, bx_off = b.top - uy1, b.left - ux1
    canvas[by_off:by_off + b.mask.shape[0], bx_off:bx_off + b.mask.shape[1]] |= b.mask

    # 3) 对合成后的 mask 做紧致裁剪（去掉全零边框）
    if not canvas.any():
        # 理论上不会发生（因为 a、b 至少一个有像素）
        new_mask = canvas
        tight_x1 = tight_y1 = 0
    else        :
        ys, xs = np.where(canvas)
        tight_x1, tight_x2 = xs.min(), xs.max() + 1
        tight_y1, tight_y2 = ys.min(), ys.max() + 1
        new_mask = canvas[tight_y1:tight_y2, tight_x1:tight_x2]

    # 4) 组合后的全局左上角、bbox、面积
    new_left = int(ux1 + tight_x1)
    new_top  = int(uy1 + tight_y1)
    new_h, new_w = new_mask.shape[0], new_mask.shape[1]
    new_bbox = np.array([new_left, new_top, new_left + new_w, new_top + new_h], dtype=float)
    new_area = int(new_mask.sum())

    # 5) 分数策略
    if score_mode == "area_weighted":
        # 面积加权（更平滑）：注意避免除零
        denom = max(1, a.area + b.area)
        new_score = float((a.score * a.area + b.score * b.area) / denom)
    else:
        # 默认：保留较大分数（保持“以高分实例为主”的语义）
        new_score = float(max(a.score, b.score))

    return DetProposal(
        bbox=new_bbox,
        score=new_score,
        label=a.label,          # 如需也可做多数表决
        mask=new_mask.astype(bool),
        top=int(new_top),
        left=int(new_left),
        area=int(new_area),
    )


def mask_nms_with_merge(proposals: List[DetProposal],
                        iou_thr: float,
                        iou_type: IOUType,
                        score_mode: str = "keep_max") -> List[DetProposal]:
    """
    合并版 Mask-NMS：
    - 先按 score 从高到低排序
    - 取当前最高分为“基准实例”，将与其 IOU/IOU_Min ≥ 阈值的其他实例合并到它上面（mask 取并，重算 bbox/area/score）
    - 合并完成后，得到新的“基准实例”，继续扫描余下实例是否还与它重叠，直到无可合并为止
    - 进入下一高分实例重复上述过程
    """
    if not proposals:
        return []

    # 按分数从大到小
    order = sorted(range(len(proposals)), key=lambda i: proposals[i].score, reverse=True)
    used = np.zeros(len(proposals), dtype=bool)
    merged_results: List[DetProposal] = []

    for idx in order:
        if used[idx]:
            continue

        # 以当前最高分为“基准实例”，逐步把可合并的并进来
        base = proposals[idx]
        used[idx] = True

        changed = True
        # 因为 base 在合并过程中会变化，这里迭代扫描直到不再发生合并
        while changed:
            changed = False
            for j in order:
                if used[j]:
                    continue
                iou = mask_iou_between(base, proposals[j], iou_type)
                if iou >= iou_thr:
                    base = _merge_two_proposals(base, proposals[j], score_mode=score_mode)
                    used[j] = True
                    changed = True  # 发生了合并，需要再扫一遍（新的 base 可能与更多实例相交）

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
        # 数据源延迟到 run_search / run_infer_only 才构建：只用 load_model/infer_tile 的调用方
        # （如 coco_add_pseudo_labels.py）传的是空 root，不该在这里就因目录校验失败。
        self.source: Optional[DataSource] = None
        self.limit: int = 0   # >0 时每个 split 只处理前 N 张（冒烟测试用）

    # hooks
    def load_model(self): raise NotImplementedError
    def infer_tile(self, tile_img: np.ndarray) -> List[Dict[str, Any]]: raise NotImplementedError

    def _images_for(self, split: str) -> List[str]:
        images = self.source.images(split)
        if self.limit and self.limit > 0:
            images = images[:self.limit]
            print(f"[limit] 只处理前 {len(images)} 张图")
        if not images:
            raise FileNotFoundError(f"没有找到任何图片（split={split!r}）")
        return images

    # ====== Modes ======
    def run_search(self, search_split: str = "val", eval_split: str = "test"):
        """Grid-search on search_split, then run eval_split with the best params."""
        ensure_dir(self.outputs.out_dir)
        random.seed(0); np.random.seed(0)
        self.source = build_data_source(self.dataset)
        if not self.source.supports_search:
            raise SystemExit("普通目录没有 GT，无法网格搜参数，请用 --mode infer")
        print("==> Loading model ..."); self.load_model()

        tag = self.source.tag(search_split)
        print(f"==> Collecting raw proposals on {tag.upper()} ...")
        search_images = self._images_for(search_split)
        val_cache, val_gt_counts = {}, {}
        for img_path in tqdm(search_images, desc=f"{tag.upper()} images"):
            val_cache[img_path] = self._proposals_for_image(img_path)
            val_gt_counts[img_path] = self.source.gt_count(img_path, search_split)
        self.source.finish()

        print(f"==> Grid searching best parameters on {tag.upper()} ...")
        best_cfg, best_sum_diff = self._grid_search(val_cache, val_gt_counts)
        summary = (f"[Best on {tag.upper()}] iou_type={best_cfg['iou_type'].value}, "
                   f"score_thr={best_cfg['score_thr']:.4f}, nms_iou_thr={best_cfg['nms_iou_thr']:.4f}, "
                   f"sum|GT-Pred|={best_sum_diff}")
        print(summary)

        # 将 print 输出保存到文件
        with open(os.path.join(self.outputs.out_dir, f"best_params_on_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(summary + "\n")

        print(f"==> Running {self.source.tag(eval_split).upper()} with best parameters ...")
        self._run_on_split(eval_split, best_cfg['iou_type'], best_cfg['score_thr'], best_cfg['nms_iou_thr'])

    def run_infer_only(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float):
        """Direct inference with fixed params on a chosen split (普通目录时 split 被忽略)."""
        ensure_dir(self.outputs.out_dir)
        random.seed(0); np.random.seed(0)
        self.source = build_data_source(self.dataset)
        print("==> Loading model ..."); self.load_model()
        print(f"==> Running {self.source.tag(split).upper()} with fixed params: "
              f"iou_type={iou_type.value}, score_thr={score_thr}, nms_iou_thr={nms_iou_thr}")
        self._run_on_split(split, iou_type, score_thr, nms_iou_thr)

    # ====== shared helpers ======
    def _run_on_split(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float):
        tag = self.source.tag(split)
        images = self._images_for(split)

        rows: List[Tuple] = []
        header_written = False
        xs, ys = [], []
        for idx, img_path in enumerate(tqdm(images, desc=f"{tag.upper()} images", total=len(images)), 1):
            proposals = self._proposals_for_image(img_path)
            final_dets = self._apply_thresh_nms(proposals, score_thr, nms_iou_thr, iou_type)
            pred_count = len(final_dets)
            gt_count = self.source.gt_count(img_path, split)
            if not header_written:
                rows.append(("filename", "ins_counts", "pred_counts") if gt_count is not None
                            else ("filename", "pred_counts"))
                header_written = True
            if gt_count is None:
                rows.append((os.path.basename(img_path), pred_count))
            else:
                rows.append((os.path.basename(img_path), gt_count, pred_count))
                xs.append(gt_count); ys.append(pred_count)
            save_coco = False
            if save_coco:
                img_id = idx * 100000
                start_ins_id = img_id
                self._save_coco_annotation(img_id, start_ins_id, img_path, final_dets, self.outputs.out_dir)
            if self.outputs.vis:
                self._visualize_and_save(img_path, final_dets, self.outputs.out_dir)

        self.source.finish()

        if self.outputs.excel:
            csv_path = os.path.join(self.outputs.out_dir, f"{tag}_pred_vs_gt.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                for r in rows: f.write(",".join(str(v) for v in r) + "\n")
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

        if self.outputs.draw_scatter:
            if xs:
                scatter_path = os.path.join(self.outputs.out_dir, f"scatter_{tag}.png")
                self._draw_scatter(xs, ys, scatter_path)
                print(f"[Saved] Scatter -> {scatter_path}")
            else:
                print("[skip] 无 GT，跳过散点图")

        print(f"{tag.upper()} finished.")

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

        for y0 in y_list:
            for x0 in x_list:
                tile_img = np.full((tile, tile, 3), padv, dtype=np.uint8)
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                h, w = y1 - y0, x1 - x0
                tile_img[:h, :w] = img[y0:y1, x0:x1]
                valid_mask = np.zeros((tile, tile), dtype=bool); valid_mask[:h, :w] = True

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
                        # print(1)
                        continue

                    mask_crop = mask[y1t:y2t, x1t:x2t]
                    area = int(mask_crop.sum())
                    if area <= 0:
                        # print(2)
                        continue

                    gx1, gy1 = x0 + x1t, y0 + y1t
                    gx2, gy2 = x0 + x2t, y0 + y2t
                    gx1 = max(0, min(W, gx1)); gx2 = max(0, min(W, gx2))
                    gy1 = max(0, min(H, gy1)); gy2 = max(0, min(H, gy2))
                    if gx2 <= gx1 or gy2 <= gy1:
                        # print(3)
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
        # return mask_nms(fil, nms_thr, iou_type) if fil else []
        # 'keep_max'：合并后保留较大 score；想更平滑可改 'area_weighted'
        return mask_nms_with_merge(fil, nms_thr, iou_type, score_mode="keep_max") if fil else []

    def _save_coco_annotation(self, img_id, start_ins_id, img_path: str, dets: List[DetProposal], out_dir: str):
            """
            将检测结果保存为 COCO 格式的标注文件
            segmentation 使用 polygon 形式
            """
            import json

            img = cv2.imread(img_path)
            if img is None:
                return

            H, W = img.shape[:2]
            img_name = os.path.basename(img_path)

            # COCO 格式结构
            coco_output = {
                "info": {},
                "licenses": [],
                "images": [{
                    "id": img_id,
                    "file_name": img_name,
                    "width": W,
                    "height": H
                }],
                "annotations": [],
                "categories": [{"id": 1, "name": "wheat"}]
            }

            ins_id = start_ins_id
            for idx, d in enumerate(dets, start=1):
                # 获取 mask 的轮廓
                mask_uint8 = (d.mask.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    continue

                # 转换为 COCO polygon 格式 (全局坐标)
                segmentation = []
                for contour in contours:
                    if len(contour) < 3:  # 至少需要3个点
                        continue
                    polygon = []
                    for point in contour:
                        x = float(d.left + point[0][0])
                        y = float(d.top + point[0][1])
                        polygon.extend([x, y])

                    if len(polygon) >= 6:  # 至少3个点(6个坐标值)
                        segmentation.append(polygon)

                if not segmentation:
                    continue

                # 计算 bbox (COCO 格式: [x, y, width, height])
                x1, y1, x2, y2 = d.bbox
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                annotation = {
                    "id": ins_id + idx,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": float(d.area),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
                coco_output["annotations"].append(annotation)

            # 保存 JSON 文件
            ensure_dir(out_dir)
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(out_dir, json_name)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(coco_output, f, indent=2, ensure_ascii=False)

    # 每个类别一种颜色（BGR），按 label 索引
    _CATEGORY_COLORS = [
        (0, 0, 255),      # 0: 红色
        (255, 0, 0),      # 1: 蓝色
        (0, 255, 0),      # 2: 绿色
        (255, 255, 0),    # 3: 青色
        (255, 0, 255),    # 4: 品红
        (0, 255, 255),    # 5: 黄色
        (128, 0, 128),    # 6: 紫色
        (128, 128, 0),    # 7: 橄榄绿
        (0, 128, 128),    # 8: 深青
        (128, 0, 0),      # 9: 深蓝
    ]

    def _get_category_color(self, label: int) -> tuple:
        # “””根据类别标签返回固定颜色，超出预定义范围则循环使用”””
        colors = self._CATEGORY_COLORS
        return colors[label % len(colors)]

    def _visualize_and_save(self, img_path: str, dets: List[DetProposal],
                            out_dir: str, draw_center: bool = False):
        img = cv2.imread(img_path)
        if img is None:
            return

        ensure_dir(out_dir)
        vis = img.copy()

        for d in dets:
            # --- 1. 按类别取颜色 ---
            color = self._get_category_color(d.label)

            # --- 2. 取对应 ROI（严格对齐 mask 与 roi）---
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

            # 计算 roi 相对 det-mask 的偏移，并裁剪 mask 使其与 roi 尺寸一致
            my1 = y1 - y1o
            my2 = my1 + (y2 - y1)
            mx1 = x1 - x1o
            mx2 = mx1 + (x2 - x1)

            m = d.mask[my1:my2, mx1:mx2].astype(bool)

            if m.shape[0] != vis[y1:y2, x1:x2].shape[0] or m.shape[1] != vis[y1:y2, x1:x2].shape[1]:
                m = m[:vis[y1:y2, x1:x2].shape[0], :vis[y1:y2, x1:x2].shape[1]]

            # --- 3. 只画轮廓，不填充颜色 ---
            m8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis[y1:y2, x1:x2], contours, -1, color, 2)

            # --- 4. 按 score 画中心点 ---
            if draw_center:
                ys, xs = np.where(m > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int(x1 + np.mean(xs))
                    cy = int(y1 + np.mean(ys))
                    cv2.circle(vis, (cx, cy), radius=6,
                               color=color, thickness=-1)

        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, vis)

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

        # 2.5) --opts 覆盖项放在最后，可以改上面任何一项（含 WEIGHTS / DEVICE）
        if self.model_cfg.opts:
            opts = list(self.model_cfg.opts)
            if len(opts) % 2 != 0:
                raise SystemExit(f"--opts 必须是 KEY VALUE 成对出现，收到 {len(opts)} 个: {opts}")
            cfg.merge_from_list(opts)
            print("[opts]    -> " + ", ".join(f"{k}={v}" for k, v in zip(opts[::2], opts[1::2])))

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
def build_argparser(suppress_defaults: bool = False):
    p = argparse.ArgumentParser(description="Wheat Segmentation (Template Method) on Detectron2 with search/infer modes")

    # Config file
    p.add_argument("--config", "-c", type=str, default="",
                   help="YAML 配置文件；键名与 --xxx 一一对应。命令行显式传入的参数覆盖它")

    # Modes
    p.add_argument("--mode", choices=["search", "infer"], default="search",
                   help="search: grid search on val then test; infer: direct inference with fixed params")

    # Dataset
    p.add_argument("--source", choices=["yolo", "coco", "dir"], default="yolo",
                   help="数据组织形式：yolo=<root>/images/<split>；coco=<root>/<split>+annotations；dir=直接列图（无 GT，仅 infer）")
    p.add_argument("--data_root", type=str, default="",
                   help="yolo/coco 的数据集根目录，或 dir 模式下的图片目录")
    p.add_argument("--yolo_root", type=str,
                   default="/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image",
                   help="[deprecated] --data_root 的旧名字，未指定 --data_root 时生效")
    p.add_argument("--coco_ann", type=str, default="",
                   help="coco 模式下显式指定标注 json，覆盖 <root>/annotations/instances_<split>.json")
    p.add_argument("--gt_cat_ids", type=int, nargs="*", default=[],
                   help="GT 计数只统计这些类别（coco 用 category_id，yolo 用 txt 首列）；不填 = 全算")

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
    p.add_argument("--opts", type=str, nargs="*", default=[],
                   help="覆盖 detectron2 cfg 的 KEY VALUE 对，在 model_config_rel 之后生效，"
                        "例如 --opts TEST.DETECTIONS_PER_IMAGE 300 INPUT.MIN_SIZE_TEST 1024")

    # Sliding window
    p.add_argument("--tile", type=int, default=800)
    p.add_argument("--overlap", type=int, default=256)

    # Search space (used only in --mode search)
    p.add_argument("--search_split", type=str, default="val",
                   help="[mode=search] 在哪个子目录上搜参数（coco 传子目录名，如 val2017）")
    p.add_argument("--eval_split", type=str, default="test",
                   help="[mode=search] 用最优参数在哪个子目录上跑最终评测")
    p.add_argument("--score_start", type=float, default=0.1)
    p.add_argument("--score_end", type=float, default=0.6)
    p.add_argument("--score_num", type=int, default=6)
    p.add_argument("--nms_start", type=float, default=0.1)
    p.add_argument("--nms_end", type=float, default=0.6)
    p.add_argument("--nms_num", type=int, default=12)
    p.add_argument("--iou_types", type=str, nargs="+", default=["IOU", "IOU_Min"])
    p.add_argument("--base_keep_score", type=float, default=0.01)

    # Infer-only fixed params
    p.add_argument("--split", type=str, default="",
                   help="[mode=infer] 推理的子目录名（yolo: val/test/train；coco: val2017 等）；dir 模式不需要")
    p.add_argument("--infer_split", type=str, default="test",
                   help="[deprecated] --split 的旧名字，未指定 --split 时生效")
    p.add_argument("--fix_iou_type", choices=["IOU", "IOU_Min"], default="IOU_Min")
    p.add_argument("--fix_score_thr", type=float, default=0.4)
    p.add_argument("--fix_nms_iou_thr", type=float, default=0.372)

    # Outputs
    p.add_argument("--out_dir", type=str, default="./swinl_guo_big_images")
    p.add_argument("--no_vis", action="store_true")
    p.add_argument("--no_excel", action="store_true")
    p.add_argument("--no_scatter", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 时每个 split 只处理前 N 张图（冒烟测试用）")

    if suppress_defaults:
        # 探测模式：默认值全换成 SUPPRESS，解析结果里只剩用户显式传入的参数。
        # 不能用 p.set_defaults()：它同时写 parser._defaults，而那份会被原样塞进
        # Namespace（每个键都变成字面量 '==SUPPRESS=='），反而分不清传没传。
        for a in p._actions:
            a.default = argparse.SUPPRESS

    return p


# -----------------------------
# Run config (YAML) —— 优先级：argparse 默认值 < 配置文件 < 命令行显式传入
# -----------------------------
def _cli_provided_keys(argv: List[str]) -> set:
    """返回用户在命令行上**显式传过**的参数名集合。

    不能用「值 != argparse 默认值」来判断：显式传一个恰好等于默认值的值会被误判成没传，
    于是被配置文件反向覆盖。这里用一个默认值全为 SUPPRESS 的同构 parser 解析一遍，
    Namespace 里出现的键就是显式传过的。
    """
    return set(vars(build_argparser(suppress_defaults=True).parse_args(argv)))


def load_run_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise SystemExit(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"配置文件顶层必须是 key: value 映射（键名与 --xxx 一一对应）: {path}")
    return data


def _arg_types() -> Dict[str, Any]:
    """dest -> argparse 声明的 type，用于把 YAML 里写成字符串的值归一化。"""
    return {a.dest: a.type for a in build_argparser()._actions if a.type is not None}


def merge_run_config(args: argparse.Namespace, cfg: Dict[str, Any], provided: set) -> List[str]:
    """把配置文件合并进 args（跳过命令行已显式指定的键），返回真正生效的配置文件键名。"""
    known = set(vars(args))
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise SystemExit(f"配置文件里有未知项: {unknown}\n"
                         f"键名需与命令行参数一一对应（去掉 -- 前缀），可用项见 --help")

    types = _arg_types()
    applied: List[str] = []
    for k, v in cfg.items():
        if k in provided:          # 命令行显式传入的优先，配置文件不覆盖
            continue
        t = types.get(k)
        if t is not None and v is not None:
            try:
                v = [t(x) for x in v] if isinstance(v, list) else t(v)
            except (TypeError, ValueError) as e:
                raise SystemExit(f"配置项 {k}: {v!r} 无法转成 {getattr(t, '__name__', t)}: {e}")
        setattr(args, k, v)
        applied.append(k)
    return applied


def dump_effective_config(args: argparse.Namespace, provided: set, from_cfg: List[str]):
    """把合并后的最终配置写到 <out_dir>/effective_config.yaml，可直接 --config 喂回去复现。"""
    effective = {k: v for k, v in sorted(vars(args).items()) if k != "config"}
    cli_keys = sorted(k for k in provided if k in effective)

    lines = ["# 本次运行的最终配置（可直接 --config 复现）"]
    if cli_keys:
        lines.append(f"# [cli]    命令行显式覆盖: {', '.join(cli_keys)}")
    if from_cfg:
        lines.append(f"# [config] 来自 {args.config}: {', '.join(sorted(from_cfg))}")
    lines.append("# 其余为脚本默认值")

    ensure_dir(args.out_dir)
    out_path = os.path.join(args.out_dir, "effective_config.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        yaml.safe_dump(effective, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"[Saved] Effective config -> {out_path}")


def _resolve_splits(args) -> Tuple[str, str, str]:
    """校验数据源/split 参数，返回 (infer_split, search_split, eval_split)。"""
    # --yolo_root 带默认值，非 yolo 源忘记 --data_root 会静默落到那个默认路径上，这里挡住
    if args.source != "yolo" and not args.data_root:
        raise SystemExit(f"--source {args.source} 必须用 --data_root 指定数据目录")

    infer_split = args.split or args.infer_split

    if args.source == "dir":
        if args.mode == "search":
            raise SystemExit("普通目录（--source dir）没有 GT，无法网格搜参数，请用 --mode infer")
        if args.split:
            print(f"[warn] --source dir 不需要子目录，已忽略 --split {args.split!r}")
        return "", "", ""

    if args.mode == "infer":
        if not infer_split:
            raise SystemExit(f"--source {args.source} 需要用 --split 指定推理子目录"
                             f"（yolo 如 test，coco 如 val2017）")
        return infer_split, "", ""

    if not args.search_split or not args.eval_split:
        raise SystemExit(f"--source {args.source} 的 search 模式需要 --search_split 与 --eval_split")
    return "", args.search_split, args.eval_split


def main():
    argv = sys.argv[1:]
    args = build_argparser().parse_args(argv)      # argparse 默认值 + 命令行
    provided = _cli_provided_keys(argv)            # 其中哪些是用户显式传的

    from_cfg: List[str] = []
    if args.config:
        cfg = load_run_config(args.config)
        from_cfg = merge_run_config(args, cfg, provided)
        overridden = sorted(set(cfg) & provided)
        print(f"[config] {args.config} -> 生效 {len(from_cfg)} 项"
              + (f"；被命令行覆盖: {', '.join(overridden)}" if overridden else ""))

    # 校验必须在合并之后：配置文件里的 split / data_root 要先生效
    infer_split, search_split, eval_split = _resolve_splits(args)
    dump_effective_config(args, provided, from_cfg)

    dataset = DatasetConfig(
        yolo_root=args.yolo_root,
        source=args.source,
        data_root=args.data_root,
        coco_ann=args.coco_ann,
        gt_cat_ids=tuple(args.gt_cat_ids),
    )
    model = ModelConfig(
        project_root=args.project_root,
        config_rel=args.model_config_rel,
        checkpoint_rel=args.checkpoint_rel if args.checkpoint_rel else "",
        device=args.device,
        imports=tuple(args.imports),
        opts=tuple(str(o) for o in args.opts),
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
    runner.limit = args.limit

    if args.mode == "search":
        runner.run_search(search_split=search_split, eval_split=eval_split)
    else:
        runner.run_infer_only(
            split=infer_split,
            iou_type=IOUType(args.fix_iou_type),
            score_thr=args.fix_score_thr,
            nms_iou_thr=args.fix_nms_iou_thr
        )



if __name__ == "__main__":
    # 延迟导入 torch（仅在需要时），避免无 GPU 环境卡顿

    main()
