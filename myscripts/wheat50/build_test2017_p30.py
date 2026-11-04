#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 coco_2cat_with_points/test2017（300 张 1024×1024 tile + 模型伪标注，含小穗点）。

与历史流程（clean-wheat50-model-1.1:myscripts/wheat50/build_test2017_from_selected.py）
的三点区别：
  1) 模型：单阶段联合训练 P=30（output_v2/exp-0922-single-stage-p30/model_final.pth），
     分割与小穗点一次前向同时产出；wheat_2 实例的 points 一并写入伪标注。
  2) 候选池：只用 5 个 D6/D8 A50 目录（与 train/val 同域），3 个 D9 目录不纳入；
     原图按“未被 train/val 使用”过滤后确定性抽样（seed 固定）。
  3) 选择：不设每原图入选上限，纯按「截断 wheat_2 少 -> 完整 wheat_2 多 -> 总实例多」排名取 300 张。

人工修订：COCO 写好后用 coco_test2017_to_labelme.py 转一份 LabelMe，在 labelme 里改，
改完再回写 COCO。

用法（MaskDINO conda 环境，把 A100 映射为逻辑 0 号卡）：
  CUDA_VISIBLE_DEVICES=1 /public/home/zncui/anaconda3/envs/MaskDINO/bin/python \
      myscripts/wheat50/build_test2017_p30.py
"""
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJ = "/public/home/zncui/pycharm_project/maskdino-wheat50"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "myscripts"))

from myscripts.infer.seg_infer import (  # noqa: E402
    Detectron2SegInference, DatasetConfig, ModelConfig,
    SlidingWindowConfig, SearchSpace, OutputOptions, IOUType,
    DetProposal, mask_iou_between, ensure_dir,
)

# ---------------- 路径配置 ----------------
RAW_ROOT = Path("/public/lab_share/nfs_field/raw_data/DJI/P1/zhaoxian2023/wheat/20240516DJI/DCIM")
# 只用与 train/val 同域的 5 个 D6/D8 目录（D9 三个目录完全未参与训练，不纳入候选池）
SOURCE_DIRS = [
    "DJI_202405161402_007_2023D6-250cm-A50",
    "DJI_202405161402_008_2023D6-250cm-A50",
    "DJI_202405161548_009_2023D6-250cm-A50",
    "DJI_202405161548_011_2023D8-250cm-A50",
    "DJI_202405161548_012_2023D8-250cm-A50",
]
DATASET_ROOT = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points"
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test2017")
TEST_JSON = os.path.join(DATASET_ROOT, "annotations", "instances_test2017.json")

CONFIG = os.path.join(PROJ, "output_v2/exp-0922-single-stage-p30/config.yaml")
WEIGHTS = os.path.join(PROJ, "output_v2/exp-0922-single-stage-p30/model_final.pth")
# 本机 A100 是物理 GPU1；ms_deform_attn 自定义 op 在非 0 号逻辑设备上会非法访存，
# 因此用 CUDA_VISIBLE_DEVICES=1 把它映射成逻辑 cuda:0（见文件头部用法）。
DEVICE = "cuda:0"

# ---------------- 采样 / 切分 / 推理参数 ----------------
# score_thr=0.3 / nms=0.35：2026-10-09 与 0.4/0.372 在同 300 tile 上对比，
# wheat 召回 +10.3%、小穗点 +15.7% 且未见明显误检；伪标注底稿宁多勿漏（删比补省劲）。
SEED = 20261008
POOL_SIZE = 50          # 首轮抽取的未见原图数；候选不足 300 时按 BATCH 扩容
EXPAND_BATCH = 25
TILE = 1024
OVERLAP = 128
SCORE_THR = 0.3         # 实例分数阈值
NMS_IOU_THR = 0.35      # 类别感知合并 NMS 阈值（IOU_Min）
IOU_TYPE = IOUType.IOU_Min
DUP_IOU_THR = 0.9       # 跨类孪生去重阈值:真实重复全是 IoU_Min=1.0(同 mask 两次检测),
                        # 相邻异类穗实测 <0.5,0.9 只杀重复不伤邻穗
BASE_KEEP = 0.05        # 收集原始 proposal 的粗筛阈值

# ---------------- 类别 / 选择参数 ----------------
SELECT_LABEL = 1        # contiguous label：0 -> wheat，1 -> wheat_2
NUM_SELECT = 300
LABEL_TO_CATID = {0: 1, 1: 2}
CATEGORIES = [
    {"id": 1, "name": "wheat", "supercategory": "none"},
    {"id": 2, "name": "wheat_2", "supercategory": "none"},
]
BORDER_MARGIN = 2


# =====================================================================
# 候选池：未被 train/val 使用的原图
# =====================================================================
def collect_used_raw_stems() -> set:
    """train 文件名 = {A50目录名}__{原图stem}__x..y..w..h..；val 只有 {原图stem}__x...
    两处出现的原图 stem 都视为已使用（宁可错杀，保证测试集独立性）。"""
    used = set()
    for split in ("train2017", "val2017"):
        d = Path(DATASET_ROOT) / split
        for p in d.glob("*.jpg"):
            ps = p.stem.split("__")
            if len(ps) >= 2:
                used.add(ps[-2])  # 倒数第二段必为原图 stem
    return used


def sample_pool(seed: int, pool_size: int) -> List[Path]:
    """按各目录未见原图数比例（最大余数法）分配名额，目录内用固定 seed 抽样。"""
    used = collect_used_raw_stems()
    unused: Dict[str, List[Path]] = {}
    for name in SOURCE_DIRS:
        imgs = sorted(p for p in (RAW_ROOT / name).iterdir()
                      if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
                      and p.stem not in used)
        unused[name] = imgs
        print(f"[pool] {name}: unused={len(imgs)}")
    total = sum(len(v) for v in unused.values())
    quota, fracs = {}, []
    for name, imgs in unused.items():
        q = len(imgs) * pool_size / total
        quota[name] = int(q)
        fracs.append((q - int(q), name))
    for _, name in sorted(fracs, reverse=True)[:pool_size - sum(quota.values())]:
        quota[name] += 1
    rng = random.Random(seed)
    pool = []
    for name in SOURCE_DIRS:
        k = min(quota[name], len(unused[name]))
        picked = sorted(rng.sample(unused[name], k)) if k else []
        quota[name] -= k
        print(f"[pool] {name}: pick {k}")
        pool.extend(picked)
    return pool


# =====================================================================
# 推理：单阶段 P=30，infer_tile 额外取小穗点
# =====================================================================
class P30SegInference(Detectron2SegInference):
    @torch.no_grad()
    def infer_tile(self, tile_img: np.ndarray) -> List[Dict[str, Any]]:
        outputs = self.model(tile_img)
        if "instances" not in outputs or len(outputs["instances"]) == 0:
            return []
        inst = outputs["instances"]
        bboxes = inst.pred_boxes.tensor.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        labels = inst.pred_classes.detach().cpu().numpy()
        if not inst.has("pred_masks"):
            return []
        masks = inst.pred_masks.detach().cpu().numpy().astype(bool)
        sp_points = inst.pred_spikelet_points if inst.has("pred_spikelet_points") else [None] * len(inst)

        out = []
        for i in range(len(inst)):
            pts = sp_points[i]
            pts = pts.detach().cpu().numpy().astype(np.float32) if pts is not None else np.zeros((0, 2), np.float32)
            out.append({
                "bbox": bboxes[i].tolist(),
                "score": float(scores[i]),
                "label": int(labels[i]),
                "mask": masks[i],
                "points": pts,  # tile 内绝对像素坐标；wheat 为 (0,2)
            })
        return out


def tile_proposals(runner: P30SegInference, tile_img: np.ndarray) -> List[DetProposal]:
    th, tw = tile_img.shape[:2]
    proposals = []
    for ins in runner.infer_tile(tile_img):
        if ins["score"] < BASE_KEEP:
            continue
        x1t = max(0, min(tw, int(math.floor(ins["bbox"][0]))))
        y1t = max(0, min(th, int(math.floor(ins["bbox"][1]))))
        x2t = max(0, min(tw, int(math.ceil(ins["bbox"][2]))))
        y2t = max(0, min(th, int(math.ceil(ins["bbox"][3]))))
        if x2t <= x1t or y2t <= y1t:
            continue
        mask_crop = ins["mask"][y1t:y2t, x1t:x2t]
        if mask_crop.sum() <= 0:
            continue
        p = DetProposal(
            bbox=np.array([x1t, y1t, x2t, y2t], dtype=float),
            score=ins["score"], label=ins["label"], mask=mask_crop,
            top=y1t, left=x1t, area=int(mask_crop.sum()),
        )
        p.points = ins["points"]
        proposals.append(p)
    return proposals


# ---- 小穗点感知的合并 NMS（逻辑同 seg_infer._merge_two_proposals / mask_nms_with_merge，
# ---- 仅在合并时把两实例的 points 拼接，避免合并丢点） ----
def _merge_two_with_points(a: DetProposal, b: DetProposal) -> DetProposal:
    ax1, ay1 = a.left, a.top
    ax2, ay2 = a.left + a.mask.shape[1], a.top + a.mask.shape[0]
    bx1, by1 = b.left, b.top
    bx2, by2 = b.left + b.mask.shape[1], b.top + b.mask.shape[0]
    ux1, uy1, ux2, uy2 = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
    canvas = np.zeros((int(uy2 - uy1), int(ux2 - ux1)), dtype=bool)
    canvas[a.top - uy1:a.top - uy1 + a.mask.shape[0], a.left - ux1:a.left - ux1 + a.mask.shape[1]] |= a.mask
    canvas[b.top - uy1:b.top - uy1 + b.mask.shape[0], b.left - ux1:b.left - ux1 + b.mask.shape[1]] |= b.mask
    ys, xs = np.where(canvas)
    tx1, tx2, ty1, ty2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    new_mask = canvas[ty1:ty2, tx1:tx2]
    new_left, new_top = int(ux1 + tx1), int(uy1 + ty1)
    merged = DetProposal(
        bbox=np.array([new_left, new_top, new_left + new_mask.shape[1], new_top + new_mask.shape[0]], dtype=float),
        score=float(max(a.score, b.score)), label=a.label, mask=new_mask,
        top=new_top, left=new_left, area=int(new_mask.sum()),
    )
    merged.points = np.concatenate([a.points, b.points], axis=0)
    return merged


def mask_nms_merge_points(proposals: List[DetProposal], iou_thr: float,
                          iou_type: IOUType,
                          dup_iou_thr: float = DUP_IOU_THR) -> List[DetProposal]:
    """两阶段类别感知 NMS：
    1) 同标签合并:IoU >= iou_thr 才并(mask 取并、points 拼接),逻辑同历史 seg_infer;
       跨标签不合并不抑制,避免「wheat 标签带小穗点」的嵌合实例
       (2026-10-09 实测 300 tile 有 236 个)。
    2) 跨类孪生去重:合并后仍可能有同一穗被检测两次、分判两类的重复
       (实测 684 对,IoU_Min 全部恰好 1.0,占 wheat_2 的 35%)。
       对 IoU >= dup_iou_thr(0.9)的对抑制低分者,保留高分者的类别与点;
       相邻异类穗 IoU 实测 <0.5,不会被误伤。
    """
    if not proposals:
        return []
    order = sorted(range(len(proposals)), key=lambda i: proposals[i].score, reverse=True)
    used = np.zeros(len(proposals), dtype=bool)
    results = []
    for idx in order:
        if used[idx]:
            continue
        base = proposals[idx]
        used[idx] = True
        changed = True
        while changed:
            changed = False
            for j in order:
                if used[j] or proposals[j].label != base.label:
                    continue
                if mask_iou_between(base, proposals[j], iou_type) >= iou_thr:
                    base = _merge_two_with_points(base, proposals[j])
                    used[j] = True
                    changed = True
        results.append(base)

    # 阶段 2:跨类孪生去重(此时同标签对 IoU 都 < iou_thr,>= dup_iou_thr 的必为跨类重复)
    order2 = sorted(range(len(results)), key=lambda i: results[i].score, reverse=True)
    suppressed = np.zeros(len(results), dtype=bool)
    kept = []
    for i in order2:
        if suppressed[i]:
            continue
        kept.append(results[i])
        for j in order2:
            if suppressed[j] or j == i:
                continue
            if mask_iou_between(results[i], results[j], iou_type) >= dup_iou_thr:
                suppressed[j] = True
    return kept


# =====================================================================
# COCO 导出
# =====================================================================
def dets_to_coco_anns(dets: List[DetProposal], image_id: int,
                      start_ann_id: int) -> Tuple[List[Dict[str, Any]], int]:
    anns, aid = [], start_ann_id
    for d in dets:
        m8 = d.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for c in contours:
            if len(c) < 3:
                continue
            poly = []
            for pt in c:
                poly.extend([float(d.left + pt[0][0]), float(d.top + pt[0][1])])
            if len(poly) >= 6:
                segmentation.append(poly)
        if not segmentation:
            continue
        x1, y1, x2, y2 = [float(v) for v in d.bbox]
        points = [[float(x), float(y)] for x, y in d.points] if int(d.label) == SELECT_LABEL else []
        anns.append({
            "id": aid, "image_id": image_id,
            "category_id": LABEL_TO_CATID[int(d.label)],
            "segmentation": segmentation,
            "area": float(d.area),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "iscrowd": 0,
            "points": points,
        })
        aid += 1
    return anns, aid


def tile_origins(H: int, W: int, tile: int, overlap: int) -> Tuple[List[int], List[int]]:
    """边缘内收滑窗：最后一格对齐图像边缘，保证 tile 永远是 1024×1024。"""
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    return ys, xs


def touches_border(d: DetProposal, tile: int, margin: int) -> bool:
    x1, y1, x2, y2 = d.bbox
    return x1 <= margin or y1 <= margin or x2 >= tile - margin or y2 >= tile - margin


# =====================================================================
def main():
    ensure_dir(TEST_IMG_DIR)
    ensure_dir(os.path.dirname(TEST_JSON))

    dataset = DatasetConfig(yolo_root="")
    model = ModelConfig(project_root=PROJ, config_rel=CONFIG, checkpoint_rel=WEIGHTS,
                        device=DEVICE, imports=("maskdino",))
    sw = SlidingWindowConfig(tile=TILE, overlap=OVERLAP)
    search = SearchSpace(base_keep_score=BASE_KEEP)
    outputs = OutputOptions(vis=False, excel=False, draw_scatter=False, out_dir=DATASET_ROOT)
    runner = P30SegInference(dataset, model, sw, search, outputs)
    print("==> Loading model ...")
    runner.load_model()

    # ---- 候选扫描：首轮 POOL_SIZE 张，不足 NUM_SELECT 个候选则确定性扩容 ----
    candidates: List[Dict[str, Any]] = []
    scanned: List[Path] = []
    batch_idx = 0
    while True:
        pool = sample_pool(SEED + batch_idx, POOL_SIZE if batch_idx == 0 else EXPAND_BATCH)
        pool = [p for p in pool if p not in scanned]
        scanned.extend(pool)
        print(f"[scan] batch {batch_idx}: {len(pool)} big images (total scanned: {len(scanned)})")
        for img_path in tqdm(pool, desc=f"batch {batch_idx}"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[skip] cannot read {img_path}")
                continue
            H, W = img.shape[:2]
            stem = img_path.stem
            ys, xs = tile_origins(H, W, TILE, OVERLAP)
            for y0 in ys:
                for x0 in xs:
                    crop = img[y0:y0 + TILE, x0:x0 + TILE]
                    props = tile_proposals(runner, crop)
                    dets = [p for p in props if p.score >= SCORE_THR]
                    dets = mask_nms_merge_points(dets, NMS_IOU_THR, IOU_TYPE) if dets else []
                    sel = [d for d in dets if d.label == SELECT_LABEL]
                    if not sel:
                        continue
                    n_trunc = sum(1 for d in sel if touches_border(d, TILE, BORDER_MARGIN))
                    candidates.append({
                        "n_sel": len(sel), "n_trunc": n_trunc,
                        "n_complete": len(sel) - n_trunc, "n_tot": len(dets),
                        "name": f"{stem}__x{x0}_y{y0}_w{TILE}_h{TILE}.jpg",
                        "crop": crop, "dets": dets,
                    })
        n_clean = sum(1 for c in candidates if c["n_trunc"] == 0)
        print(f"[info] candidates with wheat_2: {len(candidates)} (无截断: {n_clean})")
        if len(candidates) >= NUM_SELECT or len(pool) == 0:
            break
        batch_idx += 1
        print(f"[info] fewer than {NUM_SELECT} candidates, expanding pool ...")

    # ---- 密集优先排名：截断少 -> 完整 wheat_2 多 -> 总实例多 -> 文件名 ----
    candidates.sort(key=lambda c: (c["n_trunc"], -c["n_complete"], -c["n_tot"], c["name"]))
    selected = candidates[:NUM_SELECT]
    n_sel_clean = sum(1 for c in selected if c["n_trunc"] == 0)
    print(f"[info] selecting {len(selected)} tiles (requested {NUM_SELECT}); 无截断: {n_sel_clean}")

    images, annotations, ann_id = [], [], 1
    for img_id, c in enumerate(selected, start=1):
        cv2.imwrite(os.path.join(TEST_IMG_DIR, c["name"]), c["crop"])
        images.append({"id": img_id, "file_name": c["name"], "width": TILE, "height": TILE,
                       "source": "coco_2cat_with_points"})
        anns, ann_id = dets_to_coco_anns(c["dets"], img_id, ann_id)
        annotations.extend(anns)

    coco = {"info": {"description": "test2017 pseudo-labeled by exp-0922-single-stage-p30 "
                                    f"(score_thr={SCORE_THR}, label-aware merge NMS "
                                    f"{NMS_IOU_THR} IOU_Min, point_thr=0.35); "
                                    "pending manual revision"},
            "licenses": [], "images": images,
            "annotations": annotations, "categories": CATEGORIES}
    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    n_pts = sum(len(a["points"]) for a in annotations)
    print(f"[Saved] {len(images)} tiles -> {TEST_IMG_DIR}")
    print(f"[Saved] COCO annotations ({len(annotations)} anns, {n_pts} spikelet points) -> {TEST_JSON}")
    print(f"[done] test2017: {len(images)} images, {len(annotations)} annotations")


if __name__ == "__main__":
    main()
