#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用 wheat50_r50_cat2_default 模型对 CVAT 导出的 COCO 标注图片再次推理，
把新预测出来的 wheat 实例「追加」到原标注里，生成一份新的 COCO json。

核心约束
--------
- 原有标注（人工标注，全部正确）**绝对不删除**：NMS 只在「预测之间」做，
  GT 只读、永远原样保留。
- 只有「与任何 GT 的 IoU_Min >= dedup_iou_min」的预测会被丢弃（避免重复标注），
  GT 永远不会因此被丢弃。

复用 seg_infer.py 的：Detectron2SegInference(load_model/infer_tile)、
DetProposal、IOUType、mask_iou_between、mask_nms_with_merge、ensure_dir。
"""

import os
import sys
import json
import math
import argparse
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# 让 `import seg_infer`（同目录）与 `import maskdino`（项目根）都可用
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from seg_infer import (  # noqa: E402
    Detectron2SegInference,
    DatasetConfig, ModelConfig, SlidingWindowConfig, SearchSpace, OutputOptions,
    DetProposal, IOUType, mask_iou_between, mask_nms_with_merge, ensure_dir,
)

ORANGE_BGR = (0, 165, 255)   # 新增预测：橙色
GREEN_BGR = (0, 200, 0)      # 原有 GT：绿色


# -----------------------------
# GT polygon -> DetProposal（mask 裁剪到 bbox，全图坐标）
# -----------------------------
def gt_ann_to_proposal(ann: Dict[str, Any], H: int, W: int) -> DetProposal:
    """把一个 COCO GT 标注（polygon segmentation）栅格化成 DetProposal，
    mask 仅保留到其紧致 bbox 区域，便于复用 mask_iou_between。"""
    seg = ann.get("segmentation", [])
    # 收集所有多边形点，确定全图整型 bbox
    polys = []
    if isinstance(seg, list):
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
                polys.append(pts)
    if not polys:
        # 退化：用 bbox 当作矩形 mask
        bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
        x1 = max(0, int(math.floor(bx))); y1 = max(0, int(math.floor(by)))
        x2 = min(W, int(math.ceil(bx + bw))); y2 = min(H, int(math.ceil(by + bh)))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, 1, 1
        m = np.ones((y2 - y1, x2 - x1), dtype=bool)
        return DetProposal(bbox=np.array([x1, y1, x2, y2], float), score=1.0,
                           label=0, mask=m, top=y1, left=x1, area=int(m.sum()))

    all_pts = np.concatenate(polys, axis=0)
    x1 = max(0, int(math.floor(all_pts[:, 0].min())))
    y1 = max(0, int(math.floor(all_pts[:, 1].min())))
    x2 = min(W, int(math.ceil(all_pts[:, 0].max())))
    y2 = min(H, int(math.ceil(all_pts[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, 1, 1

    canvas = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for pts in polys:
        ipts = np.round(pts - np.array([x1, y1])).astype(np.int32)
        cv2.fillPoly(canvas, [ipts], 1)
    m = canvas.astype(bool)
    area = int(m.sum())
    if area <= 0:
        m = np.ones_like(m, dtype=bool); area = int(m.sum())
    return DetProposal(bbox=np.array([x1, y1, x2, y2], float), score=1.0,
                       label=0, mask=m, top=y1, left=x1, area=area)


# -----------------------------
# 模型预测 -> DetProposal（全图直接推理，无滑窗）
# -----------------------------
def predictions_for_image(runner: Detectron2SegInference, img: np.ndarray,
                          score_thr: float, keep_labels: Tuple[int, ...]) -> List[DetProposal]:
    H, W = img.shape[:2]
    insts = runner.infer_tile(img)  # [{bbox,score,label,mask(HxW bool)}]
    props: List[DetProposal] = []
    for ins in insts:
        score = float(ins["score"])
        if score < score_thr:
            continue
        label = int(ins.get("label", 0))
        if keep_labels and label not in keep_labels:
            continue
        bbox = np.array(ins["bbox"], dtype=float)
        mask = ins["mask"].astype(bool)
        x1 = max(0, min(W, int(math.floor(bbox[0]))))
        y1 = max(0, min(H, int(math.floor(bbox[1]))))
        x2 = max(0, min(W, int(math.ceil(bbox[2]))))
        y2 = max(0, min(H, int(math.ceil(bbox[3]))))
        if x2 <= x1 or y2 <= y1:
            continue
        mask_crop = mask[y1:y2, x1:x2]
        area = int(mask_crop.sum())
        if area <= 0:
            continue
        props.append(DetProposal(bbox=np.array([x1, y1, x2, y2], float), score=score,
                                 label=label, mask=mask_crop, top=y1, left=x1, area=area))
    return props


def proposal_to_coco_ann(d: DetProposal, ann_id: int, image_id: int,
                         category_id: int) -> Dict[str, Any]:
    """DetProposal -> COCO annotation（polygon segmentation，全图坐标）。"""
    m8 = (d.mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = []
        for p in c:
            poly.extend([float(d.left + p[0][0]), float(d.top + p[0][1])])
        if len(poly) >= 6:
            segmentation.append(poly)
    if not segmentation:
        return None
    x1, y1, x2, y2 = [float(v) for v in d.bbox]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": float(d.area),
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "iscrowd": 0,
        "attributes": {"occluded": False, "source": "auto_pred", "score": round(float(d.score), 4)},
    }


def draw_outline(vis: np.ndarray, d: DetProposal, color, thickness=2):
    H, W = vis.shape[:2]
    y1o, x1o = int(d.top), int(d.left)
    y2o, x2o = int(d.top + d.mask.shape[0]), int(d.left + d.mask.shape[1])
    y1 = max(0, min(H, y1o)); y2 = max(0, min(H, y2o))
    x1 = max(0, min(W, x1o)); x2 = max(0, min(W, x2o))
    if y2 <= y1 or x2 <= x1:
        return
    m = d.mask[y1 - y1o:y1 - y1o + (y2 - y1), x1 - x1o:x1 - x1o + (x2 - x1)].astype(np.uint8) * 255
    sub = vis[y1:y2, x1:x2]
    if m.shape[:2] != sub.shape[:2]:
        m = m[:sub.shape[0], :sub.shape[1]]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sub, contours, -1, color, thickness)


def build_runner(config: str, weights: str, device: str) -> Detectron2SegInference:
    dataset = DatasetConfig(yolo_root="")
    model = ModelConfig(project_root=".", config_rel=config, checkpoint_rel=weights,
                        device=device, imports=("maskdino",))
    sw = SlidingWindowConfig()
    # base_keep_score 作为模型 ROI 粗筛阈值，设低一些不影响（MaskDINO 自带 topk）
    search = SearchSpace(base_keep_score=0.05)
    outputs = OutputOptions(vis=False, excel=False, draw_scatter=False, out_dir=".")
    runner = Detectron2SegInference(dataset, model, sw, search, outputs)
    runner.load_model()
    return runner


def main():
    p = argparse.ArgumentParser(description="Append MaskDINO wheat predictions to a CVAT COCO json (GT never deleted).")
    p.add_argument("--coco_json", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--viz_dir", default=None)
    p.add_argument("--score_thr", type=float, default=0.3)
    p.add_argument("--dedup_iou_min", type=float, default=0.2)
    p.add_argument("--pred_nms_iou", type=float, default=0.5)
    p.add_argument("--keep_labels", type=int, nargs="*", default=[0],
                   help="模型 label（0=wheat,1=wheat_2）。默认只保留 0=wheat")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no_viz", action="store_true")
    p.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 张图（试运行）")
    args = p.parse_args()

    keep_labels = tuple(args.keep_labels) if args.keep_labels else tuple()
    # 模型 label -> COCO category_id（0->1 wheat, 1->2 wheat_2）
    label_to_catid = {0: 1, 1: 2}

    print(f"==> Loading COCO json: {args.coco_json}")
    with open(args.coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = coco["images"]
    anns = coco["annotations"]
    max_ann_id = max((a["id"] for a in anns), default=0)
    next_ann_id = max_ann_id + 1
    print(f"    images={len(images)}  existing_anns={len(anns)}  max_ann_id={max_ann_id}")

    # image_id -> 其 GT 标注列表
    gt_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns:
        gt_by_img.setdefault(a["image_id"], []).append(a)

    print("==> Loading model ...")
    runner = build_runner(args.config, args.weights, args.device)

    do_viz = (args.viz_dir is not None) and (not args.no_viz)
    if do_viz:
        ensure_dir(args.viz_dir)

    img_list = images if args.limit <= 0 else images[:args.limit]
    new_anns: List[Dict[str, Any]] = []
    rows = [("file_name", "gt_count", "pred_kept", "added")]
    total_added = 0

    for im in tqdm(img_list, desc="Images"):
        img_id = im["id"]
        fname = im["file_name"]
        H = int(im.get("height", 0)); W = int(im.get("width", 0))
        img_path = os.path.join(args.images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[skip] cannot read {img_path}")
            rows.append((fname, len(gt_by_img.get(img_id, [])), 0, 0))
            continue
        if H == 0 or W == 0:
            H, W = img.shape[:2]

        # 1) 全图推理 -> 预测 proposals（按 score/label 过滤）
        preds = predictions_for_image(runner, img, args.score_thr, keep_labels)
        # 2) 预测之间做 mask-NMS（仅作用于预测，GT 不参与）
        if preds:
            preds = mask_nms_with_merge(preds, args.pred_nms_iou, IOUType.IOU_Min, score_mode="keep_max")

        # 3) GT 栅格化（只读），逐预测做去重：与任一 GT 的 IoU_Min >= 阈值则丢弃该预测
        gts = [gt_ann_to_proposal(a, H, W) for a in gt_by_img.get(img_id, [])]
        kept: List[DetProposal] = []
        for d in preds:
            dup = False
            for g in gts:
                if mask_iou_between(d, g, IOUType.IOU_Min) >= args.dedup_iou_min:
                    dup = True
                    break
            if not dup:
                kept.append(d)

        # 4) 追加为新标注
        added = 0
        for d in kept:
            catid = label_to_catid.get(int(d.label), int(d.label) + 1)
            ann = proposal_to_coco_ann(d, next_ann_id, img_id, catid)
            if ann is None:
                continue
            new_anns.append(ann)
            next_ann_id += 1
            added += 1
        total_added += added
        rows.append((fname, len(gts), len(kept), added))

        # 5) 可视化：GT=绿，新增=橙
        if do_viz:
            vis = img.copy()
            for g in gts:
                draw_outline(vis, g, GREEN_BGR, 1)
            for d in kept:
                draw_outline(vis, d, ORANGE_BGR, 2)
            cv2.imwrite(os.path.join(args.viz_dir, os.path.splitext(os.path.basename(fname))[0] + ".jpg"), vis)

    # 合并：原标注全部保留 + 新增
    coco["annotations"] = anns + new_anns
    ensure_dir(os.path.dirname(args.out_json))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"\n[Done] original_anns={len(anns)}  added={total_added}  total={len(coco['annotations'])}")
    print(f"[Saved] merged COCO -> {args.out_json}")
    if do_viz:
        print(f"[Saved] visualizations -> {args.viz_dir}")

    # CSV 摘要
    csv_path = os.path.splitext(args.out_json)[0] + "_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"[Saved] per-image summary -> {csv_path}")


if __name__ == "__main__":
    main()
