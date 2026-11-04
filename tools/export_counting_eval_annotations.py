#!/usr/bin/env python3
"""Export mask-assignment regions and GT references for external counting baselines.

Requires numpy, scipy and pycocotools; does not load a model or modify images.
This is a counting evaluation protocol, NOT a replacement COCO ground-truth file.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util
from scipy.optimize import linear_sum_assignment


def to_rle(segmentation, height, width):
    if isinstance(segmentation, list):
        if not segmentation:
            raise ValueError("Empty polygon segmentation")
        rle = mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict):
        if isinstance(segmentation["counts"], list):
            rle = mask_util.frPyObjects(segmentation, height, width)
        else:
            rle = dict(segmentation)
    else:
        raise ValueError("Expected polygon or RLE segmentation")
    if list(rle["size"]) != [height, width]:
        raise ValueError("Mask dimensions do not match the original image")
    return rle


def json_rle(rle):
    counts = rle["counts"]
    return {"size": list(rle["size"]),
            "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


def make_owner_map(rles, scores, height, width):
    """Return zero for background, local prediction index + 1 for owned pixels.

    Resolve overlaps over ALL score-filtered predictions, before GT matching.
    Ties use original prediction order. Matching never influences ownership.
    """
    owner = np.zeros((height, width), dtype=np.int32)
    overlap = np.zeros((height, width), dtype=bool)
    for index in sorted(range(len(rles)), key=lambda i: (-scores[i], i)):
        mask = mask_util.decode(rles[index]).astype(bool)
        overlap |= mask & (owner != 0)
        owner[mask & (owner == 0)] = index + 1
    return owner, int(overlap.sum())


def points_inside(points, mask):
    """Original-image (x,y), indexed with floor; no clipping of outside points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("Non-finite point coordinates")
    height, width = mask.shape
    valid = ((points[:, 0] >= 0) & (points[:, 0] < width)
             & (points[:, 1] >= 0) & (points[:, 1] < height))
    xy = np.floor(points[valid]).astype(np.int64)
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[xy[:, 1], xy[:, 0]] != 0
    return inside


def file_source(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def export_protocol(gt, predictions, category_id, score_thresh, iou_thresh):
    images = gt["images"]
    image_by_id = {im["id"]: im for im in images}
    if len(image_by_id) != len(images):
        raise ValueError("Duplicate image IDs")
    if category_id not in {cat["id"] for cat in gt["categories"]}:
        raise ValueError("Target category is absent from ground truth")
    ground_truth = [a for a in gt["annotations"] if a["category_id"] == category_id]
    if len({a["id"] for a in ground_truth}) != len(ground_truth):
        raise ValueError("Duplicate ground-truth annotation IDs")
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for ann in ground_truth:
        if ann["image_id"] not in image_by_id or "points" not in ann:
            raise ValueError("Invalid GT image reference or missing points field")
        # Match the current SpikeletPointEvaluator, including GT annotation order.
        gt_by_image[ann["image_id"]].append(ann)
    for index, pred in enumerate(predictions):
        if pred["image_id"] not in image_by_id:
            raise ValueError("Prediction references an image outside this split")
        if not np.isfinite(pred["score"]):
            raise ValueError("Non-finite prediction score")
        if pred["category_id"] == category_id and pred["score"] >= score_thresh:
            pred_by_image[pred["image_id"]].append((index, pred))

    annotations, image_records = [], []
    summary = Counter(images=len(images), gt_instances=len(ground_truth),
                      gt_points=sum(len(a["points"] or []) for a in ground_truth),
                      predictions=0, matched_instances=0, unmatched_predictions=0,
                      unmatched_gt_instances=0, overlap_images=0, overlap_pixels=0,
                      matched_gt_points=0, matched_gt_points_outside_pred_mask=0,
                      matched_gt_points_outside_owned_mask=0, empty_owned_masks=0)
    for im in images:
        height, width = im["height"], im["width"]
        pp, gg = pred_by_image[im["id"]], gt_by_image[im["id"]]
        pr = [to_rle(p["segmentation"], height, width) for _, p in pp]
        gr = [to_rle(g["segmentation"], height, width) for g in gg]
        pairs = {}
        if pr and gr:
            # Deliberately identical to SpikeletPointEvaluator: assign first,
            # then reject pairs below IoU threshold (no thresholded assignment).
            iou = np.asarray(mask_util.iou(pr, gr, [0] * len(gr)))
            ri, ci = linear_sum_assignment(-iou)
            pairs = {int(a): (int(b), float(iou[a, b]))
                     for a, b in zip(ri, ci) if iou[a, b] >= iou_thresh}
        owner, overlap = make_owner_map(pr, [p["score"] for _, p in pp], height, width)
        matched_gt_ids = {gg[gi]["id"] for gi, _ in pairs.values()}
        summary.update(predictions=len(pp), matched_instances=len(pairs),
                       unmatched_predictions=len(pp) - len(pairs),
                       unmatched_gt_instances=len(gg) - len(pairs),
                       overlap_images=int(overlap > 0), overlap_pixels=overlap)
        ids = []
        for pi, (source_index, pred) in enumerate(pp):
            ann_id = source_index + 1
            ids.append(ann_id)
            owned_mask = owner == pi + 1
            area = int(owned_mask.sum())
            summary["empty_owned_masks"] += int(area == 0)
            rec = {"id": ann_id, "image_id": im["id"], "category_id": category_id,
                   "source_prediction_index": source_index,
                   "score": pred["score"], "bbox": pred["bbox"],
                   "segmentation": json_rle(pr[pi]),
                   "evaluation_segmentation": json_rle(mask_util.encode(
                       np.asfortranarray(owned_mask.astype(np.uint8)))),
                   "evaluation_area": area, "matched": pi in pairs,
                   "gt_annotation_id": None, "match_iou": None, "gt_count": None}
            if pi in pairs:
                gi, match_iou = pairs[pi]
                target = gg[gi]
                points = target["points"] or []
                outside_pred = int((~points_inside(points, mask_util.decode(pr[pi]))).sum())
                outside_owned = int((~points_inside(points, owned_mask)).sum())
                rec.update(gt_annotation_id=target["id"], match_iou=match_iou,
                           gt_count=len(points), gt_points=points,
                           gt_points_outside_pred_mask=outside_pred,
                           gt_points_outside_owned_mask=outside_owned)
                summary.update(matched_gt_points=len(points),
                               matched_gt_points_outside_pred_mask=outside_pred,
                               matched_gt_points_outside_owned_mask=outside_owned)
            annotations.append(rec)
        image_records.append(dict(im, prediction_annotation_ids=ids,
                                  gt_count=sum(len(g["points"] or []) for g in gg),
                                  gt_instance_count=len(gg), matched_instance_count=len(pairs),
                                  unmatched_gt_annotation_ids=[g["id"] for g in gg
                                                               if g["id"] not in matched_gt_ids]))
    return {"images": image_records, "annotations": annotations,
            "ground_truth_annotations": ground_truth,
            "categories": gt["categories"], "summary": dict(summary)}


def check_reference_csvs(data, image_csv=None, pair_csv=None):
    """Verify the selected cohort against the existing model evaluation exports."""
    checks = {}
    names = {im["id"]: Path(im["file_name"]).name for im in data["images"]}
    if image_csv:
        with open(image_csv, newline="") as stream:
            rows = list(csv.DictReader(stream))
        actual = Counter((names[im["id"]], len(im["prediction_annotation_ids"]),
                          im["gt_instance_count"], im["gt_count"], im["matched_instance_count"])
                         for im in data["images"])
        expected = Counter((r["file_name"], int(r["pred_spikes"]), int(r["gt_spikes"]),
                            int(r["gt_points"]), int(r["matched_spikes"])) for r in rows)
        if actual != expected:
            raise ValueError("Per-image cohort does not reproduce the reference CSV")
        checks["image_csv"] = file_source(image_csv)
    if pair_csv:
        with open(pair_csv, newline="") as stream:
            rows = list(csv.DictReader(stream))
        actual = Counter((names[a["image_id"]], a["gt_count"], round(a["match_iou"], 10))
                         for a in data["annotations"] if a["matched"])
        expected = Counter((r["file_name"], int(r["gt_points"]), round(float(r["iou"]), 10))
                           for r in rows)
        if actual != expected:
            raise ValueError("Matched pairs do not reproduce the reference CSV")
        checks["pair_csv"] = file_source(pair_csv)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--pred-json", type=Path, required=True,
                        help="COCO results with original dataset category IDs (wheat_2=2)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category-id", type=int, default=2)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--checkpoint-label", default="unspecified",
                        help="Provenance note only; the checkpoint itself is not loaded or verified")
    parser.add_argument("--image-reference-csv", type=Path)
    parser.add_argument("--pair-reference-csv", type=Path)
    args = parser.parse_args()
    if not (0 <= args.score_thresh <= 1 and 0 <= args.iou_thresh <= 1):
        parser.error("Thresholds must be between 0 and 1")
    if args.output.exists():
        parser.error("Output already exists; choose a new filename to preserve the frozen protocol")
    sources = {"ground_truth": file_source(args.gt_json),
               "predictions": file_source(args.pred_json)}
    if args.source_config:
        sources["config"] = file_source(args.source_config)
    with args.gt_json.open() as stream:
        gt = json.load(stream)
    with args.pred_json.open() as stream:
        predictions = json.load(stream)
    data = export_protocol(gt, predictions, args.category_id, args.score_thresh, args.iou_thresh)
    checks = check_reference_csvs(data, args.image_reference_csv, args.pair_reference_csv)
    data["protocol"] = {
        "schema": "wheat_counting_eval_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "evaluation_only; annotations are predicted regions, not COCO ground truth",
        "sources": sources, "checkpoint_label_unverified": args.checkpoint_label,
        "reference_checks": checks, "category_id": args.category_id,
        "instance_score_threshold_inclusive": args.score_thresh,
        "matching": "Hungarian on original mask IoU, then reject below threshold",
        "matching_iou_threshold_inclusive": args.iou_thresh,
        "region_field": "evaluation_segmentation",
        "overlap_policy": "highest instance score over all retained predictions; ties by source order",
        "point_coordinates": "original image pixels (x,y); floor to pixel; outside image rejected",
        "density": "sum count mass within owned region; align spatially and preserve total mass",
        "spike_target": "all GT points of matched instance, including those outside predicted region",
        "spike_cohort": "matched=true; unmatched GT/predictions excluded only from per-spike metrics",
        "image_target": "all target-category GT points, including unmatched GT; retain zero-prediction images",
        "assignment_scope": "mask assignment applies only to external counting models without instance outputs",
        "baseline": "full/combo retains native instance point assignments and existing evaluation; no mask reassignment or baseline recomputation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "summary": data["summary"],
                      "reference_checks_passed": list(checks)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
