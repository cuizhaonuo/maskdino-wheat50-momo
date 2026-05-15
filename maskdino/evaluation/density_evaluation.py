# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Density map counting evaluation.
# ------------------------------------------------------------------------
import itertools
import logging
import numpy as np
from collections import OrderedDict

import torch

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator


def _compute_mask_iou_batch(pred_masks, gt_masks):
    """
    Compute IoU between N pred masks and M gt masks.

    Args:
        pred_masks (Tensor): (N, H, W) binary float masks.
        gt_masks (Tensor): (M, H, W) binary float masks.

    Returns:
        Tensor: (N, M) IoU matrix.
    """
    pred_bool = pred_masks.bool()   # (N, H, W)
    gt_bool = gt_masks.bool()       # (M, H, W)

    # (N, M) = sum over spatial dims
    pred_area = pred_masks.reshape(pred_masks.shape[0], -1).sum(dim=1)  # (N,)
    gt_area = gt_masks.reshape(gt_masks.shape[0], -1).sum(dim=1)        # (M,)

    intersection = torch.einsum("nhw,mhw->nm", pred_masks.float(), gt_masks.float())
    union = pred_area[:, None] + gt_area[None, :] - intersection

    iou = torch.zeros_like(intersection)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou


class DensityEvaluator(DatasetEvaluator):
    """
    Evaluate counting accuracy of predicted density maps against ground truth.

    Uses the same greedy IoU-based matching as COCO instance segmentation
    (sort predictions by score, match each prediction to the highest-IoU
    unmatched ground truth of the same class, with IoU >= 0.5).

    Metrics:
        MAE:  Mean Absolute Error of per-instance counts.
        RMSE: Root Mean Squared Error of per-instance counts.
        MAPE: Mean Absolute Percentage Error (%).
        Bias: Mean(pred_count - gt_count), positive = over-counting.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        iou_threshold=0.5,
    ):
        """
        Args:
            dataset_name (str): name of the dataset.
            distributed (bool): if True, gather results from all ranks.
            output_dir (str): optional output directory.
            iou_threshold (float): mask IoU threshold for matching.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._iou_threshold = iou_threshold
        self._cpu_device = torch.device("cpu")

        self.reset()

    def reset(self):
        self._per_image_data = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: list of dicts, each containing keys like "image_id",
                "file_name", and optionally "instances" (GT).
            outputs: list of dicts, each containing key "instances" (pred).
        """
        for input, output in zip(inputs, outputs):
            image_id = input.get("image_id", -1)

            pred_instances = output.get("instances", None)
            if pred_instances is not None:
                pred_instances = pred_instances.to(self._cpu_device)

            gt_instances = input.get("instances", None)
            if gt_instances is not None:
                gt_instances = gt_instances.to(self._cpu_device)

            self._per_image_data.append({
                "image_id": image_id,
                "pred": pred_instances,
                "gt": gt_instances,
            })

    def evaluate(self):
        """Compute and return density counting metrics."""
        # Gather data from all workers
        if self._distributed:
            comm.synchronize()
            per_image_data = comm.gather(self._per_image_data, dst=0)
            if not comm.is_main_process():
                return {}
            per_image_data = list(itertools.chain(*per_image_data))
        else:
            per_image_data = self._per_image_data

        if len(per_image_data) == 0:
            self._logger.warning("[DensityEvaluator] No predictions collected.")
            return {}

        # ---- Check if GT density maps are available ----
        has_gt_density = any(
            data["gt"] is not None and hasattr(data["gt"], "gt_density_maps") and len(data["gt"]) > 0
            for data in per_image_data
        )
        has_pred_density = any(
            data["pred"] is not None and hasattr(data["pred"], "pred_density_maps") and len(data["pred"]) > 0
            for data in per_image_data
        )

        if not has_gt_density:
            self._logger.warning(
                "[DensityEvaluator] No GT density maps found. "
                "Ensure the test dataset mapper generates gt_density_maps (non-training mode)."
            )
            return {}

        if not has_pred_density:
            self._logger.warning("[DensityEvaluator] No predicted density maps found.")
            return {}

        # ---- Per-image matching and counting ----
        all_pred_counts = []
        all_gt_counts = []
        num_matched = 0
        num_skipped_no_points = 0
        num_fp = 0
        num_fn = 0

        for data in per_image_data:
            pred = data["pred"]
            gt = data["gt"]

            if pred is None or gt is None:
                continue
            if not hasattr(pred, "pred_density_maps") or len(pred) == 0:
                num_fn += len(gt) if hasattr(gt, "gt_density_maps") else 0
                continue
            if not hasattr(gt, "gt_density_maps") or len(gt) == 0:
                num_fp += len(pred)
                continue

            pred_density = pred.pred_density_maps  # (N_pred, H, W)
            pred_masks = pred.pred_masks            # (N_pred, H, W)
            pred_scores = pred.scores               # (N_pred,)
            pred_classes = pred.pred_classes        # (N_pred,)

            gt_density = gt.gt_density_maps          # (N_gt, H, W)
            gt_masks = gt.gt_masks                   # (N_gt, H, W)
            gt_classes = gt.gt_classes               # (N_gt,)

            # Ensure consistent spatial size via interpolation if needed
            if pred_density.shape[-2:] != gt_density.shape[-2:]:
                pred_density = torch.nn.functional.interpolate(
                    pred_density[None, :],
                    size=gt_density.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks = torch.nn.functional.interpolate(
                    pred_masks[None, :].float(),
                    size=gt_masks.shape[-2:],
                    mode="nearest",
                )[0].bool().float()

            # ---- Greedy IoU matching per class ----
            unique_classes = torch.unique(torch.cat([pred_classes.long(), gt_classes.long()]))

            for cls in unique_classes:
                cls = cls.item()
                pred_idx = (pred_classes.long() == cls).nonzero(as_tuple=True)[0]
                gt_idx = (gt_classes.long() == cls).nonzero(as_tuple=True)[0]

                if len(pred_idx) == 0:
                    num_fn += len(gt_idx)
                    continue
                if len(gt_idx) == 0:
                    num_fp += len(pred_idx)
                    continue

                # Sort predictions by score descending
                sorted_idx = pred_idx[torch.argsort(pred_scores[pred_idx], descending=True)]

                # Compute IoU between all pred and GT masks of this class
                iou_matrix = _compute_mask_iou_batch(
                    pred_masks[sorted_idx], gt_masks[gt_idx]
                )  # (P, G)

                matched_gt = set()
                for p_local, p_global in enumerate(sorted_idx):
                    best_iou = 0.0
                    best_g = None

                    for g_local, g_global in enumerate(gt_idx):
                        if g_global.item() in matched_gt:
                            continue
                        iou_val = iou_matrix[p_local, g_local].item()
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_g = g_global

                    if best_iou >= self._iou_threshold and best_g is not None:
                        matched_gt.add(best_g.item())

                        pred_count = pred_density[p_global].sum().item()
                        gt_count = gt_density[best_g].sum().item()

                        if gt_count > 0:
                            all_pred_counts.append(pred_count)
                            all_gt_counts.append(gt_count)
                            num_matched += 1
                        else:
                            num_skipped_no_points += 1
                    else:
                        num_fp += 1

                num_fn += len(gt_idx) - len(matched_gt)

        # ---- Compute metrics ----
        if num_matched == 0:
            self._logger.warning("[DensityEvaluator] No matched prediction-GT pairs.")
            return {
                "density": OrderedDict({
                    "MAE": -1.0,
                    "RMSE": -1.0,
                    "MAPE": -1.0,
                    "Bias": -1.0,
                    "Matched": 0,
                })
            }

        errors = np.abs(np.array(all_pred_counts) - np.array(all_gt_counts))
        bias_vals = np.array(all_pred_counts) - np.array(all_gt_counts)

        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(bias_vals))

        # MAPE (avoid division by zero)
        eps = 1e-6
        mape = float(np.mean(errors / np.maximum(np.array(all_gt_counts), eps)) * 100.0)

        result = {
            "density": OrderedDict({
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Bias": bias,
                "Matched": num_matched,
            })
        }

        self._logger.info(
            "Density evaluation: MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%, Bias=%.4f, "
            "Matched=%d, Skipped(no_points)=%d, FP=%d, FN=%d",
            mae, rmse, mape, bias, num_matched, num_skipped_no_points, num_fp, num_fn,
        )

        return result
