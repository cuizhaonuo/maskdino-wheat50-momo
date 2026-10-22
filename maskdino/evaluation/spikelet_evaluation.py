"""小穗点（spikelet point）评估器。

COCOEvaluator 只评 mask/box AP，看不出点回归头学得怎么样。本评估器做两级匹配后给出
点级 P/R/F1（固定像素距离阈值 σ）与计数误差，挂进 train_net.py 的 build_evaluator，
每个 TEST.EVAL_PERIOD 都会写进 metrics.json / TensorBoard。

两级匹配：
  1. 实例级：GT 的 wheat_2 穗 ↔ 预测的 wheat_2 穗，按 mask IoU 做匈牙利匹配，IoU >= 阈值才算配上。
  2. 点级：在每对配上的实例内部，对欧氏距离矩阵做一次匈牙利匹配，再按各档 σ 判 d <= σ 记 TP。
     一次分配、多档阈值，保证各档口径一致。结构上与 criterion.py 的 _inner_point_match 同构，
     只是 cost 用纯欧氏距离、不带 focal 项。

指标分两套：
  - 端到端（P@σ / R@σ / F1@σ）：未配对预测穗内的点全记 FP，未配对 GT 穗内的点全记 FN，
    因此包含了检测误差，是部署时真正关心的数。
  - 仅配对实例（matched_*）：把检测误差剥离，单看点头本身。

坐标系：模型输出的 pred_spikelet_points 已经在原图像素系（maskdino.py 在 instance_inference
之前就把 mask/box 还原回了原图尺寸，点是纯框相对还原），GT 的 points 也是原图绝对像素，
两者可直接比对，σ 的像素单位即原图像素。
"""

import copy
import csv
import itertools
import logging
import os
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util
from scipy.optimize import linear_sum_assignment

import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator


def _polys_to_rle(segm, height, width):
    """COCO segmentation（多边形 list / RLE）-> 单个合并后的 RLE。"""
    if isinstance(segm, list):
        rles = mask_util.frPyObjects(segm, height, width)
        return mask_util.merge(rles)
    if isinstance(segm, dict):
        if isinstance(segm.get("counts"), list):   # uncompressed RLE
            return mask_util.frPyObjects(segm, height, width)
        return segm
    raise TypeError(f"无法识别的 segmentation 类型: {type(segm)}")


def _match_points(pred_xy, gt_xy):
    """点级匈牙利匹配。返回 (pred_idx, gt_idx, dist)，长度均为 min(len(pred), len(gt))。"""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)
    d = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=-1)
    ri, ci = linear_sum_assignment(d)
    return ri, ci, d[ri, ci]


class SpikeletPointEvaluator(DatasetEvaluator):
    """评估每个 wheat_2 实例内部的小穗点定位与计数。"""

    def __init__(self, dataset_name, output_dir=None, sigmas=(3.0, 5.0, 8.0),
                 score_thresh=0.5, iou_thresh=0.5, spikelet_class_id=1, distributed=True):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._sigmas = tuple(float(s) for s in sigmas)
        self._score_thresh = float(score_thresh)
        self._iou_thresh = float(iou_thresh)
        self._spikelet_class_id = int(spikelet_class_id)
        self._distributed = distributed

        # GT 直接取自 DatasetCatalog：wheat 数据集没有设 json_file metadata，COCOEvaluator 会回退到
        # convert_to_coco_json，而那个转换不保留 points；DatasetCatalog 的记录里则带着 points。
        self._gt = {}
        n_inst = n_pts = 0
        for rec in DatasetCatalog.get(dataset_name):
            h, w = rec["height"], rec["width"]
            points, rles = [], []
            for ann in rec.get("annotations", []):
                if int(ann["category_id"]) != self._spikelet_class_id:
                    continue
                pts = np.asarray(ann.get("points") or [], dtype=np.float64).reshape(-1, 2)
                points.append(pts)
                rles.append(_polys_to_rle(ann["segmentation"], h, w))
                n_pts += len(pts)
            n_inst += len(points)
            self._gt[rec["image_id"]] = {"points": points, "rles": rles,
                                         "file_name": os.path.basename(rec["file_name"])}
        self._logger.info(
            f"[SpikeletPointEvaluator] {dataset_name}: {len(self._gt)} 图, "
            f"{n_inst} 个 wheat_2 实例, {n_pts} 个 GT 小穗点 | "
            f"σ={list(self._sigmas)}px, score>={self._score_thresh}, IoU>={self._iou_thresh}"
        )
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if "instances" not in out:
                continue
            inst = out["instances"]
            # 注意：绝不能对 Instances 做布尔索引 —— pred_spikelet_points 是 python list，
            # Instances.__getitem__ 会用同一个索引去切它而抛 TypeError。这里只按整数逐个取。
            classes = inst.pred_classes.detach().cpu().numpy()
            scores = inst.scores.detach().cpu().numpy()
            has_points = inst.has("pred_spikelet_points")
            # 诊断字段（POINT_ECOUNT_WEIGHT > 0 时模型才输出）：CORAL 头的期望计数 E，只进逐穗明细的 ecount 列
            ecount = (inst.pred_spikelet_ecount.detach().cpu().numpy()
                      if inst.has("pred_spikelet_ecount") else None)

            points, rles, keep_scores, ecounts = [], [], [], []
            for i in range(len(inst)):
                if int(classes[i]) != self._spikelet_class_id or scores[i] < self._score_thresh:
                    continue
                pts = inst.pred_spikelet_points[i] if has_points else None
                pts = np.zeros((0, 2)) if pts is None else pts.detach().cpu().numpy()
                m = inst.pred_masks[i].detach().cpu().numpy().astype(np.uint8)
                rles.append(mask_util.encode(np.asfortranarray(m)))
                points.append(np.asarray(pts, dtype=np.float64).reshape(-1, 2))
                keep_scores.append(float(scores[i]))
                if ecount is not None:
                    ecounts.append(float(ecount[i]))

            # 零预测的图也要保留：它们对 recall 与计数 MAE 有贡献。
            rec = {
                "image_id": inp["image_id"], "points": points,
                "rles": rles, "scores": keep_scores,
            }
            if ecount is not None:
                rec["ecounts"] = ecounts
            self._predictions.append(rec)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            gathered = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*gathered))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if not predictions:
            self._logger.warning("[SpikeletPointEvaluator] 没有收到任何预测。")
            return {}

        n_sig = len(self._sigmas)
        tp = np.zeros(n_sig)            # 端到端 TP（按 σ）
        tp_m = np.zeros(n_sig)          # 仅配对实例内的 TP
        n_pred_pts = n_gt_pts = 0
        n_pred_pts_m = n_gt_pts_m = 0
        dist_sum = 0.0
        dist_n = 0
        spike_count_abs_err = []        # 配对实例上的每穗点数误差
        pair_rows = []                  # 逐 TP 穗的 (预测点数, GT点数)，供散点分析
        img_pred_tot, img_gt_tot = [], []
        n_gt_spike = n_pred_spike = n_matched_spike = 0
        rows = []
        # 模型输出了诊断字段 pred_spikelet_ecount 时，逐穗明细多一列 ecount（E，不进任何指标）
        has_ecount = any("ecounts" in p for p in predictions)

        for pred in predictions:
            gt = self._gt.get(pred["image_id"])
            if gt is None:
                self._logger.warning(f"[SpikeletPointEvaluator] image_id {pred['image_id']} 不在 GT 中，跳过")
                continue
            gp, pp = gt["points"], pred["points"]
            n_gt_spike += len(gp)
            n_pred_spike += len(pp)
            gt_tot = int(sum(len(x) for x in gp))
            pred_tot = int(sum(len(x) for x in pp))
            n_gt_pts += gt_tot
            n_pred_pts += pred_tot
            img_gt_tot.append(gt_tot)
            img_pred_tot.append(pred_tot)

            # --- 实例级：mask IoU + 匈牙利 ---
            pairs = []
            if gp and pp:
                iou = mask_util.iou(pred["rles"], gt["rles"], [0] * len(gt["rles"]))
                iou = np.asarray(iou).reshape(len(pp), len(gp))
                ri, ci = linear_sum_assignment(-iou)
                pairs = [(int(a), int(b)) for a, b in zip(ri, ci) if iou[a, b] >= self._iou_thresh]
            n_matched_spike += len(pairs)

            # --- 点级：配对实例内部匈牙利，一次分配、多档阈值 ---
            for pi, gi in pairs:
                _, _, d = _match_points(pp[pi], gp[gi])
                for k, s in enumerate(self._sigmas):
                    hit = int((d <= s).sum())
                    tp[k] += hit
                    tp_m[k] += hit
                if d.size:
                    within = d[d <= max(self._sigmas)]
                    dist_sum += float(within.sum())
                    dist_n += int(within.size)
                n_pred_pts_m += len(pp[pi])
                n_gt_pts_m += len(gp[gi])
                spike_count_abs_err.append(abs(len(pp[pi]) - len(gp[gi])))
                row = (gt["file_name"], len(pp[pi]), len(gp[gi]), float(iou[pi, gi]))
                if has_ecount:
                    row = row + (round(pred["ecounts"][pi], 4),)
                pair_rows.append(row)

            rows.append((gt["file_name"], len(pp), pred_tot, len(gp), gt_tot, len(pairs)))

        def prf(tp_arr, n_pred, n_gt, k):
            p = tp_arr[k] / n_pred if n_pred else float("nan")
            r = tp_arr[k] / n_gt if n_gt else float("nan")
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return float(p), float(r), float(f)

        res = {}
        f1s = []
        for k, s in enumerate(self._sigmas):
            tag = f"{s:g}"
            p, r, f = prf(tp, n_pred_pts, n_gt_pts, k)
            res[f"P@{tag}"], res[f"R@{tag}"], res[f"F1@{tag}"] = p, r, f
            f1s.append(f)
            res[f"matched_F1@{tag}"] = prf(tp_m, n_pred_pts_m, n_gt_pts_m, k)[2]
        res["mF1"] = float(np.mean(f1s)) if f1s else float("nan")
        res["dist_mean_px"] = dist_sum / dist_n if dist_n else float("nan")
        res["count_MAE_spike"] = float(np.mean(spike_count_abs_err)) if spike_count_abs_err else float("nan")

        pt_, gt_ = np.asarray(img_pred_tot, float), np.asarray(img_gt_tot, float)
        res["count_MAE_img"] = float(np.mean(np.abs(pt_ - gt_))) if pt_.size else float("nan")
        res["count_RMSE_img"] = float(np.sqrt(np.mean((pt_ - gt_) ** 2))) if pt_.size else float("nan")
        res["spike_recall"] = n_matched_spike / n_gt_spike if n_gt_spike else float("nan")
        res["spike_precision"] = n_matched_spike / n_pred_spike if n_pred_spike else float("nan")
        # ---- 假阳 wheat_2 的产出量 ----
        # count_MAE_spike 与 matched_F1@σ 都只在配对上的穗内部计算，
        # 被误判为 wheat_2 的实例照常输出小穗点，却完全不进这两个指标。
        # 这三项把那部分产出显式记下来，与上面两个主指标配套读。
        res["fp_spike_count"] = float(n_pred_spike - n_matched_spike)
        res["fp_point_count"] = float(n_pred_pts - n_pred_pts_m)
        res["fp_point_ratio"] = (n_pred_pts - n_pred_pts_m) / n_pred_pts if n_pred_pts else float("nan")
        res["n_gt_points"] = float(n_gt_pts)
        res["n_pred_points"] = float(n_pred_pts)

        self._logger.info(
            f"[SpikeletPointEvaluator] GT {n_gt_spike} 穗/{n_gt_pts} 点 | "
            f"预测 {n_pred_spike} 穗/{n_pred_pts} 点 | 实例配对 {n_matched_spike} | "
            + "  ".join(f"F1@{s:g}={res[f'F1@{s:g}']:.4f}" for s in self._sigmas)
        )

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            path = os.path.join(self._output_dir, "spikelet_pred_vs_gt.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["file_name", "pred_spikes", "pred_points",
                               "gt_spikes", "gt_points", "matched_spikes"])
                wcsv.writerows(sorted(rows))
            self._logger.info(f"[SpikeletPointEvaluator] 逐图明细 -> {path}")

            # 逐 TP 穗明细：与 count_MAE_spike 完全同一批配对（mask IoU 匈牙利 + IoU>=阈值），
            # count_MAE_spike == mean(|pred_points - gt_points|)。供散点/偏差分析。
            ppath = os.path.join(self._output_dir, "spikelet_pair_counts.csv")
            with open(ppath, "w", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["file_name", "pred_points", "gt_points", "iou"]
                              + (["ecount"] if has_ecount else []))
                wcsv.writerows(pair_rows)
            self._logger.info(f"[SpikeletPointEvaluator] 逐 TP 穗明细 -> {ppath}")

        return OrderedDict([("spikelet", copy.deepcopy(res))])
