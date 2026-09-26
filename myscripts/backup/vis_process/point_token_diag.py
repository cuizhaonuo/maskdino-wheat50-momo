"""小穗点诊断：mini/iso-norm（E1）与 mini/group-decoder（E3）预注册里的机制判据。

在 val 上跑一遍推理，把点存在性阈值临时放到 -1，拿到每个实例的全部 K 个点及其分数。
对配对上的 wheat_2（wheat_2 预测、score >= 阈值、mask IoU >= 阈值，与 SpikeletPointEvaluator 同口径）报：
  - axis：阈值化后的点与 GT 做点级匈牙利（评估器同一函数），误差向量投影到该 GT 穗的主轴
          （GT 点 PCA）上，给沿轴 / 横轴绝对误差的中位数（全部配对，以及距离 <= σ 的配对）；
  - auc： 全部 K 个点与 GT 做点级匈牙利，「配上且距离 <= σ」为正，存在性分数的 AUC（Mann-Whitney）；
  - dup： 阈值化后同一穗内距离 < dup_px 的预测点对数 / 预测点数；
  - check：用同一批配对重算 matched_F1@σ，应与训练日志 metrics.json 里同阈值的值一致（自检）；
  - sweep：一遍推理拿到全部点与分数后，离线扫 POINT_THRESHOLD，给每档的 matched_F1@3/5/8 与
           count_MAE_spike（二者只依赖配对穗内阈值化后的点；穗配对与点阈值无关，所以与逐档重跑推理完全等价）。

只依赖 MaskDINO.instance_inference 的签名与 point_threshold 属性，任何分支的 checkpoint 都能分析；
POINT_GROUP_DECODER 的 run 必须在带分组 decoder 的代码下跑（本分支的代码开关关着时与基线逐位一致）。

用法（仓库根目录）：
  python myscripts/backup/vis_process/point_token_diag.py --run-dir <output_v2/run> \
      [--weights model_final.pth] --out <json> [--sigma 8] [--dup-px 3] [--limit 0] [opts ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import pycocotools.mask as mask_util
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import train_net  # noqa: E402,F401  注册数据集（import 副作用）
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import build_detection_test_loader  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from maskdino import add_maskdino_config  # noqa: E402
from maskdino.evaluation.spikelet_evaluation import SpikeletPointEvaluator, _match_points  # noqa: E402


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _wrap_instance_inference(model):
    """instance_inference 保留全部 K 个点，并按同一套 topk 顺序附上每个点的存在性分数 pred_spikelet_scores。"""
    orig = model.instance_inference
    n_cls = model.sem_seg_head.num_classes

    def wrapped(mask_cls, mask_pred, mask_box_result, point_coords=None, point_logits=None):
        thr = model.point_threshold
        model.point_threshold = -1.0
        try:
            result = orig(mask_cls, mask_pred, mask_box_result, point_coords, point_logits)
        finally:
            model.point_threshold = thr
        if point_logits is not None:
            # 与 instance_inference 完全相同的 topk 调用；用类别序列逐个比对，确认顺序对齐
            _, topk_full = mask_cls.sigmoid().flatten(0, 1).topk(model.test_topk_per_image, sorted=False)
            labels = torch.arange(n_cls, device=mask_cls.device).repeat(mask_cls.shape[0])
            assert torch.equal(labels[topk_full].to(result.pred_classes.device), result.pred_classes), \
                "topk 顺序与 instance_inference 不一致"
            pl = point_logits[topk_full // n_cls].sigmoid().squeeze(-1)   # [N_inst, P]
            result.pred_spikelet_scores = [
                pl[k] if int(result.pred_classes[k]) == model.spikelet_class_id else pl.new_zeros((0,))
                for k in range(len(result))]
        return result

    model.instance_inference = wrapped


def _principal_axes(g):
    """GT 点的主轴 u 与其法向 v（Ni >= 2）。"""
    c = g - g.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    u = vt[0]
    return u, np.array([-u[1], u[0]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--weights", default="model_final.pth", help="相对 run-dir 或绝对路径")
    p.add_argument("--out", required=True)
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--dup-px", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 张图（0 = 全部）")
    p.add_argument("--sweep", type=float, nargs="*",
                   default=[round(0.10 + 0.05 * i, 2) for i in range(11)],
                   help="离线扫描的 POINT_THRESHOLD 档位（默认 0.10~0.60 步长 0.05）")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(os.path.join(a.run_dir, "config.yaml"))
    cfg.merge_from_list(a.opts)
    cfg.freeze()

    weights = a.weights if os.path.isabs(a.weights) else os.path.join(a.run_dir, a.weights)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights)
    model.eval()
    _wrap_instance_inference(model)

    ds = cfg.DATASETS.TEST[0]
    t = cfg.MODEL.MaskDINO.TEST
    score_thr, iou_thr, pt_thr = t.SPIKELET_EVAL_SCORE_THRESH, t.SPIKELET_EVAL_IOU_THRESH, t.POINT_THRESHOLD
    gt_all = SpikeletPointEvaluator(ds, sigmas=(a.sigma,), score_thresh=score_thr, iou_thresh=iou_thr,
                                    distributed=False)._gt
    loader = build_detection_test_loader(cfg, ds)

    auc_s, auc_l = [], []
    along, across, along_s, across_s = [], [], [], []
    n_pred_thr = n_dup = n_pairs = n_gt_spike = 0
    tp_m = n_pred_m = n_gt_m = 0
    pair_data = []   # 每个配对穗的 (全部点, 分数, GT 点)，供阈值扫描
    with torch.no_grad():
        for it, inputs in enumerate(loader):
            outputs = model(inputs)
            for inp, out in zip(inputs, outputs):
                gt = gt_all[inp["image_id"]]
                n_gt_spike += len(gt["points"])
                inst = out["instances"]
                classes = inst.pred_classes.cpu().numpy()
                scores = inst.scores.cpu().numpy()
                pts_all, sc_all, rles = [], [], []
                for i in range(len(inst)):
                    if int(classes[i]) != model.spikelet_class_id or scores[i] < score_thr:
                        continue
                    pts_all.append(inst.pred_spikelet_points[i].float().cpu().numpy().reshape(-1, 2))
                    sc_all.append(inst.pred_spikelet_scores[i].float().cpu().numpy().reshape(-1))
                    m = inst.pred_masks[i].cpu().numpy().astype(np.uint8)
                    rles.append(mask_util.encode(np.asfortranarray(m)))
                if not rles or not gt["rles"]:
                    continue
                iou = np.asarray(mask_util.iou(rles, gt["rles"], [0] * len(gt["rles"]))).reshape(len(rles), -1)
                ri, ci = linear_sum_assignment(-iou)
                for pi, gi in zip(ri, ci):
                    if iou[pi, gi] < iou_thr:
                        continue
                    n_pairs += 1
                    P_all, s, G = pts_all[pi], sc_all[pi], gt["points"][gi]
                    pair_data.append((P_all, s, G))
                    # auc：全部 K 个点
                    r_a, _, d_a = _match_points(P_all, G)
                    lab = np.zeros(len(P_all), dtype=bool)
                    lab[r_a[d_a <= a.sigma]] = True
                    auc_s.extend(s.tolist())
                    auc_l.extend(lab.tolist())
                    # 阈值化后的点（与推理 / 评估同一阈值规则：score > POINT_THRESHOLD）
                    P = P_all[s > pt_thr]
                    r_t, c_t, d_t = _match_points(P, G)
                    tp_m += int((d_t <= a.sigma).sum())
                    n_pred_m += len(P)
                    n_gt_m += len(G)
                    if len(G) >= 2 and len(r_t):
                        u, v = _principal_axes(G)
                        e = P[r_t] - G[c_t]
                        ea, ec = np.abs(e @ u), np.abs(e @ v)
                        along.extend(ea.tolist())
                        across.extend(ec.tolist())
                        w = d_t <= a.sigma
                        along_s.extend(ea[w].tolist())
                        across_s.extend(ec[w].tolist())
                    if len(P) > 1:
                        dd = np.linalg.norm(P[:, None] - P[None], axis=-1)[np.triu_indices(len(P), 1)]
                        n_dup += int((dd < a.dup_px).sum())
                    n_pred_thr += len(P)
            if a.limit and it + 1 >= a.limit:
                break

    prec = tp_m / n_pred_m if n_pred_m else float("nan")
    rec = tp_m / n_gt_m if n_gt_m else float("nan")
    med = lambda x: float(np.median(x)) if len(x) else float("nan")  # noqa: E731
    res = {
        "run_dir": a.run_dir, "weights": weights, "dataset": ds,
        "point_threshold": pt_thr, "score_thresh": score_thr, "iou_thresh": iou_thr,
        "sigma": a.sigma, "dup_px": a.dup_px,
        "n_matched_spikes": n_pairs, "n_gt_spikes": n_gt_spike,
        "auc": {"value": _auc(auc_s, auc_l), "n_tokens": len(auc_l),
                "pos_rate": float(np.mean(auc_l)) if auc_l else float("nan")},
        "axis_all_pairs": {"along_median_px": med(along), "across_median_px": med(across), "n": len(along)},
        "axis_within_sigma": {"along_median_px": med(along_s), "across_median_px": med(across_s), "n": len(along_s)},
        "dup": {"ratio": n_dup / n_pred_thr if n_pred_thr else float("nan"),
                "n_close_pairs": n_dup, "n_pred_points": n_pred_thr},
        "check_matched_F1": (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0,
    }

    # 阈值扫描：与训练期评估同口径（配对穗内、score > 阈值、点级匈牙利、d <= σ 记 TP）
    sigmas = (3.0, 5.0, 8.0)
    sweep = []
    for thr in a.sweep:
        tp = np.zeros(len(sigmas))
        n_pred = n_gt = 0
        cnt_err = []
        for P_all, s, G in pair_data:
            P = P_all[s > thr]
            _, _, d = _match_points(P, G)
            for k, sg in enumerate(sigmas):
                tp[k] += int((d <= sg).sum())
            n_pred += len(P)
            n_gt += len(G)
            cnt_err.append(abs(len(P) - len(G)))
        row = {"threshold": thr, "n_pred_points": n_pred,
               "count_MAE_spike": float(np.mean(cnt_err)) if cnt_err else float("nan")}
        for k, sg in enumerate(sigmas):
            pr = tp[k] / n_pred if n_pred else 0.0
            rc = tp[k] / n_gt if n_gt else 0.0
            row[f"matched_F1@{sg:g}"] = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        sweep.append(row)
    res["sweep"] = sweep

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
