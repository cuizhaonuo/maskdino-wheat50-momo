"""小穗点诊断：mini/iso-norm（E1）、mini/group-decoder（E3）与 mini/group-nms-{match,prune} 预注册里的机制判据。

在 val 上跑一遍推理，把点存在性阈值临时放到 -1、推理侧点 NMS 临时关掉，拿到每个实例的全部 K 个点及其分数
（模型给出存活掩码 pred_point_keep 时一并取出）。对配对上的 wheat_2（wheat_2 预测、score >= 阈值、mask IoU >= 阈值，
与 SpikeletPointEvaluator 同口径）报：

旧指标（口径不变）：
  - axis：阈值化后的点与 GT 做点级匈牙利（评估器同一函数），误差向量投影到该 GT 穗的主轴
          （GT 点 PCA）上，给沿轴 / 横轴绝对误差的中位数（全部配对，以及距离 <= σ 的配对）；
  - auc： 全部 K 个点与 GT 做点级匈牙利，「配上且距离 <= σ」为正，存在性分数的 AUC（Mann-Whitney）；
  - dup： 阈值化后同一穗内距离 < dup_px 的预测点对数 / 预测点数；
  - check：用同一批配对重算 matched_F1@σ（只按 score > POINT_THRESHOLD，不做 NMS），
           推理侧 NMS 关着时应与训练日志 metrics.json 里同阈值的值一致（自检）；
  - sweep：一遍推理拿到全部点与分数后，离线扫 POINT_THRESHOLD，给每档的 matched_F1@3/5/8 与
           count_MAE_spike（二者只依赖配对穗内阈值化后的点；穗配对与点阈值无关，所以与逐档重跑推理完全等价）。

新指标（mini/group-nms-*；只看分组 decoder 的小穗 token）：
  - flip：用 forward hook 截 decoder 第 1~9 层小穗 token 的坐标与分数（aux_outputs[1..8] 与最终输出），
          每层都按 criterion 的规则把 token 配到 GT 小穗：criterion._point_nms_keep（匹配前 NMS，见下）+
          criterion._inner_point_match（robust=True，iso_scale 用 criterion._point_iso_scale）。
          GT 框取 GT mask 的外接框（mask_util.toBbox，与训练 mapper 用 mask 外接框同口径），坐标换到训练用的
          图像归一化系。得到每个 GT 小穗配到的 token 下标（没配上记 -1），报相邻层（1→2 … 8→9）下标不同的比例，
          逐对报，再报平均；做了匹配前 NMS 时另报每层的被压比例与饿死比例（留下的 token 少于 Ni 的穗占比）。
          匹配前 NMS 的来源：截到了 pred_point_keep 就用它；否则 --match-nms-kappa
          （默认按配置：POINT_NMS_MODE != "none" 时用 POINT_NMS_KAPPA，否则不做）。
  - auc_surv：最终层上「存活（有 keep 时）∧ 过了推理 NMS」的 token 的 AUC。正样本定义与旧 auc 相同
          （纯距离匈牙利、配上且 <= σ），只是参与匹配的 token 换成这些留下的；另附「标签取自全部 K 个
          token 的匹配」的版本（value_labels_from_all_k）作对照。
  - coverage：全部 K 个 token、只用留下的 token，分别与 GT 做纯距离一对一匈牙利，<= σ 的 GT 比例，
          及两者之差（百分点；即 NMS 造成的 σ 覆盖损失）。
  - readout_model：模型自己的读数规则，即 存活 ∧ 推理 NMS ∧ score > τ，F1 应与内置评估一致（自检）；
    readout_no_nms：只按 score > τ。各报 matched_F1@3/5/8、count_MAE_spike、偏差（pred − gt 均值）、
          Pearson r（每穗预测点数 vs GT 点数）、去偏移 MAE（mean|e − mean(e)|，e = pred − gt）。
  推理侧 NMS（auc_surv / coverage / readout_model 用）：--nms-kappa（默认取配置的 TEST.POINT_NMS_KAPPA），
  半径 κ·sqrt(w·h)，w、h 为预测框像素宽高，与 MaskDINO.instance_inference 同一套写法（同设备、同表达式）。

只依赖 MaskDINO.instance_inference 的签名与 point_threshold / point_nms_kappa 属性，任何分支的 checkpoint
都能分析；POINT_GROUP_DECODER 的 run 必须在带分组 decoder 的代码下跑（本分支的代码开关关着时与基线逐位一致）。
非分组 decoder 的 run 没有小穗 token，flip 记为 None。

用法（仓库根目录）：
  python myscripts/backup/vis_process/point_token_diag.py --sweep --run-dir <output_v2/run> \
      [--weights model_final.pth] --out <json> [--sigma 8] [--dup-px 3] [--limit 0] \
      [--match-nms-kappa K] [--nms-kappa K] [opts ...]
  --sweep 不带值时不做阈值扫描。它是 nargs="*"，要放在所有参数的最前面，放在后面会吞掉后续参数；
  不传 --sweep 则按默认档位扫描（训练结束后不要擅自扫，见 CLAUDE.md）。
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
from maskdino.utils.point_nms import greedy_point_nms  # noqa: E402

_SIGMAS = (3.0, 5.0, 8.0)


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _wrap_instance_inference(model):
    """instance_inference 保留全部 K 个点（阈值临时 -1、推理侧 NMS 临时关、不传存活掩码），并按同一套 topk 顺序附上：
    pred_spikelet_scores（每点存在性分数）、pred_query_index（实例的 query 下标，= topk 下标 // 类别数），
    以及模型给出存活掩码时 topk 之后的 pred_spikelet_keep。"""
    orig = model.instance_inference
    n_cls = model.sem_seg_head.num_classes

    def wrapped(mask_cls, mask_pred, mask_box_result, point_coords=None, point_logits=None, point_keep=None):
        thr, kappa = model.point_threshold, getattr(model, "point_nms_kappa", 0.0)
        model.point_threshold, model.point_nms_kappa = -1.0, 0.0
        try:
            result = orig(mask_cls, mask_pred, mask_box_result, point_coords, point_logits, point_keep=None)
        finally:
            model.point_threshold, model.point_nms_kappa = thr, kappa
        # 与 instance_inference 完全相同的 topk 调用；用类别序列逐个比对，确认顺序对齐
        _, topk_full = mask_cls.sigmoid().flatten(0, 1).topk(model.test_topk_per_image, sorted=False)
        labels = torch.arange(n_cls, device=mask_cls.device).repeat(mask_cls.shape[0])
        assert torch.equal(labels[topk_full].to(result.pred_classes.device), result.pred_classes), \
            "topk 顺序与 instance_inference 不一致"
        q_idx = topk_full // n_cls
        result.pred_query_index = q_idx.to(result.pred_classes.device)
        if point_logits is not None:
            is_sp = [int(result.pred_classes[k]) == model.spikelet_class_id for k in range(len(result))]
            pl = point_logits[q_idx].sigmoid().squeeze(-1)   # [N_inst, P]
            result.pred_spikelet_scores = [pl[k] if is_sp[k] else pl.new_zeros((0,)) for k in range(len(result))]
            if point_keep is not None:
                pk = point_keep[q_idx].bool()                 # [N_inst, P]
                result.pred_spikelet_keep = [pk[k] if is_sp[k] else pk.new_zeros((0,)) for k in range(len(result))]
        return result

    model.instance_inference = wrapped


def _capture_decoder_outputs(model):
    """forward hook 截分组 decoder 第 1~9 层小穗 token 的输出（aux_outputs[1..] 与最终输出），
    以及 padded 输入尺寸（backbone 的输入 images.tensor），供 flip 按 criterion 的规则逐层重做内层匹配。"""
    cap = {}

    def backbone_pre_hook(module, args):
        cap["pad_hw"] = tuple(int(v) for v in args[0].shape[-2:])

    def head_hook(module, inputs, output):
        out = output[0]
        layers = list(out["aux_outputs"][1:]) + [out]
        cap["abs"] = [bool(l.get("pred_point_abs", False)) for l in layers]
        cap["coords"] = [l["pred_point_coords"] for l in layers]   # 每层 [B, Q, K, 2]，图像归一化（padded 尺度）
        cap["logits"] = [l["pred_point_logits"] for l in layers]   # 每层 [B, Q, K, 1]
        cap["keep"] = [l.get("pred_point_keep") for l in layers]   # 每层 [B, Q, K] bool 或 None

    model.backbone.register_forward_pre_hook(backbone_pre_hook)
    model.sem_seg_head.register_forward_hook(head_hook)
    return cap


def _criterion_assign(crit, pred_c, pred_lg, gt_pts, box_cxcywh, image_wh, keep=None, kappa=None):
    """与 SetCriterion.loss_points 对一个 (实例, 层) 的做法逐行一致：GT 框相对坐标、iso_scale、匹配前 NMS、
    只让留下的 token 进 _inner_point_match（robust=True），下标映射回 K 个。
    Returns: (assign [Ni]：每个 GT 小穗配到的 token 下标、没配上为 -1, keep 或 None)"""
    Ni = gt_pts.shape[0]
    cx, cy, bw, bh = box_cxcywh
    x0 = cx - bw / 2
    y0 = cy - bh / 2
    wh = torch.stack([bw, bh]).clamp(min=1e-6)
    pred_rel = crit._pred_to_box_rel(pred_c, torch.stack([x0, y0]), wh, True)
    if Ni > 0:
        gt_rel = ((gt_pts - torch.stack([x0, y0])) / wh).clamp(0.0, 1.0)
    else:
        gt_rel = gt_pts.new_zeros((0, 2))
    iso_scale = crit._point_iso_scale(wh, image_wh)
    if keep is None and kappa is not None:
        keep = crit._point_nms_keep(pred_rel, pred_lg, wh, image_wh, kappa=kappa)
    if keep is None:
        row, col = crit._inner_point_match(pred_rel, pred_lg, gt_rel, iso_scale, robust=True)
    else:
        kidx = keep.cpu().nonzero(as_tuple=True)[0].to(pred_rel.device)
        row, col = crit._inner_point_match(pred_rel[kidx], pred_lg[kidx], gt_rel, iso_scale, robust=True)
        row = kidx[row]
    assign = np.full(Ni, -1, dtype=np.int64)
    assign[col.cpu().numpy()] = row.cpu().numpy()
    return assign, keep


def _infer_nms_keep(xy_all, score, box_xyxy, kappa):
    """推理侧 NMS，与 MaskDINO.instance_inference 同一套写法（同设备、同表达式），留下的点逐个一致。"""
    if kappa <= 0:
        return torch.ones(score.shape[0], dtype=torch.bool, device=score.device)
    x0, y0, x1, y1 = box_xyxy
    bw = (x1 - x0).clamp(min=0)
    bh = (y1 - y0).clamp(min=0)
    radius = kappa * torch.sqrt(bw.clamp(min=1e-6) * bh.clamp(min=1e-6))
    return greedy_point_nms(xy_all, score, radius)


def _pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


class _Readout:
    """一种读数规则下配对穗上的累计：matched_F1@σ（与评估器同式）与每穗点数误差。"""

    def __init__(self):
        self.tp = np.zeros(len(_SIGMAS))
        self.n_pred = self.n_gt = 0
        self.pred_cnt, self.gt_cnt = [], []

    def add(self, P, G):
        _, _, d = _match_points(P, G)
        for k, sg in enumerate(_SIGMAS):
            self.tp[k] += int((d <= sg).sum())
        self.n_pred += len(P)
        self.n_gt += len(G)
        self.pred_cnt.append(len(P))
        self.gt_cnt.append(len(G))

    def summary(self):
        res = {}
        for k, sg in enumerate(_SIGMAS):
            pr = self.tp[k] / self.n_pred if self.n_pred else 0.0
            rc = self.tp[k] / self.n_gt if self.n_gt else 0.0
            res[f"matched_F1@{sg:g}"] = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        e = np.asarray(self.pred_cnt, dtype=np.float64) - np.asarray(self.gt_cnt, dtype=np.float64)
        nan = float("nan")
        res["count_MAE_spike"] = float(np.mean(np.abs(e))) if e.size else nan
        res["count_bias"] = float(e.mean()) if e.size else nan
        res["count_pearson_r"] = _pearson(self.pred_cnt, self.gt_cnt)
        res["count_MAE_debiased"] = float(np.mean(np.abs(e - e.mean()))) if e.size else nan
        res["n_pred_points"] = int(self.n_pred)
        res["n_gt_points"] = int(self.n_gt)
        res["n_spikes"] = int(e.size)
        return res


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
                   help="离线扫描的 POINT_THRESHOLD 档位（默认 0.10~0.60 步长 0.05）；不带值 = 不扫，须放在最前面")
    p.add_argument("--match-nms-kappa", type=float, default=None,
                   help="flip 的匹配前 NMS 半径 κ（iso 空间，用 GT 框）。默认按配置：POINT_NMS_MODE != none 时取 "
                        "POINT_NMS_KAPPA，否则不做；<= 0 = 不做。截到 pred_point_keep 时优先用它")
    p.add_argument("--nms-kappa", type=float, default=None,
                   help="推理侧 NMS 半径 κ（× 预测框 sqrt(w·h) 像素），用于 auc_surv / coverage / readout_model。"
                        "默认取配置的 TEST.POINT_NMS_KAPPA；<= 0 = 不做")
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
    cap = _capture_decoder_outputs(model)
    crit = model.criterion

    md = cfg.MODEL.MaskDINO
    if a.match_nms_kappa is not None:
        match_kappa = a.match_nms_kappa if a.match_nms_kappa > 0 else None
    else:
        match_kappa = md.POINT_NMS_KAPPA if md.POINT_NMS_MODE != "none" else None
    infer_kappa = float(md.TEST.POINT_NMS_KAPPA if a.nms_kappa is None else a.nms_kappa)

    ds = cfg.DATASETS.TEST[0]
    t = md.TEST
    score_thr, iou_thr, pt_thr = t.SPIKELET_EVAL_SCORE_THRESH, t.SPIKELET_EVAL_IOU_THRESH, t.POINT_THRESHOLD
    gt_all = SpikeletPointEvaluator(ds, sigmas=(a.sigma,), score_thresh=score_thr, iou_thresh=iou_thr,
                                    distributed=False)._gt
    loader = build_detection_test_loader(cfg, ds)

    auc_s, auc_l = [], []
    along, across, along_s, across_s = [], [], [], []
    n_pred_thr = n_dup = n_pairs = n_gt_spike = 0
    tp_m = n_pred_m = n_gt_m = 0
    pair_data = []   # 每个配对穗的 (全部点, 分数, GT 点)，供阈值扫描
    # 新指标的累计
    surv_s, surv_l, surv_l_allk = [], [], []
    cov_all = cov_kept = cov_gt = n_tok_all = n_tok_kept = 0
    ro_model, ro_nonms = _Readout(), _Readout()
    flip_ok = None          # 是否是分组 decoder（第 1~9 层都是绝对坐标的小穗 token）
    flip_diff = flip_gt = None
    lay_sup = lay_tok = lay_starved = None
    flip_inst = flip_inst_pos = 0
    keep_src = set()
    model_keep_seen = False
    with torch.no_grad():
        for it, inputs in enumerate(loader):
            outputs = model(inputs)
            if flip_ok is None:
                flip_ok = len(cap["abs"]) > 0 and all(cap["abs"])
                if flip_ok:
                    L = len(cap["abs"])
                    flip_diff, flip_gt = np.zeros(L - 1, dtype=np.int64), 0
                    lay_sup, lay_tok, lay_starved = (np.zeros(L, dtype=np.int64) for _ in range(3))
            for bi, (inp, out) in enumerate(zip(inputs, outputs)):
                gt = gt_all[inp["image_id"]]
                n_gt_spike += len(gt["points"])
                inst = out["instances"]
                classes = inst.pred_classes.cpu().numpy()
                scores = inst.scores.cpu().numpy()
                has_keep = inst.has("pred_spikelet_keep")
                model_keep_seen |= has_keep
                sel, pts_all, sc_all, rles = [], [], [], []
                for i in range(len(inst)):
                    if int(classes[i]) != model.spikelet_class_id or scores[i] < score_thr:
                        continue
                    sel.append(i)
                    pts_all.append(inst.pred_spikelet_points[i].float().cpu().numpy().reshape(-1, 2))
                    sc_all.append(inst.pred_spikelet_scores[i].float().cpu().numpy().reshape(-1))
                    m = inst.pred_masks[i].cpu().numpy().astype(np.uint8)
                    rles.append(mask_util.encode(np.asfortranarray(m)))
                if not rles or not gt["rles"]:
                    continue
                iou = np.asarray(mask_util.iou(rles, gt["rles"], [0] * len(gt["rles"]))).reshape(len(rles), -1)
                ri, ci = linear_sum_assignment(-iou)

                # 截到的 decoder 输出：与 MaskDINO.forward 推理分支同一套缩放（padded 归一化 → 原图像素），
                # 再除以原图 (W, H) 换到训练用的图像归一化系
                h_in, w_in = inp["image"].shape[-2:]
                height, width = inp.get("height", h_in), inp.get("width", w_in)
                H_pad, W_pad = cap["pad_hw"]
                height_s = H_pad / h_in * height
                width_s = W_pad / w_in * width

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

                    # ---- 新指标：最终层的留下 token（存活 ∧ 推理 NMS），与 instance_inference 同设备同写法 ----
                    i_inst = sel[pi]
                    xy_t = inst.pred_spikelet_points[i_inst]
                    s_t = inst.pred_spikelet_scores[i_inst]
                    surv_t = _infer_nms_keep(xy_t, s_t, inst.pred_boxes.tensor[i_inst], infer_kappa)
                    if has_keep:
                        surv_t = surv_t & inst.pred_spikelet_keep[i_inst].to(surv_t.device)
                    surv = surv_t.cpu().numpy()
                    P_s, s_s = P_all[surv], s[surv]
                    r_v, _, d_v = _match_points(P_s, G)
                    lab_v = np.zeros(len(P_s), dtype=bool)
                    lab_v[r_v[d_v <= a.sigma]] = True
                    surv_s.extend(s_s.tolist())
                    surv_l.extend(lab_v.tolist())
                    surv_l_allk.extend(lab[surv].tolist())
                    cov_all += int((d_a <= a.sigma).sum())
                    cov_kept += int((d_v <= a.sigma).sum())
                    cov_gt += len(G)
                    n_tok_all += len(P_all)
                    n_tok_kept += len(P_s)
                    ro_model.add(P_all[surv & (s > pt_thr)], G)
                    ro_nonms.add(P_all[s > pt_thr], G)

                    # ---- flip：第 1~9 层按 criterion 的规则逐层配对 ----
                    if not flip_ok:
                        continue
                    q = int(inst.pred_query_index[i_inst])
                    dev = cap["coords"][-1].device
                    x, y, bw_g, bh_g = mask_util.toBbox(gt["rles"][gi]).tolist()   # GT mask 外接框（像素 xywh）
                    box = torch.tensor([(x + bw_g / 2) / width, (y + bh_g / 2) / height,
                                        bw_g / width, bh_g / height], dtype=torch.float32, device=dev)
                    image_wh = torch.tensor([width, height], dtype=torch.float32, device=dev)
                    gt_n = torch.as_tensor(G / np.array([width, height]), dtype=torch.float32, device=dev)
                    norm = torch.tensor([width_s / width, height_s / height], dtype=torch.float32, device=dev)
                    assigns = []
                    for li in range(len(cap["coords"])):
                        pc = cap["coords"][li][bi, q].float() * norm
                        plg = cap["logits"][li][bi, q].float()
                        kl = cap["keep"][li]
                        keep_l = kl[bi, q].bool() if kl is not None else None
                        keep_src.add("pred_point_keep" if keep_l is not None
                                     else ("criterion_nms" if match_kappa is not None else "none"))
                        asg, keep_used = _criterion_assign(crit, pc, plg, gt_n, box, image_wh,
                                                           keep=keep_l, kappa=match_kappa)
                        assigns.append(asg)
                        if keep_used is not None:
                            nk = int(keep_used.sum())
                            lay_sup[li] += int(keep_used.numel()) - nk
                            lay_tok[li] += int(keep_used.numel())
                            lay_starved[li] += int(nk < len(G))
                    A = np.stack(assigns)                       # [L, Ni]
                    flip_diff += (A[1:] != A[:-1]).sum(1)
                    flip_gt += len(G)
                    flip_inst += 1
                    flip_inst_pos += int(len(G) > 0)
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

    # ---- 新指标 ----
    res["nms"] = {
        "infer_nms_kappa": infer_kappa,
        "match_nms_kappa": match_kappa,
        "match_keep_source": sorted(keep_src),
        "model_point_keep": bool(model_keep_seen),
        "config_POINT_NMS_MODE": md.POINT_NMS_MODE,
        "config_POINT_NMS_KAPPA": md.POINT_NMS_KAPPA,
        "config_TEST_POINT_NMS_KAPPA": md.TEST.POINT_NMS_KAPPA,
    }
    res["auc_surv"] = {"value": _auc(surv_s, surv_l), "n_tokens": len(surv_l),
                       "pos_rate": float(np.mean(surv_l)) if surv_l else float("nan"),
                       "value_labels_from_all_k": _auc(surv_s, surv_l_allk),
                       "kept_token_frac": n_tok_kept / n_tok_all if n_tok_all else float("nan")}
    c_all = cov_all / cov_gt if cov_gt else float("nan")
    c_kept = cov_kept / cov_gt if cov_gt else float("nan")
    res["coverage"] = {"sigma": a.sigma, "all_k": c_all, "kept": c_kept,
                       "loss_pct_points": 100.0 * (c_all - c_kept), "n_gt_points": cov_gt}
    res["readout_model"] = ro_model.summary()
    res["readout_no_nms"] = ro_nonms.summary()
    if flip_ok:
        rates = (flip_diff / flip_gt).tolist() if flip_gt else [float("nan")] * len(flip_diff)
        flip = {"pairs": {f"{l + 1}->{l + 2}": r for l, r in enumerate(rates)},
                "mean": float(np.mean(rates)), "n_gt_points": int(flip_gt), "n_spikes": int(flip_inst)}
        if lay_tok.sum() > 0:
            flip["per_layer"] = [
                {"layer": l + 1,
                 "suppressed_frac": float(lay_sup[l] / lay_tok[l]) if lay_tok[l] else float("nan"),
                 "starved_frac": float(lay_starved[l] / flip_inst_pos) if flip_inst_pos else float("nan")}
                for l in range(len(lay_tok))]
        res["flip"] = flip
    else:
        res["flip"] = None

    # 阈值扫描：与训练期评估同口径（配对穗内、score > 阈值、点级匈牙利、d <= σ 记 TP）
    sigmas = _SIGMAS
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
