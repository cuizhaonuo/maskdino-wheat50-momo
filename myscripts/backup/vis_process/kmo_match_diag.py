"""KMO mechanism diagnostics (M1 / M2 of experiments-bank/ideas/kmo-matcher.md).

Loads a POINT_KMO=True checkpoint, runs training-mode forward passes (no grad) on the train loader,
and at the FINAL decoder layer's inner matchings compares, on the same predictions:
  - the KMO assignment (what training used)  vs  the L1-only (Eq.2) assignment.
Reports
  M1  mean|ŷ^k - y^k| over L1-only-matched pairs (unbiased by KMO's own pair selection), vs MAE_inst;
      plus the within-instance std of ŷ^k vs y^k over the matched slots.
  M2  override rate: KMO assignment's Eq.2 cost > Eq.2 optimum + 1e-5 (KMO traded L1 for context);
      tie-swap rate: same Eq.2 cost, different assignment (KMO only picked another exact L1 optimum).
  post-hoc (added after the first run, NOT pre-registered): splits the M1 error into
      - level: |mean_inst(ŷ^k) - mean_inst(y^k)| per instance
      - within-instance Pearson r of (ŷ^k - mean_inst ŷ^k) vs (y^k - mean_inst y^k), pooled
      - oracle_inst_same_sample: mean|y^k - median_inst(y^k)| on the very same matched pairs

2026-10-20 扩展（mini/kmo-dir），上面的字段名与 dist 时的算法都不变，新字段追加在 JSON 末尾：
  - 支持 POINT_KMO_DESC="dir"。描述子形式 / K / s 从 cfg 读（POINT_KMO_DESC / POINT_KMO_K /
    POINT_KMO_DIR_SCALE）。M2 的推翻率 / 换 tie 率与 Eq.2 代价函数对两种描述子通用。
    dir 时原有字段用不乘 s 的原始两分量：M1 / err_under_KMO_match = 两分量 L1 之和的逐点均值；
    std = 两分量 std 的均值；level = 均值差向量的 L1；Pearson r 逐分量去实例均值后合并；
    oracle = 到逐分量中位数的 L1。
  - 「头 vs 由预测点推算」对照（新机制门的依据）：最后一层、L1-only 配对 (r_l, c_l)、只看 Ni>=3 的实例，
      e_head  = 每点平均 ‖ŷ[r_l] - y[c_l]‖₁
      e_proxy = 用这些配上槽自己的预测点 pred_rel[r_l]，按同一描述子、同一 K 算出 y_proxy
                （与 r_l 同序），每点平均 ‖y_proxy - y[c_l]‖₁
      e_const = 常数预测：dir 用 0 向量；dist 用该实例 GT y^k 的中位数
                （知道每穗真值的常数，偏乐观，只作参照）
    dir 时三者都是两分量乘 s 后的 L1，dist 不乘。报告点加权均值（e_head / e_proxy / e_const）、
    逐实例均值（*_inst_mean）、e_head < e_proxy 的实例比例；dir 另报 ĉ 与 c 的平均余弦
    （只统计 ‖c‖₂ >= 0.3 的点），并给出同一批点上 y_proxy 与 c 的余弦作对照。

Run from the repo root:
  PYTHONPATH=. python myscripts/backup/vis_process/kmo_match_diag.py --config <run>/config.yaml --weights <run>/model_XXXX.pth --batches 100 --out <json>
"""
import argparse
import json

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.events import EventStorage

from maskdino import add_maskdino_config
from maskdino.modeling.criterion import SetCriterion, knn_descriptor
from train_net import Trainer  # registers datasets as a side effect

COS_MIN_NORM = 0.3   # dir 的余弦只统计 ‖c‖₂ >= 该值的点（模长太小时方向无意义）


def eq2_cost(crit, pred_rel, pred_logit, gt_rel, rows, cols):
    alpha, gamma = crit.point_focal_alpha, 2.0
    prob = pred_logit.double().sigmoid()
    neg = (1 - alpha) * prob ** gamma * (-(1 - prob + 1e-8).log())
    pos = alpha * (1 - prob) ** gamma * (-(prob + 1e-8).log())
    C = crit.point_cost_coord * torch.cdist(pred_rel.double(), gt_rel.double(), p=1) \
        + crit.point_cost_class * (pos - neg)
    return C[rows, cols].sum().item()


def _cos(a, b, b_norm):
    """逐行余弦；a 为零向量时记 0。"""
    return (a * b).sum(1) / (a.norm(dim=1).clamp(min=1e-12) * b_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.SEED = args.seed
    cfg.freeze()
    assert cfg.MODEL.MaskDINO.POINT_KMO, "diagnostic needs a POINT_KMO=True run"
    # 描述子形式 / K / s 一律从 cfg 读；旧 run 的 config.yaml 没有 DESC 等键，取默认值 "dist"
    desc = cfg.MODEL.MaskDINO.POINT_KMO_DESC
    k_nn = cfg.MODEL.MaskDINO.POINT_KMO_K
    dir_scale = cfg.MODEL.MaskDINO.POINT_KMO_DIR_SCALE
    err_scale = dir_scale if desc == "dir" else 1.0   # e_head / e_proxy / e_const 的尺度：dir 乘 s，dist 不乘
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.train()
    crit = model.criterion
    assert crit.point_kmo_desc == desc and crit.point_kmo_k == k_nn

    stats = dict(inst=0, override=0, tie_swap=0, pts=0, err_l1=0.0, err_kmo=0.0,
                 std_pred=[], std_gt=[], cost_gap=[],
                 level=[], sxy=0.0, sxx=0.0, syy=0.0, oracle=0.0)
    # 「头 vs 由预测点推算」（Ni>=3）
    hp = dict(inst=0, pts=0, head=0.0, proxy=0.0, const=0.0, head_lt_proxy=0,
              inst_head=[], inst_proxy=[], inst_const=[],
              cos_pts=0, cos_head=0.0, cos_proxy=0.0)
    orig_loss_points = SetCriterion.loss_points
    orig_match = SetCriterion._inner_point_match
    state = {"top": False}

    def loss_points(self, outputs, *a, **k):
        # only the final decoder layer's call carries 'aux_outputs'; aux / interm dicts do not
        state["top"] = "aux_outputs" in outputs
        try:
            return orig_loss_points(self, outputs, *a, **k)
        finally:
            state["top"] = False

    def head_vs_proxy(pred_rel, gk, pk, r_l, c_l):
        g = gk[c_l].double()                                                 # [n, D] 配上的 GT y^k
        h = pk[r_l].double()                                                 # [n, D] 头的 ŷ^k
        y_proxy = knn_descriptor(pred_rel[r_l].float(), k_nn, desc).double()  # [n, D] 由配上槽的预测点推算
        if desc == "dir":
            c0 = torch.zeros_like(g)                                         # 常数：0 向量
        else:
            c0 = g.median().expand_as(g)                                     # 常数：该实例 GT y^k 的中位数（偏乐观）
        e_h = err_scale * (h - g).abs().sum(1)                               # [n] 逐点 L1
        e_p = err_scale * (y_proxy - g).abs().sum(1)
        e_c = err_scale * (c0 - g).abs().sum(1)
        n = int(g.shape[0])
        hp["inst"] += 1
        hp["pts"] += n
        hp["head"] += e_h.sum().item()
        hp["proxy"] += e_p.sum().item()
        hp["const"] += e_c.sum().item()
        hp["inst_head"].append(e_h.mean().item())
        hp["inst_proxy"].append(e_p.mean().item())
        hp["inst_const"].append(e_c.mean().item())
        hp["head_lt_proxy"] += int(hp["inst_head"][-1] < hp["inst_proxy"][-1])
        if desc == "dir":
            g_norm = g.norm(dim=1)
            m = g_norm >= COS_MIN_NORM
            if bool(m.any()):
                hp["cos_pts"] += int(m.sum())
                hp["cos_head"] += _cos(h[m], g[m], g_norm[m]).sum().item()
                hp["cos_proxy"] += _cos(y_proxy[m], g[m], g_norm[m]).sum().item()

    def inner_match(self, pred_rel, pred_logit, gt_rel, pred_knn=None, gt_knn=None):
        r_k, c_k = orig_match(self, pred_rel, pred_logit, gt_rel, pred_knn, gt_knn)
        if state["top"] and gt_rel.shape[0] >= 2:
            self.point_kmo = False
            r_l, c_l = orig_match(self, pred_rel, pred_logit, gt_rel)
            self.point_kmo = True
            gk = gt_knn if gt_knn is not None else knn_descriptor(gt_rel.float(), self.point_kmo_k, desc)
            pk = pred_knn.float()
            ck = eq2_cost(self, pred_rel, pred_logit, gt_rel, r_k, c_k)
            cl = eq2_cost(self, pred_rel, pred_logit, gt_rel, r_l, c_l)
            over = ck > cl + 1e-5
            stats["inst"] += 1
            stats["override"] += int(over)
            stats["tie_swap"] += int((not over) and not (torch.equal(r_k, r_l) and torch.equal(c_k, c_l)))
            stats["cost_gap"].append(ck - cl)
            n = int(r_l.numel())
            stats["pts"] += n
            stats["err_l1"] += (pk[r_l] - gk[c_l]).abs().sum().item()
            stats["err_kmo"] += (pk[r_k] - gk[c_k]).abs().sum().item()
            if pk.shape[-1] == 1:
                # dist：与扩展前的脚本同一套运算
                stats["std_pred"].append(pk[r_l].std().item())
                stats["std_gt"].append(gk[c_l].std().item())
                # post-hoc decomposition (not pre-registered)
                p_m, g_m = pk[r_l].double().flatten(), gk[c_l].double().flatten()
                stats["level"].append(abs(p_m.mean().item() - g_m.mean().item()))
                pc, gc = p_m - p_m.mean(), g_m - g_m.mean()
                stats["oracle"] += (g_m - g_m.median()).abs().sum().item()
            else:
                # dir：逐分量算（原始两分量，不乘 s），Pearson r 逐分量去实例均值后合并
                p_m, g_m = pk[r_l].double(), gk[c_l].double()                # [n, 2]
                stats["std_pred"].append(p_m.std(0).mean().item())
                stats["std_gt"].append(g_m.std(0).mean().item())
                stats["level"].append((p_m.mean(0) - g_m.mean(0)).abs().sum().item())
                pc, gc = p_m - p_m.mean(0), g_m - g_m.mean(0)
                stats["oracle"] += (g_m - g_m.median(0).values).abs().sum().item()
            stats["sxy"] += (pc * gc).sum().item()
            stats["sxx"] += (pc * pc).sum().item()
            stats["syy"] += (gc * gc).sum().item()
            if gt_rel.shape[0] >= 3:
                head_vs_proxy(pred_rel, gk, pk, r_l, c_l)
        return r_k, c_k

    SetCriterion.loss_points = loss_points
    SetCriterion._inner_point_match = inner_match

    loader = iter(Trainer.build_train_loader(cfg))
    with EventStorage(0), torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(args.batches):
            model(next(loader))

    n_i, n_p = max(stats["inst"], 1), max(stats["pts"], 1)
    res = {
        "weights": args.weights, "batches": args.batches, "instances": stats["inst"], "points": stats["pts"],
        "M1_err_under_L1_match": stats["err_l1"] / n_p,
        "err_under_KMO_match": stats["err_kmo"] / n_p,
        "within_inst_std_pred_knn": float(np.mean(stats["std_pred"])) if stats["std_pred"] else None,
        "within_inst_std_gt_knn": float(np.mean(stats["std_gt"])) if stats["std_gt"] else None,
        "M2_override_rate": stats["override"] / n_i,
        "M2_tie_swap_rate": stats["tie_swap"] / n_i,
        "median_eq2_cost_gap_when_override": float(np.median([g for g in stats["cost_gap"] if g > 1e-5]))
        if stats["override"] else None,
        "posthoc_level_bias_mean": float(np.mean(stats["level"])) if stats["level"] else None,
        "posthoc_within_inst_pearson_r": stats["sxy"] / max((stats["sxx"] * stats["syy"]) ** 0.5, 1e-12),
        "posthoc_oracle_inst_same_sample": stats["oracle"] / n_p,
    }
    # ---- 以下为 2026-10-20 扩展的字段 ----
    hp_n_p = max(hp["pts"], 1)
    res.update({
        "kmo_desc": desc,
        "kmo_k": k_nn,
        "kmo_dir_scale": dir_scale if desc == "dir" else None,
        "hp_err_scale": err_scale,
        "hp_scope": "最后一层、L1-only 配对 (r_l, c_l)、Ni>=3 的实例；误差为逐点 L1"
                    + ("（两分量乘 s）" if desc == "dir" else "（不乘系数）"),
        "hp_instances": hp["inst"],
        "hp_points": hp["pts"],
        "e_head": hp["head"] / hp_n_p,
        "e_proxy": hp["proxy"] / hp_n_p,
        "e_const": hp["const"] / hp_n_p,
        "e_const_kind": ("dir：0 向量" if desc == "dir"
                         else "dist：该实例 GT y^k 的中位数（知道每穗真值的常数，偏乐观，只作参照）"),
        "e_head_inst_mean": float(np.mean(hp["inst_head"])) if hp["inst_head"] else None,
        "e_proxy_inst_mean": float(np.mean(hp["inst_proxy"])) if hp["inst_proxy"] else None,
        "e_const_inst_mean": float(np.mean(hp["inst_const"])) if hp["inst_const"] else None,
        "frac_inst_e_head_lt_e_proxy": hp["head_lt_proxy"] / max(hp["inst"], 1),
    })
    if desc == "dir":
        res.update({
            "dir_cos_min_norm": COS_MIN_NORM,
            "dir_cos_points": hp["cos_pts"],
            "dir_cos_points_frac": hp["cos_pts"] / hp_n_p,
            "dir_cos_head": hp["cos_head"] / hp["cos_pts"] if hp["cos_pts"] else None,
            "dir_cos_proxy": hp["cos_proxy"] / hp["cos_pts"] if hp["cos_pts"] else None,
        })
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
