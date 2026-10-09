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
from maskdino.modeling.criterion import SetCriterion, knn_mean_distance
from train_net import Trainer  # registers datasets as a side effect


def eq2_cost(crit, pred_rel, pred_logit, gt_rel, rows, cols):
    alpha, gamma = crit.point_focal_alpha, 2.0
    prob = pred_logit.double().sigmoid()
    neg = (1 - alpha) * prob ** gamma * (-(1 - prob + 1e-8).log())
    pos = alpha * (1 - prob) ** gamma * (-(prob + 1e-8).log())
    C = crit.point_cost_coord * torch.cdist(pred_rel.double(), gt_rel.double(), p=1) \
        + crit.point_cost_class * (pos - neg)
    return C[rows, cols].sum().item()


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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.train()
    crit = model.criterion

    stats = dict(inst=0, override=0, tie_swap=0, pts=0, err_l1=0.0, err_kmo=0.0,
                 std_pred=[], std_gt=[], cost_gap=[],
                 level=[], sxy=0.0, sxx=0.0, syy=0.0, oracle=0.0)
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

    def inner_match(self, pred_rel, pred_logit, gt_rel, pred_knn=None, gt_knn=None):
        r_k, c_k = orig_match(self, pred_rel, pred_logit, gt_rel, pred_knn, gt_knn)
        if state["top"] and gt_rel.shape[0] >= 2:
            self.point_kmo = False
            r_l, c_l = orig_match(self, pred_rel, pred_logit, gt_rel)
            self.point_kmo = True
            gk = gt_knn if gt_knn is not None else knn_mean_distance(gt_rel.float(), self.point_kmo_k)
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
            stats["std_pred"].append(pk[r_l].std().item())
            stats["std_gt"].append(gk[c_l].std().item())
            # post-hoc decomposition (not pre-registered)
            p_m, g_m = pk[r_l].double().flatten(), gk[c_l].double().flatten()
            stats["level"].append(abs(p_m.mean().item() - g_m.mean().item()))
            pc, gc = p_m - p_m.mean(), g_m - g_m.mean()
            stats["sxy"] += (pc * gc).sum().item()
            stats["sxx"] += (pc * pc).sum().item()
            stats["syy"] += (gc * gc).sum().item()
            stats["oracle"] += (g_m - g_m.median()).abs().sum().item()
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
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
