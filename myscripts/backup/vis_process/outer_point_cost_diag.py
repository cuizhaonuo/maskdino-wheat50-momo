"""外层匹配加小穗点项：零训练诊断（mini/outer-point-cost 第 0 步，用来决定要不要跑训练）。

外层匈牙利（matcher.py）的代价只有 类别 + 框 + mask，点损失是唯一没有对应匹配代价的实例级损失。
本脚本在训练好的 checkpoint 上量两件事，不训练、不改模型：

A. 改判率（train 集，训练态前向 + LSJ 增广，与训练时的外层匹配同口径）
   对每个 batch 的 11 次外层匹配（最后一层、aux _0.._8、interm），在同一份预测、
   同一组 12544 个 mask 采样点上分别求解：
     A    = LSAP(C_outer)                  现状（已逐图核对与 HungarianMatcher 的分配一致）
     B_λ  = LSAP(C_outer + λ·C_pt)         加小穗点项，λ ∈ {0.5, 1, 2}
     R    = LSAP(C_outer')                 只重抽 mask 采样点：外层匹配自身的抖动，作参照
     T    = B_1 的 top-k 预筛版             每个 GT 只给 C_outer 最小的 k 个 query 算点代价，
                                            其余取这 k 个里的最大值（检验训练时能否这样省算力）
   报告带小穗 GT（Ni>0）里，分到的 query 与 A 不同的比例。
   C_pt 照点损失原样写（代价 = 损失）：对 (q, g) 先做与 _inner_point_match 相同的内层匈牙利，
     C_pt(q,g) = (5·Σ_配上 ‖p̂−p‖₁ + 2·w_g·Σ_25槽 FL(s, t)) · N / N_pts
   5、2 取 weight_dict 里 loss_point_coord / loss_point_cls 的权重；N/N_pts 把点损失的 ÷N_pts
   换成框/mask 损失的 ÷N，与 C_outer 其余各项同一量纲。Ni=0（wheat）没有内层匹配：t 全 0、w=0.1。

B. 选择余量（val 集，评估态前向，与内置 SpikeletPointEvaluator 同口径）
   used   : 评估器实际采用的配对（top-100 → wheat_2 且 score≥0.4 → mask IoU 匈牙利，IoU≥0.5）
   oracle : 对每个 used 配上的 GT 穗，在全部 300 个 query 里取 mask IoU≥0.5、本穗 F1@8 最高的那个
   两者按评估器口径汇总 matched_F1@σ。used 的预测另交给真的 SpikeletPointEvaluator.evaluate() 复算，
   应与训练日志里同一 checkpoint 的内置评估一致。oracle 是在评价指标上直接挑最大值，带挑选偏差，只当上界。

从 worktree 根目录运行：
  PYTHONPATH=. python myscripts/backup/vis_process/outer_point_cost_diag.py \
      --config <run>/config.yaml --weights <run>/model_final.pth --batches 125 --out <json>
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.projects.point_rend.point_features import point_sample
from detectron2.structures import ImageList
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import EventStorage

from maskdino import add_maskdino_config
from maskdino.evaluation.spikelet_evaluation import SpikeletPointEvaluator, _match_points
from maskdino.modeling.matcher import batch_dice_loss, batch_sigmoid_ce_loss
from maskdino.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from train_net import Trainer  # 导入即注册数据集

LAMBDAS = (0.5, 1.0, 2.0)


def lsap_assign(C, G):
    """C [Q, G] -> 每个 GT 列分到的 query 下标 [G]。"""
    rows, cols = linear_sum_assignment(C)
    a = np.full(G, -1, dtype=np.int64)
    a[cols] = rows
    return a


def outer_cost(matcher, out, tgt, b, coords):
    """逐行照抄 HungarianMatcher.memory_efficient_forward（含 dtype 处理），返回 [Q, G]。
    coords: [1, K, 2] mask 采样点由调用方给出，便于在同一组点上比较不同代价。"""
    out_bbox = out["pred_boxes"][b]
    tgt_bbox = tgt["boxes"]
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    out_prob = out["pred_logits"][b].sigmoid()
    tgt_ids = tgt["labels"]
    alpha, gamma = 0.25, 2.0
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
    out_mask = out["pred_masks"][b][:, None]
    tgt_mask = tgt["masks"].to(out_mask)[:, None]
    tgt_mask = point_sample(tgt_mask, coords.repeat(tgt_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
    out_mask = point_sample(out_mask, coords.repeat(out_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
    with torch.amp.autocast("cuda", enabled=False):
        out_mask, tgt_mask = out_mask.float(), tgt_mask.float()
        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
        cost_dice = batch_dice_loss(out_mask, tgt_mask)
    C = (matcher.cost_mask * cost_mask + matcher.cost_class * cost_class + matcher.cost_dice * cost_dice
         + matcher.cost_box * cost_bbox + matcher.cost_giou * cost_giou)
    return C.reshape(out_bbox.shape[0], -1).float().cpu().numpy()


def point_cost(crit, out, tgt, b, w_coord, w_cls):
    """C_pt 中未乘 N/N_pts 的部分 [Q, G]，以及每对 (q,g) 内层配对下每点的坐标 L1 [Q, G]（框相对）。"""
    pc = out["pred_point_coords"][b].float()               # [Q, P, 2]
    pl = out["pred_point_logits"][b].float().squeeze(-1)    # [Q, P]
    Q = pl.shape[0]
    a, gm = crit.point_focal_alpha, 2.0
    s = pl.sigmoid()
    # 损失侧的 focal，与 loss_points 同一写法：t=1 时 p_t=s，t=0 时 p_t=1-s
    fl_pos = a * F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl), reduction="none") * (1 - s) ** gm
    fl_neg = (1 - a) * F.binary_cross_entropy_with_logits(pl, torch.zeros_like(pl), reduction="none") * s ** gm
    # 内层匹配侧的存在性代价，与 _inner_point_match 同一写法（每个槽一个行偏置）
    c_row = a * ((1 - s) ** gm) * (-(s + 1e-8).log()) - (1 - a) * (s ** gm) * (-(1 - s + 1e-8).log())
    fl_pos, fl_neg = fl_pos.cpu().numpy(), fl_neg.cpu().numpy()
    neg_sum = fl_neg.sum(1)
    boxes, pts = tgt["boxes"].float(), tgt["points"]
    G = boxes.shape[0]
    cpt = np.zeros((Q, G))
    l1pp = np.full((Q, G), np.nan)
    for g in range(G):
        gp = pts[g].float()
        Ni = gp.shape[0]
        if Ni == 0:
            cpt[:, g] = w_cls * crit.point_empty_cls_weight * neg_sum
            continue
        cx, cy, bw, bh = boxes[g]
        wh = torch.stack([bw, bh]).clamp(min=1e-6)
        gt_rel = ((gp - torch.stack([cx - bw / 2, cy - bh / 2])) / wh).clamp(0.0, 1.0)
        D = torch.cdist(pc, gt_rel[None].expand(Q, Ni, 2).contiguous(), p=1)    # [Q, P, Ni]
        Cin = (crit.point_cost_coord * D + crit.point_cost_class * c_row[:, :, None]).cpu().numpy()
        D = D.cpu().numpy()
        for q in range(Q):
            r, c = linear_sum_assignment(Cin[q])
            coord = D[q][r, c].sum()
            cls = neg_sum[q] + (fl_pos[q, r] - fl_neg[q, r]).sum()
            cpt[q, g] = w_coord * coord + w_cls * cls
            l1pp[q, g] = coord / len(r)
    return cpt, l1pp


def part_a(model, cfg, args):
    crit, matcher = model.criterion, model.criterion.matcher
    w_coord = crit.weight_dict["loss_point_coord"]
    w_cls = crit.weight_dict["loss_point_cls"]
    K, dev = matcher.num_points, model.device
    loader = iter(Trainer.build_train_loader(cfg))
    stats = defaultdict(lambda: defaultdict(float))
    detail = defaultdict(list)
    verify = dict(gt=0, same=0)
    order = []
    model.train()
    t0 = time.time()
    for it in range(args.batches):
        batch = next(loader)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            images = [(x["image"].to(dev) - model.pixel_mean) / model.pixel_std for x in batch]
            images = ImageList.from_tensors(images, model.size_divisibility)
            features = model.backbone(images.tensor)
            targets = model.prepare_targets([x["instances"].to(dev) for x in batch], images)
            outputs, _ = model.sem_seg_head(features, targets=targets)
            layers = [("final", outputs)] + [(f"aux{i}", o) for i, o in enumerate(outputs["aux_outputs"])]
            if "interm_outputs" in outputs:
                layers.append(("interm", outputs["interm_outputs"]))
            if not order:
                order = [n for n, _ in layers]

            if it < 3:
                # 核对：同一随机种子下，自己算的 C_outer 与 HungarianMatcher 给出同一个分配
                torch.manual_seed(1000 + it)
                mine = [lsap_assign(outer_cost(matcher, outputs, t, b, torch.rand(1, K, 2, device=dev)),
                                    len(t["labels"])) for b, t in enumerate(targets)]
                torch.manual_seed(1000 + it)
                ref = matcher({k: v for k, v in outputs.items() if k != "aux_outputs"}, targets)
                for a_mine, (i, j), t in zip(mine, ref, targets):
                    a_ref = np.full(len(t["labels"]), -1, dtype=np.int64)
                    a_ref[j.numpy()] = i.numpy()
                    verify["gt"] += len(a_ref)
                    verify["same"] += int((a_ref == a_mine).sum())

            n_gt = sum(len(t["labels"]) for t in targets)
            n_pts = sum(int(p.shape[0]) for t in targets for p in t["points"])
            scale = max(n_gt, 1) / max(n_pts, 1)        # 与 criterion 的 num_masks / denom 相同
            for name, out in layers:
                for b, tgt in enumerate(targets):
                    G = len(tgt["labels"])
                    if G == 0:
                        continue
                    co = outer_cost(matcher, out, tgt, b, torch.rand(1, K, 2, device=dev))
                    co2 = outer_cost(matcher, out, tgt, b, torch.rand(1, K, 2, device=dev))
                    with torch.amp.autocast("cuda", enabled=False):
                        cpt_raw, l1pp = point_cost(crit, out, tgt, b, w_coord, w_cls)
                    cpt = cpt_raw * scale
                    has = np.array([p.shape[0] > 0 for p in tgt["points"]])
                    A = lsap_assign(co, G)
                    R = lsap_assign(co2, G)
                    Bs = {lam: lsap_assign(co + lam * cpt, G) for lam in LAMBDAS}
                    k = min(args.topk, co.shape[0])
                    cand = np.zeros_like(co, dtype=bool)
                    cand[np.argsort(co, axis=0)[:k], np.arange(G)[None, :]] = True
                    cap = np.where(cand, cpt, -np.inf).max(0)
                    T = lsap_assign(co + np.where(cand, cpt, cap[None, :]), G)
                    st = stats[name]
                    for grp, sel in (("pts", has), ("empty", ~has)):
                        st[f"n_{grp}"] += int(sel.sum())
                        st[f"chg_resample_{grp}"] += int((R != A)[sel].sum())
                        for lam in LAMBDAS:
                            st[f"chg_l{lam:g}_{grp}"] += int((Bs[lam] != A)[sel].sum())
                        st[f"chg_l1_also_resample_{grp}"] += int(((Bs[1.0] != A) & (R != A))[sel].sum())
                        st[f"topk_disagree_{grp}"] += int((T != Bs[1.0])[sel].sum())
                    if name == "final":
                        gs = np.where(has)[0]
                        if gs.size:
                            srt = np.sort(co[:, gs], axis=0)
                            detail["outer_gap_2nd"].extend((srt[1] - srt[0]).tolist())
                            detail["cpt_matched"].extend(cpt[A[gs], gs].tolist())
                            cand_cpt = np.where(cand[:, gs], cpt[:, gs], np.nan)
                            detail["cpt_spread_topk"].extend(
                                (np.nanmax(cand_cpt, 0) - np.nanmin(cand_cpt, 0)).tolist())
                        B1 = Bs[1.0]
                        for g in np.where((B1 != A) & has)[0]:
                            qa, qb = A[g], B1[g]
                            detail["d_outer"].append(float(co[qb, g] - co[qa, g]))
                            detail["d_pt"].append(float(cpt[qb, g] - cpt[qa, g]))
                            detail["l1pp_old"].append(float(l1pp[qa, g]))
                            detail["l1pp_new"].append(float(l1pp[qb, g]))
                            ma, mb = out["pred_masks"][b][qa] > 0, out["pred_masks"][b][qb] > 0
                            detail["mask_iou_old_new"].append(
                                ((ma & mb).sum() / (ma | mb).sum().clamp(min=1)).item())
        if (it + 1) % 25 == 0:
            print(f"[A] {it + 1}/{args.batches} batches, {time.time() - t0:.0f}s", flush=True)

    def rate(st, key, grp):
        n = st[f"n_{grp}"]
        return st[f"{key}_{grp}"] / n if n else None

    keys = ["chg_resample"] + [f"chg_l{lam:g}" for lam in LAMBDAS] + ["chg_l1_also_resample", "topk_disagree"]
    per_layer = {}
    pooled = defaultdict(float)
    for name in order:
        st = stats[name]
        for k_, v in st.items():
            pooled[k_] += v
        per_layer[name] = {"n_gt_pts": st["n_pts"], "n_gt_empty": st["n_empty"],
                           **{f"{k_}_pts": rate(st, k_, "pts") for k_ in keys},
                           "chg_resample_empty": rate(st, "chg_resample", "empty"),
                           "chg_l1_empty": rate(st, "chg_l1", "empty")}
    q = lambda xs, p: float(np.percentile(xs, p)) if len(xs) else None
    final_detail = {k_: {"n": len(v), "p10": q(v, 10), "median": q(v, 50), "p90": q(v, 90),
                         "mean": float(np.mean(v)) if v else None} for k_, v in detail.items()}
    return {
        "batches": args.batches, "topk": args.topk, "seconds": round(time.time() - t0, 1),
        "verify_matcher_same_assignment": f"{verify['same']}/{verify['gt']}",
        "per_layer": per_layer,
        "all_layers_pooled": {f"{k_}_pts": rate(pooled, k_, "pts") for k_ in keys},
        "final_layer_detail": final_detail,
    }


def part_b(model, cfg, args):
    ds = cfg.DATASETS.TEST[0]
    tc = cfg.MODEL.MaskDINO.TEST
    ev = SpikeletPointEvaluator(ds, sigmas=tc.SPIKELET_EVAL_SIGMAS, score_thresh=tc.SPIKELET_EVAL_SCORE_THRESH,
                                iou_thresh=tc.SPIKELET_EVAL_IOU_THRESH, distributed=False)
    sig = ev._sigmas
    k8 = sig.index(8.0)
    loader = Trainer.build_test_loader(cfg, ds)
    model.eval()
    acc = {k: dict(tp=np.zeros(len(sig)), npred=0, ngt=0) for k in ("used", "oracle")}
    n_match = n_better = 0
    n_cand, gains, others_f1 = [], [], []
    ev_preds = []
    t0 = time.time()

    def spike_stats(xy, gp):
        _, _, d = _match_points(xy, gp)
        return np.array([(d <= s).sum() for s in sig]), len(xy), len(gp)

    def f1_at(tp, npred, ngt):
        return 2 * tp[k8] / (npred + ngt) if (npred + ngt) else 0.0

    for batch in loader:
        with torch.no_grad():
            images = [(x["image"].to(model.device) - model.pixel_mean) / model.pixel_std for x in batch]
            images = ImageList.from_tensors(images, model.size_divisibility)
            outputs, _ = model.sem_seg_head(model.backbone(images.tensor))
            mask_up = F.interpolate(outputs["pred_masks"], size=images.tensor.shape[-2:],
                                    mode="bilinear", align_corners=False)
            for i, inp in enumerate(batch):
                # ---- 照 MaskDINO.forward / instance_inference 复现推理后处理，但保留全部 300 个 query ----
                H, W = inp.get("height"), inp.get("width")
                image_size = images.image_sizes[i]
                mp = sem_seg_postprocess(mask_up[i], image_size, H, W)            # [Q, H, W] logits
                new_size = mask_up[i].shape[-2:]
                boxes = model.box_postprocess(outputs["pred_boxes"][i].to(mp),
                                              new_size[0] / image_size[0] * H, new_size[1] / image_size[1] * W)
                binm = mp > 0
                mscore = (mp.sigmoid().flatten(1) * binm.flatten(1)).sum(1) / (binm.flatten(1).sum(1) + 1e-6)
                cls = outputs["pred_logits"][i].sigmoid()
                C = cls.shape[1]
                sc, idx = cls.flatten(0, 1).topk(model.test_topk_per_image, sorted=False)
                q_of, lab = idx // C, idx % C
                final = sc if model.focus_on_box else sc * mscore[q_of]
                keep = (lab == model.spikelet_class_id) & (final >= ev._score_thresh)
                used_q = q_of[keep].cpu().numpy()
                used_s = final[keep].cpu().numpy()

                rel = outputs["pred_point_coords"][i]                                # [Q, P, 2]
                ex = outputs["pred_point_logits"][i].sigmoid().squeeze(-1) > model.point_threshold
                wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
                xy_all = (boxes[:, None, :2] + rel * wh[:, None, :]).cpu().numpy()
                ex = ex.cpu().numpy()
                xy = [xy_all[q][ex[q]].astype(np.float64) for q in range(xy_all.shape[0])]

                gt = ev._gt[inp["image_id"]]
                gpts = gt["points"]
                ev_preds.append({"image_id": inp["image_id"], "points": [xy[q] for q in used_q],
                                 "rles": [mask_util.encode(np.asfortranarray(binm[q].cpu().numpy().astype(np.uint8)))
                                          for q in used_q],
                                 "scores": used_s.tolist()})
                if not gpts or used_q.size == 0:
                    continue
                gm = torch.as_tensor(np.stack([mask_util.decode(r) for r in gt["rles"]]),
                                     device=binm.device).bool()
                inter = torch.stack([(binm & gm[g]).flatten(1).sum(1) for g in range(gm.shape[0])], 1).double()
                union = binm.flatten(1).sum(1)[:, None].double() + gm.flatten(1).sum(1)[None, :].double() - inter
                iou = (inter / union.clamp(min=1)).cpu().numpy()                    # [Q, G]
                ri, ci = linear_sum_assignment(-iou[used_q])
                for u, g in zip(ri, ci):
                    if iou[used_q[u], g] < ev._iou_thresh:
                        continue
                    n_match += 1
                    qu = used_q[u]
                    su = spike_stats(xy[qu], gpts[g])
                    best, best_f1 = su, f1_at(*su)
                    cands = np.where(iou[:, g] >= ev._iou_thresh)[0]
                    n_cand.append(len(cands))
                    oth = []
                    for qc in cands:
                        if qc == qu:
                            continue
                        sc_ = spike_stats(xy[qc], gpts[g])
                        f = f1_at(*sc_)
                        oth.append(f)
                        if f > best_f1:
                            best, best_f1 = sc_, f
                    if oth:
                        others_f1.append(float(np.mean(oth)))
                    gains.append(best_f1 - f1_at(*su))
                    n_better += int(best_f1 > f1_at(*su))
                    for key, s_ in (("used", su), ("oracle", best)):
                        acc[key]["tp"] += s_[0]
                        acc[key]["npred"] += s_[1]
                        acc[key]["ngt"] += s_[2]

    def prf(a, k):
        p = a["tp"][k] / a["npred"] if a["npred"] else float("nan")
        r = a["tp"][k] / a["ngt"] if a["ngt"] else float("nan")
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    ev._predictions = ev_preds
    official = ev.evaluate()["spikelet"]
    res = {"seconds": round(time.time() - t0, 1), "gt_spikes_matched": n_match,
           "official_evaluator_on_used": {k: official[k] for k in official
                                          if k.startswith("matched_F1") or k in ("F1@8", "count_MAE_spike",
                                                                                 "spike_recall", "spike_precision")}}
    for key in ("used", "oracle"):
        for k, s in enumerate(sig):
            res[f"{key}_matched_F1@{s:g}"] = prf(acc[key], k)
        res[f"{key}_pred_points"] = int(acc[key]["npred"])
    res["oracle_minus_used_matched_F1@8"] = res["oracle_matched_F1@8"] - res["used_matched_F1@8"]
    res["frac_spikes_oracle_strictly_better"] = n_better / n_match if n_match else None
    res["candidates_per_spike_mean"] = float(np.mean(n_cand)) if n_cand else None
    res["candidates_per_spike_hist"] = {str(v): int(c) for v, c in zip(*np.unique(n_cand, return_counts=True))}
    res["per_spike_F1@8_gain_mean"] = float(np.mean(gains)) if gains else None
    res["per_spike_F1@8_gain_p90"] = float(np.percentile(gains, 90)) if gains else None
    res["other_candidates_mean_F1@8"] = float(np.mean(others_f1)) if others_f1 else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batches", type=int, default=125, help="A 部分的 train batch 数（mini train 500 图 = 125）")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.SEED = args.seed
    cfg.freeze()
    seed_all_rng(args.seed)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    res = {"weights": args.weights, "seed": args.seed}
    with EventStorage(0):
        # 先跑评估态的 B，再跑训练态的 A（骨干是 FrozenBN，顺序本不影响结果，这样排更稳妥）
        if not args.skip_b:
            res["B_headroom"] = part_b(model, cfg, args)
            print(json.dumps(res["B_headroom"], indent=2, ensure_ascii=False), flush=True)
        if not args.skip_a:
            res["A_override"] = part_a(model, cfg, args)
            print(json.dumps(res["A_override"], indent=2, ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
