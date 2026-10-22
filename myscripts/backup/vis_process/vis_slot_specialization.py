"""诊断实验：内层匹配的 slot 是否分化，以及标签是否在迭代间抖动。

背景。loss_points 用内层匈牙利匹配把 P 个预测点槽位配到 Ni 个 GT 小穗，被配到的
slot 监督为"存在"、其余为"不存在"。如果 slot 语义稳定（slot k 总是接第 k 个小穗），
存在性头收到的标签就是一致的，能收敛到 0/1 两端；如果分配每轮都在变，同一个 slot
一会儿是正样本一会儿是负样本，头只会停在中间的平衡点。

本模块跨迭代累计两个量：

1. 每个 slot 被匹配的频率。三条曲线对照：
   * 实测频率
   * 随机基线   E[Ni]/P —— slot 完全没分化时的水平（平的）
   * 完美分化   P(Ni > k) —— slot 按 rank 分化时的阶梯（slot k 在 Ni>k 时必被匹配）
2. 同一个实例前后两次被看到时，匹配 slot 集合的 Jaccard。
   1.0 = 完全稳定；越低说明标签抖得越厉害。

配套的 overfit 配置关掉了几何增广（MIN_SCALE=MAX_SCALE=1.0、RANDOM_FLIP none），
DROPOUT=0、backbone 是 FrozenBN，所以同一张图每次前向输入完全一致，
集合的变化只可能来自权重更新——这是个干净的受控测量。

开关和参数是下面的模块常量，直接改。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from myscripts.backup.vis_process.vis_match_debug import _CJK_FONT  # noqa: F401

# ============================ debug 开关 ============================
ENABLED = False
OUT_DIR = "output_v2/vis_slot_stats"   # 相对路径按仓库根解析
DUMP_EVERY = 100          # 每 N 次 criterion 调用 dump 一次图 + CSV
MIN_OBS = 30              # 至少累计这么多个实例观测才开始 dump
ROLL_WINDOW = 50          # Jaccard 趋势线的滑动窗口
# ===================================================================

_REPO_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("maskdino.vis_slot")

_C_OBS = "#2563eb"        # 实测频率
_C_RAND = "#ef4444"       # 随机基线
_C_PERFECT = "#22c55e"    # 完美 rank 分化

_S = {
    "calls": 0,
    "obs": 0,                       # 实例观测次数（只统计 Ni>0）
    "hits": None,                   # [P] 每个 slot 被匹配的次数
    "ni": [],                       # 每次观测的 Ni
    "prev": {},                     # (stem, g) -> 上次的 matched slot 集合
    "jac": [],                      # (call_idx, jaccard, flip_frac)
    "conf": {},                     # (stem, g) -> (conf[P], matched[P])，只保留最后一次观测
    "dumped": 0,
}


def reset():
    _S.update(calls=0, obs=0, hits=None, ni=[], prev={}, jac=[], conf={}, dumped=0)


def record(criterion, outputs, targets, indices, batched_inputs=None):
    """累计一次 criterion 调用里所有有小穗点实例的匹配结果。"""
    pred_coords = outputs["pred_point_coords"].detach().float()
    pred_lgts = outputs["pred_point_logits"].detach().float()
    pred_knns = outputs.get("pred_point_knn")    # [B, Q, P, D]，仅 POINT_KMO=True 时存在（D=1 dist / 2 dir）
    if pred_knns is not None:
        pred_knns = pred_knns.detach().float()
    P = pred_coords.shape[2]
    if _S["hits"] is None:
        _S["hits"] = np.zeros(P, dtype=np.int64)

    for b, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        boxes = targets[b]["boxes"].float()
        points = targets[b]["points"]
        stem = "img%d" % b
        if batched_inputs is not None and b < len(batched_inputs):
            stem = Path(batched_inputs[b].get("file_name", stem)).stem

        for q_i, g_j in zip(src_idx.tolist(), tgt_idx.tolist()):
            gt_pts = points[g_j].to(pred_coords.device).float()
            Ni = int(gt_pts.shape[0])
            if Ni == 0:                      # 只统计有小穗点的实例
                continue

            cx, cy, bw, bh = boxes[g_j].to(pred_coords.device)
            wh = torch.stack([bw, bh]).clamp(min=1e-6)
            gt_rel = ((gt_pts - torch.stack([cx - bw / 2, cy - bh / 2])) / wh).clamp(0.0, 1.0)
            pred_knn = pred_knns[b, q_i] if pred_knns is not None else None
            row_ind, _ = criterion._inner_point_match(pred_coords[b, q_i], pred_lgts[b, q_i], gt_rel, pred_knn)

            slots = frozenset(row_ind.cpu().numpy().tolist())
            _S["hits"][list(slots)] += 1
            _S["ni"].append(Ni)
            _S["obs"] += 1

            m = np.zeros(P, bool)
            m[list(slots)] = True
            _S["conf"][(stem, g_j)] = (
                pred_lgts[b, q_i].sigmoid().squeeze(-1).cpu().numpy().copy(), m)

            key = (stem, g_j)
            prev = _S["prev"].get(key)
            if prev is not None and (prev or slots):
                inter = len(prev & slots)
                union = len(prev | slots)
                # |A Δ B| / (2*Ni)：这一轮有多大比例的正样本槽位换人了
                flip = (len(prev ^ slots) / 2.0) / max(len(slots), 1)
                _S["jac"].append((_S["calls"], inter / max(union, 1), flip))
            _S["prev"][key] = slots


def _staircase(ni, P):
    """完美 rank 分化下每个 slot 的匹配频率：slot k 在 Ni > k 时必被匹配。"""
    ni = np.asarray(ni)
    return np.array([(ni > k).mean() for k in range(P)])


def dump(tag=""):
    """把累计结果画成四宫格 + 写 CSV，返回图片路径。"""
    hits, obs = _S["hits"], _S["obs"]
    if hits is None or obs < MIN_OBS:
        return None
    out_dir = Path(OUT_DIR)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    P = hits.size
    freq = hits / obs
    ni = np.asarray(_S["ni"])
    rand_base = ni.mean() / P                       # slot 未分化时的水平
    perfect = _staircase(ni, P)                     # slot 按 rank 分化时的阶梯
    freq_sorted = np.sort(freq)[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.885, bottom=0.075,
                        wspace=0.18, hspace=0.30)

    # --- 左上：按 slot 序号的实测频率 ---
    ax = axes[0, 0]
    ax.bar(np.arange(P), freq, width=0.85, color=_C_OBS, linewidth=0)
    ax.axhline(rand_base, color=_C_RAND, ls="--", lw=1.3,
               label=f"随机基线 E[Ni]/P = {rand_base:.3f}")
    ax.set_xlabel("slot 序号", fontsize=9)
    ax.set_ylabel("被内层匹配选中的频率", fontsize=9)
    ax.set_title("每个 slot 的匹配频率（按序号）", fontsize=10)
    ax.set_ylim(0, max(1.0, freq.max() * 1.1))
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # --- 右上：排序后的频率 vs 两条参考曲线 ---
    ax = axes[0, 1]
    ax.plot(freq_sorted, color=_C_OBS, lw=2.0, label="实测（降序）")
    ax.plot(perfect, color=_C_PERFECT, lw=1.8, ls="-",
            label="完美 rank 分化 P(Ni>k)")
    ax.axhline(rand_base, color=_C_RAND, ls="--", lw=1.3, label="随机基线（完全未分化）")
    ax.set_xlabel("slot（按频率降序）", fontsize=9)
    ax.set_ylabel("匹配频率", fontsize=9)
    ax.set_title("实测曲线更贴近哪一条？", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # --- 左下：Jaccard 直方图 ---
    ax = axes[1, 0]
    jac = np.array([j for _, j, _ in _S["jac"]]) if _S["jac"] else np.zeros(0)
    if jac.size:
        ax.hist(jac, bins=np.linspace(0, 1, 26), color="#8b5cf6", edgecolor="white", linewidth=0.4)
        ax.axvline(jac.mean(), color="#111827", ls=":", lw=1.4, label=f"均值 {jac.mean():.3f}")
        ax.legend(fontsize=8.5)
    ax.set_xlabel("同一实例前后两次观测的匹配 slot 集合 Jaccard", fontsize=9)
    ax.set_ylabel("次数", fontsize=9)
    ax.set_title("标签稳定性：1.0 = 匹配集合完全没变", fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # --- 右下：Jaccard 随迭代的趋势 + 统计 ---
    ax = axes[1, 1]
    if jac.size:
        calls = np.array([c for c, _, _ in _S["jac"]])
        ax.scatter(calls, jac, s=5, alpha=0.25, color="#8b5cf6", linewidths=0)
        if jac.size >= ROLL_WINDOW:
            k = np.ones(ROLL_WINDOW) / ROLL_WINDOW
            ax.plot(calls[ROLL_WINDOW - 1:], np.convolve(jac, k, mode="valid"),
                    color="#111827", lw=1.8, label=f"滑动均值（窗口 {ROLL_WINDOW}）")
            ax.legend(fontsize=8.5, loc="lower right")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("criterion 调用次数", fontsize=9)
    ax.set_ylabel("Jaccard", fontsize=9)
    ax.set_title("标签抖动是否随训练收敛？", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # 判据：实测曲线离哪条参考更近（在降序坐标下比 L1 距离）
    d_rand = float(np.abs(freq_sorted - rand_base).mean())
    d_perfect = float(np.abs(freq_sorted - perfect).mean())
    n_hi = int((freq > 0.9).sum())
    n_lo = int((freq < 0.1).sum())
    flip = np.array([f for _, _, f in _S["jac"]]) if _S["jac"] else np.zeros(0)
    txt = (f"实例观测 {obs} 次（{len(_S['prev'])} 个不同实例）   "
           f"Ni: mean={ni.mean():.1f} min={ni.min()} max={ni.max()}   P={P}\n"
           f"到随机基线的平均距离 {d_rand:.4f}   到完美分化阶梯的平均距离 {d_perfect:.4f}"
           f"   ->  更接近【{'随机（未分化）' if d_rand < d_perfect else '完美分化'}】\n"
           f"频率 >0.9 的 slot: {n_hi}/{P}    <0.1 的 slot: {n_lo}/{P}"
           f"    实测频率 std={freq.std():.4f}  vs  完美分化 std={perfect.std():.4f}")
    if jac.size:
        txt += (f"\n匹配集合 Jaccard: 均值={jac.mean():.3f} 中位数={np.median(jac):.3f}"
                f"  完全不变的比例={100.0 * (jac > 0.999).mean():.1f}%"
                f"    每轮换掉的正样本槽位占比: 均值={flip.mean() * 100:.1f}%")
    fig.suptitle(f"内层匹配 slot 分化诊断   {tag}\n{txt}", fontsize=10, y=0.985,
                 va="top", linespacing=1.7)

    path = out_dir / f"slot_stats_{tag or 'final'}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)

    csv = out_dir / f"slot_stats_{tag or 'final'}.csv"
    with open(csv, "w") as f:
        f.write("slot,hits,freq,perfect_rank_freq,random_baseline\n")
        for k in range(P):
            f.write(f"{k},{hits[k]},{freq[k]:.6f},{perfect[k]:.6f},{rand_base:.6f}\n")
    return path


def maybe_record_slot_stats(criterion, outputs, targets, indices, batched_inputs=None):
    """criterion 里的调用入口。异常只 warning，绝不打断训练。"""
    if not ENABLED or targets is None or indices is None:
        return
    if "pred_point_coords" not in outputs or "pred_point_logits" not in outputs:
        return
    try:
        _S["calls"] += 1
        record(criterion, outputs, targets, indices, batched_inputs)
        if _S["calls"] % max(1, DUMP_EVERY) == 0:
            tag = f"call{_S['calls']:06d}"
            for p in (dump(tag=tag), dump_discrimination(criterion, tag=tag)):
                if p is not None:
                    _S["dumped"] += 1
                    logger.info("[vis_slot] %s  (累计 %d 次实例观测)", p, _S["obs"])
    except Exception:
        logger.warning("[vis_slot] recording failed, skipped", exc_info=True)


# --------------------------------------------------------------------------
# 判别力诊断：存在性头到底是在预测"这个实例的哪些 slot 该亮"，
# 还是只学到了"每个 slot 平均有多大概率亮"（即边际频率）。
# --------------------------------------------------------------------------
def _auc(scores, labels):
    """Mann-Whitney AUC。labels 为 bool。两类缺一则返回 nan。"""
    pos, neg = scores[labels], scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _equilibrium(alpha, f, gamma=2.0):
    """槽位以频率 f 为正样本时，focal loss 把置信度推到的平衡点。"""
    p = np.linspace(1e-4, 1 - 1e-4, 20000)
    fl_pos = alpha * (1 - p) ** gamma * (-np.log(p))
    fl_neg = (1 - alpha) * p ** gamma * (-np.log(1 - p))
    return np.array([p[np.argmin(fi * fl_pos + (1 - fi) * fl_neg)] for fi in np.atleast_1d(f)])


def dump_discrimination(criterion, tag=""):
    """出第二张图：存在性头有没有实例区分能力。"""
    rows = list(_S["conf"].values())
    if len(rows) < 5:
        return None
    out_dir = Path(OUT_DIR)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    C = np.stack([r[0] for r in rows])        # [N 实例, P] 置信度
    M = np.stack([r[1] for r in rows])        # [N 实例, P] 是否被匹配
    N, P = C.shape
    freq = M.mean(0)                          # 每个 slot 的边际匹配频率
    mean_conf = C.mean(0)                     # 每个 slot 的平均置信度
    alpha = getattr(criterion, "focal_alpha", 0.25)

    auc_all = _auc(C.ravel(), M.ravel())
    # 关键量：把 slot 固定住，只看跨实例的变化能不能预测该 slot 是否被匹配
    per_slot = np.array([_auc(C[:, k], M[:, k]) for k in range(P)])
    auc_within = float(np.nanmean(per_slot))
    # 方差分解：置信度的波动有多少来自"slot 不同"，多少来自"实例不同"
    v_total = float(C.var())
    v_between = float(mean_conf.var())
    frac_slot = v_between / max(v_total, 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.865, bottom=0.075, wspace=0.2, hspace=0.32)

    ax = axes[0, 0]
    xs = np.linspace(0.01, 0.99, 200)
    ax.plot(xs, _equilibrium(alpha, xs), color=_C_RAND, lw=1.8,
            label=f"理论平衡点 equilibrium(α={alpha}, f)")
    ax.scatter(freq, mean_conf, s=26, color=_C_OBS, alpha=0.85, linewidths=0, label="每个 slot")
    ax.set_xlabel("该 slot 的边际匹配频率 f", fontsize=9)
    ax.set_ylabel("该 slot 的平均置信度", fontsize=9)
    ax.set_title("slot 的平均置信度是否就等于它的边际频率对应的平衡点？", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)

    ax = axes[0, 1]
    bins = np.linspace(0, 1, 41)
    ax.hist(C[M], bins=bins, alpha=0.75, color=_C_PERFECT, label=f"匹配 (n={int(M.sum())})")
    ax.hist(C[~M], bins=bins, alpha=0.65, color="#94a3b8", label=f"未匹配 (n={int((~M).sum())})")
    ax.set_xlabel("存在性置信度", fontsize=9); ax.set_ylabel("次数", fontsize=9)
    ax.set_title(f"全局分布   AUC={auc_all:.3f}", fontsize=10)
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_axisbelow(True)

    ax = axes[1, 0]
    ok = per_slot[~np.isnan(per_slot)]
    ax.hist(ok, bins=np.linspace(0, 1, 26), color="#8b5cf6", edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, color=_C_RAND, ls="--", lw=1.4, label="0.5 = 毫无区分能力")
    ax.axvline(float(ok.mean()), color="#111827", ls=":", lw=1.4, label=f"均值 {ok.mean():.3f}")
    ax.set_xlabel("固定 slot、跨实例的 AUC", fontsize=9); ax.set_ylabel("slot 数", fontsize=9)
    ax.set_title("去掉 slot 边际效应后，还剩多少判别力？", fontsize=10)
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_axisbelow(True)

    ax = axes[1, 1]
    ax.imshow(C[np.argsort(-M.sum(1))], aspect="auto", cmap="plasma", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xlabel("slot", fontsize=9); ax.set_ylabel("实例（按 Ni 降序）", fontsize=9)
    ax.set_title("置信度矩阵：竖条纹 = 只看 slot；横向差异 = 认得出实例", fontsize=10)

    txt = (f"实例 {N} 个   P={P}   α={alpha}\n"
           f"全局 AUC={auc_all:.3f}    固定 slot 后的跨实例平均 AUC={auc_within:.3f}"
           f"    （0.5 = 完全不认识实例）\n"
           f"置信度方差中由 slot 边际差异解释的比例={100 * frac_slot:.1f}%"
           f"    平均置信度与边际频率的相关系数={np.corrcoef(freq, mean_conf)[0, 1]:.3f}")
    fig.suptitle(f"存在性头的实例判别力诊断   {tag}\n{txt}", fontsize=10, y=0.985,
                 va="top", linespacing=1.7)

    path = out_dir / f"slot_discrim_{tag or 'final'}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
