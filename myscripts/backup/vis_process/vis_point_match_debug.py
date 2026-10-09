"""逐实例小穗点（内层匹配）可视化：每个 GT 实例一个小图。

配合 `SetCriterion.loss_points` 使用，一次收集数据、出两张图：

A. 空间图（DRAW_POINT_PANELS，落在 OUT_DIR）
   每个 GT 实例一个 panel：
   * 底图     GT 框往外扩 BOX_MARGIN 后的原图裁剪 + 该实例的 GT mask 轮廓
   * 预测点   外层匹配到该实例的那个 query 的全部 P 个点，颜色深浅 = sigmoid(存在性 logit)
   * GT 点    只有 wheat_2 有小穗点标注（wheat 的 Ni=0），画成白色方块
   * 匹配对   内层匈牙利匹配上的预测点加绿色描边，并连线到它配到的 GT 点；
              这条线的长度就是 loss_point_coord 在惩罚的 L1 残差

B. 置信度分布图（DRAW_CONF_BARS，落在 CONF_OUT_DIR）
   每个 GT 实例一个条形图，一根柱子 = 一个预测点的存在性置信度，匹配上的柱子标绿。
   SORT_BY_CONF 时按置信度降序排，于是可以直接读出关键问题：
   **存在性头认为最可能存在的那 Ni 个点，是不是就是内层匹配挑中的那 Ni 个点。**
   两者一致时绿柱全部挤在左端；散开则说明存在性头和坐标头各说各话。

C. 损失分解图（DRAW_LOSS_BARS，落在 LOSS_OUT_DIR）
   一张堆叠条形图，每根柱子 = 一个 GT 实例对 total_loss 的贡献，按三种成分着色：
   坐标 L1（只对匹配点求和）/ focal 正样本部分（匹配点）/ focal 负样本部分（未匹配点）。
   数值口径完全对齐 loss_points：乘 weight_dict 权重、除以整个 batch 的 num_pos，
   因此所有柱子之和 = 最后一层 decoder 的点损失对 total_loss 的贡献。图里同时打印
   criterion.loss_points 的真实返回值做交叉校验。

坐标口径与 loss_points 完全一致：预测点是"GT 框相对 [0,1]"，用 GT 框换算回原图像素；
GT 点是图像归一化，直接乘 (W, H)。内层匹配不重新实现，直接调用
`criterion._inner_point_match`，保证画的就是 loss 真正用的那组配对。

开关和参数都是下面的模块常量，直接改，不走 config 也不用环境变量。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from myscripts.backup.vis_process.vis_match_debug import _CJK_FONT, _contour, _to_hwc_uint8  # noqa: F401

# ============================ debug 开关 ============================
ENABLED = False            # False 则 maybe_visualize_point_match 立即返回
DRAW_POINT_PANELS = True  # A 图：GT 裁剪 + 预测点 / GT 点 / 匹配连线
DRAW_CONF_BARS = True     # B 图：每个实例一个置信度条形图
DRAW_LOSS_BARS = True     # C 图：每个实例的损失分解堆叠条形图
OUT_DIR = "output_v2/vis_point_match"        # A 图输出目录，相对路径按仓库根解析
CONF_OUT_DIR = "output_v2/vis_point_conf"    # B 图输出目录
LOSS_OUT_DIR = "output_v2/vis_point_loss"    # C 图输出目录
EVERY = 1                 # 每 N 次 criterion 调用画一次
MAX_DUMP = 20             # 本进程最多画多少张图（按图片数计，不是文件数）
NUM_IMAGES = 1            # 每个 batch 画前几张图
MAX_INSTANCES = 24        # 每张图最多画多少个 GT panel（有点的排在前面）
ONLY_WITH_POINTS = False  # True 则只画有 GT 小穗点的实例（即只画 wheat_2）
BOX_MARGIN = 0.18         # A 图里 GT 框往外扩多少（占框宽高的比例）
NCOLS = 6                 # panel 网格列数
SORT_BY_CONF = True       # B 图柱子按置信度降序排（False 则按预测点槽位 0..P-1）
SORT_BY_LOSS = True       # C 图柱子按总损失降序排（False 则与 A/B 图同序）
INFER_THRESHOLD = 0.35    # B 图上的参考线，取 MaskDINO 推理用的 point_threshold 默认值
IMAGE_FORMAT = "RGB"      # 与 cfg.INPUT.FORMAT 一致
CLASS_NAMES = ["wheat", "wheat_2"]
# ===================================================================

_REPO_ROOT = Path(__file__).resolve().parents[3]

# 挂到 "maskdino" 这棵 logger 树下，train_net.py 的 setup_logger 才会打印路径
logger = logging.getLogger("maskdino.vis_point_match")

_STATE = {"calls": 0, "dumped": 0}

_CMAP = "plasma"          # 预测点置信度色标
_C_GT_PT = "#ffffff"      # GT 小穗点
_C_MATCH = "#22c55e"      # 匹配上的预测点：描边 / 连线 / 绿柱
_C_UNMATCH = "#94a3b8"    # 未匹配的预测点（B 图柱子）
_C_MASK = "#f59e0b"       # GT mask 轮廓
_C_THR = "#ef4444"        # 推理阈值参考线
_C_COORD = "#3b82f6"      # C 图：坐标 L1 成分


def _resolve(d):
    d = Path(d)
    return d if d.is_absolute() else _REPO_ROOT / d


def _cls_name(cls):
    return CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"c{cls}"


def _grid_shape(n):
    ncols = min(NCOLS, n)
    return ncols, int(np.ceil(n / ncols))


# --------------------------------------------------------------------------
# 数据收集：两张图共用，内层匹配只做一遍
# --------------------------------------------------------------------------
def _collect_items(criterion, pred_coords, pred_pt_logits, targets, indices, b, W, H, denom,
                   abs_frame=False, pred_knns=None):
    """整理第 b 张图每个 GT 实例的绘图数据。

    返回 (items, n_clamped_total)。items 已按"有小穗点优先、点多优先"排好序。
    """
    src_idx, tgt_idx = indices[b]
    boxes = targets[b]["boxes"].float()          # [Ng, 4] cxcywh 图像归一化
    points = targets[b]["points"]                # list[Tensor[Ni, 2]] 图像归一化
    labels = targets[b]["labels"].cpu().numpy()
    gt_masks = targets[b]["masks"].detach().to(torch.bool).cpu().numpy()
    alpha, gamma = criterion.focal_alpha, 2.0
    w_coord = criterion.weight_dict.get("loss_point_coord", 1.0)
    w_cls = criterion.weight_dict.get("loss_point_cls", 1.0)

    items = []
    n_clamped_total = 0
    for q_i, g_j in zip(src_idx.tolist(), tgt_idx.tolist()):
        pred_rel = pred_coords[b, q_i]                        # [P, 2]
        pred_lg = pred_pt_logits[b, q_i]                      # [P, 1]
        gt_pts_img = points[g_j].to(pred_rel.device).float()  # [Ni, 2]
        Ni = int(gt_pts_img.shape[0])
        if ONLY_WITH_POINTS and Ni == 0:
            continue

        cx, cy, bw, bh = boxes[g_j].to(pred_rel.device)
        x0, y0 = cx - bw / 2, cy - bh / 2
        wh = torch.stack([bw, bh]).clamp(min=1e-6)
        # 分组 decoder 输出绝对坐标，与 loss 一样先换到 GT 框相对系
        pred_rel = criterion._pred_to_box_rel(pred_rel.float(), torch.stack([x0, y0]), wh, abs_frame)
        if Ni > 0:
            gt_rel_raw = (gt_pts_img - torch.stack([x0, y0])) / wh
            gt_rel = gt_rel_raw.clamp(0.0, 1.0)               # loss 用的就是 clamp 后的
            n_clamped = int((gt_rel_raw != gt_rel).any(dim=1).sum())
        else:
            gt_rel = gt_pts_img.new_zeros((0, 2))
            n_clamped = 0
        n_clamped_total += n_clamped

        iso_scale = criterion._point_iso_scale(wh, targets[b].get("image_wh"))
        # POINT_KMO=True 时匹配多一项近邻代价，必须把 ŷ^k 一起传进去，画的才是 loss 用的配对
        pred_knn = pred_knns[b, q_i] if pred_knns is not None else None
        row_ind, col_ind = criterion._inner_point_match(pred_rel, pred_lg, gt_rel, iso_scale,
                                                        robust=abs_frame, pred_knn=pred_knn)

        # 框相对 -> 原图像素
        bx0, by0 = float(x0) * W, float(y0) * H
        bpw, bph = float(bw) * W, float(bh) * H
        pred_np = pred_rel.cpu().numpy()
        gt_np = gt_rel.cpu().numpy()
        pred_px = pred_np * np.array([bpw, bph]) + np.array([bx0, by0])
        # GT 点画在 loss 实际使用的（clamp 后）位置，与连线口径一致
        gt_used_px = (gt_np * np.array([bpw, bph]) + np.array([bx0, by0])
                      if Ni else np.zeros((0, 2)))
        conf = pred_lg.sigmoid().squeeze(-1).cpu().numpy()
        r = row_ind.cpu().numpy()
        c = col_ind.cpu().numpy()

        if r.size:
            l1 = float(np.abs(pred_np[r] - gt_np[c]).sum(1).mean())
            dist_px = float(np.linalg.norm(pred_px[r] - gt_used_px[c], axis=1).mean())
            conf_m = float(np.median(conf[r]))
            # 存在性头排前 Ni 的点里，有几个正是内层匹配挑中的
            hit = int(np.intersect1d(np.argsort(-conf)[:Ni], r).size)
        else:
            l1 = dist_px = conf_m = float("nan")
            hit = 0
        unmatched = np.setdiff1d(np.arange(conf.size), r)
        conf_u = float(np.median(conf[unmatched])) if unmatched.size else float("nan")

        # 逐实例损失分解，口径逐行对齐 loss_points：
        # focal 对全部 P 个点求和（按 tgt_exist 拆成正/负两块），L1 只对匹配点求和，
        # 再乘 weight_dict 权重、除以整个 batch 的 num_pos。
        tgt_exist = torch.zeros_like(pred_lg)
        if row_ind.numel():
            tgt_exist[row_ind] = 1.0
        prob = pred_lg.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_lg, tgt_exist, reduction="none")
        p_t = prob * tgt_exist + (1 - prob) * (1 - tgt_exist)
        focal = ce * ((1 - p_t) ** gamma)
        focal = (alpha * tgt_exist + (1 - alpha) * (1 - tgt_exist)) * focal
        cls_pos = float(focal[row_ind].sum()) if row_ind.numel() else 0.0
        cls_neg = float(focal.sum()) - cls_pos
        coord_raw = (float(F.l1_loss(pred_rel[row_ind], gt_rel[col_ind], reduction="sum"))
                     if row_ind.numel() else 0.0)

        items.append(dict(
            g=g_j, q=q_i, Ni=Ni, cls=int(labels[g_j]), n_clamped=n_clamped,
            box_px=(bx0, by0, bx0 + bpw, by0 + bph),
            mask=gt_masks[g_j], pred_px=pred_px, conf=conf,
            gt_used_px=gt_used_px, row=r, col=c,
            l1=l1, dist_px=dist_px, conf_m=conf_m, conf_u=conf_u, hit=hit,
            n_over=int((conf > INFER_THRESHOLD).sum()),
            loss_coord=w_coord * coord_raw / denom,
            loss_cls_pos=w_cls * cls_pos / denom,
            loss_cls_neg=w_cls * cls_neg / denom,
        ))

    # 有点的排前面，点多的优先
    items.sort(key=lambda d: (0 if d["Ni"] else 1, -d["Ni"]))
    return items, n_clamped_total


# --------------------------------------------------------------------------
# A 图：空间分布
# --------------------------------------------------------------------------
def _panel(ax, img, gt_mask, box_px, gt_pts_px, pred_px, conf, row_ind, col_ind, title, W, H):
    """画一个 GT 实例的小图。所有坐标都是原图像素。"""
    bx0, by0, bx1, by1 = box_px
    mx = BOX_MARGIN * max(bx1 - bx0, 1.0)
    my = BOX_MARGIN * max(by1 - by0, 1.0)
    cx0 = int(np.floor(max(0.0, bx0 - mx)))
    cy0 = int(np.floor(max(0.0, by0 - my)))
    cx1 = int(np.ceil(min(float(W), bx1 + mx)))
    cy1 = int(np.ceil(min(float(H), by1 + my)))
    if cx1 <= cx0 + 1 or cy1 <= cy0 + 1:
        ax.axis("off")
        return

    crop = (img[cy0:cy1, cx0:cx1].astype(np.float32) * 0.55).astype(np.uint8)
    ax.imshow(crop, extent=(cx0, cx1, cy1, cy0), interpolation="nearest")

    # GT mask 轮廓（只取裁剪窗口那一块，避免整图 contour）
    if gt_mask is not None:
        _contour(ax, gt_mask[cy0:cy1, cx0:cx1], (cx0, cx1, cy1, cy0), _C_MASK, lw=1.0)

    ax.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
            color=_C_MASK, lw=0.7, ls="--", alpha=0.8)

    # 全部 P 个预测点，颜色深浅 = 存在性置信度
    ax.scatter(pred_px[:, 0], pred_px[:, 1], c=conf, cmap=_CMAP, vmin=0.0, vmax=1.0,
               s=26, linewidths=0.4, edgecolors="#00000066", zorder=3)

    if gt_pts_px.shape[0]:
        ax.scatter(gt_pts_px[:, 0], gt_pts_px[:, 1], marker="s", s=22,
                   facecolors="none", edgecolors=_C_GT_PT, linewidths=1.1, zorder=4)

    # 内层匹配对：预测点加绿环 + 连线到它配到的 GT 点
    for r, c in zip(row_ind, col_ind):
        ax.scatter(pred_px[r, 0], pred_px[r, 1], s=64, facecolors="none",
                   edgecolors=_C_MATCH, linewidths=1.1, zorder=5)
        ax.plot([pred_px[r, 0], gt_pts_px[c, 0]], [pred_px[r, 1], gt_pts_px[c, 1]],
                color=_C_MATCH, lw=0.9, alpha=0.9, zorder=4)

    ax.set_xlim(cx0, cx1)
    ax.set_ylim(cy1, cy0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.5, linespacing=1.35)


def _draw_point_panels(img, items, shown, n_clamped_total, W, H, head_prefix, path):
    ncols, nrows = _grid_shape(len(shown))
    # 每个 panel 的标题有三行，行距要留够，否则会压到上一行的图上
    fig_w = 2.5 * ncols + 1.2          # +1.2 给右侧 colorbar
    fig_h = 3.4 * nrows + 2.0          # +2.0 给顶部 suptitle（两行）和底部图例
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.015, right=1 - 1.05 / fig_w,
                        top=1 - 1.45 / fig_h, bottom=0.55 / fig_h,
                        wspace=0.10, hspace=0.55)

    for ax in axes.ravel():
        ax.axis("off")
    for idx, it in enumerate(shown):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("on")
        name = _cls_name(it["cls"])
        if it["Ni"]:
            t = (f"g{it['g']} q{it['q']} {name}  Ni={it['Ni']}\n"
                 f"L1={it['l1']:.3f}  d={it['dist_px']:.1f}px\n"
                 f"conf m={it['conf_m']:.2f} / u={it['conf_u']:.2f}")
            if it["n_clamped"]:
                t += f"  clamp={it['n_clamped']}"
        else:
            t = (f"g{it['g']} q{it['q']} {name}  Ni=0\n"
                 f"(无小穗点，P 点全监督为不存在)\n"
                 f"conf max={it['conf'].max():.2f} med={np.median(it['conf']):.2f}")
        _panel(ax, img, it["mask"], it["box_px"], it["gt_used_px"],
               it["pred_px"], it["conf"], it["row"], it["col"], t, W, H)

    with_pts = [d for d in items if d["Ni"]]
    head = head_prefix
    if with_pts:
        head += (f"\n匹配点平均 L1(框相对)={np.nanmean([d['l1'] for d in with_pts]):.3f}"
                 f"   平均距离={np.nanmean([d['dist_px'] for d in with_pts]):.1f}px"
                 f"   匹配点 conf 中位数={np.nanmedian([d['conf_m'] for d in with_pts]):.2f}"
                 f"   未匹配点 conf 中位数={np.nanmedian([d['conf_u'] for d in with_pts]):.2f}")
    if n_clamped_total:
        head += f"   [GT 点被 clamp 到框内: {n_clamped_total}]"
    fig.suptitle(head, fontsize=9.5, y=1 - 0.18 / fig_h, va="top", linespacing=1.6)

    # colorbar 用独立 axes，不从 panel 里抢空间（否则会挤在网格中间）
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=plt.Normalize(0.0, 1.0))
    cbar = fig.colorbar(sm, cax=fig.add_axes([1 - 0.75 / fig_w, 0.28, 0.10 / fig_w, 0.44]))
    cbar.set_label("predicted point existence  sigmoid(logit)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", mfc="#bf5b17", mec="#00000066",
                   label="predicted point (color = confidence)"),
            Line2D([], [], marker="s", ls="", mfc="none", mec=_C_GT_PT,
                   label="GT spikelet point"),
            Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=_C_MATCH,
                   label="inner-matched prediction (+ link = L1 residual)"),
            Line2D([], [], color=_C_MASK, lw=1.2, label="GT mask / box"),
        ],
        loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.004),
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# B 图：置信度分布
# --------------------------------------------------------------------------
def _bar_panel(ax, it):
    """一个 GT 实例的置信度条形图：一根柱子 = 一个预测点，匹配上的标绿。"""
    conf = it["conf"]
    matched = np.zeros(conf.size, bool)
    matched[it["row"]] = True

    if SORT_BY_CONF:
        order = np.argsort(-conf)
        xlabel = "point rank by conf"
    else:
        order = np.arange(conf.size)
        xlabel = "point slot"

    x = np.arange(conf.size)
    ax.bar(x, conf[order], width=1.0, linewidth=0,
           color=np.where(matched[order], _C_MATCH, _C_UNMATCH))

    ax.axhline(INFER_THRESHOLD, color=_C_THR, ls="--", lw=0.8)
    if SORT_BY_CONF and it["Ni"]:
        # 排序后这条竖线左边就是"存在性头最看好的 Ni 个点"
        ax.axvline(it["Ni"] - 0.5, color="#111827", ls=":", lw=1.1)

    ax.set_xlim(-0.5, conf.size - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.set_xlabel(xlabel, fontsize=5.5, labelpad=1)
    ax.grid(axis="y", alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    name = _cls_name(it["cls"])
    if it["Ni"]:
        t = (f"g{it['g']} q{it['q']} {name}  Ni={it['Ni']}\n"
             f"top-Ni 命中 {it['hit']}/{it['Ni']}   超阈值 {it['n_over']}\n"
             f"匹配 conf {conf[it['row']].min():.2f}~{conf[it['row']].max():.2f}"
             f"   未匹配 max={conf[~matched].max():.2f}")
    else:
        t = (f"g{it['g']} q{it['q']} {name}  Ni=0\n"
             f"(全部监督为不存在)   超阈值 {it['n_over']}\n"
             f"conf max={conf.max():.2f}   med={np.median(conf):.2f}")
    ax.set_title(t, fontsize=6.5, linespacing=1.35)


def _draw_conf_bars(items, shown, head_prefix, path):
    ncols, nrows = _grid_shape(len(shown))
    fig_w = 2.6 * ncols
    fig_h = 2.2 * nrows + 1.9
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.99,
                        top=1 - 1.35 / fig_h, bottom=0.5 / fig_h,
                        wspace=0.28, hspace=0.95)

    for ax in axes.ravel():
        ax.axis("off")
    for idx, it in enumerate(shown):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("on")
        _bar_panel(ax, it)

    with_pts = [d for d in items if d["Ni"]]
    head = head_prefix
    if with_pts:
        hit = sum(d["hit"] for d in with_pts)
        n_gt = sum(d["Ni"] for d in with_pts)
        over_p = sum(d["n_over"] for d in with_pts)
        over_n = sum(d["n_over"] for d in items if d["Ni"] == 0)
        head += (f"\ntop-Ni 命中率 {hit}/{n_gt} ({100.0 * hit / max(n_gt, 1):.1f}%)"
                 f"   有点实例超阈值点数 {over_p} (GT 共 {n_gt})"
                 f"   无点实例超阈值点数 {over_n}"
                 f"   匹配 conf 中位数={np.nanmedian([d['conf_m'] for d in with_pts]):.2f}"
                 f" / 未匹配={np.nanmedian([d['conf_u'] for d in with_pts]):.2f}")
    fig.suptitle(head, fontsize=9.5, y=1 - 0.18 / fig_h, va="top", linespacing=1.6)

    fig.legend(
        handles=[
            Patch(facecolor=_C_MATCH, label="inner-matched point (positive)"),
            Patch(facecolor=_C_UNMATCH, label="unmatched point (supervised as absent)"),
            Line2D([], [], color=_C_THR, ls="--", lw=1.2,
                   label=f"inference threshold {INFER_THRESHOLD:g}"),
            Line2D([], [], color="#111827", ls=":", lw=1.2, label="top-Ni boundary"),
        ],
        loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.004),
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# C 图：逐实例损失分解
# --------------------------------------------------------------------------
def _draw_loss_bars(items, head_prefix, path, real=None):
    """堆叠条形图：每根柱子 = 一个 GT 实例对 total_loss 的贡献，按三种成分着色。"""
    tot_of = lambda d: d["loss_coord"] + d["loss_cls_pos"] + d["loss_cls_neg"]
    its = sorted(items, key=lambda d: -tot_of(d)) if SORT_BY_LOSS else list(items)

    x = np.arange(len(its))
    coord = np.array([d["loss_coord"] for d in its])
    cpos = np.array([d["loss_cls_pos"] for d in its])
    cneg = np.array([d["loss_cls_neg"] for d in its])
    tot = coord + cpos + cneg

    fig_w = max(11.0, 0.34 * len(its) + 2.5)
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(fig_w, 9.0), sharex=True,
        gridspec_kw=dict(height_ratios=[2.3, 1.0], hspace=0.10))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.855, bottom=0.135)

    specs = [(coord, _C_COORD, "坐标 L1（仅匹配点）"),
             (cpos, _C_MATCH, "存在性 focal · 正样本（匹配点）"),
             (cneg, _C_UNMATCH, "存在性 focal · 负样本（未匹配点）")]

    bottom = np.zeros_like(tot)
    for vals, color, label in specs:
        ax0.bar(x, vals, bottom=bottom, width=0.82, color=color, label=label, linewidth=0)
        bottom = bottom + vals
    ax0.axhline(float(tot.mean()), color="#111827", ls=":", lw=1.0,
                label=f"逐实例均值 {tot.mean():.4f}")
    ax0.set_ylabel("对 total_loss 的贡献", fontsize=9)
    ax0.grid(axis="y", alpha=0.25, lw=0.5)
    ax0.set_axisbelow(True)
    for side in ("top", "right"):
        ax0.spines[side].set_visible(False)
    ax0.legend(fontsize=8.5, ncol=4, loc="upper right", framealpha=0.9)

    # 下面板：同序的成分占比，小损失实例的构成也看得清
    share = np.maximum(tot, 1e-12)
    bottom = np.zeros_like(tot)
    for vals, color, _ in specs:
        v = vals / share * 100.0
        ax1.bar(x, v, bottom=bottom, width=0.82, color=color, linewidth=0)
        bottom = bottom + v
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_ylabel("成分占比 %", fontsize=9)
    ax1.grid(axis="y", alpha=0.25, lw=0.5)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    ax1.set_xlim(-0.7, len(its) - 0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"g{d['g']}" for d in its], rotation=90, fontsize=6.5)
    # 有小穗点的实例（wheat_2）用深色标签，无点的用浅灰，便于区分两类
    for tick, d in zip(ax1.get_xticklabels(), its):
        tick.set_color("#111827" if d["Ni"] else "#9ca3af")
    ax1.set_xlabel("GT 实例（深色标签 = 有小穗点；浅灰 = 无小穗点）", fontsize=8.5)

    with_pts = [d for d in items if d["Ni"]]
    head = head_prefix + (
        f"\n本图合计={tot.sum():.4f}"
        f"   坐标={coord.sum():.4f} ({100 * coord.sum() / max(tot.sum(), 1e-12):.1f}%)"
        f"   focal正={cpos.sum():.4f} ({100 * cpos.sum() / max(tot.sum(), 1e-12):.1f}%)"
        f"   focal负={cneg.sum():.4f} ({100 * cneg.sum() / max(tot.sum(), 1e-12):.1f}%)")
    if with_pts:
        head += (f"\n有点实例贡献 {sum(tot_of(d) for d in with_pts):.4f}"
                 f" / 无点实例 {sum(tot_of(d) for d in items if not d['Ni']):.4f}"
                 f"   （无点实例只有 focal 负样本一项）")
    if real is not None:
        rc, rl, nb = real
        head += (f"\n[交叉校验] criterion.loss_points 实际值 x 权重: "
                 f"坐标={rc:.4f}  存在性={rl:.4f}  合计={rc + rl:.4f}")
        if nb > 1:
            head += f"  （batch 有 {nb} 张图，实际值是整个 batch 的，本图只画了其中一张）"
    fig.suptitle(head, fontsize=9.5, y=0.985, va="top", linespacing=1.6)

    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
def visualize_point_match(criterion, outputs, targets, indices, batched_inputs=None, *, tag=""):
    """画 batch 前 NUM_IMAGES 张图的逐实例点匹配，返回 (写出的路径列表, 处理的图片数)。"""
    if "pred_point_coords" not in outputs or "pred_point_logits" not in outputs:
        raise ValueError("outputs 里没有 pred_point_coords / pred_point_logits")

    out_dir, conf_dir = _resolve(OUT_DIR), _resolve(CONF_OUT_DIR)
    loss_dir = _resolve(LOSS_OUT_DIR)
    if DRAW_POINT_PANELS:
        out_dir.mkdir(parents=True, exist_ok=True)
    if DRAW_CONF_BARS:
        conf_dir.mkdir(parents=True, exist_ok=True)
    if DRAW_LOSS_BARS:
        loss_dir.mkdir(parents=True, exist_ok=True)

    pred_coords = outputs["pred_point_coords"].detach().float()     # [B, Q, P, 2] 框相对
    pred_pt_logits = outputs["pred_point_logits"].detach().float()  # [B, Q, P, 1]
    pred_knns = outputs.get("pred_point_knn")                       # [B, Q, P, 1]，仅 POINT_KMO=True
    if pred_knns is not None:
        pred_knns = pred_knns.detach().float()
    P = pred_coords.shape[2]

    # loss_points 的归一化分母 num_pos：整个 batch 里所有外层匹配实例的 GT 小穗点总数。
    # 逐实例损失要除以它才能和真实 loss 数值对得上，所以必须跨图统计。
    denom = max(sum(int(targets[bb]["points"][g].shape[0])
                    for bb in range(len(indices))
                    for g in indices[bb][1].tolist()), 1)
    real = None
    if DRAW_LOSS_BARS:
        d = criterion.loss_points(outputs, targets, indices, 1)
        real = (float(d["loss_point_coord"]) * criterion.weight_dict.get("loss_point_coord", 1.0),
                float(d["loss_point_cls"]) * criterion.weight_dict.get("loss_point_cls", 1.0),
                len(indices))

    written, n_imgs = [], 0
    for b in range(min(NUM_IMAGES, pred_coords.shape[0])):
        if indices[b][0].numel() == 0:
            continue

        h_pad, w_pad = targets[b]["masks"].shape[-2:]
        if batched_inputs is not None and b < len(batched_inputs) and "image" in batched_inputs[b]:
            img = _to_hwc_uint8(batched_inputs[b]["image"], IMAGE_FORMAT)
            stem = Path(batched_inputs[b].get("file_name", f"img{b}")).stem
        else:
            img = np.full((h_pad, w_pad, 3), 96, dtype=np.uint8)
            stem = f"img{b}"
        H, W = img.shape[:2]

        items, n_clamped = _collect_items(criterion, pred_coords, pred_pt_logits,
                                          targets, indices, b, W, H, denom,
                                          abs_frame=bool(outputs.get("pred_point_abs", False)),
                                          pred_knns=pred_knns)
        if not items:
            continue
        shown = items[:MAX_INSTANCES]
        n_with = sum(1 for d in items if d["Ni"])
        head_prefix = (f"{tag}  {stem}   GT={len(items)} "
                       f"(有小穗点 {n_with} / 无 {len(items) - n_with})"
                       f"   P={P}   显示前 {len(shown)} 个")

        if DRAW_POINT_PANELS:
            path = out_dir / f"{tag}_b{b}_{stem}.png"
            _draw_point_panels(img, items, shown, n_clamped, W, H, head_prefix, path)
            written.append(path)
        if DRAW_CONF_BARS:
            path = conf_dir / f"{tag}_b{b}_{stem}_conf.png"
            _draw_conf_bars(items, shown, head_prefix, path)
            written.append(path)
        if DRAW_LOSS_BARS:
            # C 图是一张总览，不受 MAX_INSTANCES 限制，画全部实例
            path = loss_dir / f"{tag}_b{b}_{stem}_loss.png"
            _draw_loss_bars(items, f"{tag}  {stem}   GT={len(items)}   P={P}   "
                                   f"num_pos(batch)={denom}", path, real)
            written.append(path)
        n_imgs += 1

    return written, n_imgs


def maybe_visualize_point_match(criterion, outputs, targets, indices, batched_inputs=None):
    """criterion 里的调用入口。开关见本文件顶部 ENABLED；异常只 warning，绝不打断训练。"""
    if not ENABLED or targets is None or indices is None:
        return

    _STATE["calls"] += 1
    call = _STATE["calls"]
    if (call - 1) % max(1, EVERY) != 0 or _STATE["dumped"] >= MAX_DUMP:
        return

    try:
        paths, n_imgs = visualize_point_match(criterion, outputs, targets, indices,
                                              batched_inputs, tag=f"call{call:06d}")
        _STATE["dumped"] += n_imgs
        for p in paths:
            logger.info("[vis_point_match] %s", p)
    except Exception:  # 可视化永远不该让训练挂掉
        logger.warning("[vis_point_match] visualization failed, skipped", exc_info=True)
