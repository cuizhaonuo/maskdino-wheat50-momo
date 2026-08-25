"""逐实例小穗点（内层匹配）可视化：每个 GT 实例一个小图。

配合 `SetCriterion.loss_points` 使用。每个 GT 实例画成一个 panel：

  * 底图     GT 框往外扩 BOX_MARGIN 后的原图裁剪 + 该实例的 GT mask 轮廓
  * 预测点   外层匹配到该实例的那个 query 的全部 P 个点，颜色深浅 = sigmoid(存在性 logit)
  * GT 点    只有 wheat_2 有小穗点标注（wheat 的 Ni=0），画成白色方块
  * 匹配对   内层匈牙利匹配上的预测点加绿色描边，并连线到它配到的 GT 点；
             这条线的长度就是 loss_point_coord 在惩罚的 L1 残差

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from myscripts.backup.vis_process.vis_match_debug import _CJK_FONT, _contour, _to_hwc_uint8  # noqa: F401

# ============================ debug 开关 ============================
ENABLED = True            # False 则 maybe_visualize_point_match 立即返回
OUT_DIR = "output_v2/vis_point_match"   # 相对路径按仓库根解析，与 cwd 无关
EVERY = 1                 # 每 N 次 criterion 调用画一次
MAX_DUMP = 20             # 本进程最多画多少张
NUM_IMAGES = 1            # 每个 batch 画前几张图
MAX_INSTANCES = 24        # 每张图最多画多少个 GT panel（有点的排在前面）
ONLY_WITH_POINTS = False  # True 则只画有 GT 小穗点的实例（即只画 wheat_2）
BOX_MARGIN = 0.18         # GT 框往外扩多少（占框宽高的比例）
NCOLS = 6                 # panel 网格列数
IMAGE_FORMAT = "RGB"      # 与 cfg.INPUT.FORMAT 一致
CLASS_NAMES = ["wheat", "wheat_2"]
# ===================================================================

_REPO_ROOT = Path(__file__).resolve().parents[3]

# 挂到 "maskdino" 这棵 logger 树下，train_net.py 的 setup_logger 才会打印路径
logger = logging.getLogger("maskdino.vis_point_match")

_STATE = {"calls": 0, "dumped": 0}

_CMAP = "plasma"          # 预测点置信度色标
_C_GT_PT = "#ffffff"      # GT 小穗点
_C_MATCH = "#22c55e"      # 匹配上的预测点描边 + 连线
_C_MASK = "#f59e0b"       # GT mask 轮廓


def _panel(ax, img, gt_mask, box_px, gt_pts_px, gt_rel_used, pred_px, conf,
           row_ind, col_ind, title, W, H):
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
        sub = gt_mask[cy0:cy1, cx0:cx1]
        _contour(ax, sub, (cx0, cx1, cy1, cy0), _C_MASK, lw=1.0)

    # GT 框
    ax.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
            color=_C_MASK, lw=0.7, ls="--", alpha=0.8)

    # 全部 P 个预测点，颜色深浅 = 存在性置信度
    ax.scatter(pred_px[:, 0], pred_px[:, 1], c=conf, cmap=_CMAP, vmin=0.0, vmax=1.0,
               s=26, linewidths=0.4, edgecolors="#00000066", zorder=3)

    # GT 小穗点
    if gt_pts_px.shape[0]:
        ax.scatter(gt_pts_px[:, 0], gt_pts_px[:, 1], marker="s", s=22,
                   facecolors="none", edgecolors=_C_GT_PT, linewidths=1.1, zorder=4)

    # 内层匹配对：预测点加绿环 + 连线到它配到的 GT 点
    for r, c in zip(row_ind, col_ind):
        ax.scatter(pred_px[r, 0], pred_px[r, 1], s=64, facecolors="none",
                   edgecolors=_C_MATCH, linewidths=1.1, zorder=5)
        ax.plot([pred_px[r, 0], gt_rel_used[c, 0]], [pred_px[r, 1], gt_rel_used[c, 1]],
                color=_C_MATCH, lw=0.9, alpha=0.9, zorder=4)

    ax.set_xlim(cx0, cx1)
    ax.set_ylim(cy1, cy0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.5, linespacing=1.35)


def visualize_point_match(criterion, outputs, targets, indices, batched_inputs=None, *,
                          out_dir=OUT_DIR, tag=""):
    """把 batch 前 NUM_IMAGES 张图的逐实例内层点匹配画成 PNG，返回写出的路径列表。"""
    if "pred_point_coords" not in outputs or "pred_point_logits" not in outputs:
        raise ValueError("outputs 里没有 pred_point_coords / pred_point_logits")

    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_coords = outputs["pred_point_coords"].detach().float()   # [B, Q, P, 2] 框相对
    pred_pt_logits = outputs["pred_point_logits"].detach().float()  # [B, Q, P, 1]

    written = []
    for b in range(min(NUM_IMAGES, pred_coords.shape[0])):
        src_idx, tgt_idx = indices[b]
        if src_idx.numel() == 0:
            continue

        gt_masks_t = targets[b]["masks"]
        h_pad, w_pad = gt_masks_t.shape[-2:]
        if batched_inputs is not None and b < len(batched_inputs) and "image" in batched_inputs[b]:
            img = _to_hwc_uint8(batched_inputs[b]["image"], IMAGE_FORMAT)
            stem = Path(batched_inputs[b].get("file_name", f"img{b}")).stem
        else:
            img = np.full((h_pad, w_pad, 3), 96, dtype=np.uint8)
            stem = f"img{b}"
        H, W = img.shape[:2]

        boxes = targets[b]["boxes"].float()          # [Ng, 4] cxcywh 图像归一化
        points = targets[b]["points"]                # list[Tensor[Ni, 2]] 图像归一化
        labels = targets[b]["labels"].cpu().numpy()
        gt_masks = gt_masks_t.detach().to(torch.bool).cpu().numpy()

        # 收集每个 GT 实例的绘图数据（内层匹配用 criterion 自己的实现，保证与 loss 一致）
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
            if Ni > 0:
                gt_rel_raw = (gt_pts_img - torch.stack([x0, y0])) / wh
                gt_rel = gt_rel_raw.clamp(0.0, 1.0)               # loss 用的就是 clamp 后的
                n_clamped = int((gt_rel_raw != gt_rel).any(dim=1).sum())
            else:
                gt_rel = gt_pts_img.new_zeros((0, 2))
                n_clamped = 0
            n_clamped_total += n_clamped

            row_ind, col_ind = criterion._inner_point_match(pred_rel, pred_lg, gt_rel)

            # 框相对 -> 原图像素
            bx0, by0 = float(x0) * W, float(y0) * H
            bpw, bph = float(bw) * W, float(bh) * H
            pred_px = pred_rel.cpu().numpy() * np.array([bpw, bph]) + np.array([bx0, by0])
            # GT 点：画在 loss 实际使用的（clamp 后）位置，与连线口径一致
            gt_used_px = (gt_rel.cpu().numpy() * np.array([bpw, bph]) + np.array([bx0, by0])
                          if Ni else np.zeros((0, 2)))
            conf = pred_lg.sigmoid().squeeze(-1).cpu().numpy()
            r = row_ind.cpu().numpy()
            c = col_ind.cpu().numpy()

            if r.size:
                l1 = float(np.abs(pred_rel.cpu().numpy()[r] - gt_rel.cpu().numpy()[c]).sum(1).mean())
                dist_px = float(np.linalg.norm(pred_px[r] - gt_used_px[c], axis=1).mean())
                conf_m = float(np.median(conf[r]))
            else:
                l1 = dist_px = float("nan")
                conf_m = float("nan")
            conf_un = conf[np.setdiff1d(np.arange(conf.size), r)]
            conf_u = float(np.median(conf_un)) if conf_un.size else float("nan")

            items.append(dict(
                g=g_j, q=q_i, Ni=Ni, cls=int(labels[g_j]), n_clamped=n_clamped,
                box_px=(bx0, by0, bx0 + bpw, by0 + bph),
                mask=gt_masks[g_j], pred_px=pred_px, conf=conf,
                gt_used_px=gt_used_px, row=r, col=c,
                l1=l1, dist_px=dist_px, conf_m=conf_m, conf_u=conf_u,
            ))

        if not items:
            continue
        # 有点的排前面，点多的优先
        items.sort(key=lambda d: (0 if d["Ni"] else 1, -d["Ni"]))
        shown = items[:MAX_INSTANCES]

        ncols = min(NCOLS, len(shown))
        nrows = int(np.ceil(len(shown) / ncols))
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
            name = CLASS_NAMES[it["cls"]] if it["cls"] < len(CLASS_NAMES) else f"c{it['cls']}"
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
            _panel(ax, img, it["mask"], it["box_px"], it["gt_used_px"], it["gt_used_px"],
                   it["pred_px"], it["conf"], it["row"], it["col"], t, W, H)

        with_pts = [d for d in items if d["Ni"]]
        head = (f"{tag}  {stem}   GT={len(items)} "
                f"(有小穗点 {len(with_pts)} / 无 {len(items) - len(with_pts)})"
                f"   P={pred_coords.shape[2]}   显示前 {len(shown)} 个")
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
        cax = fig.add_axes([1 - 0.75 / fig_w, 0.28, 0.10 / fig_w, 0.44])
        cbar = fig.colorbar(sm, cax=cax)
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
            loc="lower center", ncol=4, fontsize=8.5, frameon=False,
            bbox_to_anchor=(0.5, 0.004),
        )

        path = out_dir / f"{tag}_b{b}_{stem}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    return written


def maybe_visualize_point_match(criterion, outputs, targets, indices, batched_inputs=None):
    """criterion 里的调用入口。开关见本文件顶部 ENABLED；异常只 warning，绝不打断训练。"""
    if not ENABLED or targets is None or indices is None:
        return

    _STATE["calls"] += 1
    call = _STATE["calls"]
    if (call - 1) % max(1, EVERY) != 0 or _STATE["dumped"] >= MAX_DUMP:
        return

    try:
        paths = visualize_point_match(criterion, outputs, targets, indices, batched_inputs,
                                      tag=f"call{call:06d}")
        _STATE["dumped"] += len(paths)
        for p in paths:
            logger.info("[vis_point_match] %s", p)
    except Exception:  # 可视化永远不该让训练挂掉
        logger.warning("[vis_point_match] visualization failed, skipped", exc_info=True)
