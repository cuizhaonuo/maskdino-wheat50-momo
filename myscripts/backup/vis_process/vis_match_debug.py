"""训练期匈牙利匹配的正负样本可视化（debug 工具，默认关闭）。

在 `SetCriterion.forward` 里、算 loss 之前调用 `maybe_visualize_matching`，把一个
batch 里前若干张图画成四宫格 PNG：

  1. 原图 + GT 实例轮廓（按类别着色，可叠加 GT 小穗点）
  2. 原图 + 置信度 Top-K 的 query mask（颜色编码 score）
  3. 原图 + 匹配上的 query（正样本，绿）/ GT（橙虚线）/ Top-K 里没匹配上的 query（灰）
  4. query 按 score 排序的曲线，正样本在曲线上的位置 + 匹配统计

第 3、4 两幅是看"正负样本匹配情况"的主力：第 3 幅看空间上匹配到哪个穗子、
漏了哪些穗子；第 4 幅看正样本是不是集中在高分区（分类头和匹配是否自洽）。

开关和参数都是本文件顶部的模块常量，直接改这里即可，不经过 config 也不用环境变量。
ENABLED 默认 True；正式训练前把它改成 False（或把 MAX_DUMP 调小）。

注意 mask 一律以 1/4 或抽稀后的分辨率画等值线，再用 extent 映射回图像坐标，
避免把 300 张 1024x1024 的 mask 插值到原图尺寸而吃掉几个 G 显存/内存。
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

# 挂到 "maskdino" 这棵 logger 树下，train_net.py 的 setup_logger(name="maskdino")
# 才会把输出路径同时打到控制台和 OUTPUT_DIR/log.txt；用 __name__ 会因为没有 handler 而静默。
logger = logging.getLogger("maskdino.vis_match")


# matplotlib 自带字体没有 CJK，标题里的中文会渲染成方框。系统里装了文泉驿/Droid
# Fallback，但 matplotlib 的字体缓存不收录 .ttc，需要显式 addfont。找不到就静默跳过。
_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _setup_cjk_font():
    from matplotlib import font_manager as fm

    for path in _CJK_FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
        plt.rcParams["axes.unicode_minus"] = False
        return name
    logger.warning("[vis] 未找到中文字体，图中中文会显示成方框")
    return None


_CJK_FONT = _setup_cjk_font()

# ============================ debug 开关 ============================
ENABLED = True           # False 则 maybe_visualize_matching 立即返回，零开销
OUT_DIR = "output_v2/vis_match"   # 相对路径按仓库根目录解析，与工作目录无关
EVERY = 1                # 每 N 次 criterion 调用画一次
MAX_DUMP = 20            # 本进程最多画多少张，画满即停（防止长训练刷屏）
TOPK = 100               # "置信度前 K 个穗子" 的 K
NUM_IMAGES = 1           # 每个 batch 画前几张图
DRAW_POINTS = True       # 在 GT 面板上叠加 GT 小穗点
IMAGE_FORMAT = "RGB"     # 与 cfg.INPUT.FORMAT 一致
CLASS_NAMES = ["wheat", "wheat_2"]   # 图例用的类别名，None 则退化成 c0/c1/...
# ===================================================================

# 仓库根目录：myscripts/backup/vis_process/vis_match_debug.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]

# 调用计数与已产出图数，用于 EVERY / MAX_DUMP 限流
_STATE = {"calls": 0, "dumped": 0}

# 与 data_register/wheat_data_cat2.py 的 thing_colors 保持一致
_DEFAULT_CLASS_COLORS = [(1.0, 0.0, 0.0), (0.0, 0.35, 1.0), (0.0, 0.7, 0.2)]

_C_POS = "#22c55e"       # 匹配上的 query（正样本）
_C_GT = "#f59e0b"        # GT
_C_NEG = "#9ca3af"       # Top-K 里没匹配上的 query（高置信度负样本）
_C_LINK = "#22c55e"      # pred 质心 -> GT 质心 的连线


def _to_hwc_uint8(img_t: torch.Tensor, image_format: str = "RGB") -> np.ndarray:
    """batched_inputs[i]["image"] 是归一化之前的 CHW 张量，转成可 imshow 的 HWC uint8。"""
    arr = img_t.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if image_format.upper() == "BGR":
        arr = arr[:, :, ::-1]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _grid(shape, extent):
    """把数组格点中心映射到图像坐标，返回 (xs, ys)。extent 与 imshow 同义：(左, 右, 下, 上)。

    不能把这个 extent 直接丢给 ax.contour：origin 为 None（默认）时 matplotlib 把它解释成
    "Z[0,0] 落在 (x0, y0)、Z[-1,-1] 落在 (x1, y1)"，于是 imshow 那套 (left, right, bottom, top)
    会让 mask 上下镜像（第 0 行被钉到图像底部），表现为"轮廓和原图对不上"。
    显式给出 X/Y 坐标轴则没有任何歧义。
    """
    nrow, ncol = shape
    x0, x1, y_bottom, y_top = extent
    xs = x0 + (np.arange(ncol) + 0.5) * (x1 - x0) / ncol
    ys = y_top + (np.arange(nrow) + 0.5) * (y_bottom - y_top) / nrow
    return xs, ys


def _contour(ax, mask2d: np.ndarray, extent, color, lw=0.8, ls="-"):
    """把一个二值 mask 画成等值线。mask 可以是任意分辨率，按 extent 贴回图像坐标。"""
    if mask2d.size == 0 or not mask2d.any():
        return
    xs, ys = _grid(mask2d.shape, extent)
    ax.contour(
        xs,
        ys,
        mask2d.astype(np.float32),
        levels=[0.5],
        colors=[color],
        linewidths=lw,
        linestyles=ls,
    )


def _centroid(mask2d: np.ndarray, extent):
    """mask 质心，返回图像坐标 (x, y)；空 mask 返回 None。与 _contour 共用同一套映射。"""
    rows, cols = np.nonzero(mask2d)
    if rows.size == 0:
        return None
    xs, ys = _grid(mask2d.shape, extent)
    return float(xs[cols].mean()), float(ys[rows].mean())


def _blank_axes(ax, img, W, H, title, dim=1.0):
    """dim < 1 时把底图压暗，否则密集的等值线会淹没在麦田纹理里。"""
    if dim < 1.0:
        img = (img.astype(np.float32) * dim).astype(np.uint8)
    ax.imshow(img, extent=(0, W, H, 0), interpolation="nearest")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def visualize_matching(
    outputs,
    targets,
    indices,
    batched_inputs=None,
    *,
    out_dir,
    tag="",
    topk=100,
    num_images=1,
    mask_stride=2,
    image_format="RGB",
    draw_points=False,
    class_names=None,
):
    """把 batch 中前 num_images 张图的匹配情况各存成一张 PNG，返回写出的路径列表。

    outputs / targets / indices 直接用 criterion.forward 里的同名变量：
      outputs["pred_logits"]: [B, Q, C]，focal loss，取 sigmoid 当置信度
      outputs["pred_masks"] : [B, Q, h_pad/4, w_pad/4] 的 logits
      targets[b]["masks"]   : [N, h_pad, w_pad] bool
      targets[b]["points"]  : list[Tensor[Ni, 2]]，按 (w, h) 归一化
      indices[b]            : (src_idx, tgt_idx)，src 是 query 下标
    """
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():          # 相对路径按仓库根解析，避免受启动 cwd 影响
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_logits = outputs["pred_logits"].detach().float()
    pred_masks = outputs.get("pred_masks", None)
    if pred_masks is None:
        raise ValueError("outputs 里没有 pred_masks，无法画 mask 轮廓")
    pred_masks = pred_masks.detach().float()

    B, Q, C = pred_logits.shape
    if class_names is None:
        class_names = [f"c{i}" for i in range(C)]

    written = []
    for b in range(min(num_images, B)):
        gt_masks_t = targets[b]["masks"]
        h_pad, w_pad = gt_masks_t.shape[-2:]

        if batched_inputs is not None and b < len(batched_inputs) and "image" in batched_inputs[b]:
            img = _to_hwc_uint8(batched_inputs[b]["image"], image_format)
            stem = Path(batched_inputs[b].get("file_name", f"img{b}")).stem
        else:
            img = np.full((h_pad, w_pad, 3), 96, dtype=np.uint8)
            stem = f"img{b}"
        H, W = img.shape[:2]

        # mask 一律降采样后画等值线，用 extent 贴回 padding 后的图像坐标系。
        pad_extent = (0, w_pad, h_pad, 0)
        s = max(1, int(mask_stride))
        gt_masks = gt_masks_t.detach().to(torch.bool).cpu().numpy()[:, ::s, ::s]
        gt_labels = targets[b]["labels"].detach().cpu().numpy()
        n_gt = gt_masks.shape[0]

        scores_all = pred_logits[b].sigmoid()                       # [Q, C]
        score, label = scores_all.max(dim=-1)
        score = score.cpu().numpy()
        label = label.cpu().numpy()

        order = np.argsort(-score)                                   # 按置信度降序
        rank = np.empty(Q, dtype=np.int64)
        rank[order] = np.arange(Q)
        k = int(min(topk, Q))
        topk_idx = order[:k]

        src, tgt = indices[b]
        src = np.asarray(src.detach().cpu() if torch.is_tensor(src) else src, dtype=np.int64)
        tgt = np.asarray(tgt.detach().cpu() if torch.is_tensor(tgt) else tgt, dtype=np.int64)
        matched_set = set(src.tolist())

        # 只把要画的 query 的 mask 搬到 CPU，避免 300 张全量拷贝
        draw_q = np.unique(np.concatenate([topk_idx, src])) if src.size else topk_idx
        qmask = {}
        for qi in draw_q.tolist():
            qmask[qi] = (pred_masks[b, qi] > 0).cpu().numpy()[::1, ::1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # ---- 面板 1：GT ----
        ax = axes[0, 0]
        _blank_axes(ax, img, W, H, f"GT instances (n={n_gt})")
        for i in range(n_gt):
            c = _DEFAULT_CLASS_COLORS[int(gt_labels[i]) % len(_DEFAULT_CLASS_COLORS)]
            _contour(ax, gt_masks[i], pad_extent, c, lw=0.9)
        if draw_points and "points" in targets[b]:
            pts = [p.detach().cpu().numpy() for p in targets[b]["points"] if p.numel()]
            if pts:
                pts = np.concatenate(pts, axis=0) * np.array([W, H])
                ax.scatter(pts[:, 0], pts[:, 1], s=2.0, c="#ffffff",
                           edgecolors="none", alpha=0.85, zorder=4)
        cls_handles = [
            Line2D([], [], color=_DEFAULT_CLASS_COLORS[i % len(_DEFAULT_CLASS_COLORS)],
                   lw=1.5, label=f"{class_names[i]} ({int((gt_labels == i).sum())})")
            for i in range(len(class_names))
        ]
        ax.legend(handles=cls_handles, loc="lower right", fontsize=7, framealpha=0.7)

        # ---- 面板 2：Top-K 预测 ----
        ax = axes[0, 1]
        _blank_axes(ax, img, W, H,
                    f"Top-{k} queries by score  (score {score[topk_idx].min():.3f} ~ {score[topk_idx].max():.3f})",
                    dim=0.45)
        cmap = plt.get_cmap("viridis")
        vmin, vmax = float(score[topk_idx].min()), float(score[topk_idx].max())
        denom = max(vmax - vmin, 1e-6)
        for qi in topk_idx.tolist():
            _contour(ax, qmask[qi], pad_extent, cmap((score[qi] - vmin) / denom), lw=0.7)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02).set_label("query score", fontsize=8)

        # ---- 面板 3：匹配（正样本） vs Top-K 中的负样本 ----
        ax = axes[1, 0]
        n_neg_topk = k - len(matched_set & set(topk_idx.tolist()))
        _blank_axes(ax, img, W, H,
                    f"matched (pos, n={src.size})  vs  unmatched in Top-{k} (neg, n={n_neg_topk})",
                    dim=0.45)
        for qi in topk_idx.tolist():          # 先画负样本压在底层
            if qi not in matched_set:
                _contour(ax, qmask[qi], pad_extent, _C_NEG, lw=0.6)
        for i in range(n_gt):                 # GT 橙色虚线
            _contour(ax, gt_masks[i], pad_extent, _C_GT, lw=1.8, ls="-")
        for q, g in zip(src.tolist(), tgt.tolist()):
            _contour(ax, qmask[q], pad_extent, _C_POS, lw=0.9)
            pc = _centroid(qmask[q], pad_extent)
            gc = _centroid(gt_masks[g], pad_extent)
            if pc is not None and gc is not None:
                ax.plot([pc[0], gc[0]], [pc[1], gc[1]], color=_C_LINK,
                        lw=0.6, alpha=0.75, zorder=3)
        ax.legend(
            handles=[
                Line2D([], [], color=_C_POS, lw=1.5, label="matched query (positive)"),
                Line2D([], [], color=_C_GT, lw=2.0, label="GT instance"),
                Line2D([], [], color=_C_NEG, lw=1.0, label=f"Top-{k} unmatched (negative)"),
            ],
            loc="lower right", fontsize=7, framealpha=0.7,
        )

        # ---- 面板 4：score 排序曲线 + 正样本落点 ----
        ax = axes[1, 1]
        ax.plot(np.arange(Q), score[order], color="#4b5563", lw=1.2,
                label="all queries (sorted by score)")
        if src.size:
            ax.scatter(rank[src], score[src], s=18, color=_C_POS, zorder=3,
                       edgecolors="#065f46", linewidths=0.4,
                       label=f"matched (positive, n={src.size})")
        ax.axvline(k, color="#ef4444", ls="--", lw=1.0, label=f"Top-{k} boundary")
        ax.set_xlabel("query rank (0 = highest score)", fontsize=9)
        ax.set_ylabel("max-class sigmoid score", fontsize=9)
        ax.set_title("positive samples' position in the score ranking", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")

        if src.size:
            mr = rank[src]
            ms = score[src]
            neg_topk = np.array([q for q in topk_idx.tolist() if q not in matched_set], dtype=np.int64)
            neg_med = f"{np.median(score[neg_topk]):.3f}" if neg_topk.size else "-"
            stats = (
                f"GT = {n_gt}   queries = {Q}   matched = {src.size}\n"
                f"matched inside Top-{k}: {int((mr < k).sum())} / {src.size}"
                f"  ({100.0 * (mr < k).mean():.1f}%)\n"
                f"matched rank: min {mr.min()}  median {int(np.median(mr))}  max {mr.max()}\n"
                f"matched score: min {ms.min():.3f}  median {np.median(ms):.3f}  max {ms.max():.3f}\n"
                f"Top-{k} unmatched score median: {neg_med}"
            )
        else:
            stats = f"GT = {n_gt}   queries = {Q}   matched = 0"
        ax.text(0.02, 0.03, stats, transform=ax.transAxes, fontsize=8, va="bottom",
                family="monospace",
                bbox=dict(boxstyle="round", fc="#f9fafb", ec="#d1d5db", alpha=0.92))

        fig.suptitle(f"{tag}  {stem}   (image {W}x{H}, padded {w_pad}x{h_pad})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.975))

        path = out_dir / f"{tag}_b{b}_{stem}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    return written


def maybe_visualize_matching(outputs, targets, indices, batched_inputs=None):
    """criterion 里的调用入口。开关见本文件顶部 ENABLED；异常只 warning，绝不打断训练。"""
    if not ENABLED or targets is None or indices is None:
        return

    _STATE["calls"] += 1
    call = _STATE["calls"]
    if (call - 1) % max(1, EVERY) != 0 or _STATE["dumped"] >= MAX_DUMP:
        return

    try:
        paths = visualize_matching(
            outputs,
            targets,
            indices,
            batched_inputs,
            out_dir=OUT_DIR,
            tag=f"call{call:06d}",
            topk=TOPK,
            num_images=NUM_IMAGES,
            image_format=IMAGE_FORMAT,
            draw_points=DRAW_POINTS,
            class_names=CLASS_NAMES,
        )
        _STATE["dumped"] += len(paths)
        for p in paths:
            logger.info("[vis_match] %s", p)
    except Exception:  # 可视化永远不该让训练挂掉
        logger.warning("[vis_match] visualization failed, skipped", exc_info=True)
