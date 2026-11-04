
"""full/combo 论文用网络结构图（矢量输出：PDF / SVG / PNG）。
用法: python figures/fig_combo_architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import os

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.5,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

W, H = 140.0, 84.0
fig = plt.figure(figsize=(7.0, 4.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.canvas.draw()
RENDER = fig.canvas.get_renderer()
INV = ax.transData.inverted()

INK, MUTED = "#22262C", "#6B7280"
BASE_FC, BASE_EC = "#EDF1F6", "#78859A"
INN_FC, INN_EC = "#FBE3D0", "#C05621"
COR_FC, COR_EC = "#D9EBF4", "#2C7A9B"
OUT_FC, OUT_EC = "#E6F2E7", "#4B7F52"

def fit_text(x, y, s, max_w, size, **kw):
    """画文字，超宽就缩字号。"""
    t = ax.text(x, y, s, fontsize=size, **kw)
    for _ in range(30):
        bb = t.get_window_extent(RENDER)
        w = (INV.transform((bb.x1, bb.y1))[0] - INV.transform((bb.x0, bb.y0))[0])
        if w <= max_w or size <= 4.2:
            break
        size -= 0.2
        t.set_fontsize(size)
    return t

def box(x, y, w, h, title, sub=None, fc=BASE_FC, ec=BASE_EC, lw=0.9, ls="-",
        ts=6.4, fs=5.4, bold=True, r=1.2, tc=INK, pad=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=2))
    cw = w - pad
    if sub:
        fit_text(x + w/2, y + h*0.64, title, cw, ts, ha="center", va="center",
                 color=tc, zorder=3, fontweight="bold" if bold else "normal")
        fit_text(x + w/2, y + h*0.28, sub, cw, fs, ha="center", va="center",
                 color=MUTED, zorder=3)
    else:
        fit_text(x + w/2, y + h/2, title, cw, ts, ha="center", va="center",
                 color=tc, zorder=3, fontweight="bold" if bold else "normal")

def arrow(p, q, ls="-", lw=0.9, color=INK, rad=0.0, ms=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))

def badge(x, y, n):
    ax.add_patch(Circle((x, y), 1.4, facecolor=INN_EC, edgecolor="white",
                        linewidth=0.7, zorder=5))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=5.2,
            color="white", fontweight="bold", zorder=6)

# ---------------- 标题 ----------------
ax.text(1.5, 81.8, "Grouped Spikelet-Aware MaskDINO", fontsize=8.4,
        fontweight="bold", color=INK, ha="left", va="center")
ax.text(60.5, 81.8, "full/combo:  grouped decoder  +  NMS-matched points  +  CORAL counting",
        fontsize=5.8, color=MUTED, ha="left", va="center", style="italic")

# ---------------- 主干 ----------------
Y0, HH = 62.0, 14.0
box(1.5, Y0, 14.0, HH, "Input tile", "1024 x 1024")
box(18.0, Y0, 16.0, HH, "ResNet-50", "res3/4/5")
box(36.5, Y0, 18.5, HH, "Pixel Decoder", "MSDeformAttn")
box(57.5, Y0, 21.5, HH, "Transformer Encoder", "6 layers, 4 scales")
box(81.5, Y0, 18.5, HH, "Query Selection", "top-100 queries")
for x0, x1 in [(15.5, 18.0), (34.0, 36.5), (55.0, 57.5), (79.0, 81.5)]:
    arrow((x0, Y0 + HH/2), (x1, Y0 + HH/2))
ax.text(46.0, Y0 - 1.9, r"$F_3$-$F_5$ + mask features, $d$=256", ha="center",
        va="center", fontsize=5.2, color=MUTED, style="italic")

# ---------------- Decoder 面板 ----------------
PX, PY, PW, PH = 1.5, 37.5, 102.5, 20.5
ax.add_patch(FancyBboxPatch((PX, PY), PW, PH, boxstyle="round,pad=0,rounding_size=1.6",
                            facecolor="#FBFCFE", edgecolor="#B9C2CF",
                            linewidth=1.0, linestyle=(0, (3, 2)), zorder=0))
fit_text(PX + 1.5, PY + PH - 1.8, "Transformer Decoder  x 9 layers  (1 instance token + K spikelet tokens)",
         PW - 3.0, 6.8, ha="left", va="center",
         fontweight="bold", color=INK, zorder=3)
arrow((90.8, Y0), (90.8, PY + PH))

yA, yB, BH = PY + 11.4, PY + 2.1, 5.8
box(3.0, yA, 16.0, BH, "Instance token", "x 100, d=256")
box(21.5, yA, 21.0, BH, "Within-inst. SA", "1+K group, gated", fc=INN_FC, ec=INN_EC)
box(45.0, yA, 21.0, BH, "SA / Def. Cross-Attn", "box ref point")
box(68.5, yA, 13.0, BH, "FFN", r"$d_{ff}$=2048")
box(84.0, yA, 13.0, BH, "Box refine", "per layer")
for x0, x1 in [(19.0, 21.5), (42.5, 45.0), (66.0, 68.5), (81.5, 84.0)]:
    arrow((x0, yA + BH/2), (x1, yA + BH/2))

box(3.0, yB, 16.0, BH, "Spikelet token", "K=30, slot emb.")
box(21.5, yB, 21.0, BH, "Slot-wise SA", "across instances", fc=INN_FC, ec=INN_EC)
box(45.0, yB, 21.0, BH, "Point Cross-Attn", "2-D deformable")
box(68.5, yB, 13.0, BH, "FFN", r"$d_{ff}$=1024")
box(84.0, yB, 13.0, BH, "Point refine", "per layer")
for x0, x1 in [(19.0, 21.5), (42.5, 45.0), (66.0, 68.5), (81.5, 84.0)]:
    arrow((x0, yB + BH/2), (x1, yB + BH/2))

arrow((31.0, yA), (31.0, yB + BH), ls=(0, (2, 1.5)), lw=0.8, color=MUTED)
ax.text(31.8, (yA + yB + BH)/2, "grouped", fontsize=4.8, color=MUTED,
        ha="left", va="center", style="italic")

# ---------------- 预测头 ----------------
HX, HW = 107.5, 31.0
box(HX, 51.5, HW, 6.5, "Class / Box / Mask heads", "wheat vs. wheat-2")
box(HX, 43.5, HW, 6.5, "Point coordinate head", "P=30 x (x,y) offsets")
box(HX, 35.5, HW, 6.5, "Point existence head", "CORAL logits, T=30", fc=COR_FC, ec=COR_EC)

arrow((97.0, yA + BH/2), (HX, 54.5), rad=-0.15)
arrow((97.0, yB + BH/2), (HX, 47.0), rad=-0.1)
arrow((97.0, yB + 0.6), (HX, 38.5), rad=0.16)

# ---------------- 输出 ----------------
box(HX, 21.5, HW, 8.0, "Spikelet output", "points + count (n_k) + masks", fc=OUT_FC, ec=OUT_EC)
arrow((123.0, 35.5), (123.0, 29.5))

# ---------------- 四个创新点卡片 ----------------
CY, CH = 21.0, 12.2
cards = [
    ("1", "Grouped decoding",
     ["each query owns K=30", "spikelet tokens;", "gated within-instance SA"], INN_EC),
    ("2", "Layer-wise point NMS",
     ["greedy NMS r=kappa*sqrt(wh)", "before inner Hungarian;", "suppressed tokens: no loss"], INN_EC),
    ("3", "kNN point loss",
     ["slot layer keeps L1 +", "kNN(k=3) assignment", "cost, weight 5.0"], INN_EC),
    ("4", "CORAL counting",
     ["one score + ordered biases;", r"$n_k=\sum_k \sigma(l_k)>0.5$"], COR_EC),
]
CW, GAPX = 24.0, 1.6
for i, (n, title, lines, acc) in enumerate(cards):
    x = PX + i*(CW + GAPX)
    fc = COR_FC if n == "4" else "#FFFFFF"
    ec = COR_EC if n == "4" else INN_EC
    ax.add_patch(FancyBboxPatch((x, CY), CW, CH, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=2))
    badge(x + 1.9, CY + CH - 1.9, n)
    fit_text(x + 4.0, CY + CH - 1.9, title, CW - 5.5, 5.9, ha="left", va="center",
             fontweight="bold", color=INK, zorder=3)
    for j, ln in enumerate(lines):
        fit_text(x + 1.4, CY + CH - 5.6 - j*2.5, ln, CW - 2.4, 4.9,
                 ha="left", va="center", color=MUTED, zorder=3)

# ---------------- 训练 / 推理 ----------------
box(1.5, 3.0, 63.0, 13.0, "", None, fc="#FFFFFF", ec="#C9D1DC", lw=0.8)
ax.text(3.4, 13.4, "Training objectives", fontsize=6.6, fontweight="bold", color=INK, ha="left")
for i, t in enumerate(["mask / dice / box / GIoU", "point coords (iso, robust L1)",
                        "point existence (focal BCE)", "+ point NMS matching / kNN / CORAL"]):
    ax.text(3.4, 10.9 - i*2.15, "- " + t, fontsize=5.2, color=MUTED, ha="left", va="center")

box(66.0, 3.0, 72.5, 13.0, "", None, fc="#FFFFFF", ec="#C9D1DC", lw=0.8)
ax.text(67.9, 13.4, "Inference: coral decides how many, focal+NMS decides which",
        fontsize=6.0, fontweight="bold", color=INK, ha="left")
box(68.5, 4.3, 15.0, 6.6, "coral count", None, fc=COR_FC, ec=COR_EC, ts=5.4)
box(86.0, 4.3, 24.0, 6.6, "candidates", None, ts=5.4)
box(112.5, 4.3, 24.5, 6.6, "keep top-n_k", "points + count", fc=OUT_FC, ec=OUT_EC, ts=5.4, fs=5.0)
ax.text(98.0, 6.2, r"$s>0.35\ \wedge$ NMS $\kappa$=0.05", fontsize=5.2, ha="center", va="center", color=MUTED)
arrow((83.5, 7.6), (86.0, 7.6), lw=0.8)
arrow((110.0, 7.6), (112.5, 7.6), lw=0.8)

# ---------------- 图例 ----------------
for i, (fc, ec, label) in enumerate([(BASE_FC, BASE_EC, "baseline MaskDINO"),
                                     (INN_FC, INN_EC, "grouped-decoder / matching change"),
                                     (COR_FC, COR_EC, "CORAL counting"),
                                     (OUT_FC, OUT_EC, "output")]):
    ax.add_patch(Rectangle((1.5 + i*30.0, 0.25), 2.2, 1.0, facecolor=fc,
                           edgecolor=ec, linewidth=0.7))
    ax.text(4.3 + i*30.0, 0.75, label, fontsize=4.9, color=MUTED, va="center", ha="left")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper")
os.makedirs(out_dir, exist_ok=True)
for ext in ("pdf", "svg", "png"):
    fig.savefig(os.path.join(out_dir, f"fig_combo_architecture.{ext}"),
                dpi=400 if ext == "png" else None, facecolor="white")
print("written to", out_dir)
