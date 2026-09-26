"""小穗点的「按位置取特征」头（IFI）与辅助点引导（APG）。

来源：Chen et al., *Improving Point-based Crowd Counting and Localization Based on Auxiliary Point
Guidance*（APGCC, ECCV 2024, arXiv 2405.10589），本地源码 /public/home/zncui/pycharm_project/APGCC。

- IFI（论文 Eq.8）：任意位置 (x, y) 取特征图上最近的 4 个格 Z_i，连同相对偏移 δ_i 及其位置编码 φ(δ_i)
  过 MLP f_θ，再按对角面积 S_i/S（即双线性权重）加权求和。官方实现的局部加权循环写成了
  `x = ret + conte * area`，ret 从不累加，实际只用了最后一个邻格；这里按论文公式累加 4 个邻格。
- APG（论文 Eq.5/6）：训练时在每个 GT 点周围撒正/负辅助点，不经匈牙利匹配，直接送进同一个预测头。
  官方仓库里这一支是 `raise NotImplemented`，这里照论文写。

坐标约定（全文件统一）：
- 位置 pos 是 padded 图像张量上的归一化坐标 (x, y) ∈ [0,1]²，x 沿宽度，与 decoder 的 pred_boxes 同一坐标系。
- 像素按 detectron2 的连续约定：第 i 个像素覆盖 [i, i+1)。stride s 的特征图第 k 格中心在 s·k + s/2 像素，
  即归一化 (k + 0.5) / W_f（align_corners=False 语义）。
- 格点坐标 g = pos · [W_f, H_f] − 0.5，格中心落在整数上。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class SpatialEncoding(nn.Module):
    """APGCC 的 SpatialEncoding：x -> [x, sin(f·x), cos(f·x)]，f = 2^linspace(0, σ, n)。

    与官方实现相同的频率与输出维度（in_dim + pe_dim），但用逐元素乘代替 matmul，
    这样在 autocast 下仍保持 fp32；频率可学习（对应官方 require_grad=True）。
    """

    def __init__(self, in_dim=2, pe_dim=32, sigma=6.0, learnable=True):
        super().__init__()
        assert pe_dim % (2 * in_dim) == 0, "pe_dim must be divisible by 2*in_dim"
        n = pe_dim // 2 // in_dim
        freq = 2.0 ** torch.linspace(0, sigma, n)
        if learnable:
            self.freq = nn.Parameter(freq)
        else:
            self.register_buffer("freq", freq, persistent=False)
        self.in_dim = in_dim
        self.out_dim = in_dim + pe_dim

    def forward(self, x):
        x = x.float()
        y = (x[..., None] * self.freq.float()).flatten(-2)   # [..., in_dim * n]
        return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)


def ifi_neighbors(pos, H, W):
    """pos [N,2]（归一化 x,y）在 H×W 特征图上的 4 个邻格。

    Returns:
        ix, iy: [N,4] long，邻格列/行索引（已夹到图内）
        w:      [N,4] float32，双线性权重，非负且和为 1（= 论文 Eq.8 的 S_i/S）
        delta:  [N,4,2] float32，查询点相对各邻格中心的偏移（格为单位，x,y）
    邻格顺序：(x0,y0), (x1,y0), (x0,y1), (x1,y1)。越界的一侧两个邻格退化为同一格，
    与 grid_sample(padding_mode="border") 等价。
    """
    pos = pos.float()
    gx = (pos[:, 0] * W - 0.5).clamp(-0.5, W - 0.5)
    gy = (pos[:, 1] * H - 0.5).clamp(-0.5, H - 0.5)
    x0 = torch.floor(gx)
    y0 = torch.floor(gy)
    tx = gx - x0
    ty = gy - y0
    xs = (x0, x0 + 1)
    ys = (y0, y0 + 1)
    wxs = (1 - tx, tx)
    wys = (1 - ty, ty)
    ix, iy, w, delta = [], [], [], []
    for b in (0, 1):
        for a in (0, 1):
            cx = xs[a].clamp(0, W - 1)
            cy = ys[b].clamp(0, H - 1)
            ix.append(cx.long())
            iy.append(cy.long())
            w.append(wxs[a] * wys[b])
            delta.append(torch.stack([gx - cx, gy - cy], dim=-1))
    return (torch.stack(ix, 1), torch.stack(iy, 1),
            torch.stack(w, 1), torch.stack(delta, 1))


class IFISampler(nn.Module):
    """单层特征上的隐式特征插值：F(pos) = Σ_i w_i · f_θ(Z_i, φ(2δ_i))。"""

    def __init__(self, in_channels, feat_dim=64, hidden=128, pe_dim=32, pe_sigma=6.0):
        super().__init__()
        # 先把整张图 1×1 投影到 feat_dim 再 gather，省显存（官方 IFI 的 inner_planes=64）
        self.proj = nn.Sequential(nn.Conv2d(in_channels, feat_dim, kernel_size=1), nn.ReLU(inplace=True))
        self.pe = SpatialEncoding(2, pe_dim, pe_sigma)
        self.f = nn.Sequential(
            nn.Linear(feat_dim + self.pe.out_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.out_dim = hidden

    def project(self, feat):
        return self.proj(feat)

    def forward(self, proj, b_idx, pos):
        H, W = proj.shape[-2:]
        ix, iy, w, delta = ifi_neighbors(pos, H, W)
        out = 0.0
        for j in range(4):
            z = proj[b_idx, :, iy[:, j], ix[:, j]]            # [N, C]
            # 官方把相对坐标放在 [-1,1] 空间里再乘特征图边长，数值上等于 2δ（格为单位）
            h = self.f(torch.cat([z, self.pe(2.0 * delta[:, j])], dim=-1))
            out = out + w[:, j:j + 1] * h
        return out

    def sample_linear(self, feat, b_idx, pos):
        """Σ_i w_i · Z_i：不过 MLP 的双线性采样，仅供单元测试对照 grid_sample。"""
        H, W = feat.shape[-2:]
        ix, iy, w, _ = ifi_neighbors(pos, H, W)
        out = 0.0
        for j in range(4):
            out = out + w[:, j:j + 1] * feat[b_idx, :, iy[:, j], ix[:, j]]
        return out


class IFIPointHead(nn.Module):
    """给定位置（可选再给实例 query）-> (存在性 logit, 偏移 Δ)。

    多层 IFI 特征拼接（第 0 层是最细的 1/4 层，偏移以它的格为单位），可选拼上实例 query 的线性投影，
    过 2 层 MLP 后分出存在性与偏移两个输出。偏移输出零初始化：开训时 Δ ≡ 0。
    """

    def __init__(self, in_channels=(256, 256), feat_dim=64, ifi_hidden=128, pe_dim=32, pe_sigma=6.0,
                 q_dim=None, hidden=256, offset_scale_cells=1.0, cls_prior=None):
        super().__init__()
        self.samplers = nn.ModuleList(
            [IFISampler(c, feat_dim, ifi_hidden, pe_dim, pe_sigma) for c in in_channels])
        in_dim = ifi_hidden * len(in_channels)
        if q_dim is not None:
            self.q_proj = nn.Linear(q_dim, ifi_hidden)
            in_dim += ifi_hidden
        else:
            self.q_proj = None
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.cls_out = nn.Linear(hidden, 1)
        self.off_out = nn.Linear(hidden, 2)
        nn.init.zeros_(self.off_out.weight)
        nn.init.zeros_(self.off_out.bias)
        if cls_prior is not None:
            nn.init.constant_(self.cls_out.bias, -math.log((1 - cls_prior) / cls_prior))
        self.offset_scale_cells = offset_scale_cells

    def prepare(self, feats):
        """feats: 与 in_channels 一一对应的特征图列表（第 0 个最细）。每次前向调用一次。"""
        return [s.project(f) for s, f in zip(self.samplers, feats)]

    def forward(self, proj, b_idx, pos, qemb=None):
        """proj: prepare() 的结果；b_idx [N] long；pos [N,2] 归一化；qemb [N, q_dim] 或 None。
        Returns: logit [N] fp32，delta [N,2] fp32（归一化坐标下的偏移）。"""
        pos = pos.float()
        hs = [s(p, b_idx, pos) for s, p in zip(self.samplers, proj)]
        if self.q_proj is not None:
            hs.append(F.relu(self.q_proj(qemb)))
        h = self.trunk(torch.cat(hs, dim=-1))
        logit = self.cls_out(h).float().squeeze(-1)
        H0, W0 = proj[0].shape[-2:]
        delta = self.off_out(h).float() * self.offset_scale_cells / pos.new_tensor([W0, H0])
        return logit, delta


class PointHeadProbe:
    """给 criterion 用的延迟求值闭包：持有头、本次前向投影好的特征、（可选）实例 query。

    训练时 decoder 不预先算满 B×Q×P 个位置，而是把它放进输出字典，由 criterion 只对匹配上的实例和
    APG 辅助点求值。它不是 nn.Module，也不会进入 aux_outputs / interm_outputs / mask_dict。
    """

    def __init__(self, head, proj, qemb=None):
        self.head = head
        self.proj = proj
        self.qemb = qemb

    def __call__(self, b_idx, pos, q_idx=None):
        q = self.qemb[b_idx, q_idx] if self.qemb is not None else None
        return self.head(self.proj, b_idx, pos, q)


def pick_stride8(feats, mask_features):
    """从 decoder 输入的多尺度特征里按 shape 挑出 1/8 层（mask_features 是 1/4 层）。"""
    H4, W4 = mask_features.shape[-2:]
    cand = [f for f in feats if f.shape[-2] * 2 == H4 and f.shape[-1] * 2 == W4]
    assert len(cand) == 1, f"expect exactly one 1/8 level, got {[tuple(f.shape) for f in feats]}"
    return cand[0]


def dense_anchor_grid(H4, W4, device=None):
    """每个 1/4 格中心一个锚点，行优先，[H4*W4, 2] 归一化 (x, y)。
    与 APGCC「stride 8、每格 2×2 个锚点（8i+4±2 px）」的像素位置完全相同（都是 4k+2 px）。"""
    ys = (torch.arange(H4, device=device, dtype=torch.float32) + 0.5) / H4
    xs = (torch.arange(W4, device=device, dtype=torch.float32) + 0.5) / W4
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def instance_dbar_px(pts_px, min_nn=1.0):
    """一个穗内 GT 小穗的平均最近邻距离（像素）。Ni < 2 时返回 None。"""
    if pts_px.shape[0] < 2:
        return None
    d = torch.cdist(pts_px.float(), pts_px.float())
    d.fill_diagonal_(float("inf"))
    return d.min(dim=1).values.clamp(min=min_nn).mean()


def sample_apg(gt_px, dbar, img_wh, num=(2, 2), radii=(0.125, 0.5), neg_exclude=0.125, generator=None):
    """APG 辅助点（论文 §4.1，半径按每穗 d̄ 自适应）。

    Args:
        gt_px:  [Ni,2] 该穗 GT 小穗（像素，x,y）
        dbar:   该穗 d̄（像素，标量）
        img_wh: (W, H) 像素，辅助点落到图外即丢弃
        num:    (k_pos, k_neg)，每个 GT 点各撒几个
        radii:  (n_pos, n_neg) 以 d̄ 为单位。正点每轴 U(−n_pos, n_pos)；负点每轴 sign·U(n_pos, n_neg)
        neg_exclude: 负点离本穗任一 GT 小穗近于 neg_exclude·d̄ 即丢弃
        generator: CPU 上的 torch.Generator（与 DN 用的全局 RNG 隔离）
    Returns:
        pos_px [Kp,2], pos_near [Kp] long, neg_px [Kn,2], neg_near [Kn] long
        （*_near 是离该辅助点最近的 GT 下标，用作吸附目标）
    """
    device = gt_px.device
    gt = gt_px.detach().float().cpu()
    dbar = float(dbar)
    Ni = gt.shape[0]
    n_pos, n_neg = radii[0] * dbar, radii[1] * dbar
    k_pos, k_neg = num
    W, H = float(img_wh[0]), float(img_wh[1])

    def _inside(p):
        return (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)

    pos = gt.repeat_interleave(k_pos, 0) + (torch.rand(Ni * k_pos, 2, generator=generator) * 2 - 1) * n_pos
    mag = n_pos + torch.rand(Ni * k_neg, 2, generator=generator) * (n_neg - n_pos)
    sign = torch.randint(0, 2, (Ni * k_neg, 2), generator=generator).float() * 2 - 1
    neg = gt.repeat_interleave(k_neg, 0) + sign * mag

    pos = pos[_inside(pos)]
    neg = neg[_inside(neg)]
    if neg.shape[0] > 0:
        dn = torch.cdist(neg, gt)
        neg = neg[dn.min(dim=1).values >= neg_exclude * dbar]
    pos_near = torch.cdist(pos, gt).argmin(dim=1) if pos.shape[0] > 0 else torch.zeros(0, dtype=torch.long)
    neg_near = torch.cdist(neg, gt).argmin(dim=1) if neg.shape[0] > 0 else torch.zeros(0, dtype=torch.long)
    return pos.to(device), pos_near.to(device), neg.to(device), neg_near.to(device)
