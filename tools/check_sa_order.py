# -*- coding: utf-8 -*-
"""full/sa-order 的回归探针：用模块级小模型对比「原 ①②③ 链路」与「统一 CA 链路」。

两种用法：
  1) 逐位回归（POINT_UNIFIED_CA=False 的新代码 vs git stash 后的原版代码）
       # 原版代码（先 git stash）
       python tools/check_sa_order.py --dump /tmp/ref.pt
       # 新代码（git stash pop 后）
       python tools/check_sa_order.py --compare /tmp/ref.pt
  2) 统一链路自检（6 种 SA 顺序各前向一次，看形状 / NaN）
       python tools/check_sa_order.py --orders

探针刻意复刻 MaskDINODecoder.__init__ 里挂 pt_layers / pt_ref_head / pt_delta_embed 的那几行
（含 xavier 与 MSDeformAttn._reset_parameters 的二次初始化），这样两种代码在同一 seed 下的
初始权重逐位相同；比较的是 TransformerDecoder.forward 的四组输出（hs / references /
pt_hs / pt_refs），即 PointGroupLayer 参与的那条链路。
"""
import argparse
import copy
import inspect
import itertools
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskdino.modeling.transformer_decoder.dino_decoder import (  # noqa: E402
    TransformerDecoder, DeformableTransformerDecoderLayer, PointGroupLayer)
from maskdino.modeling.pixel_decoder.ops.modules import MSDeformAttn  # noqa: E402
from maskdino.utils.utils import MLP, _get_clones  # noqa: E402

# 必须 256：gen_sineembed_for_position 的宽度写死为 256（2-D）/ 512（4-D），
# 与 ref_point_head / pt_ref_head 的输入维绑死。
D = 256           # hidden_dim
NQ = 6            # 带小穗组的实例 query 数（N）
K = 4             # 每个实例的小穗 token 数
B = 2
DN = 4            # DN pad（tgt_mask 非 None，走 DN 分支）
NLEVEL = 2
SHAPES = [(20, 20), (10, 10)]
HW = sum(h * w for h, w in SHAPES)
NLAYER = 2
SEED = 1234


def build(order=None):
    """order=None → 原链路（pt_sa_order=None）；否则统一链路 + 给定顺序。"""
    torch.manual_seed(SEED)
    layer = DeformableTransformerDecoderLayer(D, 2 * D, 0.0, "relu", NLEVEL, 8, 4)
    dec = TransformerDecoder(layer, NLAYER, nn.LayerNorm(D), return_intermediate=True,
                            d_model=D, query_dim=4, num_feature_levels=NLEVEL)
    bbox = MLP(D, D, 4, 3)
    dec.bbox_embed = nn.ModuleList([copy.deepcopy(bbox) for _ in range(NLAYER)])
    # 与 MaskDINODecoder 相同的挂载 / 二次初始化（unified_ca 只在 full/sa-order 及之后存在，
    # 这样同一份探针能在 git stash 前后的两份代码上跑）
    pt_kw = {}
    if "unified_ca" in inspect.signature(PointGroupLayer.__init__).parameters:
        pt_kw["unified_ca"] = order is not None
    pt_layer = PointGroupLayer(D, 2 * D, 0.0, "relu", NLEVEL, 8, 4,
                              across="slot", within_gate=True, **pt_kw)
    dec.pt_layers = _get_clones(pt_layer, NLAYER)
    dec.pt_ref_head = MLP(D, D, D, 2)
    for p in list(dec.pt_layers.parameters()) + list(dec.pt_ref_head.parameters()):
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for m in dec.pt_layers.modules():
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()
    ptd = MLP(D, D, 2, 3)
    nn.init.constant_(ptd.layers[-1].weight.data, 0)
    nn.init.constant_(ptd.layers[-1].bias.data, 0)
    dec.pt_delta_embed = nn.ModuleList([copy.deepcopy(ptd) for _ in range(NLAYER)])
    dec.pt_sa_order = tuple(order) if order is not None else None
    dec.eval()
    return dec


def make_inputs(device, dn=DN):
    """dn=0 复刻推理（无 DN：group_offset=0、tgt_mask=None），dn>0 复刻训练。"""
    g = torch.Generator().manual_seed(7)
    n_q = dn + NQ
    tgt = torch.randn(n_q, B, D, generator=g)
    ref_logit = torch.randn(n_q, B, 4, generator=g) * 2.0
    valid_ratios = torch.ones(B, NLEVEL, 2)
    memory = torch.randn(HW, B, D, generator=g)
    spatial_shapes = torch.tensor(SHAPES)
    level_start_index = torch.tensor([0] + list(torch.tensor(SHAPES).prod(1).cumsum(0)[:-1]))
    pt_tgt = torch.randn(NQ, B, K, D, generator=g)
    pt_ref = torch.rand(NQ, B, K, 2, generator=g)
    # DN 段：只看得到自己；matching 段：彼此全可见（与 prepare_for_dn 的 mask 同构）
    if dn > 0:
        tgt_mask = torch.zeros(n_q, n_q, dtype=torch.bool)
        tgt_mask[:dn, :] = True
        for i in range(dn):
            tgt_mask[i, i] = False
    else:
        tgt_mask = None
    out = dict(tgt=tgt, refpoints_unsigmoid=ref_logit, valid_ratios=valid_ratios,
               memory=memory, spatial_shapes=spatial_shapes, level_start_index=level_start_index,
               pt_tgt=pt_tgt, pt_refpoints=pt_ref, tgt_mask=tgt_mask, group_offset=dn)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}


@torch.no_grad()
def run(dec, inputs):
    ret = dec(**inputs)
    hs, refs, pt_hs, pt_refs = ret
    return dict(hs=torch.stack(hs), refs=torch.stack(refs),
                pt_hs=torch.stack(pt_hs), pt_refs=torch.stack(pt_refs))


def to_dev(x, device):
    return {k: v.to(device) for k, v in x.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--compare")
    ap.add_argument("--orders", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    if args.dump:
        dec = build(None).to(dev)
        out = to_dev(run(dec, make_inputs(dev)), "cpu")
        torch.save(out, args.dump)
        print(f"[dump] {args.dump}  " + "  ".join(f"{k}{tuple(v.shape)}" for k, v in out.items()))
        return

    if args.compare:
        ref = torch.load(args.compare)
        dec = build(None).to(dev)
        out = to_dev(run(dec, make_inputs(dev)), "cpu")
        worst = 0.0
        for k in ("hs", "refs", "pt_hs", "pt_refs"):
            d = (out[k].double() - ref[k].double()).abs().max().item()
            worst = max(worst, d)
            print(f"[compare] {k:8s} max|Δ| = {d:.3e}   shape {tuple(out[k].shape)}")
        print(f"[compare] WORST max|Δ| = {worst:.3e}   ({'PASS' if worst < 1e-6 else 'FAIL'} < 1e-6)")
        return

    if args.orders:
        for dn in (DN, 0):
            tag = "train(DN)" if dn else "infer(no DN)"
            for order in itertools.permutations(("group", "ww", "ss")):
                dec = build(order).to(dev)
                out = run(dec, make_inputs(dev, dn))
                fin = all(torch.isfinite(v).all().item() for v in out.values())
                print(f"[order] {tag:14s} {'-'.join(order):18s} hs{tuple(out['hs'].shape)} "
                      f"pt_hs{tuple(out['pt_hs'].shape)} pt_refs{tuple(out['pt_refs'].shape)} "
                      f"finite={fin}")
        return

    ap.error("需要 --dump / --compare / --orders 之一")


if __name__ == "__main__":
    main()
