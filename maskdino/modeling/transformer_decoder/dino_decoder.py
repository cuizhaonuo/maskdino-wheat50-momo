# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------

from typing import Optional, List, Union
import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast

from ...utils.utils import MLP, _get_clones, _get_activation_fn, gen_sineembed_for_position, inverse_sigmoid
from ..pixel_decoder.ops.modules import MSDeformAttn


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=True,
                 num_feature_levels=1,
                 deformable_decoder=True,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 rm_dec_query_scale=True,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None
        # 分组 decoder（POINT_GROUP_DECODER）的三组模块，由 MaskDINODecoder 在开关打开时挂上：
        # pt_layers（每层一个 PointGroupLayer）、pt_ref_head（小穗 token 的 query pos 头）、
        # pt_delta_embed（每层的小穗参考点细化头）。关着时都是 None，不新增任何参数。
        self.pt_layers = None
        self.pt_ref_head = None
        self.pt_delta_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] ==

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                # 分组 decoder（POINT_GROUP_DECODER）
                pt_tgt: Optional[Tensor] = None,
                pt_refpoints: Optional[Tensor] = None,
                group_offset: int = 0,
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
            - pt_tgt: ng, bs, K, d_model  小穗 token；None = 原行为
            - pt_refpoints: ng, bs, K, 2  小穗参考点（sigmoid 后，图像归一化绝对坐标）
            - group_offset: 带小穗组的 ng 个实例 query 在 tgt 里的起始下标（训练时 = DN 的 pad 数）
        Output:
            [hs, references]，pt_tgt 非 None 时再追加 [pt_hs, pt_references]
            （pt_hs: 每层 (bs, ng, K, d)；pt_references: 每层细化后的小穗点 (bs, ng, K, 2)，未 detach）
        """
        output = tgt
        device = tgt.device

        intermediate = []
        # refpoints_unsigmoid (reference_points_unsigmoid) --> reference_points
        # 一共 10 组参考点，也就是 10 组预测框，N=9，初始参考点 + 9 次迭代更新参考点
        # 按照 decoder 逻辑，后面每一层 decoder 都会更新参考点，会更加精确一些
        # ref_points [reference_points, new_reference_points0, new_reference_points1, ..., new_reference_pointsN]

        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        attn_output_weights_list = []
        # 分组 decoder 的小穗 token 状态
        pt_output = pt_tgt
        pt_ref = pt_refpoints.to(device) if pt_refpoints is not None else None
        pt_intermediate = []
        pt_ref_points = []
        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            # 该行代码此处其实不用考虑，前后数值没有变化，没有 padding 区域，valid_ratios 均为 1
            # (300, 1, 1, 4) * (1, 1, 3, 4) -> (300, 1, 3, 4)
            reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
            # 位置编码的生成
            # (300, 1, 4) --> (300, 1, 512)
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # nq, bs, 256*2
            # 位置编码经过 MLP 处理
            # (300, 1, 512) --> (300, 1, 256)
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # 分组 decoder ①：实例内 SA（小穗 token 与所属实例 token 交流），在原层之前
            if pt_output is not None:
                ng, _, n_pt, _ = pt_output.shape
                pt_ref_input = pt_ref[:, :, :, None] * valid_ratios[None, :, None]  # ng, bs, K, nlevel, 2
                pt_sine = gen_sineembed_for_position(pt_ref_input[:, :, :, 0].flatten(1, 2))  # ng, bs*K, 256
                pt_query_pos = self.pt_ref_head(pt_sine).view(ng, -1, n_pt, self.d_model)
                g0, g1 = group_offset, group_offset + ng
                ins, pt_output = self.pt_layers[layer_id].forward_within(
                    output[g0:g1], query_pos[g0:g1], pt_output, pt_query_pos)
                output = torch.cat([output[:g0], ins.to(output.dtype), output[g1:]], dim=0)

            # 输入变量，有几个变量不必要过多关注
            # tgt_query_sine_embed 此处没有被使用到，位置编码均为 tgt_query_pos
            # tgt_key_padding_mask 为 None, memory_key_padding_mask 全是 False，没有 padding 区域
            # memory_pos, cross_attn_mask 此处均为 None
            # ---
            # (300, 1, 256) --> (300, 1, 256)
            output, attn_output_weights = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed, # 没有被使用到
                tgt_key_padding_mask=tgt_key_padding_mask, # None
                tgt_reference_points=reference_points_input,

                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask, # 无 padding 区域，全是 False
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos, # None

                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask # None
            )
            attn_output_weights_list.append(attn_output_weights)

            # 分组 decoder ②：小穗 token 的跨实例 SA / cross-attn / FFN，再逐层细化小穗参考点
            # （与下面框的细化同一套路：inverse-sigmoid 空间加 Δ，送入下一层前 detach）
            if pt_output is not None:
                pt_output = self.pt_layers[layer_id].forward_point(
                    pt_output, pt_query_pos, pt_ref_input, memory, memory_key_padding_mask,
                    level_start_index, spatial_shapes)
                new_pt_ref = (inverse_sigmoid(pt_ref) + self.pt_delta_embed[layer_id](pt_output)).sigmoid()
                pt_ref_points.append(new_pt_ref)
                pt_intermediate.append(pt_output)
                pt_ref = new_pt_ref.detach()

            # refinement of reference points
            # reference_points (sigmoid) --> reference_before_sigmoid (logit) --> outputs_unsig (logit)--> new_reference_points (sigmoid)
            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        # 把 batch size 放到第 0 维度
        # (300, 1, 256) --> (1, 300, 256)
        # (300, 1, 4) --> (1, 300, 4)
        # intermediate: [(nq, bs, d_model), ...] --> 【(bs, nq, d_model)， ...]
        # ref_points: [(nq, bs, 4), ...] --> 【(bs, nq, 4)， ...]
        ret = [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]
        if pt_tgt is not None:
            ret += [
                [itm.transpose(0, 1) for itm in pt_intermediate],
                [itm.transpose(0, 1) for itm in pt_ref_points],
            ]
        return ret


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 key_aware_type=None,
                 ):
        super().__init__()

        # cross attention
        if use_deformable_box_attn:
            raise NotImplementedError
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @autocast(enabled=False)
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None,  # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                ):
        '''
        tgt: nq, bs, d_model
        tgt_query_pos: nq, bs, d_model
        tgt_query_sine_embed: nq, bs, d_model
        memory: hw, bs, d_model
        需要注意，tgt_query_pos 和 tgt_query_sine_embed 虽然都表示位置 embedding，但是不一样的，前者是 MLP(Sine(pos))，后者是 Sine(pos)
        '''
        '''
        self attention
        query = key = tgt + tgt_query_pos
        value = tgt
        '''
        # self attention
        attn_output_weights = None
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]

            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        '''
        # cross attention
        query = tgt + tgt_query_pos 
        reference_points = tgt_reference_points
        
        value = input_flatten = memory
        input_spatial_shapes = memory_spatial_shapes
        input_level_start_index = memory_level_start_index 
        input_padding_mask = memory_key_padding_mask # 全是 False，意思是没有 padding 区域
        '''
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                               memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, attn_output_weights


class PointGroupLayer(nn.Module):
    """GroupPose 式分组 decoder 的小穗 token 子层（POINT_GROUP_DECODER=True 时每层一个）。

    每个实例 token 挂 K 个小穗 token，组成 (N, B, 1+K, d) 的一组。一层 decoder 的执行顺序：
      1) forward_within：实例内 SA，同一组的 1+K 个 token 互相看（本类）；
      2) 原 DeformableTransformerDecoderLayer：实例 token 的跨实例 SA / cross-attn / FFN（一行不改）；
      3) forward_point：小穗 token 的跨实例 SA / cross-attn / FFN（本类）。
    实例内 SA 之后小穗 token 与实例 token 不再交互，所以 2)、3) 可以拆开执行；实例路径全部沿用
    原模块（同名同权重），小穗 token 用本类的独立权重，关掉分组 decoder 时模型逐位退回原样。

    与 GroupPose 源码（models/grouppose/transformer_deformable.py:89-119）的差异：
      - pos 用标准写法 q=k=x+pos、v=x。GroupPose 的 with_pos_embed 是原地 +=，pos 会混进 v 与残差流；
      - 实例 token 仍以 4-D 框为参考走原 cross_attn，小穗 token 以 2-D 点为参考走本类的 cross_attn；
      - 小穗 token 的各子层不与实例 token 共享权重（GroupPose 全部共享）；
      - 实例 token 接收实例内 SA 的更新时乘一个零初始化的标量门，开训时实例路径与原模型完全一致。
    """

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, across="slot", within_gate=True):
        super().__init__()
        # 实例内 SA（序列 = 同一实例的 1+K 个 token）
        self.within_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)
        # 实例 token 这一侧不做 LayerNorm，只乘门：门为 0 时实例 token 原样通过
        self.within_gate = nn.Parameter(torch.zeros(1)) if within_gate else None

        # 小穗 token 的跨实例 SA："slot" = 第 k 号 token 与其它实例的第 k 号交流（GroupPose 原样），
        # "none" = 不做，小穗 token 只经实例内 SA 间接获得跨实例信息
        assert across in ("slot", "none"), f"未知 POINT_ACROSS_ATTN: {across}"
        if across == "slot":
            self.across_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.across_dropout = nn.Dropout(dropout)
            self.across_norm = nn.LayerNorm(d_model)
        else:
            self.across_attn = None

        # 小穗 token 的可变形 cross-attn（2-D 参考点）
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 小穗 token 的 FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @autocast(enabled=False)
    def forward_within(self, ins, ins_pos, pts, pts_pos):
        """实例内 SA。
        Args:
            ins:     (N, B, d)     带小穗组的实例 token
            ins_pos: (N, B, d)     实例 token 的 query pos（原 ref_point_head 的输出）
            pts:     (N, B, K, d)  小穗 token
            pts_pos: (N, B, K, d)  小穗 token 的 query pos
        Returns:
            (ins, pts)，形状不变
        """
        ins, ins_pos, pts, pts_pos = ins.float(), ins_pos.float(), pts.float(), pts_pos.float()
        N, B, K, d = pts.shape
        grp = torch.cat([ins.unsqueeze(2), pts], dim=2).reshape(N * B, 1 + K, d)
        grp_pos = torch.cat([ins_pos.unsqueeze(2), pts_pos], dim=2).reshape(N * B, 1 + K, d)
        q = k = grp + grp_pos
        # need_weights=False 让 torch 走 SDPA（不物化注意力矩阵）
        out = self.within_attn(q, k, grp, need_weights=False)[0]
        out = self.within_dropout(out).reshape(N, B, 1 + K, d)
        gate = self.within_gate if self.within_gate is not None else 1.0
        ins = ins + gate * out[:, :, 0]
        pts = self.within_norm(pts + out[:, :, 1:])
        return ins, pts

    @autocast(enabled=False)
    def forward_point(self, pts, pts_pos, pts_ref, memory, memory_key_padding_mask,
                      memory_level_start_index, memory_spatial_shapes):
        """小穗 token 的跨实例 SA → 可变形 cross-attn → FFN。
        Args:
            pts:     (N, B, K, d)
            pts_pos: (N, B, K, d)
            pts_ref: (N, B, K, nlevel, 2)  参考点（已乘 valid_ratios）
            memory:  (hw, B, d)
        Returns:
            pts: (N, B, K, d)
        """
        pts, pts_pos = pts.float(), pts_pos.float()
        N, B, K, d = pts.shape
        if self.across_attn is not None:
            # 同 slot 跨实例：序列维 = N，批维 = B·K
            v = pts.permute(1, 2, 0, 3).reshape(B * K, N, d)
            q = k = (pts + pts_pos).permute(1, 2, 0, 3).reshape(B * K, N, d)
            out = self.across_attn(q, k, v, need_weights=False)[0]
            out = out.reshape(B, K, N, d).permute(2, 0, 1, 3)
            pts = self.across_norm(pts + self.across_dropout(out))

        query = (pts + pts_pos).transpose(0, 1).reshape(B, N * K, d)
        ref = pts_ref.transpose(0, 1).reshape(B, N * K, pts_ref.shape[-2], 2)
        out = self.cross_attn(query, ref.contiguous(), memory.transpose(0, 1),
                              memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask)
        out = out.reshape(B, N, K, d).transpose(0, 1)
        pts = self.norm1(pts + self.dropout1(out))

        out = self.linear2(self.dropout3(self.activation(self.linear1(pts))))
        pts = self.norm3(pts + self.dropout4(out))
        return pts


