# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------

from typing import Optional, List, Union, Tuple
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.nn.functional import grid_sample

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                **kwargs
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt # (nq, bs, d_model)
        device = tgt.device

        interm_outputs = kwargs['interm_outputs'] if 'interm_outputs' in kwargs else None
        interm_pred_masks = interm_outputs['pred_masks'] if (interm_outputs is not None
                                                             and 'pred_masks' in interm_outputs) else None
        maskdino_decoder = kwargs['maskdino_decoder'] if 'maskdino_decoder' in kwargs else None
        mask_features = kwargs['mask_features'] if 'mask_features' in kwargs else None

        intermediate = []
        # refpoints_unsigmoid (reference_points_unsigmoid) --> reference_points
        # 一共 10 组参考点，也就是 10 组预测框，N=9，初始参考点 + 9 次迭代更新参考点
        # 按照 decoder 逻辑，后面每一层 decoder 都会更新参考点，会更加精确一些
        # ref_points [reference_points, new_reference_points0, new_reference_points1, ..., new_reference_pointsN]

        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        # ============================
        initial_pred_boxes = reference_points.transpose(0, 1) # (bs, nq, 4)
        initial_pred_masks = interm_pred_masks # (bs, nq, h, w)

        def get_query_query_gating_matrix(pred_boxes, pred_masks, *, soft_mode=False):
            bs, nq, h, w = pred_masks.shape
            prob_masks = pred_masks.sigmoid() # (bs, nq, h, w)
            center = pred_boxes[..., :2] # (bs, nq, 2)
            grid = 2 * center - 1 # [0, 1] --> [-1, 1]
            grid_rep = grid.reshape(bs, nq, 1, 2).repeat(nq, 1, 1, 1) # (bs*nq, nq, 1, 2)
            prob_masks_as_batch = prob_masks.reshape(bs * nq, 1, h, w) # (bs*nq, 1, h, w)
            # input: pred_masks: (bs*nq, 1, h, w)
            # grid: center: (bs*nq, q, 1, 2)
            sample_mask = grid_sample(prob_masks_as_batch, grid_rep, align_corners=False) # (bs*nq, 1, nq, 1)
            g = sample_mask.squeeze(1).squeeze(-1).reshape(bs, nq, nq)

            if soft_mode:
                gate = g
                eps = 1e-6
                bias_qq = torch.log(eps + gate)
                return bias_qq
            bias_qq = (g > 0.5).float()
            return bias_qq

        query_query_gating_matrix = get_query_query_gating_matrix(
            pred_boxes=initial_pred_boxes, # (bs, nq, 4)
            pred_masks=initial_pred_masks, # (bs, nq, h, w)
            soft_mode=False,
        ) # (bs, nq, nq)
        # ============================

        attn_output_weights_list = []
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

            # 输入变量，有几个变量不必要过多关注
            # tgt_query_sine_embed 此处没有被使用到，位置编码均为 tgt_query_pos
            # tgt_key_padding_mask 为 None, memory_key_padding_mask 全是 False，没有 padding 区域
            # memory_pos, cross_attn_mask 此处均为 None
            # ---
            # (300, 1, 256) --> (300, 1, 256)
            from maskdino.ctx import current_iter_ctx
            current_iter = current_iter_ctx.get()

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
                cross_attn_mask=memory_mask, # None
                # kwargs
                query_query_gating_matrix=query_query_gating_matrix,
            )
            attn_output_weights_list.append(attn_output_weights)
            # refinement of reference points
            # reference_points (sigmoid) --> reference_before_sigmoid (logit) --> outputs_unsig (logit)--> new_reference_points (sigmoid)
            # iter update

            tmp_pred_boxes = None
            tmp_pred_masks = None
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)
                tmp_pred_boxes = new_reference_points.transpose(0, 1) # (bs, nq, 4)
            tmp_output = self.norm(output)
            _, tmp_pred_masks = maskdino_decoder.forward_prediction_heads(tmp_output.transpose(0, 1), mask_features, True)
            tmp_pred_masks = tmp_pred_masks.transpose(0, 1)
            query_query_gating_matrix = get_query_query_gating_matrix(tmp_pred_boxes, tmp_pred_masks, soft_mode=False)

            intermediate.append(self.norm(output))

        # debug: 对每一层的 attn_output_weights 进行分析
        # stacked_attn_output_weights = torch.stack(attn_output_weights_list, dim=0).squeeze() # (9, 300, 300)
        # from maskdino.ctx import stacked_attn_output_weights_ctx
        # stacked_attn_output_weights_ctx.set(stacked_attn_output_weights)

        # topk_attn_output_weights_indices = torch.topk(stacked_attn_output_weights, k=5, dim=-1)[1].transpose(0, 1) # (300, 9, 5)
        # selected_topk_attn_output_weights_indices = topk_attn_output_weights_indices[:50] # (50, 9, 5)
        # for item in selected_topk_attn_output_weights_indices:
        #     print(item.tolist())

        # for i, attn_output_weights in enumerate(stacked_attn_output_weights):
        #     topk_attn_output_weights = torch.topk(attn_output_weights, 5, dim=-1)[0]
        #     selected_topk_attn_output_weights = topk_attn_output_weights[:50]


        # 把 batch size 放到第 0 维度
        # (300, 1, 256) --> (1, 300, 256)
        # (300, 1, 4) --> (1, 300, 4)
        # intermediate: [(nq, bs, d_model), ...] --> 【(bs, nq, d_model)， ...]
        # ref_points: [(nq, bs, 4), ...] --> 【(bs, nq, 4)， ...]
        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


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
        self.n_heads = n_heads
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cus_self_attn = CustomMultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cus_self_attn.load_from_torch_mha(self.self_attn)

        # with torch.no_grad():
        #     B, L, C = 2, 100, d_model
        #     x = torch.randn(B, L, C, device=next(self.parameters()).device)
        #
        #     old = nn.MultiheadAttention(C, n_heads, dropout=dropout, batch_first=True).to(x.device)
        #     new = CustomMultiheadAttention(C, n_heads, dropout=dropout, batch_first=True).to(x.device)
        #     new.load_from_torch_mha(old)
        #
        #     y_old, _ = old(x, x, x, need_weights=False)
        #     y_new, _ = new(x, x, x, need_weights=False)
        #
        #     print((y_old - y_new).abs().max().item())

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
                **kwargs
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
        query_query_gating_matrix = kwargs.get('query_query_gating_matrix', None) # (bs, nq, nq)
        # self attention
        attn_output_weights = None
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            # 将 Matrix 转换为 bool 类型
            # if query_query_gating_matrix is not None:
            #     self_attn_mask = (query_query_gating_matrix == 0.0) # (bs, nq, nq)
            #     self_attn_mask = self_attn_mask.repeat(self.n_heads, 1, 1) # (bs*nhead, nq, nq)
            # tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            # ============================
            # 以下属于能跑就行，形状正确即可
            tgt2 = self.cus_self_attn(q.transpose(0, 1),
                                  k.transpose(0, 1),
                                  tgt.transpose(0, 1),
                                  attn_mask=self_attn_mask,
                                  query_query_gating_matrix=query_query_gating_matrix
                                  )[0]
            tgt2 = tgt2.transpose(0, 1)
            # ============================
            # (nq, bs, d_model), (bs, nq, nq)
            # tgt2, attn_output_weights = self.self_attn(q, k, tgt, attn_mask=self_attn_mask, average_attn_weights=True)

            # debug: 可视化一个 num_queries x num_queries 的 attention mask 矩阵
            # from maskdino.analysis.tool_lab import vis_attention_matrix
            # vis_attention_matrix(attn_output_weights[0])
            # from maskdino.ctx import attn_output_weights_ctx
            # attn_output_weights_ctx.set(attn_output_weights)

            # debug: 可视化 attention weights
            # tgt2, attn_output_weights = self.self_attn(q, k, tgt, attn_mask=self_attn_mask, average_attn_weights=False)
            # from maskdino.analysis.tool_lab import visualize_attn_weights
            # visualize_attn_weights(attn_output_weights[0], average_attn_weights=False)

            # debug: 可视化每个 head 的最大 attention 权重对应的索引
            # from maskdino.analysis.tool_lab import visualize_attn_indices
            # visualize_attn_indices(attn_output_weights[0])
            # from maskdino.analysis.tool_lab import visualize_attn_indices_heatmap
            # visualize_attn_indices_heatmap(attn_output_weights[0])

            # debug: 将 attn_indices 保存到 csv 文件中，一行一行写入
            # from maskdino.analysis.tool_lab import save_attn_indices_to_csv
            # save_attn_indices_to_csv(attn_output_weights[0])

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




class CustomMultiheadAttention(nn.Module):
    """
    A customizable Multi-Head Attention module (batch_first).

    Key customization points:
      - You can inject extra_bias into attention logits (e.g., gating/decay matrix).
      - You can override how logits are computed (hook functions).
      - You can return per-head attention weights.

    Args:
      d_model: embedding dim C
      n_heads: number of heads H
      dropout: dropout on attention probabilities
      bias: whether to use bias in linear layers
      batch_first: expects inputs [B, L, C] if True
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # You can swap these projections with your own (e.g., gated linear).
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)

        # 增加 beta 可学习参数
        self.beta = nn.Parameter(torch.zeros(n_heads))  # shape: (n_heads, 1, 1)

        # Optional: if you want to keep same init style as common practice
        self._reset_parameters()

    def _reset_parameters(self):
        # A simple init; you can replace with xavier_uniform_ etc.
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, L, C] -> [B, H, L, D]
        """
        B, L, C = x.shape
        x = x.view(B, L, self.n_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    @torch.no_grad()
    def load_from_torch_mha(self, mha: nn.MultiheadAttention):
        """
        将 nn.MultiheadAttention 的权重拷贝到 CustomMultiheadAttention
        要求: mha.embed_dim == self.d_model 且 mha.num_heads == self.n_heads
        """
        assert mha.embed_dim == self.d_model
        assert mha.num_heads == self.n_heads

        d = self.d_model

        # 1) 拆 in_proj_weight: [3d, d] -> q/k/v: [d, d]
        W = mha.in_proj_weight
        self.q_proj.weight.copy_(W[0:d, :])
        self.k_proj.weight.copy_(W[d:2 * d, :])
        self.v_proj.weight.copy_(W[2 * d:3 * d, :])

        # 2) 拆 in_proj_bias: [3d] -> [d]
        if mha.in_proj_bias is not None:
            b = mha.in_proj_bias
            self.q_proj.bias.copy_(b[0:d])
            self.k_proj.bias.copy_(b[d:2 * d])
            self.v_proj.bias.copy_(b[2 * d:3 * d])
        else:
            # 你的 q/k/v 可能有 bias=True，这时建议置零，避免随机 bias 影响
            if self.q_proj.bias is not None:
                self.q_proj.bias.zero_()
                self.k_proj.bias.zero_()
                self.v_proj.bias.zero_()

        # 3) out_proj 直接拷贝
        self.out_proj.weight.copy_(mha.out_proj.weight)
        if mha.out_proj.bias is not None and self.out_proj.bias is not None:
            self.out_proj.bias.copy_(mha.out_proj.bias)

        # 4) 新增参数 beta：保持 0（等价于“不影响原模型”）
        if hasattr(self, "beta") and self.beta is not None:
            self.beta.zero_()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        extra_bias: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inputs:
          query/key/value: if batch_first=True -> [B, Lq/Lk, C]
          key_padding_mask: [B, Lk] with True for positions to mask (padding)
          attn_mask:
              - [Lq, Lk] or
              - [B, Lq, Lk] or
              - [B, H, Lq, Lk]
            where True means masked OR float mask with -inf for masked positions.
          extra_bias: custom additive bias to logits, same broadcast rules as attn_mask.
              Typical: [B, H, Lq, Lk] (e.g., query-query gating/decay)
          need_weights: return attention weights
          average_attn_weights: if True, return [B, Lq, Lk]; else [B, H, Lq, Lk]

        Returns:
          out: [B, Lq, C]
          attn_weights (optional): [B, H, Lq, Lk] or averaged [B, Lq, Lk]
        """
        query_query_gating_matrix = kwargs.get('query_query_gating_matrix', None)

        if not self.batch_first:
            # If you want, you can extend to [L, B, C], but for DETR-like code batch_first is easier.
            raise NotImplementedError("This implementation assumes batch_first=True.")

        B, Lq, C = query.shape
        _, Lk, _ = key.shape

        # 1) Project
        q = self.q_proj(query)  # [B, Lq, C]
        k = self.k_proj(key)    # [B, Lk, C]
        v = self.v_proj(value)  # [B, Lk, C]

        # 2) Split heads
        q = self._shape(q)  # [B, H, Lq, D]
        k = self._shape(k)  # [B, H, Lk, D]
        v = self._shape(v)  # [B, H, Lk, D]

        # 3) Scaled dot-product logits
        # logits: [B, H, Lq, Lk]
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 4) Apply masks / bias (all are additive on logits)
        # key_padding_mask: True means mask out => add -inf
        if key_padding_mask is not None:
            # [B, 1, 1, Lk]
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            logits = logits.masked_fill(mask, float("-inf"))

        if attn_mask is not None:
            logits = self._apply_attn_mask(logits, attn_mask)

        if extra_bias is not None:
            logits = logits + self._broadcast_bias(extra_bias, B, Lq, Lk)

        if query_query_gating_matrix is not None:
            # query_query_gating_matrix: [B, Lq, Lk]
            gating_matrix = query_query_gating_matrix.unsqueeze(1)  # [B, 1, Lq, Lk]
            # logits = logits * gating_matrix # hard gating
            # logits = logits + self.beta[None, :, None, None] * gating_matrix # soft gating
            logits = logits + gating_matrix # soft gating

        # 5) Softmax -> probs
        attn = F.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)

        # 6) Weighted sum
        out = torch.matmul(attn, v)  # [B, H, Lq, D]

        # 7) Merge heads
        out = out.transpose(1, 2).contiguous().view(B, Lq, C)  # [B, Lq, C]
        out = self.out_proj(out)

        if need_weights:
            if average_attn_weights:
                return out, attn.mean(dim=1)  # [B, Lq, Lk]
            return out, attn  # [B, H, Lq, Lk]
        return out, None

    @staticmethod
    def _apply_attn_mask(logits: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, H, Lq, Lk]
        attn_mask can be:
          - bool mask (True masked)
          - float mask (additive, e.g., 0 or -inf)
          shapes: [Lq, Lk] or [B, Lq, Lk] or [B, H, Lq, Lk]
        """
        if attn_mask.dtype == torch.bool:
            m = attn_mask
            if m.dim() == 2:
                m = m[None, None, :, :]
            elif m.dim() == 3:
                m = m[:, None, :, :]
            elif m.dim() == 4:
                pass
            else:
                raise ValueError(f"attn_mask dim {m.dim()} not supported")
            return logits.masked_fill(m, float("-inf"))
        else:
            # additive float mask
            if attn_mask.dim() == 2:
                m = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                m = attn_mask[:, None, :, :]
            elif attn_mask.dim() == 4:
                m = attn_mask
            else:
                raise ValueError(f"attn_mask dim {attn_mask.dim()} not supported")
            return logits + m

    @staticmethod
    def _broadcast_bias(bias: torch.Tensor, B: int, Lq: int, Lk: int) -> torch.Tensor:
        """
        Make extra_bias broadcastable to [B, H, Lq, Lk].
        Commonly you will pass [B, H, Lq, Lk] directly.
        """
        if bias.dim() == 2:
            # [Lq, Lk]
            return bias[None, None, :, :]
        if bias.dim() == 3:
            # [B, Lq, Lk]
            return bias[:, None, :, :]
        if bias.dim() == 4:
            # [B, H, Lq, Lk]
            return bias
        raise ValueError(f"extra_bias dim {bias.dim()} not supported")

