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
import torch.nn.functional as F
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

        # 定义一个 MLP
        self.relation_mlp = MLP(1, 16, 1, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
        """
        boxes: [bs, nq, 4], 格式 cx, cy, w, h (归一化到 [0,1])
        返回: [bs, nq, 4], xyxy 同样归一化
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

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
        output = tgt
        device = tgt.device

        intermediate = []
        # refpoints_unsigmoid (reference_points_unsigmoid) --> reference_points
        # 一共 10 组参考点，也就是 10 组预测框，N=9，初始参考点 + 9 次迭代更新参考点
        # 按照 decoder 逻辑，后面每一层 decoder 都会更新参考点，会更加精确一些
        # ref_points [reference_points, new_reference_points0, new_reference_points1, ..., new_reference_pointsN]

        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        layer_box_and_class = []
        layer_relation = None
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
            output = layer(
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
                layer_relation = layer_relation,
            )
            # refinement of reference points
            # reference_points (sigmoid) --> reference_before_sigmoid (logit) --> outputs_unsig (logit)--> new_reference_points (sigmoid)
            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach() # nq, bs, 4
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

                # 增加 class_scores
                decoder_norm = kwargs['decoder_norm']
                class_embed = kwargs['class_embed']
                decoder_output = decoder_norm(output) # nq, b, c
                decoder_output = decoder_output.transpose(0, 1) # b, nq, c
                class_logits = class_embed(decoder_output)
                class_scores = class_logits.sigmoid().detach()
                class_scores = class_scores.max(dim=-1)[0] # b, nq

                # 计算 R_rank 以及 R_spt
                from maskdino.utils import box_ops
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(reference_points.transpose(0, 1)) # b, nq, 4


                if len(layer_box_and_class) > 0:
                    previous_box, previous_class = layer_box_and_class[-1]
                    current_box, current_class = boxes_xyxy, class_scores
                    def build_relation(prev_box, current_box, prev_class, current_class):
                        # 计算 IoU
                        iou_list = []
                        for pbox, cbox in zip(prev_box, current_box):
                            # all(pbox <= 1.0), "box coords should be normalized"
                            iou, _ = box_ops.box_iou(pbox, cbox)
                            iou_list.append(iou)
                        relation_spt = iou = torch.stack(iou_list, dim=0)  # b, nq, nq
                        # 逐元素比较 class 大小
                        relation_rank = (current_class > prev_class).int() * 2 - 1
                        relation = relation_spt * relation_rank.unsqueeze(-1)
                        return relation
                    relation = build_relation(previous_box, current_box, previous_class, current_class)
                    layer_relation = self.relation_mlp(relation.unsqueeze(-1)).squeeze(-1).sigmoid()

                layer_box_and_class.append((boxes_xyxy, class_scores))

            intermediate.append(self.norm(output))

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
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
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
        # self attention

        if self.self_attn is not None:
        # debug = False
        # if debug:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            # tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            layer_relation = kwargs.get('layer_relation', None)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask, layer_relation=layer_relation)[0]



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

        return tgt

class MultiHeadAttention(nn.Module):
    """
    一个简单版本的 Multi-Head Attention，实现方式和 nn.MultiheadAttention 类似。

    输入输出形状：
        - q: (L, N, E)   L = target_len, N = batch_size, E = d_model
        - k: (S, N, E)   S = source_len
        - v: (S, N, E)
    返回：
        - attn_output: (L, N, E)
        - attn_weights:
            如果 average_attn_weights=True -> (N, L, S)
            否则 -> (N * num_heads, L, S)
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)

        # 分别对 q/k/v 做线性映射（也可以像官方一样合成一个 in_proj）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 输出 projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,                    # (L, N, E)
        k,                    # (S, N, E)
        v,                    # (S, N, E)
        attn_mask=None,       # (L, S) or (N*num_heads, L, S)
        key_padding_mask=None,# (N, S), True 表示要 mask 的位置
        need_weights=True,
        average_attn_weights=True,
        **kwargs
    ):
        L, N, E = q.shape
        S, N2, E2 = k.shape
        assert N == N2 and E == self.d_model and E2 == self.d_model

        # 1. 线性投影 + 拆成多头
        # q_proj: (L, N, E) -> (L, N, h, d_k) -> (N, h, L, d_k)
        q_proj = self.q_proj(q).view(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k_proj = self.k_proj(k).view(S, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v_proj = self.v_proj(v).view(S, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        # 现在形状：
        #   q_proj: (N, h, L, d_k)
        #   k_proj: (N, h, S, d_k)
        #   v_proj: (N, h, S, d_k)

        # 2. 计算注意力分数
        #   scores = q * k^T / sqrt(d_k)
        #   scores: (N, h, L, S)
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale

        # 引入 relation
        layer_relation = kwargs.get('layer_relation', None)
        if layer_relation is not None:
            scores = scores * layer_relation.unsqueeze(1)

        # 3. 处理 key_padding_mask: (N, S)，True 表示 padding，需要 mask 掉
        if key_padding_mask is not None:
            # key_padding_mask: (N, S) -> (N, 1, 1, S)，方便 broadcast 到 (N, h, L, S)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, S)
            scores = scores.masked_fill(mask, float('-inf'))

        # 4. 处理 attn_mask:
        #   - (L, S): 会 broadcast 到 (N, h, L, S)
        #   - (N*h, L, S): 直接 reshape 为 (N, h, L, S)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (L, S) -> (1, 1, L, S) -> broadcast
                attn_mask_ = attn_mask.unsqueeze(0).unsqueeze(0)
                scores = scores + attn_mask_
            elif attn_mask.dim() == 3:
                # (N*h, L, S) -> (N, h, L, S)
                attn_mask_ = attn_mask.view(N, self.num_heads, L, S)
                scores = scores + attn_mask_
            else:
                raise ValueError("attn_mask 应该是 (L, S) 或 (N*num_heads, L, S)")

        # 5. Softmax -> 注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (N, h, L, S)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. 加权求和得到输出
        #   (N, h, L, S) @ (N, h, S, d_k) -> (N, h, L, d_k)
        attn_output = torch.matmul(attn_weights, v_proj)  # (N, h, L, d_k)

        # 7. 拼回 (L, N, E)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  # (L, N, h, d_k)
        attn_output = attn_output.view(L, N, self.d_model)          # (L, N, E)

        # 8. 输出 projection
        attn_output = self.out_proj(attn_output)

        # 9. 处理返回的注意力权重格式
        if not need_weights:
            return attn_output, None

        # PyTorch 的 nn.MultiheadAttention:
        #   average_attn_weights=True  -> (N, L, S)
        #   average_attn_weights=False -> (N*h, L, S)
        if average_attn_weights:
            # 在 head 维度上取平均： (N, h, L, S) -> (N, L, S)
            attn_weights_agg = attn_weights.mean(dim=1)
        else:
            # 拼成 (N*h, L, S)
            attn_weights_agg = attn_weights.view(N * self.num_heads, L, S)

        return attn_output, attn_weights_agg


