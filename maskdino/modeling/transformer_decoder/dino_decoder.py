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

from torchvision.ops import nms
from maskdino.utils import box_ops

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
                 ddq_enabled=False,
                 o2m_matching_enabled=False,
                 num_queries=900,
                 num_queries_o2m=1350,
                 sep_branch_enabled=False
                 ):
        super().__init__()
        self.ddq_enabled = ddq_enabled
        self.o2m_matching_enabled = o2m_matching_enabled
        self.num_queries = num_queries
        self.num_queries_o2m = num_queries_o2m
        self.sep_branch_enabled = sep_branch_enabled

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
        output = tgt
        device = tgt.device

        pred_head = kwargs.get('pred_head', None)
        mode = kwargs.get('mode', None)

        intermediate = []
        # refpoints_unsigmoid (reference_points_unsigmoid) --> reference_points
        # 一共 10 组参考点，也就是 10 组预测框，N=9，初始参考点 + 9 次迭代更新参考点
        # 按照 decoder 逻辑，后面每一层 decoder 都会更新参考点，会更加精确一些
        # ref_points [reference_points, new_reference_points0, new_reference_points1, ..., new_reference_pointsN]

        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        distinct_masks = []
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
                mode=mode,
            )
            attn_output_weights_list.append(attn_output_weights)
            # refinement of reference points
            # reference_points (sigmoid) --> reference_before_sigmoid (logit) --> outputs_unsig (logit)--> new_reference_points (sigmoid)
            # iter update
            layer_boxes = None
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                layer_boxes = new_reference_points
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

            # layer-wise ddq
            decoder_output = pred_head.decoder_norm(self.norm(output))
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = pred_head.class_embed(decoder_output)
            outputs_class = outputs_class.transpose(0, 1)  # (nq, bs, num_classes)
            layer_classes = outputs_class


            if mode == 'main_branch' and self.ddq_enabled and (layer_boxes is not None):
                scores = layer_classes.sigmoid().max(-1)[0]  # (nq, bs)
                boxes = layer_boxes  # (nq, bs, 4)
                scores_b = scores.transpose(0, 1).contiguous()  # (bs, nq)
                boxes_b = boxes.transpose(0, 1).contiguous()  # (bs, nq, 4)

                # ====== dn + one2one (+ optional one2many) ======
                # 你自己已有的 one2one 数
                one2one_len = self.num_queries

                # one2many 是否存在由开关控制
                one2many_len = self.num_queries_o2m if self.training and self.o2m_matching_enabled else 0

                # dn_len 从总长度里扣出来
                dn_len = boxes_b.size(1) - one2one_len - one2many_len
                total_len = dn_len + one2one_len + one2many_len

                def dqs_keep_mask_dn_one2one_one2many(
                        boxes_cxcywh, scores_1d, dn_len, one2one_len, one2many_len, iou_thr=0.8
                ):
                    """
                    boxes_cxcywh: (total_len, 4)  顺序: dn | one2one | (optional) one2many
                    scores_1d:    (total_len,)
                    return:       (total_len,) bool mask，仅在 one2one 段上置 True(keep)，其余默认 False
                    """
                    total_len_local = dn_len + one2one_len + one2many_len
                    assert boxes_cxcywh.size(0) >= total_len_local, \
                        f"expect >= {total_len_local}, got {boxes_cxcywh.size(0)}"

                    # 只对 one2one 段做 NMS
                    o2o_s = dn_len
                    o2o_e = dn_len + one2one_len

                    one2one_boxes = boxes_cxcywh[o2o_s:o2o_e]  # (one2one_len, 4)
                    one2one_scores = scores_1d[o2o_s:o2o_e]  # (one2one_len,)

                    one2one_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(one2one_boxes)
                    keep_local = nms(one2one_boxes_xyxy, one2one_scores, iou_thr)  # [0, one2one_len)

                    # mask 是全长（dn+one2one(+one2many)），但只在 one2one 段标 True
                    mask = torch.zeros((total_len_local,), device=boxes_cxcywh.device, dtype=torch.bool)
                    mask[o2o_s + keep_local] = True
                    return mask

                # 逐 batch 计算 keep mask
                keep_mask = []
                for b in range(boxes_b.size(0)):
                    keep_mask_b = dqs_keep_mask_dn_one2one_one2many(
                        boxes_b[b], scores_b[b],
                        dn_len=dn_len, one2one_len=one2one_len, one2many_len=one2many_len,
                        iou_thr=0.8
                    )
                    keep_mask.append(keep_mask_b)
                keep_mask = torch.stack(keep_mask, dim=0)  # (bs, total_len)

                distinct_masks.append(keep_mask)

                # 目前只支持 bs=1
                if keep_mask.size(0) == 1:
                    bs1_mask = keep_mask[0]  # (total_len,)

                    o2o_s = dn_len
                    o2o_e = dn_len + one2one_len

                    # one2one 段内部 local index：[0, one2one_len)
                    idx_local = bs1_mask[o2o_s:o2o_e].nonzero(as_tuple=False).squeeze(1)

                    # one2one 的 self-attn mask：默认全 True(不可见)，保留的 idx 互相可见 -> False
                    attn_mask = torch.ones((one2one_len, one2one_len), device=device, dtype=torch.bool)
                    attn_mask[idx_local, :] = False
                    attn_mask[:, idx_local] = False

                    # 写回 tgt_mask 的 one2one block（注意有 dn 偏移）
                    if tgt_mask is not None:
                        tgt_mask[o2o_s:o2o_e, o2o_s:o2o_e] = attn_mask

                else:
                    raise ValueError(f"Batch size {keep_mask.size(0)} != 1")

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            distinct_masks
        ]


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 key_aware_type=None,
                 o2m_matching_enabled=False,
                 num_queries_o2m=1350,
                 ):
        super().__init__()

        self.o2m_matching_enabled = o2m_matching_enabled
        self.num_queries_o2m = num_queries_o2m

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

        # ffn sep
        self.sep_linear1 = nn.Linear(d_model, d_ffn)
        self.sep_activation = _get_activation_fn(activation)
        self.sep_dropout3 = nn.Dropout(dropout)
        self.sep_linear2 = nn.Linear(d_ffn, d_model)
        self.sep_dropout4 = nn.Dropout(dropout)
        self.sep_norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

        # attn score
        self.d_model = d_model
        self.query_embed = nn.Linear(self.d_model, self.d_model * 2)

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

    def forward_ffn_sep(self, tgt):
        tgt2 = self.sep_linear2(self.sep_dropout3(self.sep_activation(self.sep_linear1(tgt))))
        tgt = tgt + self.sep_dropout4(tgt2)
        tgt = self.sep_norm3(tgt)
        return tgt

    @autocast(enabled=False)
    def forward(self,
                # for tgt
                tgt: Tensor,  # nq, bs, d_model
                tgt_query_pos: Tensor,  # nq, bs, d_model
                tgt_query_sine_embed: Tensor,  # nq, bs, d_model (unused)
                tgt_key_padding_mask: Tensor,  # (unused)
                tgt_reference_points: Tensor,  # nq, bs, 4

                # for memory
                memory: Tensor,  # hw, bs, d_model
                memory_key_padding_mask: Tensor,
                memory_level_start_index: Tensor,  # num_levels
                memory_spatial_shapes: Tensor,  # bs, num_levels, 2
                memory_pos: Tensor,  # (unused)

                # sa
                self_attn_mask: Tensor,  # or None
                cross_attn_mask: Tensor,  # (unused)
                **kwargs
                ):
        """
        tgt: nq, bs, d_model
        tgt_query_pos: nq, bs, d_model
        tgt_query_sine_embed: nq, bs, d_model

        注意：
        - tgt_query_pos = MLP(Sine(pos))
        - tgt_query_sine_embed = Sine(pos)
        """

        # ---------------- helpers ----------------
        def _sa_block(x: Tensor, pos: Tensor, attn_mask: Tensor) -> Tensor:
            if self.self_attn is None:
                return x
            q = k = self.with_pos_embed(x, pos)
            x2 = self.self_attn(q, k, x, attn_mask=attn_mask)[0]
            x = x + self.dropout2(x2)
            x = self.norm2(x)
            return x

        def _apply_key_aware(x: Tensor, mem: Tensor) -> Tensor:
            if self.key_aware_type is None:
                return x
            if self.key_aware_type == "mean":
                return x + mem.mean(0, keepdim=True)
            if self.key_aware_type == "proj_mean":
                return x + self.key_aware_proj(mem).mean(0, keepdim=True)
            raise NotImplementedError(f"Unknown key_aware_type: {self.key_aware_type}")

        def _ca_block(x: Tensor,
                      pos: Tensor,
                      ref_pts: Tensor,
                      mem: Tensor,
                      spatial_shapes: Tensor,
                      level_start_index: Tensor,
                      mem_pad_mask: Tensor) -> Tensor:
            q = self.with_pos_embed(x, pos).transpose(0, 1)  # bs, nq, d
            rp = ref_pts.transpose(0, 1).contiguous()  # bs, nq, 4
            v = mem.transpose(0, 1)  # bs, hw, d

            x2 = self.cross_attn(
                q, rp, v,
                spatial_shapes,
                level_start_index,
                mem_pad_mask
            ).transpose(0, 1)  # nq, bs, d

            x = x + self.dropout1(x2)
            x = self.norm1(x)
            return x

        mode = kwargs.get('mode', None)

        if self.training and mode == 'sep_branch':
            tgt = _sa_block(tgt, tgt_query_pos, self_attn_mask)
            tgt = _apply_key_aware(tgt, memory)
            tgt = _ca_block(
                tgt, tgt_query_pos, tgt_reference_points,
                memory, memory_spatial_shapes, memory_level_start_index,
                memory_key_padding_mask
            )
            tgt = self.forward_ffn_sep(tgt)
            return tgt, None

        if self.training and self.o2m_matching_enabled and mode == 'main_branch':
            n_o2m = self.num_queries_o2m

            o2o_tgt = tgt[:-n_o2m]
            o2m_tgt = tgt[-n_o2m:]
            o2o_pos = tgt_query_pos[:-n_o2m]

            if self_attn_mask is None:
                o2o_mask = None
            else:
                o2o_mask = self_attn_mask[:-n_o2m, :-n_o2m]

            o2o_tgt = _sa_block(o2o_tgt, o2o_pos, o2o_mask)
            tgt = torch.cat([o2o_tgt, o2m_tgt], dim=0)

        else:
            tgt = _sa_block(tgt, tgt_query_pos, self_attn_mask)

        # -------- main cross-attn + ffn --------
        tgt = _apply_key_aware(tgt, memory)
        tgt = _ca_block(
            tgt, tgt_query_pos, tgt_reference_points,
            memory, memory_spatial_shapes, memory_level_start_index,
            memory_key_padding_mask
        )
        tgt = self.forward_ffn(tgt)

        return tgt, None




