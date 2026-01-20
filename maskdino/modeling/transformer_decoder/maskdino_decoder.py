# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks

from .dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from ...utils.utils import MLP, gen_encoder_output_proposals, inverse_sigmoid
from ...utils import box_ops

from torchvision.ops import nms

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskDINO.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


@TRANSFORMER_DECODER_REGISTRY.register()
class MaskDINODecoder(nn.Module):
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            # num_queries_o2m: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            enforce_input_project: bool,
            two_stage: bool,
            dn: str,
            noise_scale:float,
            dn_num:int,
            initialize_box_type:bool,
            initial_pred:bool,
            learn_tgt: bool,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 4,
            dec_layer_share: bool = False,
            semantic_ce_loss: bool = False,

            num_queries_o2m: int = 1350,
            o2m_matching_enabled: bool = True,
            num_targets_k: int = 6,
            ddq_enabled: bool = True,
            ddq_iou_threshold: float = 0.8,
            layer_wise_iou_enabled: bool = False,
            align_loss_enabled: bool = False,
            sep_branch_enabled: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.o2m_matching_enabled = o2m_matching_enabled
        self.num_queries_o2m = num_queries_o2m
        self.num_targets_k = num_targets_k
        self.ddq_enabled = ddq_enabled
        self.ddq_iou_threshold = ddq_iou_threshold
        self.layer_wise_iou_enabled = layer_wise_iou_enabled
        self.align_loss_enabled = align_loss_enabled
        self.sep_branch_enabled = sep_branch_enabled

        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.dn=dn
        self.learn_tgt = learn_tgt
        self.noise_scale=noise_scale
        self.dn_num=dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage=two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        # learnable query features
        if not two_stage or self.learn_tgt:
            num_queries = self.num_queries if not self.o2m_matching_enabled else self.num_queries + self.num_queries_o2m
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(self.num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
            self.one2many_enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.one2many_enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes=num_classes
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        if self.mask_classification:
            if self.semantic_ce_loss:
                self.class_embed = nn.Linear(hidden_dim, num_classes+1)
                self.one2many_class_embed = nn.Linear(hidden_dim, num_classes+1)
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
                self.one2many_class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc=nn.Embedding(num_classes,hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.iou_head = MLP(hidden_dim, hidden_dim, 1, 1)
        self.iou_head_before_norm = nn.LayerNorm(hidden_dim)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points,
                                                          o2m_matching_enabled=self.o2m_matching_enabled,
                                                          num_queries_o2m=self.num_queries_o2m,)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          ddq_enabled = self.ddq_enabled,
                                          o2m_matching_enabled = self.o2m_matching_enabled,
                                          num_queries = self.num_queries,
                                          num_queries_o2m = self.num_queries_o2m,
                                          sep_branch_enabled = self.sep_branch_enabled,
                                          )

        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self._one2many_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MaskDINO.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MaskDINO.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MaskDINO.DIM_FEEDFORWARD
        ret["dec_layers"] = cfg.MODEL.MaskDINO.DEC_LAYERS
        ret["enforce_input_project"] = cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["two_stage"] =cfg.MODEL.MaskDINO.TWO_STAGE
        ret["initialize_box_type"] = cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE  # ['no', 'bitmask', 'mask2box']
        ret["dn"]=cfg.MODEL.MaskDINO.DN
        ret["noise_scale"] =cfg.MODEL.MaskDINO.DN_NOISE_SCALE
        ret["dn_num"] =cfg.MODEL.MaskDINO.DN_NUM
        ret["initial_pred"] =cfg.MODEL.MaskDINO.INITIAL_PRED
        ret["learn_tgt"] = cfg.MODEL.MaskDINO.LEARN_TGT
        ret["total_num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS
        ret["semantic_ce_loss"] = cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and ~cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON

        ret["num_queries_o2m"] = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES_O2M
        ret["o2m_matching_enabled"] = cfg.MODEL.MaskDINO.O2M_MATCHING_ENABLED
        ret["num_targets_k"] = cfg.MODEL.MaskDINO.NUM_TARGETS_K
        ret["ddq_enabled"] = cfg.MODEL.MaskDINO.DDQ_ENABLED
        ret["ddq_iou_threshold"] = cfg.MODEL.MaskDINO.DDQ_IOU_THRESHOLD
        ret["layer_wise_iou_enabled"] = cfg.MODEL.MaskDINO.LAYER_WISE_IOU_ENABLED
        ret["align_loss_enabled"] = cfg.MODEL.MaskDINO.ALIGN_LOSS_ENABLED
        ret["sep_branch_enabled"] = cfg.MODEL.MaskDINO.SEP_BRANCH_ENABLED
        return ret

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//(int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of bbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 4).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label=padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_bbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label,input_query_bbox,attn_mask,mask_dict

    def dn_post_process(self, outputs_class, outputs_coord, outputs_mask, outputs_iou, mask_dict):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]

        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {
            'pred_logits': output_known_class[-1],
            'pred_boxes': output_known_coord[-1],
            'pred_masks': output_known_mask[-1],
        }
        outputs_known_iou = None
        if len(outputs_iou) > 0 and outputs_iou is not None:
            outputs_known_iou = outputs_iou[:, :, :mask_dict['pad_size'], :]
            outputs_iou = outputs_iou[:, :, mask_dict['pad_size']:, :]
            out.update({'pred_logits_iou': outputs_known_iou[-1]})

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask, outputs_known_iou, output_known_coord)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_coord, outputs_mask, outputs_iou

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        device = reference[0].device

        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0.to(device)]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, mask_features, masks, targets=None):
        """
        :param x: input, a list of multi-scale feature
        :param mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
        obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
        :param masks: mask in the original image
        :param targets: used for denoising training
        """

        assert len(x) == self.num_feature_levels
        device = x[0].device
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = [] # h_idx x w_idx，逆序列增加
        mask_flatten = [] # h_i x w_i，正序列增加，此处可以不考虑该变量的作用，因为 mask 全部是 False，不影响结果
        spatial_shapes = [] # (h_idx, w_idx)，逆序列增加
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        predictions_iou = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                src_flatten, mask_flatten, spatial_shapes
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            bs = output_memory.shape[0]
            topk_o2o = self.num_queries

            # 初始化
            enc_outputs_class_o2o = self.class_embed(output_memory)
            enc_outputs_coord_o2o = self._bbox_embed(output_memory) + output_proposals  # (bs, sum(hw), 4) unsigmoid

            use_o2m = bool(self.training and self.o2m_matching_enabled)
            if use_o2m:
                topk_o2m = self.num_queries_o2m
                enc_outputs_class_o2m = self.one2many_class_embed(output_memory)
                enc_outputs_coord_o2m = self._one2many_bbox_embed(output_memory) + output_proposals  # unsigmoid

            def padding_to(x, max_len):
                if x is None:
                    return None
                pad = max_len - x.shape[0]
                if pad <= 0:
                    return x[:max_len]
                if x.dim() > 1:
                    padding = x.new_zeros(pad, *x.size()[1:], dtype=x.dtype)
                else:
                    padding = x.new_zeros(pad, dtype=x.dtype)
                return torch.cat([x, padding], dim=0)

            def align_tensor(list_of_tensors, max_len):
                return torch.stack([padding_to(t, max_len) for t in list_of_tensors], dim=0)

            if self.ddq_enabled:
                # ddq ONLY for o2o
                iou_thr = self.ddq_iou_threshold

                o2o_scores_list, o2o_props_list, o2o_queries_list = [], [], []
                for img_id in range(bs):
                    scores = enc_outputs_class_o2o[img_id].max(-1)[0].sigmoid()  # (sum(hw),)
                    props_logits = enc_outputs_coord_o2o[img_id]  # (sum(hw), 4) unsigmoid
                    props = props_logits.sigmoid()
                    props_xyxy = box_ops.box_cxcywh_to_xyxy(props)

                    keep = nms(props_xyxy, scores, iou_thr)

                    o2o_scores_list.append(scores[keep][:topk_o2o])
                    o2o_props_list.append(props_logits[keep][:topk_o2o])
                    o2o_queries_list.append(output_memory[img_id][keep][:topk_o2o])

                o2o_proposals_tensor = align_tensor(o2o_props_list, max_len=topk_o2o)
                o2o_queries_tensor = align_tensor(o2o_queries_list, max_len=topk_o2o)

                # o2m (if enabled) uses ORIGINAL topk (no ddq)
                if use_o2m:
                    topk_idx_o2m = torch.topk(enc_outputs_class_o2m.max(-1)[0], topk_o2m, dim=1)[1]
                    o2m_proposals_tensor = torch.gather(
                        enc_outputs_coord_o2m, 1, topk_idx_o2m.unsqueeze(-1).repeat(1, 1, 4)
                    )
                    o2m_queries_tensor = torch.gather(
                        output_memory, 1, topk_idx_o2m.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
                    )

                    proposals_tensor = torch.cat([o2o_proposals_tensor, o2m_proposals_tensor], dim=1)
                    queries_tensor = torch.cat([o2o_queries_tensor, o2m_queries_tensor], dim=1)
                else:
                    proposals_tensor = o2o_proposals_tensor
                    queries_tensor = o2o_queries_tensor

            else:
                # ORIGINAL selection for o2o
                topk_idx_o2o = torch.topk(enc_outputs_class_o2o.max(-1)[0], topk_o2o, dim=1)[1]
                o2o_proposals_tensor = torch.gather(
                    enc_outputs_coord_o2o, 1, topk_idx_o2o.unsqueeze(-1).repeat(1, 1, 4)
                )
                o2o_queries_tensor = torch.gather(
                    output_memory, 1, topk_idx_o2o.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
                )

                if use_o2m:
                    topk_idx_o2m = torch.topk(enc_outputs_class_o2m.max(-1)[0], topk_o2m, dim=1)[1]
                    o2m_proposals_tensor = torch.gather(
                        enc_outputs_coord_o2m, 1, topk_idx_o2m.unsqueeze(-1).repeat(1, 1, 4)
                    )
                    o2m_queries_tensor = torch.gather(
                        output_memory, 1, topk_idx_o2m.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
                    )

                    proposals_tensor = torch.cat([o2o_proposals_tensor, o2m_proposals_tensor], dim=1)
                    queries_tensor = torch.cat([o2o_queries_tensor, o2m_queries_tensor], dim=1)
                else:
                    proposals_tensor = o2o_proposals_tensor
                    queries_tensor = o2o_queries_tensor

            refpoint_embed_undetach = proposals_tensor  # unsigmoid
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = queries_tensor
            tgt = tgt_undetach.detach()

            outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features)

            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)

            interm_outputs=dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask
            if self.layer_wise_iou_enabled:
                outputs_iou = self.iou_head(self.iou_head_before_norm(tgt_undetach))
                interm_outputs['pred_logits_iou'] = outputs_iou

            if self.initialize_box_type != 'no':
                # convert masks into boxes to better initialize box in the decoder
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(device)
                elif self.initialize_box_type == 'mask2box':  # faster conversion
                    refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
                else:
                    assert NotImplementedError
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h],
                                                                                              dtype=torch.float).to(device)
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = torch.zeros((self.num_queries, self.num_queries), device=tgt.device).bool()
        # tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)

        if self.training and self.o2m_matching_enabled:
            tgt_len = tgt.shape[1]

            # tgt_mask 不能是 None
            if tgt_mask is None:
                base_len = tgt_len - self.num_queries_o2m
                tgt_mask = torch.zeros((base_len, base_len), device=tgt.device, dtype=torch.bool)

            tgt_mask_new = torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool)
            tgt_mask_new[-self.num_queries_o2m:, -self.num_queries_o2m:] = False
            tgt_mask_new[:-self.num_queries_o2m, :-self.num_queries_o2m] = tgt_mask
            tgt_mask = tgt_mask_new

        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(tgt.transpose(0, 1), mask_features, True)

            # 注意 ddq 可以在 train/inference 同时使用
            if self.ddq_enabled:
                def get_initial_keep_mask(out_boxes, out_classes, one2one_len=900, one2many_len=1350):
                    boxes_b = out_boxes.sigmoid()
                    scores_b = out_classes.sigmoid().max(-1)[0]

                    def initial_dqs_keep_mask(boxes_cxcywh, scores, iou_thr, one2one_len, one2many_len):
                        one2one_boxes = boxes_cxcywh[:one2one_len]
                        one2one_scores = scores[:one2one_len]
                        keep = nms(one2one_boxes, one2one_scores, iou_thr)
                        mask = torch.zeros((one2one_len + one2many_len,), dtype=torch.bool, device=boxes_cxcywh.device)
                        mask[keep] = True
                        return mask

                    keep_mask = []
                    for b in range(boxes_b.size(0)):
                        keep_mask_b = initial_dqs_keep_mask(boxes_b[b], scores_b[b], iou_thr=0.8, one2one_len=one2one_len, one2many_len=one2many_len)
                        keep_mask.append(keep_mask_b)
                    keep_mask = torch.stack(keep_mask, dim=0)  # (bs, nq)
                    return keep_mask
                dn_query_len = mask_dict['pad_size'] if mask_dict is not None else 0
                ddq_boxes = refpoint_embed[:, dn_query_len:dn_query_len + self.num_queries, :]
                ddq_classes = outputs_class[:, dn_query_len:dn_query_len + self.num_queries, :]
                keep_mask = get_initial_keep_mask(ddq_boxes, ddq_classes, one2one_len=self.num_queries, one2many_len=self.num_queries_o2m)
                initial_keep_mask = keep_mask if self.training else keep_mask[:, :self.num_queries]

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            if self.layer_wise_iou_enabled:
                outputs_iou = self.iou_head(self.iou_head_before_norm(tgt))
                predictions_iou.append(outputs_iou)

        hs, references, distinct_masks = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask,
            pred_head=self,
            mode='main_branch'
        )


        out_sep = None
        if self.training and self.sep_branch_enabled:
            num_queries_dn = mask_dict['pad_size'] if mask_dict is not None else 0
            o2o_tgt = tgt[:, num_queries_dn:num_queries_dn + self.num_queries, :]
            o2o_refpoint_embed = refpoint_embed[:, num_queries_dn:num_queries_dn + self.num_queries, :]
            o2o_tgt_mask = torch.zeros((self.num_queries, self.num_queries), device=tgt.device).bool()
            sep_hs, sep_references, _ = self.decoder(
                tgt=o2o_tgt.transpose(0, 1),
                memory=src_flatten.transpose(0, 1),
                memory_key_padding_mask=mask_flatten,
                pos=None,
                refpoints_unsigmoid=o2o_refpoint_embed.transpose(0, 1),
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                tgt_mask=o2o_tgt_mask,
                pred_head=self,
                mode='sep_branch',
            )

            predictions_class_sep = []
            predictions_mask_sep = []
            predictions_iou_sep = []
            for i, sep_output in enumerate(sep_hs):
                outputs_class_sep, outputs_mask_sep = self.forward_prediction_heads(
                    sep_output.transpose(0, 1), mask_features, True)

                predictions_class_sep.append(outputs_class_sep)
                predictions_mask_sep.append(outputs_mask_sep)

                if self.layer_wise_iou_enabled:
                    outputs_iou_sep = self.iou_head(self.iou_head_before_norm(sep_output))
                    predictions_iou_sep.append(outputs_iou_sep)
            predictions_class_sep = torch.stack(predictions_class_sep, dim=0)
            predictions_mask_sep = torch.stack(predictions_mask_sep, dim=0)
            if self.layer_wise_iou_enabled:
                predictions_iou_sep = torch.stack(predictions_iou_sep, dim=0)
            out_boxes_sep = self.pred_box(sep_references, sep_hs)
            out_sep = {
                'pred_logits': predictions_class_sep[-1],
                'pred_masks': predictions_mask_sep[-1],
                'pred_boxes': out_boxes_sep[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class_sep if self.mask_classification else None,
                    predictions_mask_sep,
                    predictions_iou_sep,
                    out_boxes_sep
                )
            }
            if self.layer_wise_iou_enabled:
                out_sep['pred_logits_iou'] = predictions_iou_sep[-1]

        if mask_dict is not None:
            num_dn_queries = mask_dict['pad_size']
            distinct_masks = [mask[:, num_dn_queries:] for mask in distinct_masks]

        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), mask_features, True)

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            if self.layer_wise_iou_enabled:
                outputs_iou = self.iou_head(self.iou_head_before_norm(output))
                predictions_iou.append(outputs_iou)

        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)
        if mask_dict is not None:
            predictions_mask=torch.stack(predictions_mask)
            predictions_class=torch.stack(predictions_class)
            if self.layer_wise_iou_enabled:
                predictions_iou = torch.stack(predictions_iou)

            predictions_class, out_boxes, predictions_mask, predictions_iou = self.dn_post_process(
                predictions_class,
                out_boxes,
                predictions_mask,
                predictions_iou,
                mask_dict
                )
            predictions_class,predictions_mask,predictions_iou=list(predictions_class),list(predictions_mask), list(predictions_iou)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0*self.label_enc.weight.sum()

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
                predictions_iou,
                out_boxes
            )
        }
        if self.layer_wise_iou_enabled:
            out['pred_logits_iou'] = predictions_iou[-1]

        if self.two_stage:
            out['interm_outputs'] = interm_outputs

        # 将 out 转换为包含 one-to-many 的格式（仅在 o2m_matching_enabled 时生成）
        num_queries_one_to_one = self.num_queries
        bs = out['pred_logits'].shape[0]

        out_one_to_many = None
        if self.training and self.o2m_matching_enabled:
            aux_outputs_one_to_many = []
            for aux_output in out.get('aux_outputs', []):
                aux_output_one_to_many = {
                    'pred_logits': aux_output['pred_logits'][:, num_queries_one_to_one:],
                    'pred_masks': aux_output['pred_masks'][:, num_queries_one_to_one:] if self.training else None,
                    'pred_boxes': aux_output['pred_boxes'][:, num_queries_one_to_one:],
                }
                # layer-wise iou 可选
                if self.layer_wise_iou_enabled and ('pred_logits_iou' in aux_output) and (
                        aux_output['pred_logits_iou'] is not None):
                    aux_output_one_to_many['pred_logits_iou'] = aux_output['pred_logits_iou'][:, num_queries_one_to_one:]
                aux_outputs_one_to_many.append(aux_output_one_to_many)

            interm_outputs_one_to_many = {
                'pred_logits': interm_outputs['pred_logits'][:, num_queries_one_to_one:],
                'pred_masks': interm_outputs['pred_masks'][:, num_queries_one_to_one:],
                'pred_boxes': interm_outputs['pred_boxes'][:, num_queries_one_to_one:],
            }
            if self.layer_wise_iou_enabled and ('pred_logits_iou' in interm_outputs) and (
                    interm_outputs['pred_logits_iou'] is not None):
                interm_outputs_one_to_many['pred_logits_iou'] = interm_outputs['pred_logits_iou'][:, num_queries_one_to_one:]

            out_one_to_many = {
                'pred_logits': out['pred_logits'][:, num_queries_one_to_one:],
                'pred_masks': out['pred_masks'][:, num_queries_one_to_one:],
                'pred_boxes': out['pred_boxes'][:, num_queries_one_to_one:],
                'aux_outputs': aux_outputs_one_to_many,
                'interm_outputs': interm_outputs_one_to_many,
            }
            if self.layer_wise_iou_enabled and ('pred_logits_iou' in out) and (out['pred_logits_iou'] is not None):
                out_one_to_many['pred_logits_iou'] = out['pred_logits_iou'][:, num_queries_one_to_one:]

        if bs == 1 and self.ddq_enabled:
            distinct_masks.insert(0, initial_keep_mask)
            last_mask = distinct_masks.pop()

            last_mask_q = last_mask[0]  # (nq_o2o,)
            out['pred_logits'] = out['pred_logits'][:, last_mask_q, :]
            out['pred_boxes'] = out['pred_boxes'][:, last_mask_q, :]
            out['pred_masks'] = out['pred_masks'][:, last_mask_q, :, :] if out.get('pred_masks',None) is not None else None

            # layer-wise iou 可选
            if self.layer_wise_iou_enabled and ('pred_logits_iou' in out) and (out['pred_logits_iou'] is not None):
                out['pred_logits_iou'] = out['pred_logits_iou'][:, last_mask_q, :]

            for aux, mask in zip(out.get('aux_outputs', []), distinct_masks):
                m = mask[0]  # (nq_o2o,)
                aux['pred_logits'] = aux['pred_logits'][:, m, :]
                aux['pred_boxes'] = aux['pred_boxes'][:, m, :]
                if aux.get('pred_masks', None) is not None:
                    aux['pred_masks'] = aux['pred_masks'][:, m, :, :]

                if self.layer_wise_iou_enabled and ('pred_logits_iou' in aux) and (aux['pred_logits_iou'] is not None):
                    aux['pred_logits_iou'] = aux['pred_logits_iou'][:, m, :]

            # interm_outputs：保留 one-to-one 前半段
            interm_outputs_ = out['interm_outputs']
            interm_outputs_['pred_logits'] = interm_outputs_['pred_logits'][:, :num_queries_one_to_one]
            interm_outputs_['pred_boxes'] = interm_outputs_['pred_boxes'][:, :num_queries_one_to_one]
            interm_outputs_['pred_masks'] = interm_outputs_['pred_masks'][:, :num_queries_one_to_one]

            if self.layer_wise_iou_enabled and ('pred_logits_iou' in interm_outputs_) and (
                    interm_outputs_['pred_logits_iou'] is not None):
                interm_outputs_['pred_logits_iou'] = interm_outputs_['pred_logits_iou'][:, :num_queries_one_to_one]

            out['interm_outputs'] = interm_outputs_

        else:
            raise ValueError(f"Batch size {bs} != 1")

        return out, out_one_to_many, out_sep, mask_dict

    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_iou=None, out_boxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            if outputs_iou is not None and self.layer_wise_iou_enabled:
                return [
                    {'pred_logits': a, 'pred_masks': b, 'pred_logits_iou': c, 'pred_boxes': d}
                    for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_iou[:-1], out_boxes[:-1])
                ]
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]