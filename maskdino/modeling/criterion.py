# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from maskdino.utils import box_ops
from .point_ifi import instance_dbar_px, sample_apg

try:
    # 正负样本匹配的 debug 可视化。要关掉请改各模块顶部的 ENABLED = False，
    # 不要注释掉这两行 import —— 那样两个钩子会一起失效，且不易察觉。
    from myscripts.backup.vis_process.vis_match_debug import maybe_visualize_matching
    from myscripts.backup.vis_process.vis_point_match_debug import maybe_visualize_point_match
    from myscripts.backup.vis_process.vis_slot_specialization import maybe_record_slot_stats
except Exception:  # 仓库根不在 sys.path 等情况下静默降级，不影响训练
    maybe_visualize_matching = None
    maybe_visualize_point_match = None
    maybe_record_slot_stats = None


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # loss.mean(1), the loss of all queries in this batch averaged
    # .sun(), the loss of all batches summed
    return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,dn="no",dn_losses=[], panoptic_on=False, semantic_ce_loss=False,
                 num_spikelet_points=50, point_cost_coord=5.0, point_cost_class=2.0,
                 point_focal_alpha=0.25, point_empty_cls_weight=1.0,
                 point_refine=False, point_apg=False, point_apg_num=(2, 2), point_apg_radii=(0.125, 0.5),
                 point_apg_neg_exclude=0.125, point_apg_neg_loc="snap", point_apg_neg_loc_coef=1.0,
                 point_apg_seed=0, point_min_dbar_px=4.0, point_refine_min_iou=0.5):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn=dn
        self.dn_losses=dn_losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

        self.panoptic_on = panoptic_on
        self.semantic_ce_loss = semantic_ce_loss

        # 小穗点回归（内层集合预测）参数
        self.num_spikelet_points = num_spikelet_points   # 每实例预测点上限 P
        self.point_cost_coord = point_cost_coord         # 内层匹配坐标代价权重
        self.point_cost_class = point_cost_class         # 内层匹配存在性代价权重
        # 点存在性专用的 focal alpha，与 loss_labels 用的 self.focal_alpha 解耦
        self.point_focal_alpha = point_focal_alpha
        # Ni==0 实例的存在性损失权重（只作用于 cls_loss 的分子，不影响 denom）
        self.point_empty_cls_weight = point_empty_cls_weight

        # APGCC 移植：IFI 精修头的损失与 APG 辅助点（见 maskdino/modeling/point_ifi.py）。
        # 这两项不进 self.losses —— 它们只在最后一层上算，aux / interm 不重复。
        self.point_refine = point_refine
        self.point_apg = point_apg
        self.point_apg_num = tuple(point_apg_num)
        self.point_apg_radii = tuple(point_apg_radii)
        self.point_apg_neg_exclude = point_apg_neg_exclude
        assert point_apg_neg_loc in ("snap", "zero"), point_apg_neg_loc
        self.point_apg_neg_loc = point_apg_neg_loc
        self.point_apg_neg_loc_coef = point_apg_neg_loc_coef
        self.point_apg_seed = point_apg_seed
        self.point_min_dbar_px = point_min_dbar_px
        self.point_refine_min_iou = point_refine_min_iou
        self._apg_gen = None   # CPU Generator，首次用到时按 seed + rank 建，与 DN 用的全局 RNG 隔离

    def loss_labels_ce(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_boxes_panoptic(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        isthing=target_labels<80
        target_boxes=target_boxes[isthing]
        src_boxes=src_boxes[isthing]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    @torch.no_grad()
    def _inner_point_match(self, pred_rel, pred_logit, gt_rel):
        """逐实例内层匈牙利匹配：把 P 个预测点匹配到 Ni 个 GT 小穗。
        Args:
            pred_rel:   [P, 2]  预测点（框相对归一化坐标）
            pred_logit: [P, 1]  预测点存在性 logit
            gt_rel:     [Ni, 2] GT 小穗（框相对归一化坐标）
        Returns:
            (row_ind, col_ind): 匹配的预测点索引与 GT 索引（长度均为 min(P, Ni)=Ni）
        """
        Ni = gt_rel.shape[0]
        if Ni == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=pred_rel.device)
            return empty, empty
        alpha, gamma = self.point_focal_alpha, 2.0
        # AMP 下统一用 float32 计算代价，避免 half 精度的 cdist / log 问题
        pred_rel = pred_rel.float()
        gt_rel = gt_rel.float()
        prob = pred_logit.float().sigmoid()               # [P,1]
        neg_cost = (1 - alpha) * (prob ** gamma) * (-(1 - prob + 1e-8).log())
        pos_cost = alpha * ((1 - prob) ** gamma) * (-(prob + 1e-8).log())
        cost_class = (pos_cost - neg_cost)                # [P,1]，与 GT 无关，作为每个预测点的偏置
        cost_coord = torch.cdist(pred_rel, gt_rel, p=1)   # [P, Ni]
        C = self.point_cost_coord * cost_coord + self.point_cost_class * cost_class  # [P, Ni]
        C = C.cpu()
        row_ind, col_ind = linear_sum_assignment(C)
        row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=pred_rel.device)
        col_ind = torch.as_tensor(col_ind, dtype=torch.int64, device=pred_rel.device)
        return row_ind, col_ind

    def loss_points(self, outputs, targets, indices, num_masks):
        """逐实例小穗点损失：对每个外层匹配到的实例，做内层匈牙利匹配后计算
        点坐标 L1 损失 + 点存在性 focal 损失。
        - 点坐标采用"实例框相对归一化"[0,1]：用匹配到的 GT 框把 GT 小穗换算到框相对坐标。
        - wheat 实例（无小穗，Ni=0）：无坐标损失，P 个点全部监督为不存在（背景）。
        """
        assert 'pred_point_coords' in outputs and 'pred_point_logits' in outputs
        pred_coords = outputs['pred_point_coords']   # [B, Q, P, 2]
        pred_logits = outputs['pred_point_logits']   # [B, Q, P, 1]
        device = pred_coords.device
        alpha, gamma = self.point_focal_alpha, 2.0

        coord_loss = torch.zeros([], device=device)
        cls_loss = torch.zeros([], device=device)
        num_pos = 0  # 总匹配小穗数，用于归一化

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            boxes_b = targets[b]['boxes']      # [Ng, 4] cxcywh 图像归一化
            points_b = targets[b]['points']    # list[Tensor[Ni,2]] 图像归一化
            for q_i, g_j in zip(src_idx.tolist(), tgt_idx.tolist()):
                # AMP 下预测为 half，GT 为 float32；统一转 float32 避免 dtype 不匹配
                pred_rel = pred_coords[b, q_i].float()     # [P,2]
                pred_lg = pred_logits[b, q_i].float()      # [P,1]
                gt_pts_img = points_b[g_j].to(device).float()   # [Ni,2] 图像归一化
                Ni = gt_pts_img.shape[0]

                # GT 框（cxcywh 归一化）-> 左上角 (x0,y0) 与宽高 (w,h)
                cx, cy, bw, bh = boxes_b[g_j].float()
                x0 = cx - bw / 2
                y0 = cy - bh / 2
                wh = torch.stack([bw, bh]).clamp(min=1e-6)
                if Ni > 0:
                    gt_rel = ((gt_pts_img - torch.stack([x0, y0])) / wh).clamp(0.0, 1.0)  # [Ni,2]
                else:
                    gt_rel = gt_pts_img.new_zeros((0, 2))

                row_ind, col_ind = self._inner_point_match(pred_rel, pred_lg, gt_rel)

                # 存在性目标：匹配上的预测点为 1，其余为 0
                tgt_exist = torch.zeros((self.num_spikelet_points, 1), device=device, dtype=pred_lg.dtype)
                if row_ind.numel() > 0:
                    tgt_exist[row_ind] = 1.0
                    # 坐标 L1 损失（仅匹配点）
                    coord_loss = coord_loss + F.l1_loss(
                        pred_rel[row_ind], gt_rel[col_ind], reduction='sum')

                # 存在性 focal 损失（全部 P 个点）
                prob = pred_lg.sigmoid()
                ce = F.binary_cross_entropy_with_logits(pred_lg, tgt_exist, reduction='none')
                p_t = prob * tgt_exist + (1 - prob) * (1 - tgt_exist)
                focal = ce * ((1 - p_t) ** gamma)
                alpha_t = alpha * tgt_exist + (1 - alpha) * (1 - tgt_exist)
                focal = alpha_t * focal
                # Ni==0 的 wheat 实例贡献 P 个纯负样本，但推理时按类别门控、
                # 它们本就不输出点，这部分监督只在误分类时生效，故可降权。
                w = 1.0 if Ni > 0 else self.point_empty_cls_weight
                cls_loss = cls_loss + focal.sum() * w

                num_pos += Ni

        denom = max(num_pos, 1)
        losses = {
            'loss_point_coord': coord_loss / denom,
            'loss_point_cls': cls_loss / denom,
        }
        return losses

    def _point_focal(self, logit, tgt):
        """与 loss_points 相同的逐元素存在性 focal（α=POINT_FOCAL_ALPHA, γ=2），未求和。"""
        alpha, gamma = self.point_focal_alpha, 2.0
        prob = logit.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logit, tgt, reduction='none')
        p_t = prob * tgt + (1 - prob) * (1 - tgt)
        alpha_t = alpha * tgt + (1 - alpha) * (1 - tgt)
        return alpha_t * ce * ((1 - p_t) ** gamma)

    def _apg_generator(self):
        if self._apg_gen is None:
            rank = torch.distributed.get_rank() if is_dist_avail_and_initialized() else 0
            self._apg_gen = torch.Generator().manual_seed(int(self.point_apg_seed) + rank)
        return self._apg_gen

    def loss_points_refine(self, outputs, targets, indices):
        """IFI 精修头的点损失 + APG 辅助点损失（只在最后一层 decoder 上算）。

        位置统一用 padded 图像归一化坐标（与 pred_boxes、IFI 特征图一致）；与 GT 比较时换回 targets 的
        图像归一化坐标（二者差一个 image_wh / pad_wh，LSJ 下为 1），再按 loss_points 的口径换到 GT 框相对系。

        主项：第一遍在粗坐标 u 处求偏移，r = u + Δ(u)；第二遍在 r.detach() 处求存在性。
              然后与 loss_points 完全相同：内层匈牙利 (5·L1 + 2·focal) → L1(配上的点) + focal(全部 P 个槽)。
              Δ≡0 且 u 用 GT 框换算时，两项逐值等于 loss_point_coord / loss_point_cls。
              只对预测框与 GT 框 IoU ≥ POINT_REFINE_MIN_IOU 的配对计算：u 由预测框换算，框差时 u 离 GT
              几十个框宽，局部头无从修正（开训第 0 iter 实测 L1 是粗点损失的 76 倍，梯度灌进共享特征）；
              评估器也只评 mask IoU ≥ 0.5 的穗配对。
        APG（POINT_APG）：每个带点实例的 GT 小穗周围撒正/负辅助点，以该配对的 query 为条件送进同一个头：
              存在性 BCE（正→1，负→0，正负各自取均值再相加，同 APGCC 官方 loss_auxiliary）；
              定位 L1（GT 框相对系）：正点吸附到最近 GT；负点 "snap" 同样吸附（存在性在吸附之后判），
              "zero" 则按论文 Eq.6 要求 Δ→0。
        """
        probe = outputs['point_refine_probe']
        u_all = outputs['pred_point_ref_anchor']            # [B,Q,P,2] fp32，已 detach
        device = u_all.device
        P = u_all.shape[2]

        pairs = [(b, q, g) for b, (src, tgt) in enumerate(indices)
                 for q, g in zip(src.tolist(), tgt.tolist())]
        zero = u_all.sum() * 0.0
        losses = {'loss_point_ref_coord': zero, 'loss_point_ref_cls': zero}
        if self.point_apg:
            losses.update({'loss_point_apg_cls': zero, 'loss_point_apg_loc': zero})
        if not pairs:
            return losses
        # 主项只用预测框够准的配对（APG 围着 GT 撒点，与预测框无关，仍用全部配对）
        pb = outputs['pred_boxes'].detach().float()
        main_pairs = []
        for (b, q, g) in pairs:
            iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(pb[b, q][None]),
                                  box_ops.box_cxcywh_to_xyxy(targets[b]['boxes'][g][None].float()))[0]
            if float(iou) >= self.point_refine_min_iou:
                main_pairs.append((b, q, g))
        M = len(main_pairs)
        b_t = torch.as_tensor([p[0] for p in main_pairs], device=device, dtype=torch.long)
        q_t = torch.as_tensor([p[1] for p in main_pairs], device=device, dtype=torch.long)
        # padded 归一化 -> targets 归一化 的逐图比例（LSJ 下为 1）
        pad2img = torch.stack([t['pad_wh'] / t['image_wh'] for t in targets]).to(device)   # [B,2]

        # ---- 第一遍：在 u 处求偏移 ----
        bb = b_t[:, None].expand(M, P).reshape(-1)
        qq = q_t[:, None].expand(M, P).reshape(-1)
        u = u_all[b_t, q_t].reshape(-1, 2)                  # [M*P,2]
        _, d1 = probe(bb, u, qq)
        r = u + d1                                          # 梯度经 Δ 流向精修头

        # ---- APG 辅助点（padded 归一化坐标），与第二遍合并成一次调用 ----
        aux_pos, aux_b, aux_q, aux_is_pos, aux_tgt_px, aux_box_px = [], [], [], [], [], []
        if self.point_apg:
            gen = self._apg_generator()
            for (b, q, g) in pairs:
                gt = targets[b]['points'][g]
                if gt.shape[0] < 2:
                    continue
                img_wh = targets[b]['image_wh'].to(device)
                gt_px = gt.to(device).float() * img_wh
                dbar = instance_dbar_px(gt_px)
                if dbar is None or float(dbar) < self.point_min_dbar_px:
                    continue
                pos_px, pos_near, neg_px, neg_near = sample_apg(
                    gt_px, dbar, img_wh.tolist(), self.point_apg_num, self.point_apg_radii,
                    self.point_apg_neg_exclude, gen)
                box_px = targets[b]['boxes'][g, 2:].to(device).float() * img_wh
                for pts, near, is_pos in ((pos_px, pos_near, True), (neg_px, neg_near, False)):
                    n = pts.shape[0]
                    if n == 0:
                        continue
                    aux_pos.append(pts / targets[b]['pad_wh'].to(device))
                    aux_b.append(torch.full((n,), b, device=device, dtype=torch.long))
                    aux_q.append(torch.full((n,), q, device=device, dtype=torch.long))
                    aux_is_pos.append(torch.full((n,), is_pos, device=device, dtype=torch.bool))
                    aux_tgt_px.append(gt_px[near])
                    aux_box_px.append(box_px.expand(n, 2))
        n_aux = sum(a.shape[0] for a in aux_pos)

        # ---- 第二遍：在 r.detach() 处求存在性；APG 点同一次调用拿 logit 与 Δ ----
        if M == 0 and n_aux == 0:
            return losses
        pos2 = torch.cat([r.detach()] + aux_pos)
        b2 = torch.cat([bb] + aux_b)
        q2 = torch.cat([qq] + aux_q)
        lg2, d2 = probe(b2, pos2, q2)
        logit_r = lg2[:M * P].view(M, P)
        r = r.view(M, P, 2)

        coord_loss = torch.zeros([], device=device)
        cls_loss = torch.zeros([], device=device)
        num_pos = 0
        for i, (b, q, g) in enumerate(main_pairs):
            boxes_b = targets[b]['boxes']
            gt_pts_img = targets[b]['points'][g].to(device).float()
            Ni = gt_pts_img.shape[0]
            cx, cy, bw, bh = boxes_b[g].float()
            x0 = cx - bw / 2
            y0 = cy - bh / 2
            wh = torch.stack([bw, bh]).clamp(min=1e-6)
            ref_img = r[i] * pad2img[b]                     # padded 归一化 -> targets 归一化
            ref_rel = (ref_img - torch.stack([x0, y0])) / wh    # [P,2]，预测不 clamp
            if Ni > 0:
                gt_rel = ((gt_pts_img - torch.stack([x0, y0])) / wh).clamp(0.0, 1.0)
            else:
                gt_rel = gt_pts_img.new_zeros((0, 2))
            lg = logit_r[i][:, None]                        # [P,1]
            row_ind, col_ind = self._inner_point_match(ref_rel, lg, gt_rel)
            tgt_exist = torch.zeros((P, 1), device=device, dtype=lg.dtype)
            if row_ind.numel() > 0:
                tgt_exist[row_ind] = 1.0
                coord_loss = coord_loss + F.l1_loss(ref_rel[row_ind], gt_rel[col_ind], reduction='sum')
            w = 1.0 if Ni > 0 else self.point_empty_cls_weight
            cls_loss = cls_loss + self._point_focal(lg, tgt_exist).sum() * w
            num_pos += Ni
        denom = max(num_pos, 1)
        losses['loss_point_ref_coord'] = coord_loss / denom
        losses['loss_point_ref_cls'] = cls_loss / denom

        if self.point_apg and n_aux:
            a_lg = lg2[M * P:]
            a_d = d2[M * P:]                                # padded 归一化偏移
            is_pos = torch.cat(aux_is_pos)
            a_b = torch.cat(aux_b)
            pad_wh_all = torch.stack([t['pad_wh'] for t in targets]).to(device).float()   # [B,2]
            a_px = torch.cat(aux_pos) * pad_wh_all[a_b]
            d_px = a_d * pad_wh_all[a_b]
            tgt_px = torch.cat(aux_tgt_px)
            box_px = torch.cat(aux_box_px).clamp(min=1e-6)
            bce = F.binary_cross_entropy_with_logits(a_lg, is_pos.float(), reduction='none')
            snap = ((a_px + d_px - tgt_px) / box_px).abs().sum(-1)      # 吸附到最近 GT 的 L1（框相对）
            stay = (d_px / box_px).abs().sum(-1)                        # Δ→0（论文 Eq.6）
            neg = ~is_pos
            cls_apg = zero
            loc_apg = zero
            if is_pos.any():
                cls_apg = cls_apg + bce[is_pos].mean()
                loc_apg = loc_apg + snap[is_pos].mean()
            if neg.any():
                cls_apg = cls_apg + bce[neg].mean()
                neg_loc = snap[neg] if self.point_apg_neg_loc == "snap" else stay[neg]
                loc_apg = loc_apg + self.point_apg_neg_loc_coef * neg_loc.mean()
            losses['loss_point_apg_cls'] = cls_apg
            losses['loss_point_apg_loc'] = loc_apg
        return losses

    def prep_for_dn(self,mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar,pad_size=mask_dict['scalar'],mask_dict['pad_size']
        assert pad_size % scalar==0
        single_pad=pad_size//scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes,num_tgt,single_pad,scalar

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes,
            'points': self.loss_points,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None, batched_inputs=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.dn is not "no" and mask_dict is not None:
            output_known_lbs_bboxes,num_tgt,single_pad,scalar = self.prep_for_dn(mask_dict)
            exc_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                exc_idx.append((output_idx, tgt_idx))
        indices = self.matcher(outputs_without_aux, targets)

        # If you run code at only 1 gpu, then you can jump this part below
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # debug 可视化：正负样本匹配。此处的 indices 才是最后一层 decoder 的匹配结果
        # （forward 末尾 return 的 indices 已被 aux / interm 分支覆盖，不能用来画图）。
        if maybe_visualize_matching is not None:
            maybe_visualize_matching(outputs, targets, indices, batched_inputs)
        # 逐实例小穗点的内层匹配可视化（复用 self._inner_point_match，与 loss 同一份配对）
        if maybe_visualize_point_match is not None:
            maybe_visualize_point_match(self, outputs, targets, indices, batched_inputs)
        # 诊断实验：跨迭代累计 slot 匹配频率与标签抖动
        if maybe_record_slot_stats is not None:
            maybe_record_slot_stats(self, outputs, targets, indices, batched_inputs)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        # IFI 精修头 / APG：只用最后一层的 indices（下面的 aux 循环会覆盖 indices）
        if self.point_refine and 'point_refine_probe' in outputs:
            losses.update(self.loss_points_refine(outputs, targets, indices))

        if self.dn != "no" and mask_dict is not None:
            l_dict={}
            for loss in self.dn_losses:
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, exc_idx, num_masks*scalar))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        elif self.dn != "no":
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            if self.dn == "seg":
                l_dict['loss_mask_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_dice_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if 'interm_outputs' in outputs:
                    start = 0
                else:
                    start = 1
                if i>=start:
                    if self.dn != "no" and mask_dict is not None:
                        out_=output_known_lbs_bboxes['aux_outputs'][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            l_dict.update(
                                self.get_loss(loss, out_, targets, exc_idx, num_masks * scalar))
                        l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif self.dn != "no":
                        l_dict = dict()
                        l_dict[f'loss_bbox_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_giou_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_ce_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        if self.dn == "seg":
                            l_dict[f'loss_mask_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                            l_dict[f'loss_dice_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        losses.update(l_dict)
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_masks)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses, indices

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
