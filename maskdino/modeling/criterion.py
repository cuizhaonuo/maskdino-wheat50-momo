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
from detectron2.utils.events import get_event_storage
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils.point_nms import greedy_point_nms
from maskdino.utils import box_ops
from .coral import coral_loss

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
                 point_focal_alpha=0.25, point_empty_cls_weight=1.0, point_coord_norm="box_axis",
                 point_coord_robust_c=0.0, point_nms_mode="none", point_nms_kappa=0.07,
                 point_coral_exist=False):
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
        # 点坐标误差的归一化方式（内层代价与 L1 共用），见 _point_iso_scale
        assert point_coord_norm in ("box_axis", "iso"), f"未知 POINT_COORD_NORM: {point_coord_norm}"
        self.point_coord_norm = point_coord_norm
        # 绝对坐标输出（分组 decoder 的小穗 token）的坐标误差稳健化常数 c，见 _point_coord_rho
        self.point_coord_robust_c = point_coord_robust_c
        # 小穗 token 的点 NMS（DDQ 式，见 config.py 的 POINT_NMS_MODE / POINT_NMS_KAPPA 与 _point_nms_keep）
        assert point_nms_mode in ("none", "match"), f"未知 POINT_NMS_MODE: {point_nms_mode}"
        self.point_nms_mode = point_nms_mode
        self.point_nms_kappa = point_nms_kappa
        # 本次 forward 内累计的 NMS 统计（每次 forward 开头清零，末尾写进 EventStorage）：
        # [被压 token 数, token 总数, 饿死实例数（留下的 token 少于 Ni）, Ni>0 的实例数]
        self._point_nms_stats = [0, 0, 0, 0]
        # POINT_CORAL_EXIST：前缀内层匹配 + CORAL 存在性损失（见 _inner_point_match / loss_points）
        self.point_coral_exist = point_coral_exist

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

    @staticmethod
    def _pred_to_box_rel(pred, box_xy0, box_wh, abs_frame):
        """把预测小穗点换到 GT 框相对系（与 gt_rel 同一坐标系）。
        abs_frame=False（MLP 点头）：输出本来就是框相对坐标，原样返回；
        abs_frame=True（分组 decoder 的小穗 token）：图像归一化绝对坐标 → (p - xy0) / wh，不 clamp。
        Args:
            pred:    [P, 2]
            box_xy0: [2] GT 框左上角，图像归一化
            box_wh:  [2] GT 框 (w, h)，图像归一化
        """
        if not abs_frame:
            return pred
        return (pred - box_xy0) / box_wh

    def _point_coord_rho(self, abs_err, robust):
        """逐轴坐标误差 → 损失。robust=False 或 c<=0：L1；否则 ρ(e) = c·log(1 + |e|/c)。
        小穗 token 输出绝对坐标、在 GT 框系里算误差：开训时实例框还很差，匹配上的 token 可离 GT
        几十个框宽（实测 L1 初值 57/层，基线 1.8），L1 对每个远点都给满额梯度，小穗分支会去追框级别的误差。
        ρ 在 |e| << c 时 ≈ L1（c=1 时，|e|=0.05 处只差 2.5%），远处梯度 1/(1+|e|/c) 不为零但变弱。
        """
        c = self.point_coord_robust_c
        if not robust or c <= 0:
            return abs_err
        return c * torch.log1p(abs_err / c)

    def _point_iso_scale(self, box_wh, image_wh=None):
        """POINT_COORD_NORM="iso" 时，框相对坐标换到各向同性空间的逐轴缩放因子；"box_axis" 时返回 None。
        框相对坐标的 1 个单位在 x/y 上分别是 w/h 个像素；乘以 (w, h)/sqrt(w·h) 后两轴的
        1 个单位都等于 sqrt(w·h) 个像素，即误差 = 像素误差 / sqrt(w·h)，两轴同权。
        Args:
            box_wh:   [2] GT 框 (w, h)，图像归一化
            image_wh: [2] 图像像素 (W, H)，用于还原像素纵横比；LSJ 训练输入恒为 1024x1024，可不传
        Returns:
            [2] 或 None
        """
        if self.point_coord_norm != "iso":
            return None
        wh = box_wh.float()
        if image_wh is not None:
            wh = wh * image_wh.to(wh)
        return wh / wh.prod().sqrt().clamp(min=1e-12)

    @torch.no_grad()
    def _inner_point_match(self, pred_rel, pred_logit, gt_rel, iso_scale=None, robust=False, prefix_match=False):
        """逐实例内层匈牙利匹配：把 P 个预测点匹配到 Ni 个 GT 小穗。
        Args:
            pred_rel:   [P, 2]  预测点（框相对归一化坐标）
            pred_logit: [P, 1]  预测点存在性 logit
            gt_rel:     [Ni, 2] GT 小穗（框相对归一化坐标）
            iso_scale:  [2] 或 None，_point_iso_scale 的返回值；非 None 时坐标代价在各向同性空间算
            robust:     True 时坐标代价用 _point_coord_rho（与该输出的 L1 损失同一函数）
            prefix_match: True 时走 CORAL 前缀匹配（slot 语义层专用，由 loss_points 按
                        point_coral_exist and not abs_frame 传入；group 层 token 语义不可用）
        Returns:
            (row_ind, col_ind): 匹配的预测点索引与 GT 索引（长度均为 min(P, Ni)=Ni）
        """
        Ni = gt_rel.shape[0]
        if Ni == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=pred_rel.device)
            return empty, empty
        if prefix_match:
            # 前缀模式：只有前 m=min(Ni,P) 个 slot 可以被配。它们全是正样本，存在性代价对所有分配都一样，
            # 故代价只剩坐标 L1。Ni<=P 时是 [Ni,Ni] 的方阵，前缀内任意排列；Ni>P 时 P 个 slot 按坐标挑 P 个 GT。
            # 于是「slot k 被配上 <=> k < min(Ni,P)」，存在性目标恰好是 CORAL 的 levels。
            m = min(Ni, pred_rel.shape[0])
            C = torch.cdist(pred_rel[:m].float(), gt_rel.float(), p=1).cpu()   # [m, Ni]
            row_ind, col_ind = linear_sum_assignment(C)
            row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=pred_rel.device)
            col_ind = torch.as_tensor(col_ind, dtype=torch.int64, device=pred_rel.device)
            return row_ind, col_ind
        alpha, gamma = self.point_focal_alpha, 2.0
        # AMP 下统一用 float32 计算代价，避免 half 精度的 cdist / log 问题
        pred_rel = pred_rel.float()
        gt_rel = gt_rel.float()
        if iso_scale is not None:
            pred_rel = pred_rel * iso_scale
            gt_rel = gt_rel * iso_scale
        prob = pred_logit.float().sigmoid()               # [P,1]
        neg_cost = (1 - alpha) * (prob ** gamma) * (-(1 - prob + 1e-8).log())
        pos_cost = alpha * ((1 - prob) ** gamma) * (-(prob + 1e-8).log())
        cost_class = (pos_cost - neg_cost)                # [P,1]，与 GT 无关，作为每个预测点的偏置
        if robust and self.point_coord_robust_c > 0:
            cost_coord = self._point_coord_rho((pred_rel[:, None] - gt_rel[None]).abs(), True).sum(-1)  # [P, Ni]
        else:
            cost_coord = torch.cdist(pred_rel, gt_rel, p=1)   # [P, Ni]
        C = self.point_cost_coord * cost_coord + self.point_cost_class * cost_class  # [P, Ni]
        C = C.cpu()
        row_ind, col_ind = linear_sum_assignment(C)
        row_ind = torch.as_tensor(row_ind, dtype=torch.int64, device=pred_rel.device)
        col_ind = torch.as_tensor(col_ind, dtype=torch.int64, device=pred_rel.device)
        return row_ind, col_ind

    @torch.no_grad()
    def _point_nms_keep(self, pred_rel, pred_logit, box_wh, image_wh=None, kappa=None):
        """内层匹配前的小穗点 NMS（POINT_NMS_MODE="match"），返回留下的 token 掩码。
        诊断脚本（point_token_diag.py）逐实例调这个方法，loss_points 逐图批量调（每个实例仍各自做一次 NMS），
        两者是同一份代码；批量只是把多个实例叠在前置维度上，逐元素运算与 greedy_point_nms 的结果都与逐个做时相同。
        在 iso 空间做：框相对坐标乘 s = (w, h)/sqrt(w·h)（w、h 为像素宽高），两轴的 1 个单位都等于
        sqrt(w·h) 个像素，于是「像素中心距 < κ·sqrt(w·h)」⟺「iso 距离 < κ」。换算与 _point_iso_scale
        同一公式，但不论 POINT_COORD_NORM 取什么都这样算：NMS 永远在 iso 空间做。
        分数 = sigmoid(logit)；坐标与分数全部 detach，搬到 CPU 一次后在 CPU 上做 NMS。
        Args:
            pred_rel:   [..., P, 2] 预测点（GT 框相对坐标）
            pred_logit: [..., P, 1] 存在性 logit
            box_wh:     [..., 2] GT 框 (w, h)，图像归一化
            image_wh:   [2] 图像像素 (W, H)；None 时视为 1
            kappa:      NMS 半径 κ；None 时取 self.point_nms_kappa（诊断脚本可覆盖）
        Returns:
            BoolTensor [..., P]（CPU），True = 留下
        """
        kappa = self.point_nms_kappa if kappa is None else kappa
        wh_px = box_wh.detach().float()
        if image_wh is not None:
            wh_px = wh_px * image_wh.to(wh_px)
        s = wh_px / wh_px.prod(-1, keepdim=True).sqrt().clamp(min=1e-12)
        xy = pred_rel.detach().float() * s.unsqueeze(-2)
        score = pred_logit.detach().float().sigmoid()
        packed = torch.cat([xy, score], dim=-1).cpu()   # 一次 GPU→CPU 同步
        return greedy_point_nms(packed[..., :2], packed[..., 2], kappa)

    def _put_point_nms_stats(self):
        """把本次 forward 累计的 NMS 统计写进 EventStorage（metrics.json 的 point_nms/*）。
        不放进 losses：MaskDINO.forward 会按 weight_dict 过滤 losses，统计项会被丢掉。"""
        n_sup, n_tok, n_starved, n_pos = self._point_nms_stats
        try:
            storage = get_event_storage()
        except AssertionError:   # 不在 EventStorage 里（如独立脚本直接前向）就跳过
            return
        if n_tok > 0:
            storage.put_scalar("point_nms/suppressed_frac", n_sup / n_tok)
        if n_pos > 0:
            storage.put_scalar("point_nms/starved_frac", n_starved / n_pos)

    def loss_points(self, outputs, targets, indices, num_masks):
        """逐实例小穗点损失：对每个外层匹配到的实例，做内层匈牙利匹配后计算
        点坐标 L1 损失 + 点存在性 focal 损失。
        - 点坐标采用"实例框相对归一化"[0,1]：用匹配到的 GT 框把 GT 小穗换算到框相对坐标。
          POINT_COORD_NORM="iso" 时，误差（内层代价与 L1）再换到各向同性空间计算，见 _point_iso_scale。
        - outputs['pred_point_abs']=True（分组 decoder 的小穗 token）：预测是图像归一化绝对坐标，
          先换到 GT 框相对系，坐标误差用 _point_coord_rho（POINT_COORD_ROBUST_C>0 时有界增长）。
          缺省 False：MLP 点头的框相对坐标，与原实现一致。
        - wheat 实例（无小穗，Ni=0）：无坐标损失，P 个点全部监督为不存在（背景）。
        - 小穗点 NMS（DDQ 式，见 config.py 的 POINT_NMS_MODE）：存活掩码 keep 的来源依次是
            1) 输出里带 pred_point_keep（[B, Q, P] bool，decoder 逐层剔除时给出）→ 取 [b, q]；
            2) 否则 POINT_NMS_MODE != "none" 且是小穗 token 的绝对坐标输出 → 用所配 GT 框做 NMS（_point_nms_keep）；
            3) 否则 keep=None，走原路径（与不加 NMS 时逐位一致）。
          keep 非 None 时只有留下的 token 进内层匹配，下标再映射回 P 个；存在性 focal 只对留下的 token
          求和（被压掉的贡献 0），坐标损失仍只算配上的；Ni=0 的实例同样处理；分母不变；
          留下的少于 Ni 时不补回（多出的 GT 小穗本层没有配对）。
        """
        assert 'pred_point_coords' in outputs and 'pred_point_logits' in outputs
        pred_coords = outputs['pred_point_coords']   # [B, Q, P, 2]
        pred_logits = outputs['pred_point_logits']   # [B, Q, P, 1]
        abs_frame = bool(outputs.get('pred_point_abs', False))
        device = pred_coords.device
        alpha, gamma = self.point_focal_alpha, 2.0
        # 小穗点 NMS 的存活掩码来源（见上）。pred_point_keep 一次搬到 CPU，逐实例取用
        keep_all = outputs.get('pred_point_keep')
        if keep_all is not None:
            keep_all = keep_all.detach().bool().cpu()
        nms_here = keep_all is None and self.point_nms_mode != "none" and abs_frame

        coord_loss = torch.zeros([], device=device)
        cls_loss = torch.zeros([], device=device)
        num_pos = 0  # 总匹配小穗数，用于归一化
        # POINT_CORAL_EXIST：逐实例收集存在性 logit / 前缀目标 / 权重，循环后一次算 CORAL 损失
        coral_logits, coral_levels_, coral_weights = [], [], []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            boxes_b = targets[b]['boxes']      # [Ng, 4] cxcywh 图像归一化
            points_b = targets[b]['points']    # list[Tensor[Ni,2]] 图像归一化
            image_wh = targets[b].get('image_wh')   # [2] 图像像素 (W, H)
            # 存活掩码逐图一次算好（[M, P]，CPU）：每个实例各自做 NMS，但整图只有一次 GPU→CPU、一次 CPU→GPU。
            # 框相对坐标与下面逐实例的 pred_rel 同一套逐元素运算，NMS 的输入逐位相同。
            keep_b = None
            if keep_all is not None:
                keep_b = keep_all[b][src_idx.cpu()]
            elif nms_here:
                bx = boxes_b[tgt_idx].float()                                             # [M, 4]
                xy0_b = torch.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2], dim=-1)
                wh_b = torch.stack([bx[:, 2], bx[:, 3]], dim=-1).clamp(min=1e-6)
                rel_b = self._pred_to_box_rel(pred_coords[b, src_idx].float(), xy0_b[:, None], wh_b[:, None], True)
                keep_b = self._point_nms_keep(rel_b, pred_logits[b, src_idx].float(), wh_b, image_wh)
            if keep_b is not None:
                n_keep_b = keep_b.sum(1).tolist()
                kidx_b = keep_b.nonzero(as_tuple=True)[1].to(device)   # 各实例留下的 token 下标，按实例拼接
                off_b = [0]
                for n in n_keep_b:
                    off_b.append(off_b[-1] + n)
            for k, (q_i, g_j) in enumerate(zip(src_idx.tolist(), tgt_idx.tolist())):
                # AMP 下预测为 half，GT 为 float32；统一转 float32 避免 dtype 不匹配
                pred_c = pred_coords[b, q_i].float()       # [P,2]
                pred_lg = pred_logits[b, q_i].float()      # [P,1]
                gt_pts_img = points_b[g_j].to(device).float()   # [Ni,2] 图像归一化
                Ni = gt_pts_img.shape[0]

                # GT 框（cxcywh 归一化）-> 左上角 (x0,y0) 与宽高 (w,h)
                cx, cy, bw, bh = boxes_b[g_j].float()
                x0 = cx - bw / 2
                y0 = cy - bh / 2
                wh = torch.stack([bw, bh]).clamp(min=1e-6)
                pred_rel = self._pred_to_box_rel(pred_c, torch.stack([x0, y0]), wh, abs_frame)  # [P,2] GT 框相对
                if Ni > 0:
                    gt_rel = ((gt_pts_img - torch.stack([x0, y0])) / wh).clamp(0.0, 1.0)  # [Ni,2]
                else:
                    gt_rel = gt_pts_img.new_zeros((0, 2))

                iso_scale = self._point_iso_scale(wh, image_wh)
                keep = None if keep_b is None else keep_b[k]
                if keep is None:
                    row_ind, col_ind = self._inner_point_match(pred_rel, pred_lg, gt_rel, iso_scale, robust=abs_frame,
                                                               prefix_match=self.point_coral_exist and not abs_frame)
                else:
                    # 只让留下的 token 进内层匹配，匹配下标再映射回 P 个 token 的下标
                    # （keep 只在 abs_frame 下出现，与前缀匹配互斥，故 prefix_match=False）
                    kidx = kidx_b[off_b[k]:off_b[k + 1]]
                    n_keep = n_keep_b[k]
                    row_ind, col_ind = self._inner_point_match(
                        pred_rel[kidx], pred_lg[kidx], gt_rel, iso_scale, robust=abs_frame)
                    row_ind = kidx[row_ind]
                    st = self._point_nms_stats
                    st[0] += int(keep.numel()) - n_keep
                    st[1] += int(keep.numel())
                    if Ni > 0:
                        st[3] += 1
                        st[2] += int(n_keep < Ni)

                # 存在性目标：匹配上的预测点为 1，其余为 0
                tgt_exist = torch.zeros((self.num_spikelet_points, 1), device=device, dtype=pred_lg.dtype)
                if row_ind.numel() > 0:
                    tgt_exist[row_ind] = 1.0
                    # 坐标 L1 损失（仅匹配点），与内层代价同一个空间、同一个误差函数
                    pred_m, gt_m = pred_rel[row_ind], gt_rel[col_ind]
                    if iso_scale is not None:
                        pred_m, gt_m = pred_m * iso_scale, gt_m * iso_scale
                    if abs_frame and self.point_coord_robust_c > 0:
                        coord_loss = coord_loss + self._point_coord_rho((pred_m - gt_m).abs(), True).sum()
                    else:
                        coord_loss = coord_loss + F.l1_loss(pred_m, gt_m, reduction='sum')

                # Ni==0 的 wheat 实例贡献 P 个纯负样本，但推理时按类别门控、
                # 它们本就不输出点，这部分监督只在误分类时生效，故可降权。
                w = 1.0 if Ni > 0 else self.point_empty_cls_weight
                if self.point_coral_exist and not abs_frame:
                    # 前缀匹配下 tgt_exist 就是 levels = [1]*min(Ni,P) + [0]*(P-min(Ni,P))
                    coral_logits.append(pred_lg.squeeze(-1))
                    coral_levels_.append(tgt_exist.squeeze(-1))
                    coral_weights.append(w)
                else:
                    # 存在性 focal 损失（全部 P 个点；有 NMS 掩码时只对留下的求和，见下）
                    prob = pred_lg.sigmoid()
                    ce = F.binary_cross_entropy_with_logits(pred_lg, tgt_exist, reduction='none')
                    p_t = prob * tgt_exist + (1 - prob) * (1 - tgt_exist)
                    focal = ce * ((1 - p_t) ** gamma)
                    alpha_t = alpha * tgt_exist + (1 - alpha) * (1 - tgt_exist)
                    focal = alpha_t * focal
                    if keep is None:
                        cls_loss = cls_loss + focal.sum() * w
                    else:
                        # NMS 压掉的 token 不算存在性损失（贡献 0、梯度 0），只对留下的求和
                        cls_loss = cls_loss + focal[kidx].sum() * w

                num_pos += Ni

        denom = max(num_pos, 1)
        if self.point_coral_exist:
            # CORAL：每实例 P 个任务的 BCE 取均值，实例间按 w 加权平均（口径与基线的 sum/num_pos 不同，
            # 见 maskdino/modeling/coral.py 与 experiments-bank/ideas/coral-exist.md）
            if coral_logits:
                wt = torch.as_tensor(coral_weights, device=device, dtype=torch.float32)
                cls_term = coral_loss(torch.stack(coral_logits), torch.stack(coral_levels_), wt)
            else:
                cls_term = pred_logits.sum() * 0.0
        else:
            cls_term = cls_loss / denom
        losses = {
            'loss_point_coord': coord_loss / denom,
            'loss_point_cls': cls_term,
        }
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
        # 小穗点 NMS 统计：每次 forward 清零，loss_points 里累计，末尾写进 EventStorage
        self._point_nms_stats = [0, 0, 0, 0]

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

        if self.point_nms_mode != "none":
            self._put_point_nms_stats()
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
