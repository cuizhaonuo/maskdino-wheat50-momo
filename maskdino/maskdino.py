# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils import box_ops


@META_ARCH_REGISTRY.register()
class MaskDINO(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        point_threshold: float = 0.35,
        point_source: str = "coarse",
        dense_topk: int = 64,
        dense_owner_score: float = 0.3,
        dense_mask_thresh: float = 0.5,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            transform_eval: transform sigmoid score into softmax score to make score sharper
            semantic_ce_loss: whether use cross-entroy loss in classification
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss
        # 小穗点推理：存在性阈值 + 带小穗的类别（wheat_2 的连续 id=1，只有该类输出点）
        self.point_threshold = point_threshold
        self.spikelet_class_id = 1
        # 点从哪个头出："coarse" = 25 槽头（框相对坐标）；"dense" = 稠密锚点头（padded 图像归一化绝对坐标，
        # 按预测掩码归到各个穗，不再经预测框换算）
        assert point_source in ("coarse", "dense"), point_source
        self.point_source = point_source
        self.point_abs_coords = point_source == "dense"
        self.dense_topk = dense_topk
        self.dense_owner_score = dense_owner_score
        self.dense_mask_thresh = dense_mask_thresh

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        print('criterion.weight_dict ', self.criterion.weight_dict)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        # 小穗点损失权重
        point_coord_weight = cfg.MODEL.MaskDINO.POINT_COORD_WEIGHT
        point_cls_weight = cfg.MODEL.MaskDINO.POINT_CLASS_WEIGHT
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})
        # 小穗点损失（随 _interm / _{i} 深监督自动带上；dn 不计算点损失，其 _dn 权重存在但不会被产生，无害）
        weight_dict.update({"loss_point_coord": point_coord_weight, "loss_point_cls": point_cls_weight})
        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["labels","boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks","boxes"]
        else:
            dn_losses=[]
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        # APGCC 移植的损失每次前向只算一次，放在 interm / dn / 深监督展开之后注册，不生成 _interm/_dn/_i 副本
        if cfg.MODEL.MaskDINO.POINT_DENSE:
            weight_dict.update({"loss_point_dense_coord": cfg.MODEL.MaskDINO.POINT_DENSE_COORD_WEIGHT,
                                "loss_point_dense_cls": cfg.MODEL.MaskDINO.POINT_DENSE_CLASS_WEIGHT})
        if cfg.MODEL.MaskDINO.POINT_APG:
            assert cfg.MODEL.MaskDINO.POINT_DENSE, "POINT_APG needs POINT_DENSE (APG 训练的是稠密锚点头)"
            weight_dict.update({"loss_point_apg_cls": cfg.MODEL.MaskDINO.POINT_APG_CLS_WEIGHT,
                                "loss_point_apg_loc": cfg.MODEL.MaskDINO.POINT_APG_LOC_WEIGHT})
        point_source = cfg.MODEL.MaskDINO.TEST.POINT_SOURCE
        if point_source == "auto":
            point_source = "dense" if cfg.MODEL.MaskDINO.POINT_DENSE else "coarse"
        assert point_source != "dense" or cfg.MODEL.MaskDINO.POINT_DENSE, "POINT_SOURCE=dense needs POINT_DENSE"
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks","boxes","points"]
        else:
            losses = ["labels", "masks","points"]
        # building criterion
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            num_spikelet_points=cfg.MODEL.MaskDINO.NUM_POINTS,
            point_cost_coord=cfg.MODEL.MaskDINO.COST_POINT_COORD_WEIGHT,
            point_cost_class=cfg.MODEL.MaskDINO.COST_POINT_CLASS_WEIGHT,
            point_focal_alpha=cfg.MODEL.MaskDINO.POINT_FOCAL_ALPHA,
            point_empty_cls_weight=cfg.MODEL.MaskDINO.POINT_EMPTY_CLS_WEIGHT,
            point_dense=cfg.MODEL.MaskDINO.POINT_DENSE,
            point_dense_cost_point=cfg.MODEL.MaskDINO.POINT_DENSE_COST_POINT,
            point_dense_cost_class=cfg.MODEL.MaskDINO.POINT_DENSE_COST_CLASS,
            point_dense_region_dilate=cfg.MODEL.MaskDINO.POINT_DENSE_REGION_DILATE,
            point_dense_max_anchors=cfg.MODEL.MaskDINO.POINT_DENSE_MAX_ANCHORS,
            point_dense_wheat_weight=cfg.MODEL.MaskDINO.POINT_DENSE_WHEAT_WEIGHT,
            point_dbar_fallback=cfg.MODEL.MaskDINO.POINT_DBAR_FALLBACK,
            point_apg=cfg.MODEL.MaskDINO.POINT_APG,
            point_apg_num=cfg.MODEL.MaskDINO.POINT_APG_NUM,
            point_apg_radii=cfg.MODEL.MaskDINO.POINT_APG_RADII,
            point_apg_neg_exclude=cfg.MODEL.MaskDINO.POINT_APG_NEG_EXCLUDE,
            point_apg_neg_loc=cfg.MODEL.MaskDINO.POINT_APG_NEG_LOC,
            point_apg_neg_loc_coef=cfg.MODEL.MaskDINO.POINT_APG_NEG_LOC_COEF,
            point_apg_seed=cfg.MODEL.MaskDINO.POINT_APG_SEED,
            point_min_dbar_px=cfg.MODEL.MaskDINO.POINT_MIN_DBAR_PX,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
                or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MaskDINO.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL,
            "pano_temp": cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "point_threshold": cfg.MODEL.MaskDINO.TEST.POINT_THRESHOLD,
            "point_source": point_source,
            "dense_topk": cfg.MODEL.MaskDINO.TEST.POINT_DENSE_TOPK,
            "dense_owner_score": cfg.MODEL.MaskDINO.TEST.POINT_DENSE_OWNER_SCORE,
            "dense_mask_thresh": cfg.MODEL.MaskDINO.TEST.POINT_DENSE_MASK_THRESH,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        if self.training:
            # dn_args={"scalar":30,"noise_scale":0.4}
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                if 'detr' in self.data_loader:
                    targets = self.prepare_targets_detr(gt_instances, images)
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # 预处理时已经 pad 好了，不需要在这里 pad 了，因此不传入 mask，即 mask=None
            outputs,mask_dict = self.sem_seg_head(features, targets=targets)

            # bipartite matching-based loss
            # batched_inputs 只被 criterion 里的 debug 可视化钩子使用，不参与任何计算
            losses, indices = self.criterion(outputs, targets, mask_dict, batched_inputs=batched_inputs)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:

            # 在推理过程中 aux_outputs.pred_masks 是 None，但是在训练的过程中是 (bs, num_queries, H/4, W/4)
            outputs, _ = self.sem_seg_head(features)
            # test the outputs of interm_outputs and aux_outputs
            # ...
            # outputs = outputs['interm_outputs']
            # outputs = outputs['aux_outputs'][0]
            # ...
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            # 小穗点预测：coarse 为框相对归一化 [0,1]；dense 为 padded 图像归一化绝对坐标（每个实例 K 个候选）
            if self.point_source == "dense":
                point_coord_results, point_logit_results = self._dense_to_query_candidates(
                    mask_cls_results, mask_pred_results, outputs["pred_dense_points"], outputs["pred_dense_logits"])
            else:
                point_coord_results = outputs["pred_point_coords"]   # [B, Q, P, 2]
                point_logit_results = outputs["pred_point_logits"]   # [B, Q, P, 1]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, point_coord_result, point_logit_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, point_coord_results, point_logit_results, batched_inputs, images.image_sizes
            ):  # image_size is augmented size, not divisible to 32
                height = input_per_image.get("height", image_size[0])  # real size
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)


                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    # mask_box_result = mask_box_result.to(mask_pred_result)
                    # mask_box_result = self.box_postprocess(mask_box_result, height, width)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    mask_box_result = self.box_postprocess(mask_box_result, height, width)
                    if self.point_abs_coords:
                        # 稠密头的小穗点与框同为 padded 图像归一化坐标，用同一套缩放换到原图像素
                        point_coord_result = point_coord_result * point_coord_result.new_tensor([width, height])

                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result, point_coord_result, point_logit_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            # 小穗点：从绝对像素 [x,y] 转成图像归一化 [0,1]（与 boxes 同一 (w,h)，便于框相对换算）
            wh = torch.as_tensor([w, h], dtype=torch.float, device=self.device)
            gt_points = [p.to(self.device).float() / wh for p in targets_per_image.gt_points]
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy,
                    "points": gt_points,
                    # 图像与 padded 张量的像素尺寸 (W, H)：稠密头 / APG 做像素与两种归一化坐标之间的换算
                    "image_wh": wh,
                    "pad_wh": torch.as_tensor([w_pad, h_pad], dtype=torch.float, device=self.device),
                }
            )
        return new_targets


    def prepare_targets_detr(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            wh = torch.as_tensor([w, h], dtype=torch.float, device=self.device)
            gt_points = [p.to(self.device).float() / wh for p in targets_per_image.gt_points]
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor) / image_size_xyxy,
                    "points": gt_points,
                    # 图像与 padded 张量的像素尺寸 (W, H)：稠密头 / APG 做像素与两种归一化坐标之间的换算
                    "image_wh": wh,
                    "pad_wh": torch.as_tensor([w_pad, h_pad], dtype=torch.float, device=self.device),
                }
            )
        return new_targets

    @torch.no_grad()
    def _dense_to_query_candidates(self, mask_cls, mask_pred, dense_pts, dense_logits):
        """把稠密头的全图提议归到各个 query，整理成与 25 槽头同形的输出，供 instance_inference 与诊断脚本直接使用。

        Args:
            mask_cls:     [B,Q,C] 类别 logit
            mask_pred:    [B,Q,h,w] 掩码 logit（1/4 分辨率，padded 坐标系）
            dense_pts:    [B,A,2] 提议位置 a + Δ(a)（padded 归一化）
            dense_logits: [B,A]   提议存在性 logit
        Returns:
            coords [B,Q,K,2]（padded 归一化），logits [B,Q,K,1]。
        每图：在掩码上采样得到每个 query 对每个提议的掩码概率 M[q,j]；wheat_2 类分数 ≥ OWNER_SCORE 的 query 为候选主人，
        提议 j 归给其中 M 最大者（要求 M ≥ MASK_THRESH）；非候选主人的 query 取 M ≥ MASK_THRESH 的提议（不排他，
        它们类别分数低，评估器不会用到）。每个 query 从【自己的】提议里按存在性取前 K 个，分数 = σ(conf)。
        候选因此都在该实例掩码内，与 25 槽头「每个槽都属于这个实例」同口径；只有拥有的提议不足 K 个时，
        才用掩码概率次高的提议补位，补位分数为 0（logit −20，任何阈值都滤掉）。
        """
        B, Q = mask_cls.shape[:2]
        K = self.dense_topk
        coords = dense_pts.new_zeros((B, Q, K, 2))
        logits = dense_pts.new_full((B, Q, K, 1), -20.0)
        for b in range(B):
            pts = dense_pts[b]                                                 # [A,2]
            A = pts.shape[0]
            conf = dense_logits[b].sigmoid()
            grid = (pts * 2 - 1).view(1, 1, A, 2).expand(Q, 1, A, 2)
            M = F.grid_sample(mask_pred[b][:, None].float(), grid.float(), mode="bilinear",
                              padding_mode="zeros", align_corners=False)[:, 0, 0].sigmoid()   # [Q,A]
            eligible = mask_cls[b, :, self.spikelet_class_id].sigmoid() >= self.dense_owner_score
            own = M >= self.dense_mask_thresh                                  # [Q,A]
            if eligible.any():
                M_el = M.masked_fill(~eligible[:, None], -1.0)
                owner = M_el.argmax(dim=0)                                     # [A]
                excl = torch.zeros_like(own)
                excl[owner, torch.arange(A, device=pts.device)] = True
                own[eligible] &= excl[eligible]
            # 自己的提议按 σ(conf) 排在前面（+1 保证排在所有补位之前）；补位按掩码概率
            rank = torch.where(own, 1.0 + conf[None], M * 1e-3)
            k = min(K, A)
            _, idx = rank.topk(k, dim=1)                                       # [Q,k]
            owned_k = own.gather(1, idx)
            coords[b, :, :k] = pts[idx]
            logits[b, :, :k, 0] = torch.where(owned_k, dense_logits[b][idx].float(),
                                              torch.full_like(idx, -20, dtype=torch.float))
        return coords, logits

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result, point_coords=None, point_logits=None):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # labels = torch.ones_like(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # 同步按 topk 选出每个实例的小穗点预测
        if point_coords is not None:
            point_coords = point_coords[topk_indices]   # [K, P, 2] 框相对 [0,1]（dense 源下为像素）
            point_logits = point_logits[topk_indices]   # [K, P, 1]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
            if point_coords is not None:
                point_coords = point_coords[keep]
                point_logits = point_logits[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        # ===== 小穗点：把框相对坐标还原到图像像素，仅对 wheat_2 实例输出 =====
        if point_coords is not None:
            spikelet_points = []
            spikelet_counts = []
            boxes_xyxy = mask_box_result  # [K, 4] 像素 xyxy
            for k in range(len(result)):
                if int(labels_per_image[k].item()) == self.spikelet_class_id:
                    score_k = point_logits[k].sigmoid().squeeze(-1)   # [P]
                    keep_pt = score_k > self.point_threshold
                    if self.point_abs_coords:
                        xy = point_coords[k][keep_pt]                 # [Ki, 2] 已是像素（见 forward）
                    else:
                        x0, y0, x1, y1 = boxes_xyxy[k]
                        bw = (x1 - x0).clamp(min=0)
                        bh = (y1 - y0).clamp(min=0)
                        rel = point_coords[k][keep_pt]                # [Ki, 2] 框相对
                        xy = torch.stack([x0, y0]) + rel * torch.stack([bw, bh])  # [Ki, 2] 像素
                    spikelet_points.append(xy)
                    spikelet_counts.append(int(keep_pt.sum().item()))
                else:
                    # wheat 实例不输出任何小穗点
                    spikelet_points.append(point_coords.new_zeros((0, 2)))
                    spikelet_counts.append(0)
            result.pred_spikelet_points = spikelet_points
            result.pred_spikelet_counts = torch.as_tensor(spikelet_counts, device=self.device, dtype=torch.int64)
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes


