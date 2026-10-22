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
        point_density_head: bool = False,
        point_count_source: str = "points",
        point_count_line: Tuple[float, float] = (8.924, 0.0345),
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
        # 小穗密度头（mini/density-count）：训练时 prepare_targets 给 targets 带上密度监督要用的字段；
        # 推理时 point_count_source 决定 wheat_2 的 pred_spikelet_counts 怎么来（见 config.py）。
        self.point_density_head = point_density_head
        assert point_count_source in ("points", "density"), point_count_source
        if point_count_source == "density":
            assert point_density_head, "TEST.POINT_COUNT_SOURCE=density 需要 MODEL.MaskDINO.POINT_DENSITY_HEAD=True"
        self.point_count_source = point_count_source
        self.point_count_line = (float(point_count_line[0]), float(point_count_line[1]))

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
        # 小穗密度头的两项损失只在最后一层算一次：在 _interm / _dn / _{i} 扩展之后再加，不产生带后缀的键
        if cfg.MODEL.MaskDINO.POINT_DENSITY_HEAD:
            weight_dict.update({"loss_density": cfg.MODEL.MaskDINO.POINT_DENSITY_WEIGHT,
                                "loss_density_count": cfg.MODEL.MaskDINO.POINT_DENSITY_COUNT_WEIGHT})
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
            point_density=cfg.MODEL.MaskDINO.POINT_DENSITY_HEAD,
            point_density_sigma=cfg.MODEL.MaskDINO.POINT_DENSITY_SIGMA,
            point_density_count_min_iou=cfg.MODEL.MaskDINO.POINT_DENSITY_COUNT_MIN_IOU,
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
            "point_density_head": cfg.MODEL.MaskDINO.POINT_DENSITY_HEAD,
            "point_count_source": cfg.MODEL.MaskDINO.TEST.POINT_COUNT_SOURCE,
            "point_count_line": tuple(cfg.MODEL.MaskDINO.POINT_COUNT_LINE),
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
            # 小穗点预测（框相对归一化 [0,1]）
            point_coord_results = outputs["pred_point_coords"]   # [B, Q, P, 2]
            point_logit_results = outputs["pred_point_logits"]   # [B, Q, P, 1]
            # 小穗密度图（密度头关着时没有这个键）。每个 query 的密度计数在 1/4 分辨率、mask 上采样之前算：
            # n̂_q = Σ D ⊙ 1[mask_logit_q > 0]（与 pred_masks 同一网格），再像点预测一样随 topk 选出。
            density_results = outputs.get("pred_density")
            if density_results is not None:
                density_results = density_results.float()                          # [B, h, w]
                query_count_results = ((mask_pred_results > 0).float()
                                       * density_results[:, None]).sum(dim=(-2, -1))  # [B, Q]
            else:
                density_results = [None] * len(mask_pred_results)
                query_count_results = [None] * len(mask_pred_results)
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, point_coord_result, point_logit_result, density_result, query_count_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, point_coord_results, point_logit_results, density_results, query_count_results, batched_inputs, images.image_sizes
            ):  # image_size is augmented size, not divisible to 32
                height = input_per_image.get("height", image_size[0])  # real size
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)

                # 小穗密度图的读数（只给评估器用，不影响实例输出）：
                #   spikelet_density_sum：有效区域（去掉 padding）内 D 的总和 = 整图预测小穗数；
                #   spikelet_density：D 映射回原图尺寸 [height, width]（最近邻放大后按总和重新归一，质量守恒），
                #     评估器用它在 GT mask 内积分（与 compare/DM-Count 的「GT mask 内积分」同口径）。
                if density_result is not None:
                    sh = new_size[0] // density_result.shape[-2]
                    sw = new_size[1] // density_result.shape[-1]
                    hv = -(-int(image_size[0]) // sh)
                    wv = -(-int(image_size[1]) // sw)
                    d_valid = density_result[:hv, :wv]
                    d_sum = d_valid.sum()
                    d_img = F.interpolate(d_valid[None, None], size=(int(height), int(width)), mode="nearest")[0, 0]
                    d_img = d_img * (d_sum / d_img.sum().clamp(min=1e-12))
                    processed_results[-1]["spikelet_density_sum"] = float(d_sum)
                    processed_results[-1]["spikelet_density"] = d_img


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

                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result, point_coord_result, point_logit_result, query_count_result)
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
                }
            )
            if self.point_density_head:
                # 小穗密度头的监督（criterion.loss_density）：
                #   density_points：该图全部 GT wheat_2 小穗点，增广后图像的像素坐标 [K, 2]（稠密目标）；
                #   point_counts：每个实例的 Ni（实例计数目标；LSJ 截断降级的穗 Ni 已被清零）；
                #   demoted：mapper 降级的实例（稠密损失在它们的 mask 内忽略）。没有这个字段时视为全 False。
                labels = targets_per_image.gt_classes
                pts_px = [p.to(self.device).float().reshape(-1, 2)
                          for p, l in zip(targets_per_image.gt_points, labels.tolist())
                          if int(l) == self.spikelet_class_id]
                new_targets[-1]["density_points"] = (torch.cat(pts_px) if pts_px
                                                     else torch.zeros((0, 2), device=self.device))
                new_targets[-1]["point_counts"] = torch.as_tensor(
                    [len(p) for p in targets_per_image.gt_points], dtype=torch.float32, device=self.device)
                new_targets[-1]["demoted"] = (
                    targets_per_image.gt_demoted.to(self.device).bool()
                    if targets_per_image.has("gt_demoted")
                    else torch.zeros(len(labels), dtype=torch.bool, device=self.device))
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
                }
            )
        return new_targets

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

    def instance_inference(self, mask_cls, mask_pred, mask_box_result, point_coords=None, point_logits=None,
                           query_counts=None):
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
            point_coords = point_coords[topk_indices]   # [K, P, 2] 框相对 [0,1]
            point_logits = point_logits[topk_indices]   # [K, P, 1]
        # 密度计数（每个 query 一个数，1/4 分辨率硬 mask 内的 D 积分，未取整）同样随 topk 选出；密度头关着时为 None
        if query_counts is not None:
            query_counts = query_counts[topk_indices]   # [K]
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
            if query_counts is not None:
                query_counts = query_counts[keep]
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
                    x0, y0, x1, y1 = boxes_xyxy[k]
                    bw = (x1 - x0).clamp(min=0)
                    bh = (y1 - y0).clamp(min=0)
                    score_k = point_logits[k].sigmoid().squeeze(-1)   # [P]
                    keep_pt = score_k > self.point_threshold
                    rel = point_coords[k][keep_pt]                    # [Ki, 2] 框相对
                    xy = torch.stack([x0, y0]) + rel * torch.stack([bw, bh])  # [Ki, 2] 像素
                    spikelet_points.append(xy)
                    spikelet_counts.append(int(keep_pt.sum().item()))
                else:
                    # wheat 实例不输出任何小穗点
                    spikelet_points.append(point_coords.new_zeros((0, 2)))
                    spikelet_counts.append(0)
            result.pred_spikelet_points = spikelet_points
            result.pred_spikelet_counts = torch.as_tensor(spikelet_counts, device=self.device, dtype=torch.int64)

        # ===== density 读法：wheat_2 的每穗计数改由密度图在预测 mask 内的积分给出；小穗点本身不变 =====
        # "points" 读法什么都不改（计数 = 留下的点数，与上面逐位相同）。
        if self.point_count_source == "density":
            assert query_counts is not None, "POINT_COUNT_SOURCE=density 但模型没有输出 pred_density"
            is_spike = labels_per_image == self.spikelet_class_id
            n_hat = query_counts.float()
            cnt = torch.round(n_hat).clamp(min=0).long()
            result.pred_spikelet_counts = torch.where(is_spike, cnt, torch.zeros_like(cnt))
            # 未取整的密度计数、穗长直线（预测框，原图像素）一并挂出，评估器据此在逐穗明细里加列（同一批穗上比 r）
            result.pred_spikelet_count_density = torch.where(is_spike, n_hat, torch.zeros_like(n_hat))
            a, b_line = self.point_count_line
            bwh = (mask_box_result[:, 2:] - mask_box_result[:, :2]).clamp(min=0).float()
            line = a + b_line * torch.hypot(bwh[:, 0], bwh[:, 1])
            result.pred_spikelet_count_line = torch.where(is_spike, line, torch.zeros_like(line))
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes


