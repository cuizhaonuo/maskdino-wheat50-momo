# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
from typing import Tuple
import copy

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
        o2m_matching_enabled = False,
        num_targets_k = 6,
        layer_wise_iou_enabled = False,
        sep_branch_enabled = False,
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

        self.o2m_matching_enabled = o2m_matching_enabled
        self.num_targets_k = num_targets_k
        self.sep_branch_enabled = sep_branch_enabled

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

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        print('criterion.weight_dict ', self.criterion.weight_dict)

    @classmethod
    def from_config(cls, cfg):
        # one-to-many
        o2m_matching_enabled = cfg.MODEL.MaskDINO.O2M_MATCHING_ENABLED
        num_targets_k = cfg.MODEL.MaskDINO.NUM_TARGETS_K
        layer_wise_iou_enabled = cfg.MODEL.MaskDINO.LAYER_WISE_IOU_ENABLED
        align_loss_enabled = cfg.MODEL.MaskDINO.ALIGN_LOSS_ENABLED
        sep_branch_enabled = cfg.MODEL.MaskDINO.SEP_BRANCH_ENABLED

        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        # iou_weight = cfg.MODEL.MaskDINO.IOU_WEIGHT

        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT

        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            matcher_nms_enabled=cfg.MODEL.MaskDINO.MATCHDER_NMS_ENABLED,
        )

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})
        weight_dict.update({'loss_iou': box_weight})
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
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]
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
            layer_wise_iou_enabled=layer_wise_iou_enabled,
            align_loss_enabled=align_loss_enabled,
        )

        return {
            "o2m_matching_enabled": o2m_matching_enabled,
            "num_targets_k": num_targets_k,
            "layer_wise_iou_enabled": layer_wise_iou_enabled,
            "sep_branch_enabled": sep_branch_enabled,
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
            "semantic_ce_loss": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
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
            outputs, outputs_one2many, out_sep, mask_dict = self.sem_seg_head(features, targets=targets)

            # one-to-one loss, one-to-one query and denoising query
            # bipartite matching-based loss
            losses, _ = self.criterion(outputs, targets, mask_dict)

            def duplicate_targets(src_targets, k_one2many):
                for t in src_targets:
                    t['labels'] = t['labels'].repeat(k_one2many)
                    t['masks'] = t['masks'].repeat(k_one2many, 1, 1)
                    t['boxes'] = t['boxes'].repeat(k_one2many, 1)
                return src_targets

            # one-to-many loss, one-to-many query
            if self.o2m_matching_enabled:
                multi_targets = copy.deepcopy(targets)
                multi_targets = duplicate_targets(multi_targets, k_one2many=self.num_targets_k)
                # mask_dict = None # 表示关闭 dn-query
                losses_one2many, _ = self.criterion(outputs_one2many, multi_targets, None)
                lambda_one2many = 1.0
                for key, value in losses_one2many.items():
                    losses[key + '_one2many'] = value * lambda_one2many
                    weight = self.criterion.weight_dict[key]
                    self.criterion.weight_dict[key + '_one2many'] = weight

            if self.sep_branch_enabled:
                multi_targets = copy.deepcopy(targets)
                multi_targets = duplicate_targets(multi_targets, k_one2many=self.num_targets_k)
                losses_sep, _ = self.criterion(out_sep, multi_targets, None)
                lambda_sep = 1.0
                for key, value in losses_sep.items():
                    losses[key + '_sep'] = value * lambda_sep
                    weight = self.criterion.weight_dict[key]
                    self.criterion.weight_dict[key + '_sep'] = weight


            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    if 'loss_iou' in k or 'loss_dice_l1' in k:
                        continue
                    print(k)
                    losses.pop(k)
            return losses
        else:
            outputs, _, _, _ = self.sem_seg_head(features)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            mask_iou_results = outputs["pred_logits_iou"] if "pred_logits_iou" in outputs else \
                torch.zeros((*mask_cls_results.shape[:-1], 1), device=mask_cls_results.device)
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, mask_iou_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, mask_iou_results, batched_inputs, images.image_sizes
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

                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result, mask_iou_result)
                    processed_results[-1]["instances"] = instance_r
                    # instance_r, inst_panoptic = retry_if_cuda_oom(self.instance_inference_like_panoptic)(
                    #     mask_cls_result, mask_pred_result, mask_box_result, mask_iou_result
                    # )
                    # processed_results[-1]["instances"] = instance_r
                    # processed_results[-1]["inst_panoptic"] = inst_panoptic

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
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
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
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor) / image_size_xyxy
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

    def instance_inference(self, mask_cls, mask_pred, mask_box_result, mask_iou_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        # ious = mask_iou_result.sigmoid()
        # # scores = (scores + 1) * ious  # multiply mask iou score
        # scores = scores + ious  # multiply mask iou score
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # labels = torch.ones_like(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        flat_scores = scores.flatten(0, 1)
        k = min(self.test_topk_per_image, flat_scores.numel())
        scores_per_image, topk_indices = flat_scores.topk(k, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
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

        # mask_scores_per_image = 1.0

        pred_iou = mask_iou_result[topk_indices].sigmoid().squeeze(-1)

        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots(1,1)
        # ax.plot(scores_per_image.cpu().numpy(), label='cls score')
        # ax.plot(mask_scores_per_image.cpu().numpy(), label='mask iou score')
        # ax.plot((scores_per_image * mask_scores_per_image).cpu().numpy(), label='final score')
        # ax.plot(pred_iou.cpu().numpy(), label='pred iou score')
        # ax.legend()
        # plt.show()

        # result.scores = scores_per_image * mask_scores_per_image
        result.scores = scores_per_image * 1.0
        result.pred_classes = labels_per_image
        result.pred_iou = pred_iou
        result.mask_scores = mask_scores_per_image

        result.pred_masks = (mask_pred > 0)
        result.pred_masks_prob = mask_pred.sigmoid()
        return result

    import torch
    from detectron2.structures import Boxes, Instances

    import torch
    from detectron2.structures import Boxes, Instances

    def instance_inference_like_panoptic(self, mask_cls, mask_pred, mask_box_result, mask_iou_result):
        """
        Return:
            result: Instances (normal instance outputs)
            inst_panoptic: (panoptic_like_seg, segments_info)
                - panoptic_like_seg: IntTensor(H,W), 0=void, 1..K=instance id
                - segments_info: list of dicts with keys: id, isthing, category_id, score
        """

        # -----------------------
        # 0) Basic prep
        # -----------------------
        image_size = mask_pred.shape[-2:]  # (H, W)
        H, W = image_size
        device = mask_pred.device
        num_queries = mask_cls.shape[0]

        # -----------------------
        # 1) Classification scores & top-k (改：每个query只取max类，减少冗余)
        # -----------------------
        scores_all = mask_cls.sigmoid()  # [num_queries, num_classes]
        scores_q, labels_q = scores_all.max(dim=1)  # [num_queries], [num_queries]

        k = min(self.test_topk_per_image, scores_q.numel())
        scores_per_image, topk_query_indices = scores_q.topk(k, sorted=False)
        labels_per_image = labels_q[topk_query_indices]

        mask_pred_topk = mask_pred[topk_query_indices]  # [N,H,W] logit
        box_topk = mask_box_result[topk_query_indices]  # [N,4]

        # -----------------------
        # 2) (Optional) keep only thing classes when panoptic_on
        # -----------------------
        if self.panoptic_on:
            # 这里要求 metadata 有 thing_dataset_id_to_contiguous_id
            # 对于你的单类数据，labels_per_image通常就是0，直接保留也行
            thing_ids = set(self.metadata.thing_dataset_id_to_contiguous_id.values()) \
                if hasattr(self.metadata, "thing_dataset_id_to_contiguous_id") else None

            if thing_ids is not None and len(thing_ids) > 0:
                keep = torch.tensor(
                    [int(l.item()) in thing_ids for l in labels_per_image],
                    device=device,
                    dtype=torch.bool,
                )
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred_topk = mask_pred_topk[keep]
                box_topk = box_topk[keep]
                topk_query_indices = topk_query_indices[keep]  # for iou indexing

        # -----------------------
        # 3) Build Instances output (standard)
        # -----------------------
        result = Instances(image_size)
        result.pred_masks = (mask_pred_topk > 0).float()
        result.pred_boxes = Boxes(box_topk)

        mask_scores_per_image = (
                (mask_pred_topk.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1)
                / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        )
        if getattr(self, "focus_on_box", False):
            mask_scores_per_image = torch.ones_like(mask_scores_per_image)

        pred_iou = mask_iou_result[topk_query_indices].sigmoid().squeeze(-1)

        # 你原本是 cls 分数；如果想更稳可以用：scores_per_image * mask_scores_per_image * pred_iou
        final_scores = scores_per_image * 1.0

        result.scores = final_scores
        result.pred_classes = labels_per_image
        result.pred_iou = pred_iou
        result.mask_scores = mask_scores_per_image

        # -----------------------
        # 4) Panoptic-like pixel assignment (for visualization)
        #    改：先做 mask-NMS 去重，再 argmax 分配像素
        # -----------------------
        prob = 0.5
        overlap_threshold = getattr(self, "overlap_threshold", 0.8)

        # mask-NMS 参数：越小越“狠”，去重更多，数量更接近 GT
        mask_nms_iou_thr = getattr(self, "mask_nms_iou_thr", 0.5)
        max_instances_for_nms = getattr(self, "mask_nms_max_candidates", 300)

        N = mask_pred_topk.shape[0]
        panoptic_like_seg = torch.zeros((H, W), dtype=torch.int32, device=device)
        segments_info = []

        if N == 0:
            return result, (panoptic_like_seg, segments_info)

        masks_prob = mask_pred_topk.sigmoid()  # [N,H,W]
        bin_masks = (masks_prob >= prob)  # [N,H,W] bool

        # ---- 4.1 mask-NMS（按分数排序，IoU 过阈值则抑制）----
        # 为了不让 N 太大导致 O(N^2) 太慢，先截断候选数
        order = torch.argsort(final_scores, descending=True)
        if order.numel() > max_instances_for_nms:
            order = order[:max_instances_for_nms]

        keep_ids = []

        def _mask_iou_bool(m1, m2):
            inter = (m1 & m2).sum().float()
            union = (m1 | m2).sum().float()
            return inter / (union + 1e-6)

        for idx in order.tolist():
            m = bin_masks[idx]
            if m.sum().item() == 0:
                continue
            ok = True
            for kept in keep_ids:
                if _mask_iou_bool(m, bin_masks[kept]) > mask_nms_iou_thr:
                    ok = False
                    break
            if ok:
                keep_ids.append(idx)

        if len(keep_ids) == 0:
            return result, (panoptic_like_seg, segments_info)

        keep_ids = torch.tensor(keep_ids, device=device, dtype=torch.long)

        # 过滤候选
        final_scores_n = final_scores[keep_ids]
        labels_n = labels_per_image[keep_ids]
        masks_prob_n = masks_prob[keep_ids]
        bin_masks_n = bin_masks[keep_ids]

        # ---- 4.2 像素级竞争：score × mask_prob argmax ----
        prob_masks = final_scores_n.view(-1, 1, 1) * masks_prob_n
        winner_ids = prob_masks.argmax(0)  # [H,W], value in [0..N'-1]

        current_id = 0
        Nn = masks_prob_n.shape[0]
        for inst_k in range(Nn):
            mask_area = (winner_ids == inst_k).sum().item()
            original_area = bin_masks_n[inst_k].sum().item()
            final_mask = (winner_ids == inst_k) & bin_masks_n[inst_k]

            if mask_area == 0 or original_area == 0 or final_mask.sum().item() == 0:
                continue

            # overlap ratio 过滤：被抢走太多的实例丢掉（与你原 panoptic 逻辑一致）
            if mask_area / (original_area + 1e-6) < overlap_threshold:
                continue

            current_id += 1
            panoptic_like_seg[final_mask] = current_id
            segments_info.append(
                {
                    "id": int(current_id),
                    "isthing": True,
                    "category_id": int(labels_n[inst_k].item()),  # contiguous id (0..C-1)
                    "score": float(final_scores_n[inst_k].item()),
                }
            )

        return result, (panoptic_like_seg, segments_info)

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes


