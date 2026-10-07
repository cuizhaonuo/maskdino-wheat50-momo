# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li.
import copy
import logging

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, PolygonMasks

from pycocotools import mask as coco_mask

from maskdino.data.copy_paste import build_copy_paste

__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        demote_border_truncated=False,
        demote_from_to=(1, 0),
        demote_border_margin=2,
        copy_paste=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        # 贴边截断实例降级（见 config.py 的 INPUT.DEMOTE_* 说明）
        self.demote_border_truncated = demote_border_truncated
        self.demote_from_to = tuple(demote_from_to)
        self.demote_border_margin = int(demote_border_margin)
        # wheat_2 专项 copy-paste（见 maskdino/data/copy_paste.py），None 表示未开启
        self.copy_paste = copy_paste
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "demote_border_truncated": cfg.INPUT.DEMOTE_BORDER_TRUNCATED,
            "demote_from_to": cfg.INPUT.DEMOTE_FROM_TO,
            "demote_border_margin": cfg.INPUT.DEMOTE_BORDER_MARGIN,
            "copy_paste": build_copy_paste(cfg) if is_train else None,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # ---- wheat_2 专项 copy-paste：必须在 LSJ 增广之前、于【原图尺度】上做 ----
        # 所有图都是同 GSD 的 1024x1024 航拍裁片，原图尺度一致，贴进去无尺度失配；
        # 贴完把新标注追加进 dataset_dict["annotations"]，后续多边形变换、points 变换、
        # DEMOTE_BORDER_TRUNCATED 降级、空实例过滤全部复用下方已有链路。
        if self.is_train and self.copy_paste is not None and "annotations" in dataset_dict:
            image, dataset_dict["annotations"], _n_pasted = self.copy_paste(
                image, dataset_dict["annotations"])
            dataset_dict["n_copy_pasted"] = _n_pasted

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # 实际施加的缩放系数（新尺寸 / 原尺寸，按 x、y 分开）。残差计数头的直线锚要用原生尺度的框对角线：
        # 原生 = 增广后像素 ÷ 缩放系数（maskdino.py 的 prepare_targets）。翻转、裁剪、补边都不改尺度。
        # 只读 transform 的属性，不消耗随机数；计数头关着时这个键不被使用。
        lsj_sx = lsj_sy = 1.0
        for t in transforms.transforms:
            if isinstance(t, T.ResizeTransform):
                lsj_sx *= t.new_w / t.w
                lsj_sy *= t.new_h / t.h
        dataset_dict["lsj_scale"] = (float(lsj_sx), float(lsj_sy))
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)

            # 只保留非 crowd 的标注，points 处理与 annos 严格同序
            raw_annos = [
                obj for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # 先把每个实例的小穗 points（绝对像素 [x,y] 列表）按相同的几何增广变换到当前图像坐标系，
            # 并过滤掉增广（缩放/裁剪/翻转）后落到图像外的点。transform_instance_annotations 不处理
            # 自定义的 points 字段，因此这里单独用 transforms.apply_coords 处理。
            h_img, w_img = image_shape  # 增广后的图像尺寸 (h, w)
            points_per_anno = []
            for obj in raw_annos:
                pts = obj.get("points", None)
                if pts is None or len(pts) == 0:
                    coords = np.zeros((0, 2), dtype=np.float32)
                else:
                    coords = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
                    coords = transforms.apply_coords(coords)
                    inside = (
                        (coords[:, 0] >= 0) & (coords[:, 0] < w_img)
                        & (coords[:, 1] >= 0) & (coords[:, 1] < h_img)
                    )
                    coords = coords[inside].astype(np.float32)
                points_per_anno.append(torch.as_tensor(coords, dtype=torch.float32))

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in raw_annos
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'):  # this is to avoid empty annotation
                instances.gt_masks = PolygonMasks([])
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            # 手动计算 keep 掩码（与 detectron2.filter_empty_instances 默认行为一致：by_box & by_mask），
            # 以便把 points_per_anno 按同一掩码过滤，保持与实例严格对齐。
            keep = instances.gt_boxes.nonempty()
            if instances.has('gt_masks'):
                keep = keep & instances.gt_masks.nonempty()
            instances = instances[keep]
            kept_idx = keep.nonzero().squeeze(1).tolist()
            gt_points = [points_per_anno[i] for i in kept_idx]
            # Generate masks from polygon
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks

                # 被 ResizeScale+FixedSizeCrop 切到 crop 边界的 wheat_2 已不完整，
                # 按标注规则（截断的穗子归 wheat）降级类别，并清空其小穗点 ——
                # 截断穗的 GT 点数对应的是完整穗，留着就是系统性的 GT-图像不一致。
                # 清空后该实例 Ni=0，自然走 POINT_EMPTY_CLS_WEIGHT 的降权路径，
                # 与推理端 instance_inference 按 pred_classes 门控的行为保持一致。
                if self.demote_border_truncated and len(instances) > 0:
                    src, dst = self.demote_from_to
                    m = self.demote_border_margin
                    mk = instances.gt_masks.bool()
                    touches = (
                        mk[:, :m, :].flatten(1).any(dim=1)
                        | mk[:, -m:, :].flatten(1).any(dim=1)
                        | mk[:, :, :m].flatten(1).any(dim=1)
                        | mk[:, :, -m:].flatten(1).any(dim=1)
                    )
                    demote = touches & (instances.gt_classes == src)
                    if bool(demote.any()):
                        instances.gt_classes[demote] = dst
                        for i in demote.nonzero().squeeze(1).tolist():
                            gt_points[i] = gt_points[i].new_zeros((0, 2))

            # 把对齐过滤后的小穗点挂为自定义 list 字段（每实例 Tensor[Ni,2]，绝对像素坐标）。
            # wheat（无小穗）实例为 Tensor[0,2]。
            instances.gt_points = gt_points

            dataset_dict["instances"] = instances

        return dataset_dict
