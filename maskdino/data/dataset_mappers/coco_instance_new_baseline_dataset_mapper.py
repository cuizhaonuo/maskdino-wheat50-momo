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


import copy
import logging
import random
from typing import List, Optional

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import DatasetCatalog, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode, PolygonMasks


def build_transform_gen(cfg, is_train):
    """
    Default detectron2-style augmentations:
      - Flip
      - ResizeScale (multi-scale)
      - FixedSizeCrop
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

    augmentation.extend(
        [
            T.ResizeScale(
                min_scale=min_scale,
                max_scale=max_scale,
                target_height=image_size,
                target_width=image_size,
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ]
    )
    return augmentation


# If you already have this in your project, keep your own.
def convert_coco_poly_to_mask(polygons, height, width):
    """
    Placeholder: use your existing implementation.
    """
    raise NotImplementedError


class COCOInstanceNewBaselineDatasetMapper:
    """
    Mapper for (MaskFormer-like) instance tasks, with optional Copy-Paste augmentation.

    Copy-Paste is implemented at mapper level (NOT in tfm_gens), because it needs:
      - a second sample (donor image)
      - new annotations
      - instance-level placement logic
    """

    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        tfm_gens: List,
        image_format: str,
        # Copy-Paste controls
        enable_copypaste: bool = False,
        copypaste_prob: float = 0.5,
        copypaste_max_instances: int = 3,
        copypaste_overlap_prob: float = 0.8,
        donor_dataset_dicts: Optional[List[dict]] = None,
    ):
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.is_train = is_train

        # Copy-Paste settings
        self.enable_copypaste = enable_copypaste and is_train
        self.copypaste_prob = float(copypaste_prob)
        self.copypaste_max_instances = int(copypaste_max_instances)
        self.copypaste_overlap_prob = float(copypaste_overlap_prob)
        self.donor_dataset_dicts = donor_dataset_dicts or []

        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] TransformGens: %s", str(self.tfm_gens)
        )
        if self.enable_copypaste:
            logging.getLogger(__name__).info(
                "[COCOInstanceNewBaselineDatasetMapper] CopyPaste enabled: "
                "prob=%.3f max_instances=%d overlap_prob=%.3f donor_pool=%d",
                self.copypaste_prob,
                self.copypaste_max_instances,
                self.copypaste_overlap_prob,
                len(self.donor_dataset_dicts),
            )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        tfm_gens = build_transform_gen(cfg, is_train)

        # --- Copy-Paste config (you can move these into cfg.* if you like) ---
        enable_copypaste = getattr(cfg.INPUT, "COPYPASTE", False)
        copypaste_prob = getattr(cfg.INPUT, "COPYPASTE_PROB", 0.5)
        copypaste_max_instances = getattr(cfg.INPUT, "COPYPASTE_MAX_INSTANCES", 3)
        copypaste_overlap_prob = getattr(cfg.INPUT, "COPYPASTE_OVERLAP_PROB", 0.8)

        donor_dataset_dicts = None
        if enable_copypaste and is_train:
            # Use the first training dataset as donor pool (common setup)
            train_name = cfg.DATASETS.TRAIN[0]
            donor_dataset_dicts = DatasetCatalog.get(train_name)

        return {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "enable_copypaste": enable_copypaste,
            "copypaste_prob": copypaste_prob,
            "copypaste_max_instances": copypaste_max_instances,
            "copypaste_overlap_prob": copypaste_overlap_prob,
            "donor_dataset_dicts": donor_dataset_dicts,
        }

    def _copypaste_bbox_only(self, image: np.ndarray, dataset_dict: dict):
        """
        BBox-only Copy-Paste (no instance masks):
          - crop donor bbox patch (includes background)
          - paste patch onto target image
          - append new bbox annotation (XYWH_ABS)

        Notes:
          - This is a practical baseline for detection when masks are unavailable.
          - For more realistic results, prefer mask-based copy-paste.
        """
        if not self.donor_dataset_dicts:
            return image, dataset_dict

        # Decide whether to apply
        if random.random() >= self.copypaste_prob:
            return image, dataset_dict

        H, W = image.shape[:2]

        # sample a donor with annotations
        # (try a few times to avoid empty donors)
        donor = None
        for _ in range(10):
            cand = random.choice(self.donor_dataset_dicts)
            if "annotations" in cand and len(cand["annotations"]) > 0:
                donor = cand
                break
        if donor is None:
            return image, dataset_dict

        donor_img = utils.read_image(donor["file_name"], format=self.img_format)
        dH, dW = donor_img.shape[:2]

        donor_annos = [a for a in donor.get("annotations", []) if a.get("iscrowd", 0) == 0]
        if not donor_annos:
            return image, dataset_dict

        k = random.randint(1, max(1, self.copypaste_max_instances))
        k = min(k, len(donor_annos))
        donor_annos = random.sample(donor_annos, k)

        # Make sure target has annotations list
        dataset_dict.setdefault("annotations", [])

        # Helper: choose paste location (prefer overlap with existing instances)
        def choose_location(pw, ph):
            if dataset_dict["annotations"] and random.random() < self.copypaste_overlap_prob:
                base = random.choice(dataset_dict["annotations"])
                bx, by, bw, bh = base["bbox"]
                # paste somewhere near the center area to create occlusion
                cx = int(bx + bw * random.uniform(0.25, 0.75) - pw * 0.5)
                cy = int(by + bh * random.uniform(0.25, 0.75) - ph * 0.5)
            else:
                cx = random.randint(0, max(0, W - pw))
                cy = random.randint(0, max(0, H - ph))
            cx = max(0, min(cx, W - pw))
            cy = max(0, min(cy, H - ph))
            return cx, cy

        for anno in donor_annos:
            # donor bbox -> ints
            if anno.get("bbox_mode", None) == BoxMode.XYXY_ABS:
                x1, y1, x2, y2 = anno["bbox"]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            else:
                # assume XYWH_ABS
                x, y, w, h = anno["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)

            # clamp to donor image
            x = max(0, min(x, dW - 1))
            y = max(0, min(y, dH - 1))
            w = max(1, min(w, dW - x))
            h = max(1, min(h, dH - y))

            patch = donor_img[y : y + h, x : x + w]
            if patch.size == 0:
                continue

            cx, cy = choose_location(w, h)

            # paste (hard overwrite). You can replace with alpha blend if you want.
            image[cy : cy + h, cx : cx + w] = patch

            # append new annotation
            new_anno = {
                "bbox": [float(cx), float(cy), float(w), float(h)],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": int(anno.get("category_id", 0)),
                "iscrowd": 0,
            }
            # keep segmentation if exists (it will NOT be correct for bbox-patch paste, so skip by default)
            dataset_dict["annotations"].append(new_anno)

        return image, dataset_dict

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # --- Copy-Paste BEFORE geometric transforms ---
        if self.enable_copypaste:
            image, dataset_dict = self._copypaste_bbox_only(image, dataset_dict)

        # padding mask (same as your original)
        padding_mask = np.ones(image.shape[:2], dtype=np.uint8)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        # crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances(annos, image_shape)

            # If your pipeline expects masks but dataset is bbox-only, this keeps it from crashing.
            if not instances.has("gt_masks"):
                instances.gt_masks = PolygonMasks([])

            # Recompute boxes from masks if masks exist; otherwise keep boxes from annos
            # (In bbox-only case, instances.gt_masks is empty, so get_bounding_boxes() is empty too.)
            if instances.has("gt_masks") and len(instances.gt_masks) > 0:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            instances = utils.filter_empty_instances(instances)

            # If you truly have polygons and want to convert to bitmasks, keep your original logic:
            h, w = instances.image_size
            if hasattr(instances, "gt_masks") and isinstance(instances.gt_masks, PolygonMasks):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

        return dataset_dict

