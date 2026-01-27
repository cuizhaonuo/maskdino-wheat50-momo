# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li.

import copy
import logging
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from pycocotools import mask as coco_mask

from detectron2.config import configurable
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode, PolygonMasks, BitMasks


__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    segmentations: list[list[float]] polygons in COCO format
    return: (N,H,W) uint8 tensor
    """
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


def _build_transform_gen(cfg, is_train):
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

def build_transform_gen(cfg, is_train):
    assert is_train
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    aug = []

    if cfg.INPUT.RANDOM_FLIP == "horizontal":
        aug.append(T.RandomFlip(horizontal=True, vertical=False))
    elif cfg.INPUT.RANDOM_FLIP == "vertical":
        aug.append(T.RandomFlip(horizontal=False, vertical=True))
    elif cfg.INPUT.RANDOM_FLIP == "both":
        aug.append(T.RandomFlip(horizontal=True, vertical=True))

    aug.append(
        T.ResizeScale(
            min_scale=min_scale,
            max_scale=max_scale,
            target_height=image_size,
            target_width=image_size,
        )
    )
    if cfg.INPUT.ROTATE_AUG:
        aug.append(T.RandomRotation(angle=[-120, 120], expand=False, center=None))

    aug.append(T.FixedSizeCrop(crop_size=(image_size, image_size)))

    if cfg.INPUT.COLOR_AUG:
        aug.extend([
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            # T.RandomLighting(0.1),  # 如果你环境里有
        ])

    return aug


class COCOInstanceNewBaselineDatasetMapper:
    """
    Mapper for instance segmentation tasks, with optional MASK-based Copy-Paste augmentation.

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
        copypaste_prob: float = 0.3,
        copypaste_max_instances: int = 2,
        copypaste_overlap_prob: float = 0.7,
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
            "[COCOInstanceNewBaselineDatasetMapper] TransformGens used in training: %s",
            str(self.tfm_gens),
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

        enable_copypaste = getattr(cfg.INPUT, "COPYPASTE", False)
        copypaste_prob = getattr(cfg.INPUT, "COPYPASTE_PROB", 0.3)
        copypaste_max_instances = getattr(cfg.INPUT, "COPYPASTE_MAX_INSTANCES", 2)
        copypaste_overlap_prob = getattr(cfg.INPUT, "COPYPASTE_OVERLAP_PROB", 0.7)

        donor_dataset_dicts = None
        if enable_copypaste and is_train:
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

    # -----------------------------
    # Copy-Paste helpers (mask-based)
    # -----------------------------
    @staticmethod
    def _decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
        """
        Support COCO polygon or RLE dict. Return binary mask (H,W) uint8 {0,1}.
        """
        if segmentation is None:
            return np.zeros((height, width), dtype=np.uint8)

        # Polygon format: list[list[float]]
        if isinstance(segmentation, list):
            rles = coco_mask.frPyObjects(segmentation, height, width)
            m = coco_mask.decode(rles)
            if m.ndim == 3:
                m = np.any(m, axis=2)
            return m.astype(np.uint8)

        # RLE dict format: {"size":[h,w], "counts": ...}
        if isinstance(segmentation, dict):
            rle = segmentation
            # If counts is a list, it’s uncompressed RLE -> convert
            if isinstance(rle.get("counts", None), list):
                rle = coco_mask.frPyObjects(rle, height, width)
            m = coco_mask.decode(rle)
            if m.ndim == 3:
                m = np.any(m, axis=2)
            return m.astype(np.uint8)

        # Unknown format
        return np.zeros((height, width), dtype=np.uint8)

    @staticmethod
    def _encode_rle(mask: np.ndarray) -> dict:
        """
        Encode binary mask (H,W) to COCO RLE dict.
        """
        # pycocotools requires Fortran order
        rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
        # rle["counts"] is bytes; keep it as-is (detectron2/pycocotools compatible)
        return rle

    @staticmethod
    def _mask_to_xywh(mask: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute XYWH bbox from a binary mask.
        """
        rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
        x, y, w, h = coco_mask.toBbox(rle).tolist()
        return float(x), float(y), float(w), float(h)

    @staticmethod
    def _mask_to_polygons(mask: np.ndarray, epsilon: float = 1.0, min_area: int = 20):
        """
        mask: (H,W) uint8 {0,1}
        return: list[list[float]] COCO polygon format
        """
        mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            c = cv2.approxPolyDP(c, epsilon, True)
            if len(c) < 3:
                continue
            poly = c.reshape(-1, 2).astype(float).flatten().tolist()
            if len(poly) >= 6:
                polygons.append(poly)
        return polygons

    def _choose_paste_xy(self, W: int, H: int, pw: int, ph: int, annos: List[dict]) -> Tuple[int, int]:
        """
        Choose paste location. Prefer overlapping with existing instances to create occlusion.
        """
        if annos and random.random() < self.copypaste_overlap_prob:
            base = random.choice(annos)
            # base bbox assumed XYWH
            bx, by, bw, bh = base["bbox"]
            cx = int(bx + bw * random.uniform(0.25, 0.75) - pw * 0.5)
            cy = int(by + bh * random.uniform(0.25, 0.75) - ph * 0.5)
        else:
            cx = random.randint(0, max(0, W - pw))
            cy = random.randint(0, max(0, H - ph))

        cx = max(0, min(cx, W - pw))
        cy = max(0, min(cy, H - ph))
        return cx, cy

    def _copypaste_with_masks(self, image: np.ndarray, dataset_dict: dict):
        """
        Mask-based Copy-Paste:
          - decode donor instance mask
          - crop tight patch around donor bbox/mask
          - paste only masked pixels onto target
          - append new annotation containing:
                * bbox (XYWH_ABS)
                * segmentation (RLE) in target-image coords
        """
        if not self.donor_dataset_dicts:
            return image, dataset_dict

        if random.random() >= self.copypaste_prob:
            return image, dataset_dict

        H, W = image.shape[:2]
        dataset_dict.setdefault("annotations", [])
        target_annos = dataset_dict["annotations"]

        # pick a donor with annotations+segmentation
        donor = None
        for _ in range(20):
            cand = random.choice(self.donor_dataset_dicts)
            ann = cand.get("annotations", [])
            if not ann:
                continue
            # require at least one annotation with segmentation
            if any(("segmentation" in a and a["segmentation"] is not None) and a.get("iscrowd", 0) == 0 for a in ann):
                donor = cand
                break
        if donor is None:
            return image, dataset_dict

        donor_img = utils.read_image(donor["file_name"], format=self.img_format)
        donor_img = np.ascontiguousarray(donor_img)

        dH, dW = donor_img.shape[:2]

        donor_annos = [a for a in donor.get("annotations", []) if a.get("iscrowd", 0) == 0 and a.get("segmentation", None) is not None]
        if not donor_annos:
            return image, dataset_dict

        k = random.randint(1, max(1, self.copypaste_max_instances))
        k = min(k, len(donor_annos))
        donor_annos = random.sample(donor_annos, k)

        for anno in donor_annos:
            # donor bbox (assume XYWH in your dataset; if XYXY, convert)
            if anno.get("bbox_mode", None) == BoxMode.XYXY_ABS:
                x1, y1, x2, y2 = anno["bbox"]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            else:
                x, y, w, h = anno["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)

            # clamp
            x = max(0, min(x, dW - 1))
            y = max(0, min(y, dH - 1))
            w = max(1, min(w, dW - x))
            h = max(1, min(h, dH - y))

            # decode full donor mask then crop to bbox for speed
            full_mask = self._decode_coco_segmentation(anno["segmentation"], dH, dW)
            if full_mask.sum() == 0:
                continue

            mask_crop = full_mask[y : y + h, x : x + w]
            if mask_crop.sum() == 0:
                continue

            img_crop = donor_img[y : y + h, x : x + w]
            if img_crop.size == 0:
                continue

            # choose paste location
            cx, cy = self._choose_paste_xy(W, H, w, h, target_annos)

            # paste only masked pixels
            # broadcast mask to 3 channels
            m = mask_crop.astype(bool)
            if img_crop.ndim == 3:
                image_patch = image[cy : cy + h, cx : cx + w]
                image_patch[m] = img_crop[m]
                image[cy : cy + h, cx : cx + w] = image_patch
            else:
                # unlikely (grayscale), but handle
                image[cy : cy + h, cx : cx + w][m] = img_crop[m]

            # build target-sized mask for the pasted instance
            pasted_mask = np.zeros((H, W), dtype=np.uint8)
            pasted_mask[cy : cy + h, cx : cx + w] = mask_crop

            # bbox from pasted mask
            bx, by, bw, bh = self._mask_to_xywh(pasted_mask)
            if bw <= 1 or bh <= 1:
                continue

            # ✅ convert pasted mask to polygon segmentation (MaskDINO needs polygon)
            polys = self._mask_to_polygons(pasted_mask, epsilon=1.5, min_area=20)
            if len(polys) == 0:
                continue

            new_anno = {
                "bbox": [bx, by, bw, bh],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": polys,  # ✅ polygon list-of-polygons
                "category_id": int(anno.get("category_id", 0)),
                "iscrowd": 0,
            }
            target_annos.append(new_anno)

        return image, dataset_dict

    # -----------------------------
    # Main call
    # -----------------------------
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = np.ascontiguousarray(image).copy()

        utils.check_image_size(dataset_dict, image)

        # padding mask (same as your original)
        padding_mask = np.ones(image.shape[:2], dtype=np.uint8)

        # --- Copy-Paste BEFORE geometric transforms ---
        if self.enable_copypaste:
            image, dataset_dict = self._copypaste_with_masks(image, dataset_dict)

        # cv2.imwrite("debug_copypaste.jpg", image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        # cv2.imwrite("debug_copypaste_transforms.jpg", image)

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

            # ensure masks exist
            if not instances.has("gt_masks"):
                instances.gt_masks = PolygonMasks([])

            # filter empty first
            instances = utils.filter_empty_instances(instances)

            # Your original logic: convert polygon masks to bitmasks.
            # Here, our Copy-Paste generates RLE, and annotations_to_instances typically converts that to BitMasks already.
            # But keep polygon conversion for polygon-based anns.
            h, w = instances.image_size
            if hasattr(instances, "gt_masks") and isinstance(instances.gt_masks, PolygonMasks):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks

            # boxes: if masks exist, recompute tight boxes from masks
            if instances.has("gt_masks") and hasattr(instances.gt_masks, "get_bounding_boxes"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = instances

        return dataset_dict






