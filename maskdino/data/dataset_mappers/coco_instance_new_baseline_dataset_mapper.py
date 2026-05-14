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

__all__ = ["COCOInstanceNewBaselineDatasetMapper", "generate_density_map", "visualize_dataset_dict", "visualize_density_maps_grid"]


def generate_density_map(points, image_size, sigma=3.0):
    """
    Generate a normalized density map from point annotations using Gaussian kernels.

    Each valid point contributes 1.0 to the density-map integral.
    Therefore:
        density_map.sum() ~= number of valid points
    """
    H, W = int(image_size[0]), int(image_size[1])

    # ---- Normalize input to (N, 2) np.ndarray ----
    if isinstance(points, list) and len(points) > 0:
        if torch.is_tensor(points[0]):
            valid_tensors = [p for p in points if p.numel() > 0]
            if len(valid_tensors) == 0:
                return np.zeros((H, W), dtype=np.float32)
            points = torch.cat(valid_tensors, dim=0).detach().cpu().numpy()
        elif isinstance(points[0], np.ndarray):
            valid_arrays = [p for p in points if len(p) > 0]
            if len(valid_arrays) == 0:
                return np.zeros((H, W), dtype=np.float32)
            points = np.concatenate(valid_arrays, axis=0)
        else:
            points = np.array(points, dtype=np.float32)
    else:
        if torch.is_tensor(points):
            points = points.detach().cpu().numpy()
        else:
            points = np.array(points, dtype=np.float32)

    points = points.reshape(-1, 2).astype(np.float32)

    if len(points) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # ---- Keep only points inside image bounds ----
    valid = (
        (points[:, 0] >= 0) &
        (points[:, 0] < W) &
        (points[:, 1] >= 0) &
        (points[:, 1] < H)
    )
    points = points[valid]

    if len(points) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # ---- Determine sigma ----
    if sigma == "adaptive" and len(points) >= 3:
        try:
            from scipy.spatial import KDTree
            tree = KDTree(points)
            distances, _ = tree.query(points, k=min(4, len(points)))

            if distances.ndim > 1 and distances.shape[1] > 1:
                avg_dist = distances[:, 1:].mean()
            else:
                avg_dist = distances.mean()

            sigma = max(float(avg_dist) * 0.3, 1.0)
        except ImportError:
            sigma = 3.0
    elif isinstance(sigma, str):
        sigma = 3.0

    sigma = float(sigma)

    if sigma <= 0:
        return np.zeros((H, W), dtype=np.float32)

    # ---- Generate Gaussian kernel ----
    ksize = int(6 * sigma)
    if ksize % 2 == 0:
        ksize += 1

    ksize = max(3, min(ksize, 99))
    half = ksize // 2

    ax = np.arange(-half, half + 1, dtype=np.float32)
    gx = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel_2d = np.outer(gx, gx).astype(np.float32)

    # Normalize the full kernel first.
    kernel_sum = kernel_2d.sum()
    if kernel_sum <= 0:
        return np.zeros((H, W), dtype=np.float32)

    kernel_2d /= kernel_sum

    # ---- Add normalized Gaussians to density map ----
    density_map = np.zeros((H, W), dtype=np.float32)

    for x0_f, y0_f in points:
        x0, y0 = int(round(float(x0_f))), int(round(float(y0_f)))

        x_min = x0 - half
        x_max = x0 + half + 1
        y_min = y0 - half
        y_max = y0 + half + 1

        # Kernel crop coordinates
        kx_min = max(0, -x_min)
        kx_max = ksize - max(0, x_max - W)
        ky_min = max(0, -y_min)
        ky_max = ksize - max(0, y_max - H)

        # Image crop coordinates
        x_min = max(0, x_min)
        x_max = min(W, x_max)
        y_min = max(0, y_min)
        y_max = min(H, y_max)

        if x_min < x_max and y_min < y_max:
            patch = kernel_2d[ky_min:ky_max, kx_min:kx_max]

            # Important:
            # Re-normalize the visible patch so that boundary points
            # still contribute exactly 1.0.
            patch_sum = patch.sum()

            if patch_sum > 0:
                density_map[y_min:y_max, x_min:x_max] += patch / patch_sum

    return density_map.astype(np.float32)


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
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
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

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
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

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # ---- Transform points (spikelet positions) through augmentations ----
            for anno in annos:
                if "points" in anno and len(anno["points"]) > 0:
                    points = np.array(anno["points"], dtype=np.float32)
                    points = transforms.apply_coords(points)
                    # Filter out-of-bounds points after crop
                    inside = (points >= np.array([0, 0])) & (points < np.array(image_shape[::-1]))
                    inside = inside.all(axis=1)
                    points = points[inside]
                    anno["points"] = points.tolist()
            # ----------------------------------------------------------------

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
            instances, keep = utils.filter_empty_instances(instances, return_mask=True)

            # ---- Attach transformed points to Instances (after filtering) ----
            kept_annos = [anno for anno, keep_flag in zip(annos, keep.tolist()) if keep_flag]
            gt_points = [
                torch.as_tensor(anno["points"], dtype=torch.float32)
                if "points" in anno and len(anno["points"]) > 0
                else torch.zeros((0, 2), dtype=torch.float32)
                for anno in kept_annos
            ]
            instances.gt_points = gt_points
            # -------------------------------------------------

            # Generate masks from polygon
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks

            # ---- Generate density maps from points (per instance) ----
            if len(instances) == 0:
                instances.gt_density_maps = torch.zeros((0, h, w), dtype=torch.float32)
            else:
                density_maps = [
                    torch.as_tensor(
                        generate_density_map(
                            pts.cpu().numpy() if torch.is_tensor(pts) else pts,
                            (h, w),
                        ),
                        dtype=torch.float32,
                    )
                    for pts in instances.gt_points
                ]
                instances.gt_density_maps = torch.stack(density_maps, dim=0)
            # ---------------------------------------------------------

            dataset_dict["instances"] = instances

            # visualize_dataset_dict(dataset_dict).show()
            # visualize_density_maps_grid(dataset_dict).show()

        return dataset_dict


# ---- Visualization helper (top-level) ----
def visualize_dataset_dict(dataset_dict, show_masks=True, show_boxes=True, show_points=True):
    """
    Visualize a Detectron2-style dataset_dict.

    Args:
        dataset_dict (dict): A dataset_dict, optionally already mapped.
        show_masks (bool): Overlay instance masks if present.
        show_boxes (bool): Draw instance boxes if present.
        show_points (bool): Draw per-instance points if present (e.g., gt_points).

    Returns:
        matplotlib.figure.Figure: The created figure for further inspection/saving.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # ---- Get image in HWC uint8 ----
    if "image" in dataset_dict and torch.is_tensor(dataset_dict["image"]):
        image = dataset_dict["image"].permute(1, 2, 0).cpu().numpy()
    else:
        image = utils.read_image(dataset_dict["file_name"], format=dataset_dict.get("image_format", None))

    # Ensure uint8 for display
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")

    instances = dataset_dict.get("instances", None)
    if instances is None:
        return fig

    # ---- Masks ----
    if show_masks and hasattr(instances, "gt_masks"):
        gt_masks = instances.gt_masks
        if isinstance(gt_masks, BitMasks):
            masks = gt_masks.tensor.cpu().numpy()
        else:
            masks = gt_masks.cpu().numpy() if torch.is_tensor(gt_masks) else np.asarray(gt_masks)
        for mask in masks:
            ax.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.35)

    # ---- Boxes ----
    if show_boxes and hasattr(instances, "gt_boxes"):
        boxes = instances.gt_boxes.tensor.cpu().numpy()
        for x0, y0, x1, y1 in boxes:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=1.5))

    # ---- Points ----
    if show_points and hasattr(instances, "gt_points"):
        for pts in instances.gt_points:
            if torch.is_tensor(pts):
                pts = pts.cpu().numpy()
            if pts is None or len(pts) == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], s=10, c="red")

    return fig


# ---- Visualization helper (top-level) ----
def visualize_density_maps_grid(dataset_dict, grid_size=3, min_sum=1e-6, cmap="magma"):
    """
    Visualize a grid of density maps from Instances.

    Args:
        dataset_dict (dict): A dataset_dict containing Instances with gt_density_maps.
        grid_size (int): Grid size for visualization, default 3 -> 3x3.
        min_sum (float): Threshold to treat a density map as non-empty.
        cmap (str): Matplotlib colormap name.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    import matplotlib.pyplot as plt

    instances = dataset_dict.get("instances", None)
    if instances is None or not hasattr(instances, "gt_density_maps"):
        raise ValueError("dataset_dict.instances.gt_density_maps is required for visualization")

    density_maps = instances.gt_density_maps
    if torch.is_tensor(density_maps):
        density_maps = density_maps.cpu().numpy()

    n_total = density_maps.shape[0]
    if n_total == 0:
        raise ValueError("No density maps found in instances")

    sums = density_maps.reshape(n_total, -1).sum(axis=1)
    non_empty_idx = [i for i, s in enumerate(sums) if s > min_sum]

    num_slots = grid_size * grid_size
    selected_idx = non_empty_idx[:num_slots]
    if len(selected_idx) < num_slots:
        for i in range(n_total):
            if i not in selected_idx:
                selected_idx.append(i)
            if len(selected_idx) >= num_slots:
                break

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    axes = axes.flatten()

    for ax, idx in zip(axes, selected_idx):
        ax.imshow(density_maps[idx], cmap=cmap)
        ax.set_title(f"idx={idx}, sum={sums[idx]:.2f}")
        ax.axis("off")

    # Hide unused axes if any
    for ax in axes[len(selected_idx):]:
        ax.axis("off")

    fig.tight_layout()
    return fig

