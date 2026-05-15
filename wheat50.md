# wheat50

This document summarizes the improvements discussed and implemented in the project for wheat50, especially around points, density maps, and visualization.

## Data and Mapper Updates

- Added support for COCO-style `points` in dataset registration using `extra_annotation_keys=["points"]`.
- Implemented point augmentation in the dataset mapper by applying geometric transforms and removing points that fall outside the cropped image.
- Added a `generate_density_map` utility to build a Gaussian density map from points.
- Attached `gt_points` to `Instances` and moved the attachment before filtering empty instances to keep alignment.
- Generated `gt_density_maps` per instance and stored them in `instances.gt_density_maps`.

## Density Map Prediction Head

- Added a density prediction branch in the decoder using an MLP similar to the mask head.
- Exposed `pred_density_maps` in model outputs and propagated it through aux outputs.
- Attached `pred_density_maps` to `Instances` during inference so it is accessible in downstream scripts.

## Density Loss (wheat_2 only)

- Implemented `loss_density` in the criterion.
- Loss uses Hungarian-matched pairs and filters to only non-zero density targets (wheat_2 instances).
- Added optional interpolation of GT to match prediction resolution before loss.
- Hooked `loss_density` into `losses` and `weight_dict`.

## Visualization (seg_infer.py)

- Added density map handling in inference output parsing.
- Added heatmap overlay (JET colormap) inside instance masks.
- Added local-maximum peak extraction to visualize predicted points from density maps.
- Only visualize points for `wheat_2` (label == 1).

## Notable Fixes

- Fixed mismatch between `gt_points` and filtered instances by attaching points before filtering.
- Fixed `Instances` indexing error caused by lists in `gt_points` by reattaching after filtering.
- Fixed missing `_extract_density_peaks` definition used during inference.

## Files Touched (key)

- `maskdino/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py`
- `data_register/wheat_data_cat2.py`
- `maskdino/modeling/transformer_decoder/maskdino_decoder.py`
- `maskdino/modeling/criterion.py`
- `maskdino/maskdino.py`
- `myscripts/seg_infer.py`

## Open Considerations

- Density scale mismatch (GT vs pred) may require normalization or non-negative activation.
- The interpolation step may need area-based scaling to preserve total count after resizing.
- Peak detection parameters (kernel size / min value) can be tuned for point density.

