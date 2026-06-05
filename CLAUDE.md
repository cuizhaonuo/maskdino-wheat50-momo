# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [MaskDINO](https://github.com/IDEA-Research/MaskDINO) (Detectron2-based) adapted for
**wheat spike instance segmentation / counting** on custom datasets. Upstream MaskDINO is a unified
transformer model for detection + instance/panoptic/semantic segmentation; here it is driven almost
exclusively in **COCO instance-segmentation** mode against agricultural datasets (wheat, maize/rice,
GWHD2021, Xinxiang). The interesting code is the dataset registration glue, the wheat configs, and
the inference/visualization scripts under `myscripts/` — the `maskdino/` model package is largely
upstream.

`detectron2/` is a **vendored, pre-compiled** copy (the engine ships as `.pyc` + a `_C` CUDA `.so`),
not the standard pip install. It is gitignored and not part of the tracked source. Do not assume the
public detectron2 source layout matches what runs here.

## Commands

All commands run from the repo root. `train_net.py` is the single entry point for both training and eval.

```bash
# Train (single GPU). Pick a Wheat-* config; override solver knobs on the CLI as needed.
python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/Wheat-COCO-InstanceSegmentation.yaml

# Multi-GPU train
python train_net.py --num-gpus 8 --config-file <config> MODEL.WEIGHTS /path/to/backbone_or_ckpt.pth

# Evaluate a trained checkpoint. NOTE the flag is --eval_only (underscore), not detectron2's --eval-only.
python train_net.py --eval_only \
  --config-file <output_dir>/config.yaml \
  MODEL.WEIGHTS <output_dir>/model_final.pth \
  OUTPUT_DIR <eval_out> \
  DATASETS.TEST "('gwhd_test',)"

# Sweep-evaluate every experiment subdir against a fixed test set
bash eval_datasets.sh    # edit BASE_DIR / SAVE_ROOT / DATASETS.TEST inside first

# Inference / counting on raw images (custom pipeline, not train_net.py)
python myscripts/unified_wheat_infer.py

# Upstream demo (single images, visualization)
cd demo && python demo.py --config-file <config> --input img.jpg --opts MODEL.WEIGHTS <ckpt>
```

There is no test suite, linter, or build step for the Python code. The only build is the MSDeformAttn
CUDA kernel (see Installation), which only needs recompiling if you change `maskdino/modeling/pixel_decoder/ops`.

### Installation / CUDA kernel
Environment is Python 3.10 + PyTorch + a matching CUDA toolkit. After deps, the deformable-attention
kernel must be compiled once (already built here):
```bash
cd maskdino/modeling/pixel_decoder/ops && sh make.sh   # needs CUDA_HOME set
```
See `INSTALL.md` for the full conda recipe (note: upstream pins py3.8 / torch1.9; this checkout runs py3.10).

## Architecture

**Config-driven, registry-based (Detectron2).** `setup()` in `train_net.py` builds the config by
layering `add_deeplab_config` (poly LR) then `add_maskdino_config` (all `MODEL.MaskDINO.*` knobs,
defined in `maskdino/config.py`) then the YAML then CLI `--opts`. Everything below is selected by string
keys in that merged config — backbones, dataset mappers, and datasets are all looked up from registries.

**Datasets are registered as a side effect of import.** `train_net.py` does
`from data_register import wheat_data, maize_rice_data, xinxiang_data, gwhd2021_data, wheat_data_cat1_quality, wheat_data_cat2, wheat_data_cat3`
purely to run each module's top-level `register_coco_instances(...)` / `DatasetCatalog.register(...)`.
A config's `DATASETS.TRAIN/TEST` names (e.g. `wheat_cat2_coco_train`, `gwhd_test`) must match a name
registered in one of these modules. To add a dataset: create a `data_register/<name>.py` that registers
it, then add it to that import line. Each module also `MetadataCatalog.set(...)` with `thing_classes`,
`thing_colors`, `evaluator_type="coco"`, and a `thing_dataset_id_to_contiguous_id` map.

- **`cat1` / `cat2` / `cat3`** in dataset/config names = number of wheat **categories** (1-class vs
  2-class "wheat"/"wheat_2" vs 3-class). They point at different `instances_*2017.json` under
  `/public/home/zncui/datasets/wheat50/...`. Class count in the config (`MODEL.*NUM_CLASSES`) must match.
- **"points" annotations**: `wheat_data_cat2.py` has a `WITH_POINTS` flag that switches the dataset
  path (`coco_2cat_with_points` vs `coco_2cat`) and loads via `load_coco_json(..., extra_annotation_keys=["points"])`
  so a per-instance `points` field (e.g. spike center points for counting) survives into the dataset
  dicts. This is a project-specific extension threaded through the dataset mappers, matcher, criterion,
  and decoder (`grep -rln points maskdino/`). When touching counting/points logic, follow that chain.

**Training pipeline (`Trainer` in `train_net.py`, subclass of `DefaultTrainer`):**
- `build_train_loader` dispatches on `cfg.INPUT.DATASET_MAPPER_NAME`:
  `coco_instance_lsj` → `COCOInstanceNewBaselineDatasetMapper` (the default for wheat, LSJ aug),
  `coco_instance_detr` → `DetrDatasetMapper`, `coco_panoptic_lsj`, `mask_former_semantic`, else default.
- `build_evaluator` dispatches on the dataset's `evaluator_type` metadata; wheat uses `coco` → `COCOEvaluator`.
- `build_optimizer` builds per-parameter-group AdamW: backbone params get `BACKBONE_MULTIPLIER` LR,
  norm/embedding layers get special weight decay, full-model grad clipping is wrapped on.
- **`SOLVER.GRAD_ACCUM_STEPS`** is a project-added knob (defined in `maskdino/config.py`, default 1)
  passed into `AMPTrainer` to simulate larger batch sizes on few GPUs. Effective batch =
  `IMS_PER_BATCH * GRAD_ACCUM_STEPS`. The wheat configs run small `IMS_PER_BATCH` (4) on one GPU, so
  iteration-based knobs (`MAX_ITER`, `STEPS`, `CHECKPOINT_PERIOD`, `TEST.EVAL_PERIOD`) are hand-tuned
  per config rather than scaled from the upstream bs16/50ep schedule.

**Model package (`maskdino/`, mostly upstream):** three swappable, registry-selected components —
backbone (`modeling/backbone`, incl. a custom `dinov3_sta.py`), pixel decoder / multi-scale deformable
encoder (`modeling/pixel_decoder`, the `ops/` CUDA kernel lives here), and transformer decoder
(`modeling/transformer_decoder/maskdino_decoder.py`). `maskdino/maskdino.py` is the top-level
`MaskDINO` meta-arch. Loss/matching are `modeling/criterion.py` and `modeling/matcher.py`.

## Repo conventions & gotchas

- **Eval flag is `--eval_only`** (underscore), a custom arg added in `train_net.py`'s parser, distinct
  from detectron2's stock `--eval-only`. `eval_datasets.sh` uses the underscore form.
- **Absolute dataset paths** are hard-coded into `data_register/*.py` and the YAML configs
  (`/public/home/zncui/datasets/...`). Moving data means editing these, or set `DETECTRON2_DATASETS`.
- **Config files are heavily commented-out.** The Wheat YAMLs keep several alternative `SOLVER:` blocks
  commented out as a history of tried schedules; only the last uncommented block is active.
- **Outputs:** `output/` (v1) and `output_v2/` (v2) hold experiment dirs, each with `config.yaml` +
  `model_final.pth` — the shape `eval_datasets.sh` and the eval command expect. Both are gitignored.
- **Branches:** active line is `wheat50-model-v2` (main/PR base is `wheat50-model-v1`). Output dir and
  branch are versioned together (`output_v2`).
- **`myscripts/`** is a grab-bag of data-prep and analysis tools (COCO↔YOLO conversion, RLE→polygon,
  cropping to window size, point sampling, counting/metrics under `vis/` and `vis_process/`). These are
  standalone scripts run directly, not wired into `train_net.py`.
- Files prefixed `备份` (Chinese for "backup", e.g. `备份coco_instance_new_baseline_dataset_mapper.py`)
  are dead backup copies — ignore them; edit the un-prefixed file.
