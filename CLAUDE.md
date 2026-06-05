# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of **MaskDINO** (detectron2-based unified detection/segmentation) customized for
**wheat spike (麦穗) instance segmentation** — the "wheat50" project. The upstream MaskDINO
README/configs for COCO/ADE20K/Cityscapes are kept intact; all project-specific work lives in
`data_register/`, the `Wheat-*`/`Maize-*` configs, `myscripts/`, and the `maskdino/` model code.

Two object classes: `wheat` (category id 1 → contiguous 0) and `wheat_2` (id 2 → contiguous 1).
Only `wheat_2` spikes carry a per-instance `points` field (spikelet/小穗 coordinates) in the COCO
annotations. The research goal is a single end-to-end model that, per `wheat_2` instance, predicts
the mask **and** the internal spikelet points (for yield estimation), while `wheat` instances stay
point-free.

**Research trajectory (important context):**
- The `wheat50-model-v2` branch and `output_v2/*density_map*` experiments used a **Gaussian density-map**
  head (now abandoned).
- The current `wheat-model-v2.x` direction is **nested set prediction**: per-instance point heads with
  an inner Hungarian match. The full design lives in
  `memory-bank/小穗点回归方案.md` — read it before touching point/spikelet code. As of HEAD the data
  loads `points` (`data_register/wheat_data_cat2.py`, `WITH_POINTS=True`) but the point heads/losses
  are **not yet implemented in the working tree** (they exist on a separate branch). The plan doc is the
  source of truth for the intended file-by-file changes.
- `memory-bank/{architecture,implement-plan,progress}.md` are currently empty placeholders.

## Setup & build

```bash
pip install -r requirements.txt
# Compile the MSDeformAttn CUDA kernel (required before training/inference):
cd maskdino/modeling/pixel_decoder/ops && sh make.sh
```

**Vendored detectron2:** this repo contains a *modified* copy of detectron2 under `detectron2/`
(see commit "将修改后的 detectron2 源码纳入主项目"). It is intentionally not the pip-installed upstream —
e.g. `AMPTrainer`/`SimpleTrainer` were patched to accept `grad_accum_steps`. Install it editable
(`pip install -e detectron2`) so the patched version is what gets imported; do not assume a stock
detectron2 will work.

## Train / eval

Entry point is `train_net.py` (subclasses detectron2's `DefaultTrainer`). Dataset registration happens
purely as an **import side-effect** at the top of `train_net.py`
(`from data_register import wheat_data, ... wheat_data_cat2, ...`) — to add a dataset, create a
`data_register/*.py` that calls `DatasetCatalog.register(...)` and add it to that import line.

```bash
# Train (single GPU). Configs layer: a full MaskDINO config + project overrides.
python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
  OUTPUT_DIR output_v2/<run_name>

# Eval only
python train_net.py --eval-only --num-gpus 1 \
  --config-file <OUTPUT_DIR>/config.yaml \
  MODEL.WEIGHTS <OUTPUT_DIR>/model_final.pth \
  DATASETS.TEST "('wheat_cat2_coco_val',)"
```

- `eval_datasets.sh` batch-evaluates every experiment subdir under a base output dir.
- Any `KEY VALUE` pairs after the config file override config values (detectron2 convention).
- The `Wheat-COCO-InstanceSegmentation.yaml` / `Wheat-Swin-*` configs are **partial override files**
  (datasets + solver + backbone only, no model head) — they are not self-contained and rely on a full
  MaskDINO base config being merged in. `maskdino_R50_bs16_50ep_3s.yaml` and friends use `_BASE_:`.
- Datasets are expected at `/public/home/zncui/datasets/wheat50/<variant>/` in COCO layout
  (`annotations/instances_{train,val}2017.json`, `{train,val}2017/`). Variants: `coco_2cat`,
  `coco_2cat_with_points`, etc., selected inside the `data_register/*.py` files.

## Architecture (model code under `maskdino/`)

Standard detectron2 plug-in layout; config knobs are registered in
`maskdino/config.py:add_maskdino_config` (adds `cfg.MODEL.MaskDINO.*` and `cfg.SOLVER.GRAD_ACCUM_STEPS`).
The forward pass, for an instance-seg run:

- `maskdino/maskdino.py` — the `MaskDINO` meta-architecture (registered, `META_ARCHITECTURE: MaskDINO`).
  Owns `prepare_targets*` (builds GT dicts incl. image-normalized boxes/points), the training loss call,
  and `instance_inference` (post-processing that turns query outputs into `Instances`). This is where new
  per-instance outputs (e.g. spikelet points) get threaded into both targets and inference.
- `maskdino/modeling/meta_arch/maskdino_head.py` — wraps pixel decoder + transformer decoder.
- `maskdino/modeling/pixel_decoder/maskdino_encoder.py` — multi-scale deformable-attention encoder
  (the CUDA op compiled above lives under `pixel_decoder/ops/`).
- `maskdino/modeling/transformer_decoder/maskdino_decoder.py` — the DINO-style decoder.
  `forward_prediction_heads` (≈line 577) produces per-query `(class, box, mask)` per layer; this is the
  insertion point for additional prediction heads, and any new head must also be handled in
  `dn_post_process` (denoising split) and `_set_aux_loss` (deep supervision) to stay aligned.
- `maskdino/modeling/criterion.py` — `SetCriterion`: outer Hungarian match (via
  `maskdino/modeling/matcher.py`) + per-loss functions. New losses are registered in its `loss_map`.
- `maskdino/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py` — the LSJ instance mapper
  (`DATASET_MAPPER_NAME: "coco_instance_lsj"`). Augmentation-time handling of extra annotation keys
  (e.g. applying transforms to `points`) belongs here.

Custom backbones live in `maskdino/modeling/backbone/` (`swin.py`, `focal.py`, `dinov3_sta.py`) with
matching configs under `configs/coco/instance-segmentation/{swin,dinov3}/`.

`myscripts/` holds standalone data-prep / visualization / inference utilities (COCO↔YOLO conversion,
GT visualization, sliding-window cropping, density/point visualization, `unified_wheat_infer.py`). They
are run directly, not part of the training pipeline.

## Conventions

- Coordinate systems must stay consistent: GT boxes and GT points are both image-normalized by the same
  pre-pad `(w, h)` in `prepare_targets`. The point-regression plan additionally uses box-relative
  normalization — see `memory-bank/小穗点回归方案.md` for the exact convention before changing it.
- Denoising (DN) queries: any new per-query head added to the decoder must be split in `dn_post_process`
  exactly like masks, and DN losses/weights must **exclude** the new head, or matching/loss indices will
  desync. This is the most common breakage point when extending the model.

# 写任何代码前必须阅读 memory-bank/process.md
# 写任何代码前必须阅读 memory-bank/architecture.md
# 每完成一个重大功能或里程碑后，必须更新 memory-bank/architecture.md