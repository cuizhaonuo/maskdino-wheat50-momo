# Baseline：Stage2 小穗点头

两阶段迁移的第二阶段：冻结 Stage1 检测器，只训练小穗点头。
后续所有点回归实验都以此为对照基准。

## 配置

`configs/coco/instance-segmentation/wheat_stage2_pointhead.yaml`
（`_BASE_: maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml`）

| 项 | 值 |
|:--|:--|
| 起始权重 | `output_v2/wheat_stage1_detector_v2_cosine/model_final.pth` |
| 数据 | `wheat_cat2_pts_coco_train` / `_val` |
| 冻结 | `FREEZE_EXCEPT_POINT_HEAD: True` → 可训练 170,134 / 冻结 52,017,990 |
| 点头 | `NUM_POINTS: 50`，坐标为**实例框相对**归一化 [0,1] |
| 点损失权重 | `POINT_COORD_WEIGHT: 5.0`（L1）、`POINT_CLASS_WEIGHT: 2.0`（focal α=.25 γ=2） |
| 内层匹配代价 | `COST_POINT_COORD_WEIGHT: 0.0`、`COST_POINT_CLASS_WEIGHT: 0.0` |
| 检测部分 | `NUM_OBJECT_QUERIES: 300`，`DEC_LAYERS: 9`，`DN: seg`，`TWO_STAGE: True` |
| SOLVER | `IMS_PER_BATCH: 4`，`MAX_ITER: 8000`，`STEPS: (6400, 7400)`，`WARMUP_ITERS: 500`，`BASE_LR: 1e-4`，**WarmupMultiStepLR**，AMP on |
| 测试分辨率 | `MIN_SIZE_TEST = MAX_SIZE_TEST = 1024`（val 图原生 1024²，不缩放） |
| 评估周期 | `EVAL_PERIOD: 800`，`CHECKPOINT_PERIOD: 800` |

⚠ 输出目录名里的 `_cosine` 来自 Stage1 权重的名字，**Stage2 本身用的是 MultiStep 不是 cosine**。

## 数据

| split | 图 | 实例 | wheat_2 | 小穗点 |
|:--|--:|--:|--:|--:|
| train | 320 | 12885 | 988 | 13636 |
| val | 81 | 3187 | 257 | 3592 |

只有 `wheat_2` 带点（已清洗为可断言的不变量）。val 与 train 共享 DJI 母图，指标偏乐观。

## 评估

`COCOEvaluator`（bbox/segm AP）+ `SpikeletPointEvaluator`：
σ = {3, 5, 8} px（原图像素），实例配对 mask IoU ≥ 0.5，预测穗 score ≥ 0.4，
点存在性阈值 `POINT_THRESHOLD: 0.35`。

主看指标：`spikelet/mF1`（挑 checkpoint）、`spikelet/matched_F1@5`（点头本身的进展，
已剥离检测误差）。

## 运行

```bash
cd /public/home/zncui/pycharm_project/maskdino-wheat50
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /public/home/zncui/anaconda3/envs/MaskDINO-310/bin/python train_net.py \
  --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_stage2_pointhead.yaml \
  OUTPUT_DIR output_v2/wheat_stage2_pointhead_v2_cosine
```
