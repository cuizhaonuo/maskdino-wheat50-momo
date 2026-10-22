# 想法（完整集）：两两组合消融臂 coral-exist + kmo-matcher（A+B）· full/coral-kmo

> **结果**（2026-10-22 补记）：[exp-2026-10-14-full-coral-kmo](../experiments/exp-2026-10-14-full-coral-kmo.md) —— 落在预注册没有覆盖的格子，记 **计数保持、点位没有叠加**（count 1.3224 / `matched_F1@8` 0.6256，iter 23999）。另：KMO 匹配项因前缀分支先 return 没有生效，本 run 测的是「A + KMO 辅助损失」。

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-11 ｜ 分支 `full/coral-kmo`（从 `full/combo` 切出，merge base `eff90f5`）｜ 配置
`wheat_joint_single_stage_full_coral_kmo.yaml`（`_BASE_` 为 `wheat_joint_single_stage_full_coral_exist.yaml`）｜ 输出 `output_v2/exp-1011-full-coral-kmo`

## 动机

full 集已跑完 p30 基线与三个单点消融臂（A=coral-exist / B=kmo-matcher / C=group-nms-match）与三合一 combo，
但 2^3 消融立方体缺三个两两组合臂。本臂补 A+B：CoralHead 计数头 + KMO 匹配代价，**保持 slot 架构、不开 group 层**
（即相对 full/combo 只关掉 `POINT_GROUP_NMS_MATCH`）。
目的：在不引入 group 层语义改动的前提下，检验「CORAL 有序存在性监督」与「knn 一致性匹配代价」是否可叠加——
A 主攻计数（MAE 2.768→1.394），B 主攻点位质量（F1@8 0.6477→0.6586），二者机制正交，是最可能完全叠加的一对。

## 改动（相对 full/combo）

只关一个 flag：`POINT_GROUP_NMS_MATCH` 不开（默认 False）。
开的 flags（经 `_BASE_` 链 + 本文件）：
- `POINT_CORAL_EXIST: True`（+ `TEST.POINT_THRESHOLD: 0.5`，来自 `full_coral_exist.yaml`）
- `POINT_KMO: True`、`POINT_KMO_K: 3`、`COST_POINT_KNN_WEIGHT: 5.0`、`POINT_KNN_WEIGHT: 5.0`

代码与 full/combo 逐位一致，无 `maskdino/` 改动。

## 训练口径（与上一批一致，写死）

完整集 2092 图，bs4，50 epoch，26000 iter，`EVAL_PERIOD=1000`（共 26 次评估）。单种子。

## 主参照（写死）

- p30 基线 `exp-0922-single-stage-p30`：`count_MAE_spike` **2.768**、`matched_F1@8` **0.6477**、`segm/AP` 81.97、`bbox/AP` 80.88、`spike_precision` 0.7629。
- A 单点 `exp-1009-full-coral-exist`：MAE **1.394**、F1@8 0.6310。
- B 单点 `exp-1009-full-kmo-matcher`：MAE 2.911、F1@8 **0.6586**。
- C 单点 `exp-1009-full-group-nms-match`：MAE 2.499、F1@8 0.6493。

噪声带（写死）：`matched_F1@8` ±0.015，`count_MAE_spike` ±0.05。

## 判定线（跑之前写死）

### 主判据（叠加判定）

| 条件 | 判定 |
|:--|:--|
| `count_MAE_spike` ≤ **1.444**（A 的 1.394 + 0.05）且 `matched_F1@8` ≥ **0.6586**（B 单点峰值），同一点位达到，且护栏全过 | **完全叠加** |
| MAE ≤ 1.444 但 `matched_F1@8` ∈ [0.646, 0.6586) | **计数保持、点位部分叠加** |
| `count_MAE_spike` > **1.444** | **互相干扰** |

（指标取训练期评估曲线上的峰值/同点读数，口径与 combo 预注册一致；阈值在跑之前写死，跑完后只填数不改线。）

### 次口径（必报）

同时报告 8/16/32px 三个尺度的 P/R/F1/matched_F1（点位质量结论须三尺度同看，单看 @8 偏严）。

### 护栏（任一不过则整体不成立）

- `segm/AP` ≥ **81.47**、`bbox/AP` ≥ **80.38**（p30 − 0.5）。
- `spikelet/spike_precision` ≥ **0.753**（p30 − 0.01）。

### 机制读数（必读项，只记录）

- **CoralHead.bias 收敛性**：最终有序 bias 相邻比例与间隔（×10 lr 疗效，与 A 臂同项对照）。
- `loss_point_knn` 曲线（与 B 臂同项对照，确认 knn 监督在 coral 存在性监督共存时正常收敛）。
- slot 层 `loss_point_cls`（CORAL BCE）曲线与 A 臂对照（knn 代价进匹配后存在性头是否受扰）。
- 推理侧 coral 计数 n_k 与实际输出点数的差距分布。

## 已知限制

- 单种子。
- 本臂 slot 层匹配含 knn 代价（与 B 臂口径一致）；注意 combo 的组合语义中 slot 层**不加** knn 代价，
  本臂与 combo 的对比不能纯归因于「有无 group 层」，还夹着 slot 层 knn 代价这一项差异。
- 立方体其余两臂（A+C、B+C）结论未出前，A+B 的「完全叠加/干扰」判定不推广到其他组合。

## 成本

26000 iter，bs4，slot 架构（300 query 口径同 A/B 臂），与上一批单点臂同量级。
