# 想法（完整集）：两两组合消融臂 B+C · kmo-matcher + group-nms-match · full/kmo-gn

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-11 ｜ 分支 `full/kmo-gn`（从 `full/combo`@`eff90f5` 切出，
不改任何 `maskdino/` 代码）｜ 配置 `wheat_joint_single_stage_full_kmo_gn.yaml`（`_BASE_` 为
`wheat_joint_single_stage_full_group_nms_match.yaml`）｜ 输出 `output_v2/exp-1011-full-kmo-gn`

## 动机

full 集已跑完 4 个 run（p30 基线 / A coral-exist / B kmo-matcher / C group-nms-match / combo ABC）。
本臂补全 2^3 消融立方体的 **B+C 顶点**（KMO 匹配代价 + Group 分组解码架构，不含 coral 存在性头），
与另两个两两组合臂（A+B、A+C）一起，把「单点 → 两两 → 三合一」的交互结构填完整，
使 B、C 两点在有/无对方存在时的边际效应可分解、combo 的增益可归因。

## 改动（相对 full/combo）

只关掉一个 flag：**`POINT_CORAL_EXIST`（不开）**。其余配置与 `full/combo` 完全一致
（KMO 四键 `POINT_KMO/K=3/COST_POINT_KNN_WEIGHT=5.0/POINT_KNN_WEIGHT=5.0` 保持开启，
GN 分组解码架构继承自 `_BASE_`）。代码零改动。

## 训练口径（与上一批一致）

完整集 2092 图，bs4，50 epoch，26000 iter，`EVAL_PERIOD=1000`。单 GPU。

## 主参照（写死）

- p30 基线（`exp-0922-single-stage-p30`）：`count_MAE_spike` 2.768、`matched_F1@8` 0.6477。
- 单点臂读数：A `MAE=1.394/F1=0.6310`；B `MAE=2.911/F1=0.6586`；C `MAE=2.499/F1=0.6493`。
- combo（ABC）：作归因对照。

## 判定线（跑之前写死，原文）

主判据（交互判定）：

| 条件 | 判定 |
|:--|:--|
| `matched_F1@8` >= **0.6627**（B 单点预注册线）且 `count_MAE_spike` <= **2.49**（同点），且 guardrails 全过 | **「B×C 交互有效」** |
| 仅一项达标 | **「无增益交互」** |
| 均不达标 | **「无显著交互」** |

本臂主要为补全 2^3 消融立方体。

### 护栏（guardrails，任一不过则整体不成立）

- `segm/AP` >= **81.47**
- `bbox/AP` >= **80.38**
- `spikelet/spike_precision` >= **0.753**

### 噪声带

`matched_F1@8` ±0.015，`count_MAE_spike` ±0.05。

## 次口径

点位质量看 8/16/32px 三个尺度（@3/5/8 偏严，不单独作结论）。

## 机制指标（只记录）

- `point_nms/suppressed_frac`（GN 臂机制门）。
- `loss_point_knn` 轨迹（KMO 臂机制门）。

## 已知限制

- 单种子。
- 本臂不含 coral 计数头，推理计数走 GN 默认路径（继承 `_BASE_`），与 combo 的 top-n_k 规则不可直接对比。

## 成本

26000 iter，bs4，结构与 GN 臂同级，估 24-34 h + 26 次评估。
