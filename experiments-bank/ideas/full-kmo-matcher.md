# 想法（完整集）：KMO 内层点匹配 · full/kmo-matcher

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-09 ｜ 分支 `full/kmo-matcher`（从 `full/combo`@`3bd2071` 开出）｜ 配置
`wheat_joint_single_stage_full_kmo.yaml`（`_BASE_` 为 `wheat_joint_single_stage.yaml`）｜ 输出 `output_v2/exp-1009-full-kmo-matcher`
｜ mini 对应臂：`mini/kmo-matcher`（[exp-2026-09-24-mini-kmo-matcher-ep100](../experiments/exp-2026-09-24-mini-kmo-matcher-ep100.md)：
预注册判定**未决**——机制门通过（knn 代价确实改变配对），matched_F1@8 峰值 0.6529 过线、末值 0.6454 差 0.0008 未过；count_MAE 2.450 在噪声内）

**完整集消融口径（用户决策 2026-10-09）**：本臂是「full-combo 合并代码上只开 `POINT_KMO`」的单点消融，不用 mini 臂的代码。
合并代码 flags 全关 ≡ `mini/baseline` 逐位一致（S1 硬门禁已过）。

## 主参照（写死）

`output_v2/exp-0922-single-stage-p30` 最终读数（同批 4 臂共用）：`segm/AP` 81.97、`bbox/AP` 80.88、
`spikelet/count_MAE_spike` **2.768**、`spikelet/matched_F1@8` **0.6477**、`spikelet/spike_precision` 0.7629。

## 与基线的差异

四处配置：`POINT_KMO: True`、`POINT_KMO_K: 3`、`COST_POINT_KNN_WEIGHT: 5.0`、`POINT_KNN_WEIGHT: 5.0`
（后三项等于 config.py 默认值，显式写出）。机制：内层匹配代价加 KNN 平均距离项
`C = 5·|p−p̂|₁ + 5·|y^k−ŷ^k| + 2·focal(ĉ)`（CLTR, ECCV 2022 §3.3），ŷ^k 由 `point_knn_embed` 逐槽预测、
匹配上的槽 L1 监督（`loss_point_knn`）。schedule / 数据 / 其余旋钮逐项相同。
与 mini 臂配置差异：无（mini 臂同样四键；schedule 不同是 mini/full 批次本身的差异）。

## 预期与判否线（跑之前写死）

噪声地板 0.015（`matched_F1@8`）。CLTR Table 8 的效果量约 1 个 F1 点（UCF-QNRF）。

### 主判据：定位为主、计数不伤

| 条件 | 判定 |
|:--|:--|
| `matched_F1@8` 峰值 ≥ **0.6627**（p30 + 0.015）且 `count_MAE_spike` 峰值 ≤ **2.91**（p30 ×1.05） | **成立** |
| `matched_F1@8` 峰值 ≤ **0.6477**（p30 持平或更差） | **判否** |
| 其余 | **未决**（与 mini 判定同口径：机制门单独复核，见下） |

### 机制门（成立判定的必要补充，跑之前写死）

若主判据落在「成立」区间，必须复核 KMO 机制真的在起作用：`loss_point_knn`（×5 后）从初始 ~5·E|y^k−ŷ^k|
显著下降且收敛值远低于初始值（mini 臂实测从 ~2.9 降到 ~0.6）。若损失不降而指标动了，归因不成立。

### 护栏（任一不过则整体不成立）

- `segm/AP` 峰值 ≥ **81.47**、`bbox/AP` 峰值 ≥ **80.38**（p30 − 0.5）。
- `spikelet/spike_precision` 末值 ≥ **0.753**（p30 − 0.01）。

### 机制读数（必读项，只记录）

- `loss_point_knn` 全曲线（主输出 + 9 份 aux + interm）。
- 内层匹配配对改变率（mini 臂的做法：同一 GT 集合上 KMO 配对 vs 纯 L1 配对的一致率，若复核需要另跑诊断脚本）。

## 已知限制

- 单种子。
- mini 臂 100 epoch 下「未决」：完整集数据量 4 倍、schedule 按 iter 计（50 epoch），效果可能不同；
  本臂是一锤定音的完整集检验，不再补 mini 复跑。

## 成本

26000 iter，bs4。与基线相比多一个逐槽 knn 头（参数微量）与内层代价的一次 cdist，预计与基线近同速，约 18-25 h + 26 次评估。
