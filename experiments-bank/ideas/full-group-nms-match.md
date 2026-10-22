# 想法（完整集）：分组 decoder + 小穗 token 逐层点 NMS 后再内层匹配 · full/group-nms-match

> **结果**（2026-10-22 补记）：[exp-2026-10-09-full-group-nms-match](../experiments/exp-2026-10-09-full-group-nms-match.md) —— 按原文判定 **成立**：count 后半程峰值 2.4669 ≤ 2.49，机制读数与推理-only 对照（2026-10-22 补做）都过。附注：这条线拿本臂峰值比 p30 末值，p30 自己的后半程最低 2.4645 也在线下，同 iter 平均只低 0.17。

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-09 ｜ 分支 `full/group-nms-match`（从 `full/combo`@`3bd2071` 开出）｜ 配置
`wheat_joint_single_stage_full_group_nms_match.yaml`（`_BASE_` 为 `wheat_joint_single_stage.yaml`）｜ 输出 `output_v2/exp-1009-full-group-nms-match`
｜ mini 对应臂：`mini/group-nms-match`（[exp-2026-10-08-mini-group-nms-match](../experiments/exp-2026-10-08-mini-group-nms-match.md)：
**歧义**——主判据与 E0 持平，机制未启动，计数次判据成立但收益来自推理 NMS；MAE 2.120）

**完整集消融口径（用户决策 2026-10-09，两条）**：
1. 本臂是「full-combo 合并代码上只开分组 decoder + NMS 两组开关」的单点消融，不用 mini 臂的代码（flags 全关 ≡ `mini/baseline`，S1 已过）。
2. **NMS 也重训**：不是「在 baseline checkpoint 上只开推理 NMS 评测」，而是训练侧 NMS 匹配 + 推理侧 NMS 同开从头训（与 mini 臂口径一致）。

## 主参照（写死）

`output_v2/exp-0922-single-stage-p30` 最终读数（同批 4 臂共用）：`segm/AP` 81.97、`bbox/AP` 80.88、
`spikelet/count_MAE_spike` **2.768**、`spikelet/matched_F1@8` **0.6477**、`spikelet/spike_precision` 0.7629。

**与 p30 的架构差异声明（跑之前写死）**：本臂 `NUM_OBJECT_QUERIES=100`、`NUM_POINTS=30`；p30 是 300 query / P=30。
100 query 是分组 decoder 架构在 mini 上的验证配置（mini/group-decoder 起即 100），不是本臂的变量之一；
判读「GN 方案 vs p30」时需注意 query 数不同，归因由 mini/group-decoder（E3）与 mini/q100（E2）的对照支撑。

## 与基线的差异

八处配置（=`wheat_joint_single_stage.yaml` 之上的全量 GN 方案）：
`NUM_OBJECT_QUERIES: 100`、`NUM_POINTS: 30`、`POINT_GROUP_DECODER: True`、`POINT_ACROSS_ATTN: "slot"`、
`POINT_DIM_FEEDFORWARD: 1024`、`POINT_WITHIN_GATE: True`、`POINT_COORD_NORM: "iso"`、`POINT_COORD_ROBUST_C: 1.0`、
`POINT_NMS_MODE: "match"`、`POINT_NMS_KAPPA = TEST.POINT_NMS_KAPPA = 0.05`。
机制见 mini 臂预注册 [group-nms-match](group-nms-match.md) 与 [group-decoder](group-decoder.md)，此处不重复。

## 预期与判否线（跑之前写死）

噪声地板 0.015（`matched_F1@8`）。mini 臂的教训：**必须把「训练侧机制收益」与「推理 NMS 白捡的收益」分开**——
mini 上机制未启动（suppressed_frac 虽 12~15% 但 matched_F1 未动），完整集若重演则同样判歧义。

### 主判据：定位 × 计数二维，且要求机制读数配合

| 条件 | 判定 |
|:--|:--|
| `matched_F1@8` 峰值 ≥ **0.6627**（p30 + 0.015）或 `count_MAE_spike` 峰值 ≤ **2.49**（p30 ×0.9），且机制读数成立（下） | **成立** |
| 指标达上表但机制读数不成立 | **歧义**（收益来自推理 NMS，同 mini），记录推理-only 对照读数 |
| `matched_F1@8` 峰值 ≤ **0.6477** 且 `count_MAE_spike` 峰值 ≥ **2.77** | **判否** |
| 其余 | **未决** |

### 机制读数（成立判定的必要补充，跑之前写死）

- `point_nms/suppressed_frac`（训练侧每层被压 token 比例；mini 实测 12~15%）：落在 **5%~40%** 区间外视为机制异常。
- `point_nms/starved_frac`（留下的少于 Ni 的实例比例）：> 5% 视为过度抑制。
- **推理-only 对照**（训练完成后做一次，不算训练 run）：在 p30 的 `model_final` 上只开推理 NMS（κ=0.05）评测，
  若本臂相对 p30 的指标增量 ≤ 推理-only 的增量，则收益归因于推理 NMS 而非训练侧机制（mini 臂即此结局）。

### 护栏（任一不过则整体不成立）

- `segm/AP` 峰值 ≥ **81.47**、`bbox/AP` 峰值 ≥ **80.38**（p30 − 0.5）。
- `spikelet/spike_precision` 末值 ≥ **0.753**（p30 − 0.01）：NMS 若靠压点提精度会体现在这里（mini 护栏 G 正是此项未过）。

## 已知限制

- 单种子。
- 与 p30 比有 100q/300q 的架构差异（见上声明）；与 mini E3 的对比才是同架构对照，但 mini/full 数据分布不同
  （mini train 密度偏置：wheat_2 占比 9.9% vs 完整集 5.9%）。
- 推理 NMS 的 κ=0.05 是 mini 全 val 诊断后定的，完整集 val 与 mini val 同一集合，沿用合理。

## 成本

26000 iter，bs4。分组 decoder 每层多 100×30 个小穗 token 的注意力，mini 实测比基线慢约 1/3；
冒烟（bs1）与 point_nms 读数见启动后补记。估 24-34 h + 26 次评估。
