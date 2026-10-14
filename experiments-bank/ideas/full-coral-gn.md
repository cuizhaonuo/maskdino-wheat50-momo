# 想法（完整集）：两两组合消融臂 coral-exist + group-nms-match（A+C）· full/coral-gn

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-11 ｜ 分支 `full/coral-gn`（从 `full/combo`@`eff90f5` 切出）｜ 配置
`wheat_joint_single_stage_full_coral_gn.yaml`（`_BASE_` 为 `wheat_joint_single_stage_full_group_nms_match.yaml`）｜ 输出 `output_v2/exp-1011-full-coral-gn`

## 动机

full 集 4 个 run（p30 基线 / A=coral-exist / B=kmo-matcher / C=group-nms-match / combo=A+B+C）已跑完，
combo 的 `matched_F1@8` 掉到 0.614、低于 A 单点臂的 0.631，但三个单点 + combo 无法归因是哪两个 feature
之间产生了负交互。2^3 消融立方体缺三个两两组合臂（A+C / A+B / B+C），本臂为 **A+C**：
只开 CoralHead 计数头 + Group 分组解码架构，用于判定 **C（GN）对 A（coral）是否无害**——
若 A+C 的 F1 已掉到 A 的噪声带之下，则 C 是 combo F1 下滑的主因之一；若 A+C 与 A 同点，
则 combo 的下滑需由 A+B / B+C 臂或三方交互解释。

## 改动

相对 `full/combo` 只关掉一个 flag：**`POINT_KMO` 不开**（连带 `POINT_KMO_K` / `COST_POINT_KNN_WEIGHT` /
`POINT_KNN_WEIGHT` 均取默认关闭）。config = `full_group_nms_match.yaml` 全键 + `POINT_CORAL_EXIST: True`，
即 combo config 去掉 KM 四键，其余逐键一致。不改 `maskdino/` 任何代码。

组合语义同 combo（方案 b，用户决策 2026-10-09）：slot 层 CE 前缀内层匹配 + CoralHead 逐 slot 存在性；
group 层 GN 逐 token focal + 训练侧点 NMS 匹配，CoralHead 挂实例 query 输出 `pred_point_coral [B,Q,30]`；
推理由 coral 报 n_k、focal+NMS（κ=0.05）按分数取 top-n_k。CoralHead 根级 bias lr ×10 同 combo/CE 臂口径。

## 训练口径

与上一批完全一致：完整集 2092 图，bs4，50 epoch，26000 iter，`EVAL_PERIOD=1000`。单种子。

## 主参照（写死）

- p30 基线：`count_MAE_spike` 2.768、`matched_F1@8` 0.6477。
- A（coral-exist）：MAE **1.394**、F1 **0.6310**。
- C（group-nms-match）：MAE 2.499、F1 0.6493。
- combo：F1 0.614。
- 噪声带：`matched_F1@8` ±0.015、MAE ±0.05。

## 判定线（跑之前写死，原文）

**主判据（归因判定）：** `count_MAE_spike` ≤ **1.444** 且 `matched_F1@8` ≥ **0.616**
（A 的 0.631 − 0.015 噪声带）同点 + guardrails 全过 → 判「**C 对 A 无害**」；
若 MAE ≤ 1.444 但 F1 < 0.616 → 判「**C 拖累 A**」，即为 combo 的 F1 掉到 0.614 的主因；
若 MAE > 1.444 → 判「**C 同时损害计数收益**」。

### 护栏（任一不过则整体不成立）

- `segm/AP` 峰值 ≥ **81.47**、`bbox/AP` 峰值 ≥ **80.38**。
- `spikelet/spike_precision` 末值 ≥ **0.753**。

### 次口径

点位质量结论须同时看 8/16/32px 三个尺度（`matched_F1@8/16/32`），不以 @8 单尺度下结论。

### 机制指标（必读项，只记录）

- `point_nms/suppressed_frac`、`point_nms/starved_frac`（同 GN 臂的机制门：训练侧 NMS 是否启动、是否饿死匹配）。
- n_k vs 候选数差距（coral 报的计数与 focal>0.35 ∧ NMS 存活候选数之差的分布；验证 top-n_k 截断/补足是否主导）。
- CoralHead.bias 收敛性（30 个 bias 相邻有序比例与间隔）。

## 已知限制

- 单种子；100 query / P=30 架构（vs p30 的 300 query），与 p30 对比时需注意。
- 推理 top-n_k 依赖 coral 计数校准；coral 系统性多/少报会直接转化为点多/少报。

## 成本

26000 iter，bs4。结构同 combo 臂减 knn 头，估 24-34 h + 26 次评估。
