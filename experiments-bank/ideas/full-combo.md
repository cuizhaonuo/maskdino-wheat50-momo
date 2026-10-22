# 想法（完整集）：三合一完整模型 coral-exist + kmo-matcher + group-nms-match · full/combo

> **结果**（2026-10-22 补记）：[exp-2026-10-09-full-combo](../experiments/exp-2026-10-09-full-combo.md) —— 判定 **判否**：`matched_F1@8` 后半程峰值 0.6242 ≤ 0.6477；count 峰值 1.3393 也没到「优于各单点」的 1.3190。另含 2³ 立方体总表。

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-09 ｜ 分支 `full/combo`（merge base `mini/group-nms-match`@`bfe1d7e`，
合并 `mini/coral-exist`@`f029805`、`mini/kmo-matcher`@`3319798`，组合语义提交 `3582dbf`）｜ 配置
`wheat_joint_single_stage_full_combo.yaml`（`_BASE_` 为 `wheat_joint_single_stage_full_group_nms_match.yaml`）｜ 输出 `output_v2/exp-1009-full-combo`

本批次的 4 个实验 = 本模型 + 3 个「在本合并代码上只开对应开关」的单点消融臂（用户决策 2026-10-09）。
flags 全关 ≡ `mini/baseline` 逐位一致（S1 硬门禁已过：409 参数键 + eval 输出逐张量 `torch.equal`）。

## 组合语义（方案 b，用户决策 2026-10-09，跑之前写死）

三个创新点作用在 decoder 的不同层语义上，合并后的分派规则：

| 层 | 语义 | 训练匹配 | 存在性/计数监督 | knn |
|:--|:--|:--|:--|:--|
| layer-0 / interm（MLP 点头） | slot | **CE 前缀匹配**（纯坐标 L1，**不加 knn 代价**） | CoralHead 逐 slot CORAL BCE（`loss_point_cls`） | `loss_point_knn` 照常 |
| 第 1~9 层（小穗 token） | group | **GN 逐 token focal + 训练侧点 NMS 匹配** | token focal（`loss_point_cls`）+ **CoralHead 挂实例 query** 输出 `pred_point_coral [B,Q,30]`，监督 = `coral_levels(min(Ni,30))`（无需匹配，`loss_point_coral`，权重 = `POINT_CLASS_WEIGHT`） | 无（dict 无 `pred_point_knn` 键，criterion 按键分派） |

**推理规则（跑之前写死）**：coral 管「数几个」——`n_k` = 最终层 `pred_point_coral` sigmoid > 0.5 的个数（阈值 0.5 代码内固定，
与 `TEST.POINT_THRESHOLD` 无关）；focal+NMS 管「选哪些」——token 分数 > 0.35 ∧ 贪心 NMS（κ=0.05）存活，按分数取 **top-n_k**。

**CoralHead.bias lr ×10（用户决策 2026-10-09）**：`train_net.py` 对 CoralHead 根级 `bias` 单独 lr ×10
（mini CE 确诊 bias 被 lr 卡住）。full CE 消融臂同样受此参数组影响，mini/full 对比时需注意。

## 主参照（写死）

`output_v2/exp-0922-single-stage-p30` 最终读数：`segm/AP` 81.97、`bbox/AP` 80.88、
`spikelet/count_MAE_spike` **2.768**、`spikelet/matched_F1@8` **0.6477**、`spikelet/spike_precision` 0.7629。
以及本批三个单点消融臂（`exp-1009-full-coral-exist` / `exp-1009-full-kmo-matcher` / `exp-1009-full-group-nms-match`）。

## 与单点臂的差异

`full_combo.yaml` = `full_group_nms_match.yaml` 全键 + `POINT_CORAL_EXIST: True` + KM 四键（`POINT_KMO/K/COST/LOSS`）。
`TEST.POINT_THRESHOLD` 保持 0.35（focal 候选过滤档；n_k 由 coral 内部 0.5 决定）。
架构差异声明同 GN 臂：100 query / P=30（vs p30 的 300 query）。

## 预期与判否线（跑之前写死）

噪声地板 0.015（`matched_F1@8`）。看训练期评估曲线的峰值与末值。

### combo 判定（写死）

| 条件 | 判定 |
|:--|:--|
| `count_MAE_spike` 峰值 ≤ **min(2.49, 三个单点臂最优值 − 0.05)** 且 `matched_F1@8` 峰值 ≥ **max(0.6327, 三个单点臂最优值 + 0.005)**，且护栏全过 | **成立**（组合优于每个单点） |
| 优于 p30 但不优于至少一个单点臂 | **部分成立**：记录相对各单点的增减，归因由消融表给出 |
| `count_MAE_spike` 峰值 ≥ 2.77 或 `matched_F1@8` 峰值 ≤ 0.6477 | **判否** |

（「三个单点臂最优值」指各臂自身曲线上的峰值，逐指标取；阈值在跑之前写死，单点臂跑完后只填数不改线。）

### 护栏（任一不过则整体不成立）

- `segm/AP` 峰值 ≥ **81.47**、`bbox/AP` 峰值 ≥ **80.38**（p30 − 0.5）。
- `spikelet/spike_precision` 末值 ≥ **0.753**（p30 − 0.01）。

### 机制读数（必读项，只记录）

- **n_k vs 候选数差距**：推理侧最终层「coral 报的计数 n_k」与「focal>0.35 ∧ NMS 存活的候选数」之差的分布
  （验证 coral 与 focal 两个计数通道的一致性；差距大说明 top-n_k 截断/补足在主导）。
- **CoralHead.bias 收敛性**：最终 30 个 bias 相邻有序比例与间隔（×10 lr 的疗效；CE 臂同项对照）。
- `point_nms/suppressed_frac`、`point_nms/starved_frac`（同 GN 臂的机制门）。
- `loss_point_knn` 曲线（同 KM 臂的机制门）。
- slot 层与 group 层的 `loss_point_cls` 两条曲线（CORAL BCE 与 focal 的相对收敛）。

## 已知限制

- 单种子。
- 组合语义是新的（mini 各臂只单独验证过）：slot 层不加 knn 代价是写死的设计决策，
  若 combo 不及 KM 单点臂，「knn 代价是否应该进 slot 层前缀匹配」是后续拆解的第一候选。
- 推理 top-n_k 依赖 coral 计数校准；coral 若系统性多/少报会直接转化为点多/少报。

## 成本

26000 iter，bs4。结构同 GN 臂 + CoralHead/knn 头（参数微量），估 24-34 h + 26 次评估。
