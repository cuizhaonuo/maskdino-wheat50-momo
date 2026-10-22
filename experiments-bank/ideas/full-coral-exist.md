# 想法（完整集）：秩一致存在性头 + 前缀内层匹配 · full/coral-exist

> **结果**（2026-10-22 补记）：[exp-2026-10-09-full-coral-exist](../experiments/exp-2026-10-09-full-coral-exist.md) —— 判定 **成立**：后半程峰值 count 1.3690（iter 16999）≤ 2.49、`matched_F1@8` 0.6348（iter 24999）≥ 0.6327，护栏全过。

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-09 ｜ 分支 `full/coral-exist`（从 `full/combo`@`3bd2071` 开出）｜ 配置
`wheat_joint_single_stage_full_coral_exist.yaml`（`_BASE_` 为 `wheat_joint_single_stage.yaml`）｜ 输出 `output_v2/exp-1009-full-coral-exist`
｜ mini 对应臂：`mini/coral-exist`（[exp-2026-09-26-mini-coral-exist](../experiments/exp-2026-09-26-mini-coral-exist.md)：count_MAE_spike 峰值 1.67，
基线 2.46；matched_F1@8 峰值 0.6301 vs 基线 0.6331 在噪声内；护栏过）

**完整集消融口径（用户决策 2026-10-09）**：本臂是「full-combo 合并代码上只开 `POINT_CORAL_EXIST`」的单点消融，
不用 mini 臂的代码。合并代码在 flags 全关时与 `mini/baseline` 逐位一致（S1 硬门禁已过：409 个参数键 + eval 输出逐张量 `torch.equal`）。

## 主参照（写死）

`output_v2/exp-0922-single-stage-p30`（26000 iter / 300 query / P=30 / bs4 / lr 1e-4 cosine，完整集现行基线）最终读数：

| 指标 | p30 末值 |
|:--|--:|
| `segm/AP` | 81.97 |
| `bbox/AP` | 80.88 |
| `spikelet/count_MAE_spike` | **2.768** |
| `spikelet/matched_F1@8` | **0.6477** |
| `spikelet/spike_precision` | 0.7629 |
| `spikelet/spike_recall` | 0.9135 |

注：p30 用 P=30、本臂用 P=25（继承 `wheat_joint_single_stage.yaml`）。P 的差异对计数上限有影响
（train 集仅 6 个实例 >25 点），mini 批次已按同口径对比，此处延续并在判读时注意。

## 与基线的差异

两项配置：`POINT_CORAL_EXIST: True`、`TEST.POINT_THRESHOLD: 0.5`（机制见 mini 臂预注册
[coral-exist](coral-exist.md)，此处不重复）。schedule / 数据 / 其余旋钮与 `wheat_joint_single_stage.yaml` 逐项相同。

**与 mini 臂的非严格同配置声明（跑之前写死）**：合并代码的 `train_net.py` 对 CoralHead 根级 `bias` 单独 lr ×10
（mini CE 确诊 25 个序数 bias 被 lr 卡住未收敛：末值仍接近初始化降序、相邻 bias 间隔未拉开）。mini/full 对比时需注意本臂不等价于 mini CE 的放大版。

## 预期与判否线（跑之前写死）

噪声地板沿用 mini 实验族的 0.015（`matched_F1@8`）。看训练期评估曲线（EVAL_PERIOD 1000，26 个点）的**峰值**与**末值**。

### 主判据：计数 × 定位二维

| 条件 | 判定 |
|:--|:--|
| `count_MAE_spike` 峰值 ≤ **2.49**（p30 ×0.9）且 `matched_F1@8` 峰值 ≥ **0.6327**（p30 − 0.015） | **成立** |
| `count_MAE_spike` 峰值 ≤ 2.49，但 `matched_F1@8` 峰值 < 0.6327 | **计数成立、定位有代价**，记代价大小 |
| `count_MAE_spike` 峰值 ≥ **2.77**（p30 持平或更差） | **判否** |
| 其余 | **未决** |

### 护栏（任一不过则整体不成立）

- `segm/AP` 峰值 ≥ **81.47**、`bbox/AP` 峰值 ≥ **80.38**（p30 − 0.5）。
- `spikelet/spike_precision` 末值 ≥ **0.753**（p30 − 0.01）：计数头若靠少报点提 MAE 会体现在这里。

### 机制读数（必读项，只记录）

- **CoralHead.bias 收敛性**：最终 25 个 bias 相邻有序比例、相邻间隔分布（mini CE 的病灶；×10 lr 是否治愈）。
- 末值每穗有符号偏差（p30 参照需另算；mini CE 为 +0.24 量级）。
- `loss_point_cls`（CORAL BCE ×2 后）起步量级 ~1.3-1.4（冒烟实测 1.28-1.67，合理）。

## 已知限制

- 单种子。
- P=25 vs p30 的 P=30：Ni>25 的穗（train 6 个）计数被封顶，理论上偏向 p30；本臂 MAE 若仍优则结论更强。
- 三处改动（头结构 / 前缀匹配 / 损失口径）绑在一起检验，与 mini 臂相同。

## 成本

26000 iter，bs4。冒烟（bs1）实测 0.9 s/iter、max_mem 6.4G；bs4 估 ~2.5-3.5 s/iter，约 18-25 h + 26 次评估。
