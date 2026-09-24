# 想法：实例 query 300 → 100 的检测代价（为分组 decoder 定 Ni）

状态：**待验证（100 epoch 复跑）** ｜ 提出 2026-09-23 ｜ 分支 `mini/q100`（从 `mini/baseline`@`0f71161` 开出，100 epoch）｜ 配置
`wheat_joint_single_stage_mini_q100.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜ 输出 `output_v2/exp-<启动日期>-mini-q100`

50 epoch 版（输出 `output_v2/exp-0923-mini-q100`，原提交由 tag `archive/mini-q100-ep50` 保留）已判「Ni=100 可用」，
判据与结论保留在文末，不再适用于 100 epoch 复跑。

来源：GroupPose 结合方案讨论（E2）。**这是决策实验，不是改进实验**：它回答分组 decoder（E3）能不能
按 Ni=100 设计，也是 E3 的直接对照臂（同为 100 个实例 query）。

## 为什么要单独量

分组 decoder 给每个实例 query 挂 30 个小穗 token，token 数 = Nq × 31，所以想把 Nq 从 300 降到 100。
但这同时减少了检测的 query 预算。实测每图实例数：

| | 均值 | p90 | p99 | max | GT>100 的图 |
|:--|--:|--:|--:|--:|--:|
| mini train | 55.4 | 83 | 102 | 121 | 7 / 500 |
| 全集 train | 43.5 | 74 | 96 | 121 | 12 / 2092 |
| val | 43.2 | 61 | 82 | 88 | 0 / 200 |

query/GT 比在 mini 上从约 5.4 降到 1.8；两阶段 top-100 的编码器提案要覆盖最多 121 个目标。
GT>100 的训练图在 Nq=100 下必然有 GT 配不上 query（这些 GT 只产生不了正样本，不会报错）。

## 修法

只改 `NUM_OBJECT_QUERIES: 300 → 100`。`DN_NUM`（100，按 GT 数分组、与 query 数无关）与
`TEST.DETECTIONS_PER_IMAGE`（100，每图输出上限，基线同为 100）不变。

注意 `loss_labels` 是对所有 query 的 focal 求和（`sigmoid_focal_loss(...) * num_queries`），
query 变少时负样本随之变少，分类损失的数值会下降约 3 倍（初值实测 140.6 → 36.6），这是 query 数本身的一部分，不另做补偿。

## 100 epoch 复跑：预期与判定（跑之前写死，2026-09-24）

2026-09-24 起 `mini/baseline` 就是 100 epoch 基线（原 `mini/ep100`，`0f71161`）。本节写于任何 100 epoch E2 结果之前，
判定规则沿用 50 epoch 版，基线数值换成 100 epoch；`segm/AP` 的容差按基线实测噪声放宽（见表下说明）。

在 100 epoch 下 E2 有两个角色：一是 E3（`mini/group-decoder`）的同 query 预算对照臂，二是复核 Ni=100 的检测代价
在训练充分后是否仍然可以接受。E3 的 100 epoch 复跑不等本实验的结论，照常按 Ni=100 跑。

对照 mini/baseline（100 epoch，`output_v2/exp-0923-mini-ep100`，数值取自该 run 的 metrics.json）：
`segm/AP` 峰值 78.80 / 末值 78.80（末 4 轮摆幅 0.70），`spike_recall` 峰值 0.9511（iter 8999）/ 末值 0.9429（末 4 轮摆幅 0.011），
`matched_F1@8` 峰值 0.6331。

| 条件 | 判定 | 对 E3 的含义 |
|:--|:--|:--|
| `segm/AP` 峰值 ≥ **77.80**（跌幅 ≤ 1.0）且 `spike_recall` 峰值 ≥ **0.9311**（跌幅 ≤ 0.02） | **Ni=100 可用** | E3 的 Ni=100 设计在 100 epoch 下仍然成立 |
| 任一条不满足 | **Ni=100 代价过大** | E3 的结论只在「接受这份检测代价」的前提下成立；备选方案（实例 query 保持 300，只给编码器分数前 100 的 query 挂小穗组）需另开实验 |

`segm/AP` 的容差从 50 epoch 版的 0.5 改为 1.0：100 epoch 基线末 4 轮的平台摆幅已有 0.70，0.5 会被噪声判出「代价过大」
（与 kmo-matcher 100 epoch 复跑的护栏一致）。`spike_recall` 的 0.02 大于基线末 4 轮摆幅 0.011，保持不变。

观察量（不作判据）：`bbox/AP`、`matched_F1@8`（50 epoch 时意外升了 +0.0145，看 100 epoch 下是否仍在，作为单种子波动量级的旁证）、
`count_MAE_spike`、s/iter 与峰值显存。

## 50 epoch 版：预期与判定（2026-09-23 写死，已按此判，结论见文末）

对照 mini/baseline（`output_v2/exp-0922-single-stage-mini`）：segm AP 峰值 73.67 / 末值 73.52，
spike_recall 峰值 0.9282 / 末值 0.9103，matched_F1@8 峰值 0.5783。

| 条件 | 判定 | 对 E3 的含义 |
|:--|:--|:--|
| `segm/AP` 峰值 ≥ **73.17**（跌幅 ≤ 0.5）且 `spike_recall` 峰值 ≥ **0.9082**（跌幅 ≤ 0.02） | **Ni=100 可用** | E3 按原设计，所有实例 query 都挂小穗组 |
| 任一条不满足 | **Ni=100 代价过大** | E3 改用备选：实例 query 保持 300，只给编码器分数前 100 的 query 挂小穗组，外层匹配时 wheat_2 GT 只配给带组 query |

观察量（不作判据）：`bbox/AP`、`matched_F1@8`（点头是逐实例的，预期基本不变）、`count_MAE_spike`、s/iter 与峰值显存（预期下降）。

## 成本

100 epoch 版：12500 iter，约 10 ~ 13 h（视同卡并发而定）。50 epoch 版：6250 iter，约 6~7 h。

## 100 epoch 结论

（待填）

## 50 epoch 版结论（2026-09-24）

**Ni=100 可用。** 两条判据都满足：`segm/AP` 峰值 73.69（≥ 73.17，与基线 73.67 持平），
`spike_recall` 峰值 0.9233（≥ 0.9082，基线 0.9282）。E3 按原设计（所有实例 query 都挂小穗组）成立。
详见 [exp-2026-09-23-mini-q100](../experiments/exp-2026-09-23-mini-q100.md)。

**一个没预料到的观察**：只砍 query 数，`matched_F1@8` 却从 0.5718 升到 0.5887（末 5 轮均值 +0.0145），
与 E1（iso 归一化）的增幅几乎相同。点头是逐实例的，query 数本不该这样影响它；要么少 query 确实改善了
配对穗的点质量，要么 **单种子 run 之间 `matched_F1@8` 的波动本身就有 0.01~0.015**。在测出种子方差之前，
这个量级的差异（包括 E1 的 +0.015）都不应当作结论。
