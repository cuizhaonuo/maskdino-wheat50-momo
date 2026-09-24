# 想法：实例 query 300 → 100 的检测代价（为分组 decoder 定 Ni）

状态：**待验证** ｜ 提出 2026-09-23 ｜ 分支 `mini/q100`（从 `mini/baseline` 拉出）｜ 配置
`wheat_joint_single_stage_mini_q100.yaml` ｜ 输出 `output_v2/exp-0923-mini-q100`

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

## 预期与判定（跑之前写死）

对照 mini/baseline（`output_v2/exp-0922-single-stage-mini`）：segm AP 峰值 73.67 / 末值 73.52，
spike_recall 峰值 0.9282 / 末值 0.9103，matched_F1@8 峰值 0.5783。

| 条件 | 判定 | 对 E3 的含义 |
|:--|:--|:--|
| `segm/AP` 峰值 ≥ **73.17**（跌幅 ≤ 0.5）且 `spike_recall` 峰值 ≥ **0.9082**（跌幅 ≤ 0.02） | **Ni=100 可用** | E3 按原设计，所有实例 query 都挂小穗组 |
| 任一条不满足 | **Ni=100 代价过大** | E3 改用备选：实例 query 保持 300，只给编码器分数前 100 的 query 挂小穗组，外层匹配时 wheat_2 GT 只配给带组 query |

观察量（不作判据）：`bbox/AP`、`matched_F1@8`（点头是逐实例的，预期基本不变）、`count_MAE_spike`、s/iter 与峰值显存（预期下降）。

## 成本

6250 iter，约 6~7 h。

## 结论（2026-09-24）

**Ni=100 可用。** 两条判据都满足：`segm/AP` 峰值 73.69（≥ 73.17，与基线 73.67 持平），
`spike_recall` 峰值 0.9233（≥ 0.9082，基线 0.9282）。E3 按原设计（所有实例 query 都挂小穗组）成立。
详见 [exp-2026-09-23-mini-q100](../experiments/exp-2026-09-23-mini-q100.md)。

**一个没预料到的观察**：只砍 query 数，`matched_F1@8` 却从 0.5718 升到 0.5887（末 5 轮均值 +0.0145），
与 E1（iso 归一化）的增幅几乎相同。点头是逐实例的，query 数本不该这样影响它；要么少 query 确实改善了
配对穗的点质量，要么 **单种子 run 之间 `matched_F1@8` 的波动本身就有 0.01~0.015**。在测出种子方差之前，
这个量级的差异（包括 E1 的 +0.015）都不应当作结论。
