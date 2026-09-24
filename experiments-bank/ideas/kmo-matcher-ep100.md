# 想法：KMO 内层点匹配在 100 epoch 下复核

状态：**预注册判定：未决**（机制门通过；`matched_F1@8` 峰值 0.6529 过线、末值 0.6454 差 0.0008 未过）｜ 验证于 [exp-2026-09-24-mini-kmo-matcher-ep100](../experiments/exp-2026-09-24-mini-kmo-matcher-ep100.md) ｜ 提出 2026-09-24 ｜ 分支 `mini/kmo-matcher`（原名 `mini/kmo-matcher-ep100`，2026-09-24 改名；从 `mini/baseline`@`0f71161` 开出，
带上旧 50 epoch 分支 `mini/kmo-matcher` 的 KMO 实现与 50 epoch 结果；100 epoch 的 schedule 直接写在本分支配置里）

> 分支来源（2026-09-24 补）：本分支最初从旧的 50 epoch 分支 `mini/kmo-matcher`（挂在 `1ffe75f`，已删除，原提交见 tag `archive/mini-kmo-matcher-pre-move`）拉出，
> `output_v2/exp-0924-mini-kmo-matcher-ep100` 于 11:22 从 `1a9a3f3` 启动。run 开跑后，经用户批准，
> 把 `f3ced48` / `512f281` / `1a9a3f3` 重放到 `mini/baseline`@`0f71161` 上（提交信息带 `cherry picked from` 标注），
> 与其余 `mini/*-ep100` arm 同源。移动前后，代码与 `wheat_joint_single_stage_mini_kmo_ep100.yaml` 逐字相同，
> 只多了 `mini/baseline` 带来的 3 篇文档和 `wheat_joint_single_stage_mini_ep100.yaml`，因此该 run 的结果仍对应本分支的代码。
>
> 命名（2026-09-24 补）：用户规定 `mini/baseline` 就是 100 epoch 基线、分支名不再带 `-ep100`，本分支随之由
> `mini/kmo-matcher-ep100` 改名为 `mini/kmo-matcher`（run 的输出目录名 `exp-0924-mini-kmo-matcher-ep100` 不变）。
> `mini/ep100` 分支已删除，**下文出现的 `mini/ep100` 一律指 `mini/baseline`@`0f71161`**；
> 它的实验记录现在就在本分支上：[exp-2026-09-23-mini-longer-schedule](../experiments/exp-2026-09-23-mini-longer-schedule.md)。
> 启动提交 `1a9a3f3` 由 tag `archive/mini-kmo-matcher-ep100-pre-move` 保留。

## 动机

[kmo-matcher](kmo-matcher.md) 在 50 epoch 下的结果：`matched_F1@8` 峰值 / 末值为 0.5962 / 0.5954，
对照 mini/baseline 的 0.5783 / 0.5718，末段稳定领先约 +0.024。但预注册的机制门 M1 没过，判定为「机制未启动」。
见 [exp-2026-09-23-mini-kmo-matcher](../experiments/exp-2026-09-23-mini-kmo-matcher.md)。

随后 `mini/ep100` 分支的 `experiments-bank/experiments/exp-2026-09-23-mini-longer-schedule.md` 证明
**50 epoch 对 mini 是被 cosine 掐断的**：只把 schedule 拉到 100 epoch，基线 `matched_F1@8` 峰值就从 0.5783 升到 0.6331，
segm AP 从 73.67 升到 78.80。那份记录据此把 `mini/*` 的默认 schedule 改为 100 epoch。

所以 KMO 的 +0.024 是在一个**训练不充分的基线**上测出来的，有两种解释，50 epoch 的数据分不开：

1. **KMO 真的改善了点定位**：在训练充分的基线上，增益应当仍在。
2. **KMO 只是让训练收敛得更快**：基线训练充分后，增益会缩小甚至消失。
   50 epoch 下 KMO 臂从 ep 14 起就一直领先，这与「加速」也相容。

本实验就是把两者分开：同样的 KMO，换成 100 epoch，对照 mini/ep100。

## 修法

没有代码改动，沿用 KMO 50 epoch 的实现（`f3ced48`，本分支上对应的重放提交是 `c8525dc`）。新配置 `wheat_joint_single_stage_mini_kmo_ep100.yaml`
以 KMO 50 epoch 的配置为 `_BASE_`，把 schedule 的 5 项显式写出，数值逐项取自 `mini/ep100` 分支的
`wheat_joint_single_stage_mini_ep100.yaml`：MAX_ITER 12500、WARMUP_ITERS 250、STEPS (10000, 11500)、
CHECKPOINT_PERIOD 500、EVAL_PERIOD 500。

启动前用解析后的 cfg 逐项核对过（`mini/ep100` 的配置取自其提交版本，经 `git archive` 导出）：

- 相对 mini/ep100，**只差 `POINT_KMO`**；
- 相对 KMO 50 epoch，**只差 schedule 的 4 项**（MAX_ITER、STEPS、CHECKPOINT_PERIOD、EVAL_PERIOD；WARMUP_ITERS 两边都是 250）。

## 预期与判否线（跑之前写死）

对照 mini/ep100（`output_v2/exp-0923-mini-ep100`）。数值取自该 run 的 metrics.json，
以及 `mini/ep100` 分支上的 `exp-2026-09-23-mini-longer-schedule.md`：

| 指标 | 峰值 | 末值 | 末 4 轮摆幅 |
|:--|--:|--:|--:|
| `matched_F1@8` | **0.6331**（iter 11499） | **0.6312** | 0.004 |
| `segm/AP` | 78.80（iter 12499） | 78.80 | 0.70 |
| `bbox/AP` | 77.06 | 77.05 | 0.63 |
| `count_MAE_spike` | 2.46（iter 8999） | 2.60 | 0.07 |

噪声地板沿用 mini 实验族的 **0.015**（来自 mini/baseline 末 8 轮的摆幅）；ep100 自己的末 8 轮摆幅是 0.011，
0.015 更保守。

### 机制门（先判；按 50 epoch 的教训改写）

50 epoch 的 M1 用一个 MAE 同时考了「穗间水平」和「穗内信号」，要求赢过知道每个穗真实中位数的 oracle，过严了
（见该实验记录的「事后分解」）。KMO 只能通过 `ŷ^k` 在**同一个穗内各槽之间的差异**改变配对，所以这次直接检验穗内信号。

工具：`myscripts/backup/vis_process/kmo_match_diag.py`，参数与 50 epoch 时相同（mini train，100 个 batch，seed 0）。
在 `model_0005999.pth`（48 ep，训练过半）和 `model_final.pth` 上各跑一次。

- **M1′（穗内有信号）**：最终 checkpoint 的实例内 Pearson r（`posthoc_within_inst_pearson_r`）**≥ 0.3**。
  50 epoch 时为 0.47（iter 2999 时 0.41）。不通过 → 判**「机制未启动」**。
- **M2（不是空操作）**：最终 checkpoint 的推翻率 **≥ 1%**。若推翻率与换 tie 率都 < 1% → 判**「空操作」**。
- 50 epoch 的旧 M1（L1-only 配对下的 MAE vs 0.0158）照常报告，以便前后对照，**不作判据**。

### 主判据：`matched_F1@8`

| 条件 | 判定 |
|:--|:--|
| 峰值 ≥ **0.6481** 且末值 ≥ **0.6462**（都是 ep100 基线 +0.015），且护栏满足 | **成立**：增益在训练充分的基线上仍在，不只是加速 |
| 峰值 ≤ **0.6331** | **判否**：50 epoch 的增益来自加速或种子，100 epoch 下不存在 |
| 其余（峰值落在 (0.6331, 0.6481)，或峰值过线但末值没过） | **未决**：可能是部分加速。单种子分辨不了，需要两臂各补一个种子 |

### 护栏

- `segm/AP` 峰值 ≥ **77.80**（ep100 基线 78.80 − 1.0；基线末 4 轮的平台摆幅是 0.70）。
- `count_MAE_spike` 峰值 ≤ **2.71**（2.46 × 1.1）。
  50 epoch 的教训是这一项主要反映阈值标定：两臂都系统性多数，ep100 基线最终也是每穗平均多 2.4 个点。
  所以同时报告逐穗偏差（预测 − GT），它若只是整体平移，不按真退步处理。

### 观察量（只记录，不作判据）

- `bbox/AP`：50 epoch 时 KMO 末值 +1.97，内层点匹配不直接作用于框。看它在 100 epoch 下是否仍在。
- 加速的直接证据：KMO 臂与 ep100 基线在同 epoch 的差距随训练进程怎么变。
  若差距在前半程大、后半程收窄到噪声内，就是解释 2。

## 已知限制

- 单种子，与之前所有 mini 实验一致。
- KMO 的「匹配代价」与「`ŷ^k` 辅助损失」仍绑在一起，拆分要靠 `COST_POINT_KNN_WEIGHT=0` 的消融。
- 计划与 mini/q100 叠在 xuyu GPU1 上并发（该仓库的训练受 dataloader 限制，并发几乎不损速）。
  用户同时占用的卡数不超过 2 张。

## 成本

12500 iter。ep100 基线是 10:49（前 3000 iter 与 p30 并发）。本次全程并发，预计约 13 ~ 15 h；
另加两次诊断约 10 min。

## 结论（2026-09-24）

**预注册判定：「未决」。** 详见 [exp-2026-09-24-mini-kmo-matcher-ep100](../experiments/exp-2026-09-24-mini-kmo-matcher-ep100.md)。

| | KMO 100 epoch | ep100 基线 | 线 |
|:--|--:|--:|--:|
| M1′ 穗内 Pearson r（终） | **0.54** | — | ≥ 0.3 ✅ |
| M2 推翻率（终） | **12.6%** | — | ≥ 1% ✅ |
| `matched_F1@8` 峰值 / 末值 | **0.6529 / 0.6454** | 0.6331 / 0.6312 | ≥ 0.6481 ✅ / ≥ 0.6462 ❌（差 0.0008） |
| `segm/AP` 峰值 | 78.60 | 78.80 | 护栏 ≥ 77.80 ✅ |
| `count_MAE_spike` 峰值 | 2.49 | 2.46 | 护栏 ≤ 2.71 ✅ |

- **机制成立**：`ŷ^k` 的穗内信号比 50 epoch 时更强（r 0.47 → 0.54），KMO 仍推翻约八分之一实例的 L1 配对。
- **两种解释都部分成立**：末段对基线的领先从 50 epoch 的 +0.024 缩到 +0.015（约缩小 40%），
  `bbox/AP` 的 +1.97 完全消失，说明 50 epoch 的增益有相当一部分是「收敛更快」；
  但 100 epoch 末段仍稳定领先约 +0.015（52 ep 起 13 次评估全部领先），这个残余量恰好等于噪声地板。
- 预注册写的「未决 → 两臂各补一个种子」**不执行**：用户 2026-09-24 决定 mini 实验保持单种子。记为未决。

