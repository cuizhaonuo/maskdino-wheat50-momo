# 想法：把解冻组 A/B/C 搬到真实数据集（320 图）

状态：**待验证** ｜ 提出 2026-08-27 ｜ 承接
[unfreeze-decoder](unfreeze-decoder.md) / [unfreeze-all-heads](unfreeze-all-heads.md)

## 为什么现在必须做这一步

真实数据上我们只有**一行**可信数字：

| | 训练数据 | val mF1 | val 计数 MAE |
|:--|:--|--:|--:|
| **innercost**（冻结 trunk） | **320 图** | **0.1670** @0.20 | **7.6** @0.30 |
| 冻结 + canonical | 只有 10 图 | 0.1074 | 2.38 |
| 解冻 decoder（arm A） | 只有 10 图 | 0.1538 | 4.37 |
| 冻结 baseline | 只有 10 图 | 0.1495 | 8.60 |

除 innercost 外**没有任何改动在真实数据上训练过**。
[exp-2026-08-27-unfreeze-val-generalization](../experiments/exp-2026-08-27-unfreeze-val-generalization.md)
已经显示解冻带来的点头能力提升是真的（`matched_F1@8` 0.3429 → 0.41~0.42，+20~24%），
但端到端 `mF1` 被检测器退化抵消（`spike_precision` 0.7529 → 0.55）。
该退化有两个成因，其中类别先验偏移（overfit10 的 `wheat_2` 占 14.0%，
val 8.1% / 真实训练集 7.7%）**在真实配置上不存在**。所以这一步是决定性的。

## 三个 arm（与 overfit10 那批逐项同构，只换数据集）

| arm | `UNFREEZE_KEYS` | 可训练参数 | overfit10 上的表现 |
|:--|:--|--:|:--|
| **A** | `["predictor.decoder"]` | 14,413,082 | 点任务最好（F1@8 0.9987），val matched_F1@8 也最好 0.4236 |
| **C** | + `class_embed` + `mask_embed` | 14,610,972 | 掩码零收益，点任务反而变差 |
| **B** | `["sem_seg_head"]` | 28,733,212 | 训练集 segm AP 99.30 最高，val 50.57 最低（过拟合最重） |

新配置 `configs/coco/instance-segmentation/wheat_stage2_unfreeze.yaml`，
`UNFREEZE_KEYS` 走 CLI，其余七项损失权重全非零、schedule 沿用父配置
（320 图 / bs4 / 8000 iter / BASE_LR 1e-4），`EVAL_PERIOD` 与 `CHECKPOINT_PERIOD`
加密到 400。

**关键结构性事实：父配置的 `DATASETS.TEST` 就是 `wheat_cat2_pts_coco_val`，
所以训练期评估就是 val 评估——自带泛化曲线，不必事后再评，也能直接挑峰值 checkpoint。**

## 跑之前写死的预期与判否线

判据沿用
[exp-2026-08-27-unfreeze-val-generalization](../experiments/exp-2026-08-27-unfreeze-val-generalization.md)
末尾预注册的三条，对照是 **innercost 的 val mF1 0.1670 / 计数 MAE 7.6**
（320 图、冻结 trunk、同配置、已收敛）。噪声地板 |Δ mF1| ≈ 0.008。

**主判据（arm A）**

- **`mF1` 峰值 > 0.1670 + 0.008 = 0.1750 → 解冻在真实数据上成立**，
  两阶段设计应改为「小学习率解冻 decoder」。
- **`mF1` 峰值 < 0.1670 → 解冻在真实数据上不成立**，
  overfit10 上那个 0.98 纯属记忆，Stage2 冻结 trunk 的原设计站得住。
- 落在 0.1670~0.1750 之间 → 噪声地板内，**记为未决**，不得单方面宣称成立。

**机制判据（检验上一轮的归因是否正确）**

上一轮把 val 上的过检归因于 overfit10 的类别先验偏移，并预测真实数据上不会发生。
本次直接检验：

- **预测穗数应回到 257 附近**（overfit10 训出来的是 315~357，冻结组是 259）。
- **`spike_precision` 应显著高于 0.55、接近冻结组的 0.7529。**
- 若 `spike_precision` 仍 < 0.70 → **上一轮的归因错了**，损害不只是类别先验，
  下一步必须改小学习率或分层学习率，而不是继续加解冻范围。

**能力判据**

- `matched_F1@8` 应保持在 0.42 附近或更高（它在 overfit10 训练下就有 0.4236，
  真实数据只会更好）。若反而下降 → 说明 8000 iter 过训，取曲线峰值那一档。

**控制量**

- 可训练参数须精确等于 14,413,082 / 14,610,972 / 28,733,212。
- `spike_recall` 首次评估不得为 0（为 0 说明检测被毁，重演 08-26 那个坑）。
- 七项损失全部非零非 NaN。

**arm 间的预期排序**

overfit10 上是 A > C > B（点任务）。真实数据上样本多 32 倍、过拟合压力小得多，
**B 的劣势应当缩小甚至反转**——若 B 在真实数据上最好，说明 overfit10 的排序
是被样本量伪造的，那批结论要重读。

## 注意事项

- 三个 arm 的 `spike_recall` 各不相同，`F1@σ` 严格来说不可比；
  **跨 arm 的点头能力比较必须看 `matched_F1@σ`**。
- 8000 iter 在 320 图上是 100 epoch，解冻后大概率过拟合。
  **报告时以 val 曲线峰值为准，并同时给出末次值**，不得只报峰值。
- 对照复用 innercost（2000 iter），它是冻结 trunk 且文档记录点损失 600 iter 见底、
  已收敛，不重跑。
