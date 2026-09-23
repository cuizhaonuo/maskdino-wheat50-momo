# 想法：mini 上把 schedule 从 50 epoch 拉到 100 epoch

状态：**成立（幅度很大）** ｜ 提出 2026-09-23 ｜ 验证于 [exp-2026-09-23-mini-longer-schedule](../experiments/exp-2026-09-23-mini-longer-schedule.md) ｜ 承接 [exp-2026-09-22-mini-baseline](../experiments/exp-2026-09-22-mini-baseline.md)

## 为什么值得做

基线跑完 50 epoch 后末三轮（46/48/50 ep）指标几乎重合：
segm AP 73.67 / 73.57 / 73.52，`matched_F1@8` 0.5714 / 0.5730 / 0.5718。
**看起来收敛了，但这是 cosine 把 lr 退到 1.6e-6 造成的收敛，未必是模型容量饱和。**

区分这两者的办法就是把 cosine 的分母拉长：同样的权重，在原本已经退火的区间里
lr 仍保持在 3~5e-5，如果指标继续上行，说明 50 epoch 是**被 schedule 掐断的**，不是学到头了。

支持「还有空间」的迹象：

- `spike_recall` 到末轮还在 0.91 的高位且没有回落，`计数 MAE/穗` 单调改善到 3.88 从未反弹；
- 全程 **25 轮 val 没有出现过拟合的典型形态**（val 指标见顶后持续下滑）；
- 500 图 × 50 ep = 25000 张次曝光，相比 arm B 的 320 图 × 8000 iter × bs4 = 32000 张次**还少**。

反对的迹象：

- 你们自己记录过「8000 iter 在 320 图上是 100 epoch，解冻后大概率过拟合」
  （[unfreeze-real-dataset](unfreeze-real-dataset.md) 注意事项）。本跑是 500 图 × 100 ep，
  单图重复次数与那个警告同量级。

**所以这是一个真正的两面问题，值得用一跑来判。**

## 配置（与基线的差异只有 schedule）

| | 基线 | 本跑 |
|:--|:--|:--|
| `SOLVER.MAX_ITER` | 6250（50 ep） | **12500（100 ep）** |
| `SOLVER.WARMUP_ITERS` | 250 | **250（不变）** |
| `TEST.EVAL_PERIOD` / `CHECKPOINT_PERIOD` | 250（2 ep，25 点） | **500（4 ep，25 点）** |
| 其余全部 | — | **逐项继承，不动** |

**`WARMUP_ITERS` 故意不缩放**：保持绝对值 250 使前 250 iter 与基线逐步相同，
差异从 warmup 结束后才出现，便于归因。代价是 warmup 占比从 4% 降到 2%。

**`EVAL_PERIOD` 翻倍**保持 25 个采样点，使两条曲线的分辨率和评估开销都可比。

⚠️ **cosine 的分母绑在 `MAX_ITER` 上**（`detectron2/solver/build.py`:
`LRMultiplier(..., max_iter=cfg.SOLVER.MAX_ITER)`，warmup 比例也是 `WARMUP_ITERS / MAX_ITER`）。
所以改 `MAX_ITER` 会同时改变退火周期——**这正是本实验想要的**，不是副作用。
由此本跑**不是**基线的续训，两条 lr 路径从 iter 250 起就不同，不能用 `--resume` 衔接。

## 跑之前写死的预期与判否线

对照是基线的峰值：**segm AP 73.67 / `matched_F1@8` 0.5783 / `计数 MAE/图` 26.72**。

噪声地板：基线最后 8 轮（36~50 ep）`matched_F1@8` 在 0.5631~0.5783 之间摆，
幅度 **0.015**；segm AP 在 71.05~73.67 之间摆，幅度 **2.6**。
**这两个数就是本实验的噪声地板**，比 innercost 时代的 0.008 宽得多，因为解冻后 recall 在摆。

### 主判据（看 val 曲线峰值，不看末值）

- **`matched_F1@8` 峰值 > 0.5783 + 0.015 = 0.593 → 更长 schedule 成立**，
  50 epoch 是被 cosine 掐断的，`mini/*` 的默认 schedule 应改为 100 epoch。
- **`matched_F1@8` 峰值 < 0.5783 → 不成立**，50 epoch 已经够，且多花一倍算力。
- 落在 0.5783~0.593 之间 → **记为未决**，不得单方面宣称成立。

### 过拟合判据（本跑最需要防的失败模式）

- **val 峰值若出现在 iter < 6250（即 50 ep 之前）且此后持续下滑 → 100 epoch 是过训**，
  应当反过来缩短 schedule 而不是延长。
- 判定"持续下滑"的标准：峰值之后连续 **3 轮** val 的 `matched_F1@8` 低于峰值超过噪声地板 0.015。
- 若过训成立，**同时记录峰值出现在第几个 epoch**，那才是 mini 的最优 schedule。

### 机制判据（检验"lr 被掐断"这个归因）

本想法的核心假设是：基线末段的平台是 lr→0 造成的，不是容量饱和。

- **在 iter 6250 附近（此时本跑 lr ≈ 5e-5，基线已 ≈ 0），本跑指标应明显高于基线同 iter 的值**
  （基线 iter 6249：segm 73.52 / `matched_F1@8` 0.5718）。
- 若本跑在 iter 6250 处**不高于**基线 → 归因错误，平台不是 lr 造成的，
  那么延长 schedule 这条路整体失效，应转向扫阈值或改数据。

### 控制量（出现即判本跑无效，需重跑）

- 七项损失全部非零非 NaN。
- 首次评估的 `spike_recall` 不得为 0（为 0 说明检测被毁）。
- `wheat_cat2_pts_mini_coco_train` 加载必须是 **500 图 / 27702 实例 / 43024 点**。
- 初始权重必须是 ImageNet R-50（日志里 `Following weights matched with submodule backbone - Total num: 53`）。

## 注意事项

- **`matched_F1@σ` 在 `spike_recall` < 0.2 时不可读**（基线前 6 轮就吃过这个亏：
  12 个样本报出 0.4876，样本涨到 237 个后跌到 0.3801）。报告时必须同时给出 `spike_recall` 与配上的穗数。
- **过检倍数要逐轮记录**。基线的 `计数 MAE/图` 几乎由它主导，若本跑过检继续恶化，
  即便 `matched_F1@8` 上行，计数这个最终目标仍可能倒退——**两者要分开报，不得只报好的那个**。
- 预计耗时 **约 14 小时**（12500 iter × 3.84 s/iter + 25 次评估 × 2 min），
  与 `exp-0922-single-stage-p30` 并发时显存 27G + 39G = 66G / 81.9G，放得下。
