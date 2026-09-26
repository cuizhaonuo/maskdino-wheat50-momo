# 想法：图级 wheat_2 穗数的 CORAL 序数头（C 臂）

状态：**待验证（预注册，跑之前写死）** ｜ 提出 2026-09-26 ｜ 分支 `mini/coral-img`（从 `mini/baseline`@`0f71161` 开出，100 epoch）｜ 配置
`wheat_joint_single_stage_mini_coral_img.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜ 输出 `output_v2/exp-<启动日期>-mini-coral-img`

来源：Cao, Mirjalili, Raschka, *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*
（Pattern Recognition Letters 2020，arXiv 1901.07884），本地源码 `/public/home/zncui/pycharm_project/coral-cnn`。

同一批还有两条 CORAL 分支，都从 `mini/baseline`@`0f71161` 分出，彼此独立、不合并：
`mini/coral-count`（A：每穗 CORAL 计数头）与 `mini/coral-exist`（B：CORAL 存在性头 + 前缀匹配）。
A、B 针对的是**每穗报几个点**，本臂针对的是**每图报几个穗**，两者正交。

## 现象

`mini/*` 的头号遗留问题是**过检**：ep100 基线（`output_v2/exp-0923-mini-ep100`）末值预测 863 个 wheat_2 vs GT 613（1.41×），
`fp_spike_count` 285，`spike_precision` 0.670。`count_MAE_img`（逐图小穗点总数误差）末值 30.82，
25 轮里在 23.6~54.1 之间震荡，**不随 segm AP 变好而改善**。
baseline 与 longer-schedule 两份记录都判断：「报几个穗」是决策 / 标定问题，不是拟合问题。

逐图看穗数（本次从 ep100 末轮的 `spikelet_pred_vs_gt.csv` 算）：

| 逐图 wheat_2 穗数的预测方式 | MAE | 有符号偏差 |
|:--|--:|--:|
| ep100 基线：分数 ≥ 0.4 的 wheat_2 个数 | **1.350** | +1.25 |
| 常数 3（val 中位数附近） | 1.455 | — |

**检测阈值数出来的穗数，只比一个不看图的常数好 7%。** 报出的穗数几乎没有用上图像内容。

## 假设

每图 wheat_2 穗数是一个序数量（val 0~8，均值 3.06）。用 CORAL 直接从整图特征回归它，
推理时按分数取前 N 个 wheat_2（N 由该头给出），替代固定分数阈值 0.4：

1. 决策点由图像内容决定，而不是一个全局常数阈值，过检应当下降，`spike_precision` 上升；
2. 过检下降会直接带动 `count_MAE_img` 下降（ep100 基线里它与过检倍数强相关）。

## 修法

新增 `cfg.MODEL.MaskDINO.IMG_COUNT_HEAD`（默认 False，关着时一个参数都不建，模型逐位不变），
`IMG_COUNT_WEIGHT=2.0`、`IMG_COUNT_MAX=15`（16 个序数类，15 个二分类任务），
`TEST.IMG_COUNT_READOUT ∈ {thresh, topn}`。

- `maskdino/modeling/coral.py`（三条 CORAL 分支逐字相同）：`CoralHead`、`coral_levels`、`coral_loss`、`coral_predict`。
- decoder：对 decoder 入口的多尺度特征 `src_flatten`（input_proj 之后，256 维）在有效位置上做全局均值池化，
  接 `img_count_embed = CoralHead(256, 256, 15, 3)`，输出 `pred_img_count_logits [B,15]`，只挂在主输出上，不做深监督。
  训练（LSJ 画布）与 val 都是 1024×1024，均值与总量只差常数，所以均值池化能表达「数量」。
  这就是论文 ResNet → 全局池化 → CORAL 的做法，换成了本模型的 encoder 特征。
- criterion：新损失 `img_count`，目标是**增广并截断降级之后**这张图里 wheat_2 的实例数（与推理要数的对象一致），每图权重 1。
  aux / interm 输出里没有这个键，直接返回空。
- 推理：`N = Σ_k 1[σ(l_k) > 0.5]` 写进每图结果 `img_spike_count`。
- `SpikeletPointEvaluator`：`topn` 读法按分数取前 N 个 wheat_2 作为报出的穗，不再用 `SPIKELET_EVAL_SCORE_THRESH`；
  `thresh` 读法保持原口径。两种读法都新报 `spike_count_MAE_img`（逐图报出穗数 vs GT）；有头时再报
  `img_count_head_MAE` / `img_count_head_bias`，并写 `img_count_head.csv`。**COCO AP 不受读法影响。**

CPU 单测：`coral_loss × T` 与论文 `cost_fn` 逐值一致；`loss_img_count` 与手算一致、目标截到 15、aux 返回空；
evaluator 在真实 val 的 20 张图上配合成预测（GT 穗 0.9 分 + 每图两个假阳 0.5 / 0.3 分，头给出 GT 穗数）：
`topn` 下 `spike_precision` 1.0、`spike_count_MAE_img` 0；`thresh` 下 0.818 与 1.0。

## 与基线的差异

两项配置：`IMG_COUNT_HEAD: True`、`TEST.IMG_COUNT_READOUT: topn`（解析后的 cfg 逐项比对，只差这两个键）。
新增参数 131,856 个（`img_count_embed.*`）。开关关闭时 state_dict 的 732 个键与形状与 ep100 checkpoint 完全一致。

## ⚠️ 已知风险：训练集的穗数先验偏高

mini train 是按 wheat_2 数取的前 500 张图，**增广前每张都有 ≥ 4 个 wheat_2**。LSJ 裁剪把分布拉低了不少，
但仍偏高。本次在 CPU 上遍历 200 个 train batch（800 张增广后的图，SEED 0）实测：

| | 均值 | ≤ 4 的比例 | 0 的比例 | 最大 |
|:--|--:|--:|--:|--:|
| mini train 增广前 | 5.51 | 35.8% | 0% | 13 |
| **mini train 增广后（头实际看到的）** | **3.82** | **66.8%** | **5.9%** | 13 |
| val | 3.06 | 79.0% | 3.0% | 8 |

增广后 train/val 均值比 **1.25×**，比增广前的 1.80× 温和得多，但仍非零。
**头若只学到训练先验，会在 val 上系统性多报。** 机制门里的有符号偏差就是查这个的：
先验偏移假说预测 `img_count_head_bias` > 0。

## 预期与判否线（跑之前写死）

对照 ep100 基线（`output_v2/exp-0923-mini-ep100`，`mini/baseline`@`0f71161`）末值：

| 指标 | 末值 | 备注 |
|:--|--:|:--|
| `count_MAE_img` | **30.82** | 峰值 23.57（iter 8499），末 4 轮 30.26~32.86 |
| `spike_precision` | **0.670** | 末 4 轮 0.658~0.670 |
| `spike_recall` | 0.943 | |
| 逐图穗数 MAE（thresh 读法） | **1.350** | 常数 3 为 1.455 |
| `matched_F1@8` | 峰值 0.6331 | |
| `segm/AP` | 峰值 78.80 | |

主判据看**末值**：topn 读法把决策点交给头，曲线应当比阈值读法平稳；而 `count_MAE_img` 的峰值在基线里是震荡中的一个点，拿来比不公平。

### 机制门（先判）

`model_final.pth` 上的 `img_count_head_MAE`（头本身的逐图误差，与读法无关）：

- **< 1.350 且 < 1.455** → 通过：头比检测阈值数、也比常数都准，确实从图像里读出了穗数。
- **≥ 1.455** → 判**「只学到先验」**，主判据不再解释。
- 两者之间 → 记「弱」，主判据照判，但结论里注明。
- 同时报告 `img_count_head_bias`：> +0.5 视为先验偏移显著。

### 主判据（topn 读法，训练期评估的末值）

| 条件 | 判定 |
|:--|:--|
| `count_MAE_img` 末值 ≤ **24.7**（基线 ×0.8）且 `spike_precision` 末值 ≥ **0.75**，且护栏满足 | **成立** |
| `count_MAE_img` 末值 ≥ **30.82** | **判否** |
| 其余 | **未决**（单种子，不补种子） |

### 护栏

- `spike_recall` 末值 ≥ **0.90**（基线 0.943）：topn 不能靠少报换精度。
- `matched_F1@8` 峰值 ≥ **0.6181**（基线 − 0.015）。
- `segm/AP` 峰值 ≥ **77.80**（基线 − 1.0）。

### 观察量（只记录，不作判据）

- 同一 `model_final` 用 `thresh` 读法再评一次
  （`--eval_only ... MODEL.MaskDINO.TEST.IMG_COUNT_READOUT thresh`），与基线对比，拆开「辅助损失对表征的作用」与「读法的作用」。
- `img_count_head.csv` 里按 GT 穗数分组的偏差：先验偏移应表现为少穗图多报、多穗图少报（向训练均值回归）。
- `spike_count_MAE_img` 两种读法的值。

## 已知限制

- 单种子，与之前所有 mini 实验一致。
- 图级样本只有 500 张（×LSJ 增广），远少于论文的数万张人脸。头的泛化能力是本臂最大的不确定性。
- 均值池化对全图一视同仁，而 wheat 与 wheat_2 的区别是细粒度的（完整、可数小穗）。若机制门不过，
  下一步可以换成对 query 特征做注意力池化，但那是另一个实验。
- 读法只改点评估与计数；下游 `myscripts/infer/seg_infer.py` 的滑窗计数流程没有接这个头。
- `IMG_COUNT_WEIGHT=2.0` 未调，且只有主输出一份损失（A 臂的计数头随深监督有 11 份）。

## 成本

12500 iter。多一个全局池化 + 3 层 MLP，预计与基线同速，约 11~14 h。另加 1 次 `thresh` 读法的 `--eval_only`（约 2 min）。
GPU 冒烟（bs2，30 step）的实测显存与速度见下节。

## 冒烟

（待 qiuhao A100 空出后补记：bs2 峰值显存、s/iter、新参数梯度非零、两种读法在 20 张 val 上都能跑通。）
