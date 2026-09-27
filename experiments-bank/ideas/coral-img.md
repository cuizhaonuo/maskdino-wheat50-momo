# 想法：图级 wheat_2 穗数的 CORAL 序数头（C 臂）

状态：**机制门不过**（`img_count_head_MAE` 1.560 ≥ 1.455，比常数 3 还差；头被 bias 学习率上限卡住，属实现缺陷，不能据此否定想法）｜ 验证于 [exp-2026-09-27-mini-coral-img](../experiments/exp-2026-09-27-mini-coral-img.md) ｜ 2026-09-27 00:43:51 由队列启动于 `b307eba`（代码与 `1ab1978` 相同）｜ 预注册提交于 `1ab1978`（跑之前）｜ 提出 2026-09-26 ｜ 分支 `mini/coral-img`（从 `mini/baseline`@`0f71161` 开出，100 epoch）｜ 配置
`wheat_joint_single_stage_mini_coral_img.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜ 输出 `output_v2/exp-0927-mini-coral-img`

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

## 【2026-09-27 运行中发现的实现缺陷】（判否线不因此改动）

A、B 两臂跑完后发现：CoralHead 的 bias 是标量参数，AdamW 每步对它的位移约等于当步 lr，而整个 schedule 的累积学习率只有 0.613，
所以每个 bias 全程最多走约 0.6；A、B 的 bias 全部顶在这个上限上、没有收敛（详见那两个分支的实验记录）。
本臂更严重：训练集增广后每图 wheat_2 数的边际分布要求 15 个 bias 的 logit 跨度约 9.7（logit(P(n>0)) 2.77 到 logit(P(n>14)) −6.9），
而本臂初始跨度 0.93、跑到 iter 4999 时 1.56，到训练结束最多约 2.2。**头在结构上表达不了训练集的计数分布**，
早期（iter 499~1999）对所有 val 图都给出 N=0，训练损失到 iter 5259 时仍高于「只知道边际分布的常数预测」（×2 后 0.756 vs 0.539）。
判否线照旧执行；若机制门不过，结论要写成「在这个 bias 参数化下没学到」，不能写成「图级 CORAL 计数这个想法不成立」。

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

**GPU**（qiuhao A100 GPU1，2026-09-26 15:03，跑之前）：独立脚本 import `train_net.Trainer`（不经 `train_net.py` 的 main，
不算一次训练 run），ImageNet R-50 起点，bs2 × 30 step（AMP），再在 20 张 val 上推理 + `SpikeletPointEvaluator`。
（s/iter 不可比：第一个跑的臂要冷读 NFS 上的图，且当时别人的 vLLM 占着 GPU 的 100% 算力。）

| 项 | 结果 |
|:--|:--|
| 30 步损失 | 全部有限；`loss_img_count`（×2 后）1.561 → 1.098 |
| 每步产生的 `loss_img_count*` | 1 个（只有主输出，符合设计） |
| 第 0 步梯度 | `img_count_embed.*` 全部非零 |
| 峰值显存（bs2，torch 口径） | 11.91 GiB |
| 20 张 val | 两种读法都跑通；30 步时头对 20 张图都给出 N=2（这 20 张 GT 均值 4.5），`img_count_head_bias` −2.5 |

**CPU**（`CUDA_VISIBLE_DEVICES=""`，一点不占 GPU；15:14~15:19）：把 ep100 基线的
`output_v2/exp-0923-mini-ep100/model_final.pth` 载入本臂模型，与 `mini/baseline` 同权重的输出在 12 张 val
（top-100 里 94 个 wheat_2 条目）上逐实例比对。补上 GPU 冒烟没走到的读法分支（30 步的模型还不出 wheat_2）。

- 实例、每穗点数、点坐标与基线**逐项相同**；`thresh` 读法下原有的点评估指标全部相同，新增 `spike_count_MAE_img` 1.000；
- `topn` 读法跑通。未训练的头对每张图都给 N=15（bias 按降序正值初始化，15 个任务全部 > 0.5），
  于是报出的穗变多、`spike_precision` 0.553——这只说明读法生效，不代表训练后的行为。
  **早期几轮评估的 topn 指标会因此偏差很大，读曲线时从头学会计数之后再看。**

**显存与排队**：A、B 两臂的正式 run 在 nvidia-smi 上各占 32~39 GiB（缓存占了大头），GPU1 只剩约 8 GiB，
本臂由队列在其中一臂结束、空闲显存 ≥ 30 GiB 后自动启动。启动时 worktree 的 HEAD 可能是本文档的后续提交，
代码与 `1ab1978` 相同。
