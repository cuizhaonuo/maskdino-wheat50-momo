# 想法：分组 decoder 上，存在性改学「定位质量」（Quality Focal Loss）

状态：**预注册，待跑** ｜ 提出 2026-10-07 ｜ 分支 `mini/group-quality`（从 `mini/group-decoder`@`00a7aaf` 开出，100 epoch）｜ 配置
`wheat_joint_single_stage_mini_group_quality.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_group.yaml`，即 E3 的配置）｜ 输出 `output_v2/exp-<启动日期 MMDD>-mini-group-quality`

**分叉点与约定不同，是有意的。** mini 实验的约定是新 arm 一律从 `mini/baseline` 开。本 arm 是在 E3（分组 decoder）上续做：
要检验的是「小穗 token 已经能逐点看局部特征，只差监督」这一判断，离开分组 decoder 就不成立。所以从 `mini/group-decoder`@`00a7aaf` 开出。
主 agent 已向用户说明。对照因此是 E3，不是 E0。

## 来源

- Li et al., *Generalized Focal Loss*（NeurIPS 2020）的 Quality Focal Loss：分类分支的目标改为定位质量（IoU），用 |q − p|^γ 调制。
- Zhang et al., *VarifocalNet*（CVPR 2021）：正样本的分类目标为 IoU-aware 分数。
- Liu et al., *Detection Transformer with Stable Matching*（Stable-DINO，ICCV 2023）的 position-supervised loss：DETR 系的分类目标用位置质量。

三者的共同点：分类分数直接学「这个预测定位得有多好」，排序与定位质量对齐。

## 现象

**瓶颈在挑点，不在定位。** 在最终 checkpoint 上，把全部 token 与 GT 做一对一匹配（`point_token_diag.py` 的 auc 口径）：

| | E0 | E2 | E3 |
|:--|--:|--:|--:|
| 8 px 内有 token 的 GT 小穗比例 | 75.5% | 76.7% | 79.1% |
| 分数完美挑选时的 F1@8 上限 | 0.86 | 0.87 | 0.88 |
| 实际 F1@8（最终 checkpoint，阈值 0.35） | 0.631 | 0.642 | 0.638 |
| 存在性 AUC | 0.667 | 0.683 | 0.699 |

- 覆盖比例 = `output_v2/_diag-0926/<run>.json` 里 auc 的正样本数（n_tokens × pos_rate）÷ 该 run `inference/spikelet_pair_counts.csv` 的 GT 点数之和：
  E0 6957 / 9209，E2 7126 / 9287，E3 7330 / 9262。
- F1 上限 = 只挑出覆盖到的 token（precision 1、recall = 覆盖比例）时的 F1 = 2c / (1 + c)。
- 实际 F1@8、AUC 来自 [exp-2026-09-25-mini-group-decoder](../experiments/exp-2026-09-25-mini-group-decoder.md) 的诊断表（最终 checkpoint，阈值 0.35）。
- 上限与实际差 0.23~0.25。定位已经够好，差在分数挑不出那 75%~79%。

**只给局部特征、不改监督，存在性不会变成局部判别。**

- 三条分支都没让 AUC 明显上去（同一诊断口径，最终 checkpoint）：
  - E3（`mini/group-decoder`，逐 token 可变形采样）：0.699，比同 query 预算的 E2 只高 0.016（`output_v2/_diag-0926/`）。
  - `mini/ifi-refine`（IFI 取 1/4 + 1/8 特征）：refine 头 0.659，低于基线 0.667（`output_v2/_diag-1007-apg/B_refine.json`、`baseline.json`）。
  - `mini/apg-refine`：refine 头 0.630（`output_v2/_diag-1007-apg/A_refine.json`）。
- APG 的正负对比（2 px 对 2~8 px）低于 1/4 特征的格宽（4 px），学不出来。
- 现在的监督是「配上就是 1」：内层匈牙利配上的 token 目标一律为 1，离 GT 十几 px 的也一样。
  评估在 σ=8 下把这些点算作 FP，训练却把它们教成正样本。存在性更像在学「这个 slot 常被配上」
  （08-25 的 slot 诊断：存在性头只学到每个 slot 的边际频率），而不是「这里的点准」。

## 假设

把匹配上的 token 的存在性目标换成定位质量 q（像素距离的高斯，σ 约为 GT 最近邻间距的一半，与评估 σ=8 对齐），
存在性分数就直接学「这个点离真小穗有多近」。分组 decoder 的小穗 token 已能在自己的位置取特征，监督对上之后：

1. 存在性的排序与「≤ 8 px 配上」对齐，AUC 上升（机制，M1）；
2. 默认阈值下离 GT 远的匹配点分数被压低、少出 FP，matched_F1@8 上升（主判据）；
3. 计数随之改善（次判据，但见「归因限制」里的标定混杂）。

## 修法

开关默认关；关着时与 `mini/group-decoder` 逐位等价（见验证 1）。

**配置**（`maskdino/config.py`）：

- `MODEL.MaskDINO.POINT_EXIST_TARGET`：`"hard"`（默认，现状）或 `"quality"`。
- `MODEL.MaskDINO.POINT_QUALITY_SIGMA_NN`：默认 0.5。

**代码**：只改 `maskdino/modeling/criterion.py` 的 `loss_points`（新增 `_point_quality` 与模块级 `quality_focal_loss`），
以及 `maskdino/maskdino.py` 构造 `SetCriterion` 处的两个参数透传。decoder、matcher、推理一行不改。

`quality` 下：

- **适用实例**：外层匹配上、有 GT 小穗（Ni ≥ 1）的实例。内层匹配不变。
- **匹配上的 token**：目标 q = exp(−d² / (2σ²))。
  - d = 该 token 预测点与所配 GT 小穗的像素距离，在 `no_grad` 下算（detach）。
  - 预测与 GT 都在 GT 框相对系里，先乘框的归一化宽高，再乘 `targets[b]['image_wh']`，得到像素。
    d 是纯像素距离，不受 iso 归一化或 ρ 影响。
- **σ** = `POINT_QUALITY_SIGMA_NN` × 该实例 GT 小穗「各点到最近邻距离」的中位数（像素，增广后）。
  - mini train 原生尺度下，逐实例中位数的 p50 为 16.8 px（2754 个 wheat_2，p5 12.0、p95 21.6；val p50 17.0），
    σ ≈ 8.4 px，与评估的 σ=8 对齐。LSJ 缩放（0.1~2.0）时 σ 自动跟着变。
  - Ni < 2 时 σ = 8 px（原生尺度下 train/val 都没有 Ni < 2 的 wheat_2）。
- **没匹配上的 token**：q = 0。
- **损失**：存在性 focal 换成 α 加权的 Quality Focal Loss，
  `L = −|q − p|^γ · [α·q·log p + (1−α)·(1−q)·log(1−p)]`，γ = 2，α = `POINT_FOCAL_ALPHA`（0.5）。
  q ∈ {0,1} 时与现有 focal 逐项相等（验证 2）；软目标下在 p = q 处取最小值 0。
- **不变**：归一化（÷ 总 GT 小穗数）、`POINT_CLASS_WEIGHT`、wheat（Ni=0）实例的处理（q 全为 0，权重 `POINT_EMPTY_CLS_WEIGHT`=0.1）、
  内层匹配代价（`COST_POINT_*`）。
- **所有输出统一**：interm、layer-0 的 MLP 头、小穗 token 的 9 层都走同一个 `loss_points`，用同一种目标。
- **推理不变**，阈值仍为 0.35。q = 0.35 约对应 d ≈ 1.45σ ≈ 12 px。

实现时的取舍（方案未写明处，取最简单、忠实于方案的做法）：

- d 与 σ 都在 clamp 到 [0,1] 之后的 GT 框相对坐标上算，与坐标损失用同一个 GT 目标。
- 中位数用线性插值的中位数（`torch.quantile(·, 0.5)`，与 numpy 相同；偶数个时取中间两数的平均）。
- 数值保护：σ 下限 1e-3 px，防 GT 点重合时 0/0。mini train/val 里没有重合点，不会触发。
- Ni > P（train 最多 31 点、P=30）时只有配上的 30 个 token 拿到 q，与 E3 的处理相同。

## 与对照的差异

单因素对照 **E3**（`mini/group-decoder` 的 100 epoch 版，`output_v2/exp-0925-mini-group-decoder`）：

| 项 | E3 | 本 arm |
|:--|:--|:--|
| `POINT_EXIST_TARGET` | hard（默认） | **quality** |
| `POINT_QUALITY_SIGMA_NN` | —（hard 下不生效） | 0.5 |

其余全部继承 E3 的配置：分组 decoder（100 个实例 query、K=30、iso、ρ c=1、`POINT_EMPTY_CLS_WEIGHT` 0.1、`POINT_FOCAL_ALPHA` 0.5）、
数据、schedule（12500 iter = 100 epoch，cosine，每 500 iter 评估一次，共 25 次）、`POINT_THRESHOLD` 0.35。
冒烟落盘的 `config.yaml` 与 E3 run 的逐行 diff：除 `OUTPUT_DIR` 和冒烟覆盖的三项（`MAX_ITER` / `EVAL_PERIOD` / `CHECKPOINT_PERIOD`）外，只多出上表两个键。
正式 run 除 `OUTPUT_DIR` 外不带任何 CLI 覆盖。

## 实施前已完成的验证

脚本与原始输出：`output_v2/_smoke-1007-group-quality/verify/`（`verify_group_quality.py`，一次性脚本，不进 git）。
GPU 部分都在 qiuhao GPU1（A100）上跑。

1. **等价性（开关关着）**：固定 random / numpy / torch 种子，mini train 前 2 张图经同一个 mapper，训练态前向 1 次，
   在本 WT 与基点 worktree（`mini/group-decoder`@`00a7aaf`）各跑一次，比对 JSON：
   - 随机初始化：**127 项 loss（加权前、加权后）逐位相等**，加权总和逐位相等；参数量同为 65,322,300，参数 sha256 相同。
   - 载入 E3 最终 checkpoint：127 项 loss 同样逐位相等。
   - 正式配置 + `POINT_EXIST_TARGET hard`：与基点逐位相等（127 项）。
2. **QFL 退化检查**：
   - 逐元素（200,009 个 logit，含 ±50 的极端值；α = 0.25 / 0.5；CPU 与 GPU 都测）：
     q = 1 时**逐位相等**；q = 0 时数值相等，最大绝对差 9.5e-7，loss > 1e-6 的元素里最大相对差 5.5e-6，总和相对差 4e-10。
     q = 0 时不逐位相等的原因：原式算 (1 − p_t)^γ 时 p_t = 1 − p，先算 1 − p 再算 1 − (1 − p)，有浮点抵消；QFL 直接用 |0 − p| = p，更准。
   - 在 `loss_points` 上（真实模型输出，12 个 batch × 11 个输出）：quality 分支里把 q 强制为 1，存在性损失与 hard 分支的最大相对差 2.1e-7。
   - 软目标：p = q 时 L = 0，对 logit 的梯度也为 0（q = 0.1 / 0.35 / 0.5 / 0.9）。
3. **开关打开**（正式配置，ImageNet R-50 起步，即正式 run 的起点；4 个 batch × 2 图；为与 hard 逐位对齐，此项设 `PYTORCH_JIT=0`）：
   - 前向 + 反向跑通。735 个可训练参数张量都有梯度，全部有限。
   - 同一模型、同一 batch、同一种子下，除存在性损失外的 116 项 loss 与 hard 逐位相同（坐标、mask、框、类别都不受影响）。
   - 存在性损失初值：各输出 0.21~0.40（hard 为 0.25~0.36），量级相同。
   - 开训时 q 的分布（45 个 wheat_2 实例，每个输出 752 个匹配 token）：
     - 小穗 token 各层：d 中位约 340 px（实例框还近乎整图），98% 的 q < 0.01。开训时存在性目标几乎全是 0。
     - layer-0 MLP 头与 interm：d 中位 26 px，q 的 p75 0.06、p90 约 0.5，13%~15% ≥ 0.35。
     - σ 的 p10 / p50 / p90 = 3.4 / 5.8 / 8.4 px（LSJ 后）。
   - 训练好的模型上 q 的分布（载入 E3 最终 checkpoint，只前向；mini train 前 16 张图经 LSJ，64 个 wheat_2 实例，每个输出 1062 个匹配 token）：
     - 最终层：q 的 p10 / p25 / p50 / p75 / p90 = 0.27 / 0.49 / 0.72 / 0.88 / 0.96，均值 0.66；85% ≥ 0.35，74% ≥ 0.5；d 中位 4.4 px。
     - layer-0 MLP 头：q 中位 0.66，均值 0.62。σ 的 p10 / p50 / p90 = 2.7 / 5.8 / 13.5 px。
     - 由 q、d 反推的 σ 与独立重算（scipy 最近邻 + numpy 中位数）一致，最大相对差 1.6e-7。
     - 这是训练图，q 偏乐观；val 上应更低。
4. **冒烟**（qiuhao GPU1，独占；正式配置 + `SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 1000`，
   输出 `output_v2/_smoke-1007-group-quality`）：
   - 40 iter 与 iter 19 / 39 两次完整 val 评估都跑通，0 报错。
   - `metrics.json` 里两次记录的 256 个 loss 值全部有限。total_loss 522.3 → 508.4，`loss_point_cls`（加权后）0.72 → 0.70。
   - 2.92 s/iter（detectron2 汇总：38 iter / 1:50）。峰值显存 max_mem 25890M，与 E3 冒烟的 25.9 G 相同，本改动不增显存。
   - 未训练模型的评估指标全为 0（AP 0、matched_F1 0、预测点 0），与 E3 冒烟相同。

## 预期与判否线（跑之前写死，2026-10-07）

**读数规则**（用户的硬性规定）：

- 一律用默认读数：训练内置评估，`POINT_THRESHOLD` = 0.35（配置值）。不做阈值扫描，不用扫描最优档或在 val 上拟合的校准值做对比。
- 判据在同一次评估上读：取后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估，该 iter 的全部指标都从这次读。
- 护栏（`segm/AP`、`spike_recall`）用全程峰值。
- 单种子。

**E3 的读数**（`output_v2/exp-0925-mini-group-decoder/metrics.json`，按上面的规则重算）：

- 读数那次在 iter 9499：`count_MAE_spike` 2.422，`matched_F1@8` 0.6326，`spike_recall` 0.9364，`segm/AP` 77.40。
- 峰值：`segm/AP` 78.35（iter 9999），`spike_recall` 0.9657（iter 7999）。
- 诊断（最终 checkpoint）：存在性 AUC 0.6991，重复点 0.39%（42 / 10734）。

### 机制（先判）

用 `myscripts/backup/vis_process/point_token_diag.py`，最终 checkpoint，口径同 E3（σ = 8，`--dup-px 3`）。
必须传不带值的 `--sweep`，把扫描关掉。`--sweep` 要放在最前面、在 opts 之前：它是 `nargs="*"`，
放在 opts 前一位会把 `MODEL.DEVICE` 之类当成阈值吞掉而报错（已用同一套 argparse 定义试过）：

```
python myscripts/backup/vis_process/point_token_diag.py --sweep --run-dir <输出目录> --out <json> [opts ...]
```

- **M1（存在性变成定位质量的判别）**：存在性 AUC ≥ **0.749**（E3 0.6991 + 0.05）。不过 → 判「机制未启动」，端到端差异不归因于存在性学到了定位质量。
- **M2（护栏）**：重复点 ≤ **1%**。

### 主判据：读数那次的 `matched_F1@8`

| 条件 | 判定 |
|:--|:--|
| ≥ **0.6526**（E3 + 0.02） | **成立** |
| ≤ **0.6326**（E3） | **判否** |
| 介于两者之间 | **歧义** |

+0.02 高于 mini 实验族的噪声地板 0.015（见 [group-decoder](group-decoder.md)）。

### 次判据：计数（单独判）

读数那次的 `count_MAE_spike` ≤ **2.18**（E3 2.422 × 0.9）→ 计数成立。

### 护栏

- `segm/AP` 峰值 ≥ **77.35**（E3 峰值 78.35 − 1.0）；
- `spike_recall` 峰值 ≥ **0.9457**（E3 峰值 0.9657 − 0.02）。

### 参照（不作判据）

并列报告 E0（`mini/baseline`，`output_v2/exp-0923-mini-ep100`）与 E2（`mini/q100`，`output_v2/exp-0925-mini-q100`）的同口径读数，
用来回答「值不值得用分组 decoder」。按上面的规则从各自 `metrics.json` 重算：

| | 读数 iter | count_MAE_spike | matched_F1@8 | spike_recall | segm/AP | 峰值 segm/AP | 峰值 spike_recall |
|:--|--:|--:|--:|--:|--:|--:|--:|
| E0 | 8999 | 2.465 | 0.6228 | 0.9511 | 77.40 | 78.80 | 0.9511 |
| E2 | 8999 | 2.614 | 0.6242 | 0.9347 | 77.56 | 79.16 | 0.9576 |
| E3（对照） | 9499 | 2.422 | 0.6326 | 0.9364 | 77.40 | 78.35 | 0.9657 |

## 归因限制

- **结论只对分组 decoder 成立。** 对照是 E3；quality 目标对 MLP 点头（E0 / E2）是否有用，这里不检验。
- **标定混杂。** quality 下存在性分数学的是 q（匹配 token 在 E3 checkpoint 上的 q 均值约 0.66），不再是 1。
  固定阈值 0.35 下输出的点数会变，计数与 F1 的变化里混有「标定移动」，不全是「排序变好」。
  E3 在 0.35 下点数偏多：它的阈值扫描里计数最优档是 0.50（见 E3 结果文档）。分数整体下移本身就可能降低 `count_MAE_spike`。
  所以次判据成立不能单独说明机制；机制只看 M1（AUC 与阈值无关）。按规定不做阈值扫描，这一混杂不在本次拆开。
- **冷启动。** 开训时小穗 token 离 GT 几百 px，q 几乎全是 0（验证 3），存在性头先被教成「全 0」，定位变好后 q 才上来。
  若训练早期点数很少、F1 低，这是预期内的，不单独作为失败信号；判据只看后半程的读数。
- 单种子。

## 成本

- 12500 iter（100 epoch）。冒烟独占 A100 时 2.92 s/iter，约 10 h；与其它 run 同卡并发时按 E3 100 epoch 实测的 3.71 s/iter，约 13 h。
- 另加 25 次评估，每次约 2.5 min（冒烟实测），共约 1 h。
- 峰值显存约 26~27 GB（冒烟 max_mem 25.9 G；E3 100 epoch 实测 26.9 GB），只能用 qiuhao 的 A100（GPU1）。
- 每次存 checkpoint 785 MB × 25 个，约 20 GB 磁盘。
