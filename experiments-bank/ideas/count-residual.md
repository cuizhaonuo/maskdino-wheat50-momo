# 想法：计数交给「穗长直线 + 残差计数头」

状态：**计数过线、护栏未过（不记成立）**；机制启动 ｜ 验证于 [exp-2026-10-07-mini-count-residual](../experiments/exp-2026-10-07-mini-count-residual.md) ｜ 2026-10-07 20:41:31 启动于 `60d21a3`，xuyu A100 GPU0（本批次用户临时放开）｜ 预注册提交于 `60d21a3`（跑之前）｜ 提出 2026-10-07 ｜ 分支 `mini/count-residual`（从 `mini/baseline`@`0f71161` 开出，100 epoch）｜ 配置 `wheat_joint_single_stage_mini_count_residual.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜ 输出 `output_v2/exp-1007-mini-count-residual`

## 来源

- 2026-10-07 讨论 group-decoder 后续方向时的离线分析：每穗小穗数用穗长一条直线就能估，不用训练。
- 先例 `mini/coral-count`：每实例一个 CORAL 序数计数头。r 比直线高，但输在偏差上（见该分支的 `experiments-bank/ideas/coral-count.md` 与 `experiments/exp-2026-09-26-mini-coral-count.md`）。
- 本臂把两者接起来：直线给锚，残差头只学直线解释不了的那部分。

## 现象

**1. 默认读数的计数误差主要是系统多报。**

- 现在每穗报几个点 = 25 个点槽里存在性 > 0.35 的个数。这个数没有被直接监督。
- mini/baseline 最终评估（`output_v2/exp-0923-mini-ep100/inference/spikelet_pair_counts.csv`，578 个配对穗）：
  `count_MAE_spike` 2.599，有符号偏差 +2.40，r 0.642。

**2. 每穗点数用一条直线就能估。**

- mini train 的 2754 个带点 wheat_2 上最小二乘：**点数 = 8.924 + 0.0345 × hypot(w, h)**。
  w、h 是 GT 框宽高，原生 1024² 像素。10-07 用标注复算：a 8.9241、b 0.03451（COCO bbox 与多边形外接框结果相同）。
- val 613 穗、GT 框、取整：MAE **1.419**（系数原值；用配置里的 8.924 / 0.0345 是 1.418），偏差 −0.22、r 0.630（未取整）。
- 对照（同一批 613 穗）：常数 16 是 1.865；改用 mask 最小外接矩形长边 L（9.307 + 0.0341·L）是 1.427，r 0.635
  （10-07 主 agent 离线，脚本未入库）。

**3. 用预测框套直线同样成立，不训练就能把计数误差降一半。**

做法：离线读最终评估存下的 `inference/coco_instances_results.json`，复现 SpikeletPointEvaluator 的穗配对
（wheat_2、score ≥ 0.4、mask IoU 匈牙利、IoU ≥ 0.5），配对数与评估器逐一相同。
按预测框套直线，与同一批穗的默认读数对比（10-07 主 agent 离线；基线一行已由本文 eval-line 用代码复现，见下）：

| run | 直线（预测框） | 默认读数 |
|:--|--:|--:|
| mini/baseline（`exp-0923-mini-ep100`） | 1.419（偏差 −0.20） | 2.599（偏差 +2.40，r 0.642） |
| q100 | 1.448 | 2.859 |
| group-decoder | 1.437 | 2.764 |
| coral-count | 1.396 | 2.285 |
| coral-exist | 1.424 | 2.024 |

截至 10-07，没有任何 mini arm 的默认读数赢过这条直线。

**4. 特征里有穗长以外的信息，但 CORAL 头输在偏差上。**

- `mini/coral-count` 的计数头与真实点数的 r 是 **0.686**（最终评估，579 穗），高于直线。
  作为参照，基线最终 checkpoint 的 578 个配对穗上，直线（预测框、取整）的 r 是 0.615（本文 eval-line，见下）。
- 但它的有符号偏差是 **+1.86**。CORAL 的 25 个标量 bias 在 mini schedule 里走不远：
  全程累积学习率只有 0.613，bias 位移顶在 ±0.5 左右（见该分支文档的「实现缺陷」一节）。

## 假设

1. 直线已经解释了每穗点数的大头（r ≈ 0.62~0.63）。剩下的误差里有一部分能从实例特征里读出来，例如小穗密度、穗型。
2. 让计数 = 直线 + 残差 Δ，残差头只学直线解释不了的部分。
3. 残差头最后一层零初始化：开训时 Δ ≡ 0，计数恰好等于直线。直线是在 train 上最小二乘拟合的，
   **偏差从构造上为零**（val 上 −0.2），不是 +1.86 / +2.40 那种量级。
   不依赖任何标量参数走远：Δ 是 256 维隐层的线性组合，不受「累积学习率 0.613」的上限约束。
4. 残差学到东西 → 计数误差低于直线（1.42）；学不到 → 结果应约等于直线。

## 修法

所有开关默认关闭。关着时与 `mini/baseline` 逐位等价（见「实施前已完成的验证」）。

| 位置 | 改动 |
|:--|:--|
| `maskdino/config.py` | 新增 `POINT_COUNT_HEAD`（False）、`POINT_COUNT_WEIGHT`（1.0）、`POINT_COUNT_LINE`（[8.924, 0.0345]）、`TEST.POINT_COUNT_SOURCE`（`points` / `line` / `head`，默认 `points`） |
| decoder | `point_count_embed = MLP(256, 256, 1, 2)`，最后一层 weight、bias 置零，建在 `__init__` 末尾。interm、initial_pred、9 层都输出 Δ（`pred_point_counts`，[B, Q]）；DN 段与点预测一样切掉 |
| LSJ mapper | 从实际施加的 `ResizeTransform` 取缩放系数（新尺寸 / 原尺寸，x、y 分开），写进 `dataset_dict["lsj_scale"]`。只读属性，不消耗随机数 |
| `maskdino.py` `prepare_targets` | 计数头打开时给 targets 加 `box_diag_native`（增广后 GT 框像素对角线 ÷ 缩放系数）和 `point_counts`（Ni） |
| criterion | 新 loss `point_count`：N̂ = a + b·diag_native + Δ，在外层匹配上的 wheat_2（Ni ≥ 1）上算 \|N̂ − Ni\|，除以这类 GT 实例数（跨卡做法同 `num_masks`）。随 interm 与 9 份 aux 深监督，共 11 项，权重都是 1.0。wheat 类不监督（含 LSJ 截断后降级、Ni 被清零的穗）。点坐标、存在性、内层匹配一行不动 |
| 推理 `instance_inference` | 只对 wheat_2 生效。`line`：计数 = round(a + b·hypot(w, h))，w、h 取该实例的预测框（`mask_box_result`，原图像素）；`head`：再加该 query 最后一层的 Δ。clamp 到 ≥ 0，写进 `pred_spikelet_counts`。小穗点照旧按 0.35 输出。`points` 读法一行不改 |
| 评估器 | `count_MAE_spike`、`count_MAE_img`（含 RMSE）、两份 CSV 的 `pred_points` 列改读 `pred_spikelet_counts`；点级 P/R/F1、`n_pred_points`、`fp_point_*` 仍按实际输出的点算。`line` / `head` 读法下逐穗明细多两列：`out_points`（实际输出的点数）、`line_count`（只用直线、取整），同一批穗可直接比 r |
| `myscripts/infer/seg_infer_reg.py` | **不改**：`infer_image` 已优先读 `pred_spikelet_counts`，计数跟着配置的读法走 |
| `train_net.py` | `_POINT_HEAD_KEYS` 加 `point_count_embed`（只在 `FREEZE_EXCEPT_POINT_HEAD` 时用到，本臂不用） |

训练用 GT 框、推理用预测框：

- 训练用匹配到的 GT 框，目标更稳。
- val 上两种框套直线的 MAE 几乎相同：GT 框 1.419（偏差 −0.22），基线预测框 1.419（偏差 −0.20）。

## 与对照的差异

- 对照：mini/baseline 100 epoch（`output_v2/exp-0923-mini-ep100`，`0f71161`）。同数据、同起点、同 schedule、同 `SEED`。
- 解析后的 cfg 与 `wheat_joint_single_stage_mini_ep100.yaml` 逐项比对，**只差两个键**：
  `POINT_COUNT_HEAD` False → True，`TEST.POINT_COUNT_SOURCE` points → head。
  `POINT_COUNT_WEIGHT: 1.0` 与默认值相同，写在配置里只为显式。
- 新增参数 66,049 个（`point_count_embed.*`）。
- 内置评估的 `count_MAE_spike` 读计数头的计数。`matched_F1@σ` 等点指标仍按存在性 > 0.35 输出的点算，与基线同一读法。

## 实施前已完成的验证

脚本与产物都在 `output_v2/_smoke-1007-count-residual/`（`scripts/`、`equiv/`、`eval-*`、`smoke40/`）。全部跑在 xuyu A100 GPU0。

**1. 等价性**（10-07 20:24，`scripts/equiv_test.py`，分别在本 worktree 与 `mini-baseline` worktree 里跑）

- 做法：固定 random / numpy / torch 种子，cudnn deterministic；载 ImageNet R-50；同一份 batch
  （mini train 前 2 张图经 LSJ mapper，缩放 1.47 / 0.38，85 个实例、6 个 wheat_2）；训练态 fp32 前向一次。
- **127 项 loss 逐位相等**（总和 588.9991957247257）。
- **参数量同为 52,300,433**；state_dict 732 个键逐张量哈希相同。

**2. 开关打开**（同一脚本加 `--on`，同一份 batch）

- 参数 52,366,482（+66,049）。已有 732 个张量的初值与关闭时逐位相同（新模块建在最后），原有 127 项 loss 也逐位相同。
- 多出 11 项 `loss_point_count*`（主输出、9 份 aux、interm）。第 0 步全部等于 **1.7831**，与手算「只用直线」的 MAE 相同（该 batch 的 6 个 wheat_2）。
- Δ 的形状都是 [2, 300]，DN 段（pad 63）已切掉；第 0 步 Δ ≡ 0。
- LSJ 缩放系数取得对：原生对角线（增广后 ÷ 系数）与原始标注多边形外接框对角线的最大相对误差 3.6e-8。
- 梯度：第 0 步最后一层有梯度（weight 范数 17.0、bias 3.67），第一层为 0。最后一层零初始化时这是预期的。
  走一步真实优化器（AdamW + 全模型裁剪 0.01）后，第一层也有了梯度（weight 65024/65536 个元素非零）。
- 初值量级（×权重，同一 batch）：`loss_point_count` 每层 1.78，11 层共 19.6，占总损失 608.6 的 3.2%。
  对照 `loss_ce` 7.74、`loss_bbox` 12.24、`loss_dice` 4.97、`loss_point_coord` 1.73、`loss_point_cls` 0.55。

**3. 评估器与直线读法**（10-07 20:26，mini/baseline 的 `model_final.pth` 配它自己的 `config.yaml`，本 worktree 代码，`--eval_only`）

| 读法 | 输出目录 | `count_MAE_spike` | `count_MAE_img` | 其余指标 |
|:--|:--|--:|--:|:--|
| `points` | `eval-points` | **2.5986** | 30.82 | 与训练日志末值逐项相同 |
| `line` | `eval-line` | **1.4187** | 21.23 | 与 `points` 逐项相同 |
| `head`，计数头未训练（Δ ≡ 0） | `eval-head0` | 1.4187 | 21.23 | 与 `line` 逐项相同 |

- `points`：两份 CSV 与 `coco_instances_results.json` 都和基线最终评估**逐字节相同**；36 个指标与 `metrics.json` 末值一致。评估器改动没有副作用。
- `line`：1.4187 与离线的 1.419 一致。同 578 穗上偏差 −0.197，r 0.615。只有 `count_MAE_spike`、`count_MAE_img`、`count_RMSE_img` 三项变，点指标一项不变。
- `head`（打开计数头载基线权重，checkpoint 里没有计数头参数，保持零初始化）：逐穗明细与 `line` 逐字节相同，`head` 读法的推理路径跑通。
- 新增两列自检：`out_points` 列复现基线读数（MAE 2.5986、偏差 +2.405、r 0.642）。
- eval_only 不算训练，不进 experiments-bank 的实验记录。

**4. 冒烟**（10-07 20:28~20:39，正式配置，CLI 覆盖 `SOLVER.MAX_ITER 40`、`TEST.EVAL_PERIOD 20`，输出 `smoke40/`）

- 训练 40 iter 与两次完整 val 评估（iter 19、39）都跑通，无报错。
- 损失全部有限：`total_loss` 555.8 / 550.6，`loss_point_count`（每层，20 步窗口中位数）1.595 / 1.225。
  这 40 步还在 warmup 里，lr ≤ 1e-7，参数几乎没动，两个数的差别来自 batch 不同，不是学习。
- 未训练的模型没有 score ≥ 0.4 的 wheat_2：`count_MAE_spike` 为 nan，`count_MAE_img` 48.735，与基线 iter 499 相同。
  `head` 读法的推理路径在每图 top-100 实例上照常执行；有配对穗时的 `head` 读法已由上面的 eval-head0 验过。
- 速度 **2.63 s/iter**（xuyu GPU0 独占，`Overall training speed: 38 iterations in 0:01:39`）。
- 峰值显存 `max_mem` **22,989 MB**（torch 口径；mini/baseline 约 23.1 GB）。
- 未训练模型的一次评估约 4.3 min（预测框多且差，COCO 评估慢）。

## 预期与判否线（跑之前写死）

### 读数规则

- 一律用默认读数：内置评估，`POINT_THRESHOLD` 取配置值 0.35。本臂的默认读数里，每穗计数来自计数头（`POINT_COUNT_SOURCE: head`）。
- 不做阈值扫描。不用扫描最优档，也不用在 val 上拟合的校准值做对比。
- 判据在同一次评估上读：**后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估**，该 iter 的全部指标都从这次读。
- 护栏 segm AP、`spike_recall` 用全程峰值。
- 单种子。

### 基线读数（`output_v2/exp-0923-mini-ep100/metrics.json`）

| | iter | `count_MAE_spike` | `matched_F1@8` | `spike_recall` | segm AP |
|:--|--:|--:|--:|--:|--:|
| 后半程计数最好的那次 | 8999 | **2.465** | 0.6228 | 0.9511 | 77.40 |
| 全程峰值 | — | — | — | **0.9511** | **78.80** |

不训练的参照线：直线读法 **1.42**（val GT 框 1.419；基线最终 checkpoint 的预测框 1.4187）。

### 机制（先判，并记录）

- 在最终评估的逐穗明细（`inference/spikelet_pair_counts.csv`）上，同一批配对穗：
  **r(`pred_points`, `gt_points`) − r(`line_count`, `gt_points`) ≥ 0.03** → 残差学到了穗长以外的信息。
- 达不到 → 记「机制未启动」。
- 两个 r 都在取整后的整数计数上算：`pred_points` 是计数头的计数，`line_count` 是同一批穗只用直线的计数。
- 参照：基线最终 checkpoint 的 578 穗上，直线的 r 是 0.615；同一批穗上存在性阈值读法的 r 是 0.642（高 0.027，但偏差 +2.40）。

### 主判据：读数那次的 `count_MAE_spike`

| 条件 | 判定 |
|:--|:--|
| ≤ **1.28**（直线 1.42 × 0.9） | **成立** |
| ≥ **1.42**（不比不训练的直线好） | **判否**：直接用直线即可 |
| 介于两者之间 | **歧义** |

成立还要求下面三条护栏都满足。主判据过线但护栏未过，记「计数过线、护栏未过」，不记成立。

### 护栏

- segm AP 全程峰值 ≥ **77.80**（基线峰值 78.80 − 1.0）。
- `spike_recall` 全程峰值 ≥ **0.9311**（基线峰值 0.9511 − 0.02）。
- 读数那次的 `matched_F1@8` ≥ **0.6078**（基线读数那次 0.6228 − 0.015）。点的输出没改，F1@8 应当不降。

### 观察量（只记录，不作判据）

- 最终评估逐穗明细上：有符号偏差 mean(pred − gt)、r；同一批穗上 `line_count` 的 MAE 与偏差；
  `out_points`（存在性阈值读法）的 MAE、偏差、r，对照基线的 2.599 / +2.40 / 0.642。
  逐穗明细每次评估都会被覆盖，只有最终评估的留得下来。
- `count_MAE_img`：计数头也给误检的 wheat_2 计数，每图总数跟着变。
- `loss_point_count` 的训练曲线（分段均值）；最终 \|Δ\| 的分布（`pred_points` − `line_count`）。

## 归因限制

- 计数头的计数 = 直线 + 残差。相对基线默认读数的改善里，直线本身（不训练）就占了大头（2.60 → 1.42）。
  残差的贡献只看与直线的差距和机制判据，不看与基线的差距。
- 计数损失的梯度经共享的 decoder 回传，可能影响分割和点头。点的输出读法不变，所以 `matched_F1@8`、segm AP 的变化只能来自这一项。
- 训练时锚用 GT 框，推理时用预测框。预测框的系统误差会原样进计数，残差头学不到对它的修正。离线核过两者偏差只差 0.02。
- 直线系数在 mini train 上拟合，训练中固定不学。val 上的 1.419 是样本外的数。
- LSJ 把穗缩放 0.1~2.0 倍。锚用原生尺度对角线，Δ 要从缩放后的特征里读出与尺度无关的残差（例如小穗间距与穗长之比）。
- 单种子。

## 成本

- 12500 iter，另加 25 次评估。
- 速度：冒烟独占一张卡 2.63 s/iter；mini/baseline 当时与别的 run 并发约 3.8 s/iter。按 2.6~3.8 s/iter 算，训练 9~13 h，评估另约 1~1.5 h。
- 峰值显存：冒烟 22,989 MB，与基线（约 23.1 GB）同量级。计数头每层只多一个 2 层 MLP 和一次向量化的 L1。
- 跑在 xuyu A100 GPU0（本批次用户临时放开，一个 run 一张卡）。

## 结论（2026-10-08）

**计数过线、护栏未过，按预注册不记成立；机制启动。**
详见 [exp-2026-10-07-mini-count-residual](../experiments/exp-2026-10-07-mini-count-residual.md)。

| 判据 | 要求 | 实测 | |
|:--|:--|:--|:--:|
| 机制：r(头) − r(直线)，最终评估 561 穗 | ≥ +0.03 | **+0.127**（0.747 vs 0.621） | ✅ |
| 主判据：读数那次（iter 9499）`count_MAE_spike` | ≤ 1.28 成立 | **1.146**（基线 iter 8999 为 2.465） | ✅ 过线 |
| 护栏：segm AP 全程峰值 | ≥ 77.80 | 77.26 | ❌ |
| 护栏：`spike_recall` 全程峰值 | ≥ 0.9311 | 0.9217 | ❌ |
| 护栏：读数那次 `matched_F1@8` | ≥ 0.6078 | 0.6027 | ❌ |

- 计数：偏差 +0.02、r 0.75；后半程 13 轮全部 ≤ 1.27。这是 mini 系列第一个默认读数赢过直线（1.42）的 arm。
  同一 checkpoint 只用直线是 1.417，相对直线的收益来自残差 Δ。
- 代价：后半程与基线同 iter 比，segm AP 平均 −2.05、`spike_recall` −0.032、`matched_F1@8` −0.019。
  点的读法没改，差异来自计数损失经共享 decoder 的回传；单次 run 分不清是梯度方向冲突还是梯度裁剪下的份额挤占。
- 未验证的后续方向：计数损失不回传共享 decoder（Δ 的输入 detach），或降低 `POINT_COUNT_WEIGHT`。
