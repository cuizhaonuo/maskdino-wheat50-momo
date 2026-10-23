# 想法：残差计数头的输入 detach —— 计数损失只训练计数头

状态：**不成立：主判据歧义（1.3104），护栏三条都过 → 护栏恢复、计数退化**；机制启动 ｜ 验证于 [exp-2026-10-22-mini-count-residual-detach](../experiments/exp-2026-10-22-mini-count-residual-detach.md) ｜ 2026-10-22 15:17:35 启动于 `902225c`，baiju A100 GPU0（本批次用户临时放开；由主 agent 排卡脚本启动，全程与别人共卡）｜ 预注册提交于 `902225c`（跑之前）｜ 提出 2026-10-22 ｜ 分支 `mini/count-residual-detach`（从 `mini/count-residual`@`222d864` 开出；代码即父臂启动时的 `60d21a3`，父臂之后只有文档提交；100 epoch）｜ 配置 `wheat_joint_single_stage_mini_count_residual_detach.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_count_residual.yaml`）｜ 输出 `output_v2/exp-1022-mini-count-residual-detach`

> 本批「计数改进三臂」（2026-10-22，mini）之一：L `mini/count-residual-detach`（本臂）、D `mini/density-count`、K `mini/coral-ecount`。
> 三臂共用一组判定线：下文的读数规则、主判据、护栏、次口径、噪声地板照抄批次说明第 4 节（`output_v2/_batch-1022-count/AGENT_BRIEF.md`）。
> 机制门与归因是本臂另写的。

## 来源

- 父臂 `mini/count-residual`：每穗计数 = 穗长直线 + 零初始化残差头 Δ。
  设计见 [ideas/count-residual.md](count-residual.md)，结果见 [exp-2026-10-07-mini-count-residual](../experiments/exp-2026-10-07-mini-count-residual.md)。
  - 计数过线：主读数 iter 9499，`count_MAE_spike` 1.1459（≤ 1.28）。最终评估 561 穗，偏差 +0.02、r 0.747。
  - 机制启动：r(头) − r(直线) = +0.127。
  - 三条护栏都没过（见下），按预注册不记成立。
- 父臂文档的归因：点的读法没改，分割、检出、点的退化只能来自计数损失经共享 decoder 的回传。
  单次 run 分不清是梯度方向冲突，还是全模型梯度裁剪下的份额挤占。
- 父臂文档列的未验证后续之一：「计数损失不回传共享 decoder（Δ 的输入 detach）」。本臂只做这一件事。
- detach 的位置与做法同 `mini/kmo-detach`（KMO 头的输入 detach，`forward_prediction_heads` 里 `decoder_norm` 之后）。

## 现象：父臂的代价

| 护栏 | 要求 | 父臂 | E0 |
|:--|:--|--:|--:|
| segm AP 全程峰值 | ≥ 77.80 | 77.26 ❌ | 78.80 |
| `spike_recall` 全程峰值 | ≥ 0.9311 | 0.9217 ❌ | 0.9511 |
| 主读数那次 `matched_F1@8` | ≥ 0.6078 | 0.6027 ❌（iter 9499） | 0.6228（iter 8999） |

- 后半程 13 次评估（iter 6499~12500）与 E0 同 iter 比：
  - segm AP 13/13 次低于 E0，平均 −2.05；bbox AP 12/13 次低，平均 −2.23；
  - `matched_F1@8` 12/13 次低，平均 −0.0186；`spike_recall` 13/13 次低，平均 −0.032。
- 末段 11 项计数损失合计 8.6，占总损失 47.5 的 18%。其余训练损失末段都比 E0 略高，例如 `loss_ce` 0.275 vs 0.255、`loss_point_coord` 0.366 vs 0.342。

**零训练梯度诊断**（10-22，只作参照，不进判定）

- 设置：父臂 `model_final.pth`，下文验证用的那份 2 图 batch，训练态 fp32 前向一次。
  把共享参数的梯度拆成「11 项计数损失」与「其余 127 项」两部分（`_smoke-1022-count-residual-detach/equiv/W_off_trained_fp32.json`）。

| 模块 | ‖g 计数‖ | ‖g 其余‖ | 夹角余弦 |
|:--|--:|--:|--:|
| backbone | 261.1 | 82.3 | +0.013 |
| pixel decoder | 67.9 | 74.6 | +0.048 |
| transformer decoder 与其余输出头 | 19.8 | 109.4 | +0.006 |
| 全部共享参数 | 270.5 | 155.9 | +0.013 |

- 计数梯度的大头在 backbone，是其余损失在 backbone 上梯度的 3.2 倍。它与其余损失的方向几乎正交（cos ≈ 0），不是相反。AMP 下数值几乎相同。
- 读法：父臂末期的代价更像「份额挤占」，而不是「方向冲突」。
  一个更大、方向无关的分量占掉了共享参数的更新：AdamW 逐参数的二阶矩，以及全模型裁剪用的总范数，都被它撑大。
- 只有一个 batch，只作参照。

## 假设

1. 父臂护栏失败，来自 `loss_point_count` 经 `decoder_norm` → decoder → pixel decoder → backbone 的回传。
   它与七项原有损失争共享参数，并在全模型梯度裁剪（`CLIP_VALUE` 0.01，每步都生效）下占掉一部分更新份额。
2. 把 Δ 的输入换成 decoder 输出的 detach 副本后，共享参数拿到的梯度恰好是「去掉计数项的总损失」的梯度，与 E0 的梯度来源相同。
   剩下的唯一联系是计数头自己的梯度仍计入裁剪范数（见「已知限制」）。分割、检出、点应回到 E0 水平，三条护栏能过。
3. 计数头仍读同一份 decoder 特征，只是这份特征只由其余损失塑造。父臂的 Δ 学到了穗长以外的信息（r 比直线高 0.127）。
   - 如果这部分信息在点 / 分割训练出的特征里本来就有，计数仍能过线；
   - 如果它要靠计数损失去塑造特征，r 与计数会退向直线（1.42）。

## 修法

开关默认关闭；关着时与父臂逐位等价（见「实施前已完成的验证」）。

| 位置 | 改动 |
|:--|:--|
| `maskdino/config.py` | 新增 `MODEL.MaskDINO.POINT_COUNT_DETACH`（默认 False） |
| decoder `__init__` / `from_config` | 新参数 `point_count_detach`，存成 bool 属性。不建参数、不消耗随机数。打开时要求 `POINT_COUNT_HEAD=True`（assert） |
| decoder `forward_prediction_heads` | 开关打开时，`point_count_embed` 读 `decoder_output.detach()`；关闭时原样。interm（two-stage 的 encoder 提议）、initial_pred、9 层都经这个函数输出 Δ，一处改动覆盖全部 11 份 |
| 其余 | 一律不动：直线系数（8.924, 0.0345）、`loss_point_count` 的形式与权重（L1，1.0，深监督 11 项）、`TEST.POINT_COUNT_SOURCE: head`、评估器、LSJ mapper、`prepare_targets`、`seg_infer_reg.py` |

- **前向数值不变**：Δ、全部损失、decoder 的全部输出都与关闭时逐位相同；推理与评估不受影响。
- **反向被切断**：`loss_point_count*` 只到 `point_count_embed` 的 4 个参数张量。
  `decoder_norm` 及其上游（decoder、pixel decoder、backbone）和其余输出头，从计数损失拿到的梯度都是 None。
- 计数头的学习率、权重衰减、AdamW 状态与父臂相同（`point_count_embed` 不在 backbone 里，lr 1e-4）。

## 与对照的差异

- **主对照 E0**：`output_v2/exp-0923-mini-ep100`（`mini/baseline`@`0f71161`）。同数据、同起点（ImageNet R-50）、同 schedule、同 `SEED` 42。
  - 解析后的 cfg 与 `wheat_joint_single_stage_mini_ep100.yaml` 相比，只差三个开关：
    `POINT_COUNT_HEAD` True、`TEST.POINT_COUNT_SOURCE` head、`POINT_COUNT_DETACH` True。与 E0 run 落盘的 `config.yaml` 比，另差 `OUTPUT_DIR`。
  - 父臂配置里还写了 `POINT_COUNT_WEIGHT: 1.0` 与 `POINT_COUNT_LINE`，都等于默认值。
  - 新增参数 66,049 个（`point_count_embed.*`），与父臂相同。
- **归因参照：父臂** `output_v2/exp-1007-mini-count-residual`（`60d21a3`）。
  - 解析后的 cfg 与父臂配置逐项比对（CPU，329 个叶子键），**只差 `POINT_COUNT_DETACH`（False → True）一个键**。
  - 与父臂 run 落盘的 `config.yaml` 比，另差 `OUTPUT_DIR`。
  - 参数量相同（52,366,482）。

## 实施前已完成的验证

脚本与产物都在 `output_v2/_smoke-1022-count-residual-detach/`（`scripts/`、`equiv/`、`eval-*`、`smoke40/`）。
父臂代码用 `git archive 60d21a3` 导出到 `parent_src_60d21a3/`，不碰父臂 worktree。`222d864` 与 `60d21a3` 的代码相同，之后只有文档提交。

**1. 配置**（CPU）

- 做法：`get_cfg()` + `add_deeplab_config` + `add_maskdino_config` + `merge_from_file`，展平后逐项比对 329 个叶子键。
- 新配置 vs 父臂配置：只差 `POINT_COUNT_DETACH`。新配置 vs 父臂 run 落盘的 `config.yaml`：另差 `OUTPUT_DIR`。

**2. 开关关闭时与父臂逐位等价**（10-22 15:06~15:13，xuyu A100 GPU1，`scripts/equiv_detach.py`、`run_equiv.sh`，结果见 `equiv/compare.txt`）

- 做法同父臂的 equiv_test：
  - 固定 random / numpy / torch 种子（建模型 0、mapper 1、前向 2），cudnn deterministic；
  - 同一份 batch：mini train 前 2 张图经 LSJ mapper，缩放 1.47 / 0.38，85 个实例、6 个 wheat_2，Ni = 17 / 19、18、17、23、17；
  - 训练态前向一次。
- 父臂代码快照 vs 本 worktree，都用父臂配置（`POINT_COUNT_DETACH` 取默认 False）。两种权重 × 两种精度：
  - 权重：ImageNet 初值（同父臂 equiv_test），父臂 `model_final.pth`（计数头已训，Δ 非零）；
  - 精度：fp32，AMP（`torch.amp.autocast` float16，同 AMPTrainer）。
- 四种组合下都逐位等价：
  - **138 项 loss 逐位相等**；
  - **参数量同为 52,366,482**，state_dict 736 个键逐张量哈希相同；
  - decoder 的 66 个输出张量（含 11 份 Δ）哈希相同。
- 计数头整个关掉（ep100 配置）时也一样：127 项 loss 逐位相等，参数 52,300,433、732 个键相同。
- 梯度不做跨进程逐位比较。同一进程里对同一张图反向两次，就已有 200~390 个张量不逐位相同（最大相对差 2e-4~2e-3）：
  MSDeformAttn 与 grid_sample 的反向用 atomicAdd，本来就不确定。

**3. eval_only 核对**（10-22 14:55~15:00，qiuhao A100 GPU1，`scripts/run_evals.sh`）

父臂 run 的 `model_final.pth` 配它自己的 `config.yaml`，在本 worktree 代码上 `--eval_only`：

| 设置 | 输出目录 | 结果 |
|:--|:--|:--|
| 不加覆盖（`POINT_COUNT_DETACH` 默认 False） | `eval-off` | 36 个指标与父臂训练日志最后一次评估逐项相同（`count_MAE_spike` 1.1979、segm AP 76.8649、`matched_F1@8` 0.6074 等）；`coco_instances_results.json` 与两份 spikelet CSV 和父臂最终评估逐字节相同 |
| 加 `MODEL.MaskDINO.POINT_COUNT_DETACH True` | `eval-on` | 与 `eval-off` 逐项相同，三个文件逐字节相同。detach 只切反向，推理不受影响 |

**4. 开关打开**（同第 2 项的脚本与 batch；本 worktree，新配置 vs 父臂配置）

- 参数量不变（52,366,482），state_dict 键与初值哈希相同。
- 138 项 loss、66 个 decoder 输出、11 份 Δ 都与关闭时逐位相同（fp32 / AMP × 两种权重）。父臂权重下 |Δ| 最大 5.17。
- **只对 11 项 `loss_point_count*` 求梯度**（`torch.autograd.grad`，`allow_unused=True`）：

| | 共享参数（409 个张量） | `point_count_embed`（4 个张量） |
|:--|:--|:--|
| 开（本臂），四种组合 | **全部 None** | 父臂权重：layers.0.weight / layers.0.bias / layers.1.weight 非零（范数 8.87 / 0.50 / 36.3）。ImageNet 初值：layers.1 的 weight / bias 非零（16.97 / 3.67），layers.0 为零 |
| 关（对照，即父臂行为） | 父臂权重：375 个有梯度、374 个非零，一直到 backbone stem。初值：375 个有梯度但全为零（计数头末层零初始化） | 同上 |

  - 初值下 layers.0 为零：末层零初始化，传回第一层的梯度为零。父臂实施前验证里走一步优化器后第一层就有了梯度，这里同理。
  - 父臂权重下 layers.1.bias 的梯度恰好为 0。L1 对 bias 的梯度是 11 层 sign(N̂ − N) 之和除以 6，
    这个 batch 上每层都是 3 个多报、3 个少报（`scripts/probe_bias_sign.py`）。图是连通的，只是正负号恰好抵消。
- **对总损失求梯度**：
  - 413/413 个可训练张量都有梯度，fp32 与 AMP 相同，没有 autocast 缓存吞梯度的问题。
  - 全零的张量与父臂代码完全相同：初值下 7 个（`level_embed`、共享框头末层零初始化挡住的 4 个、计数头第一层 2 个），父臂权重下 1 个（上面那个 bias）。
  - AMP 下按 S = 1024 缩放时，初值那一组有梯度溢出成 inf，父臂代码也一样（实际训练中 GradScaler 会跳过该步）。用 S = 1 重跑后全部有限，结论不变。
- **共享参数上 g(总损失) 与 g(总损失 − 计数项)**：
  - 结构上，计数项到共享参数的梯度是 None，两者在精确算术下相同。
  - 数值上，两者的差异与反向非确定性同一量级。
    - fp32、父臂权重：不逐位相同 389/409，最大相对差 7.0e-4；同一张图两次反向的地板是 389/409、7.8e-4。
    - 四种组合（AMP 初值取 S = 1）的最大相对差在 2.5e-4~2.5e-3 之间，各自的地板在 2.1e-4~1.7e-3 之间。
  - 对照（开关关）：父臂权重下最大相对差 1.67（AMP 1.57），比地板大三个数量级。
- **计数头梯度与裁剪**：父臂权重下全模型梯度范数 160.3，其中共享参数 155.9、计数头 37.3。
  计数头自身梯度使裁剪系数 0.01/‖g‖ 缩小 2.75%（AMP 2.81%）；初值下 < 0.001%。见「已知限制」。

**5. 冒烟**（10-22 14:55~15:06，正式配置，CLI 覆盖 `SOLVER.MAX_ITER 40`、`TEST.EVAL_PERIOD 20`，输出 `smoke40/`）

- xuyu A100 GPU1，与别人一个约 45 GB 的进程共卡。训练 40 iter 与两次完整 val 评估（iter 19、39）都跑通，0 个 Traceback。
- 138 项损失全部有限，其中 11 项 `loss_point_count*`。
  - `total_loss` 555.85 / 550.51，`loss_point_count`（每层，20 步窗口中位数）1.595 / 1.225。
  - 与父臂冒烟（555.85 / 550.55，1.595 / 1.225）相同到第 3~4 位有效数字。前向逐位相同；这 40 步 lr ≤ 1e-7，还在 warmup 里，参数几乎不动；余下差异来自 GPU 非确定性。
- 未训练模型的评估读数与父臂冒烟相同：`count_MAE_spike` 为 nan（没有 score ≥ 0.4 的 wheat_2），`count_MAE_img` 48.735。
- 速度 **2.815 s/iter**（`Overall training speed: 38 iterations in 0:01:46`，共卡；父臂冒烟独占一张卡时 2.63）。
- 峰值显存 `max_mem` **22,989 MB**（torch 口径，与父臂冒烟相同）；nvidia-smi 口径约 27.0 GB（评估时进程占 26,982 MiB）。

## 预期与判否线（跑之前写死）

### 读数规则（批次统一）

- 一律用默认读数：内置评估，`POINT_THRESHOLD` 取配置值 0.35；本臂每穗计数来自计数头（`POINT_COUNT_SOURCE: head`）。
- 不扫阈值：不改 `TEST.POINT_THRESHOLD` 去扫，不跑任何 `--sweep`，不用在 val 上拟合的校准值做对比。
- **主读数**：后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估（排除 nan）。该 checkpoint 的全部指标都从这次读。
- 护栏 segm AP、`spike_recall` 用全程峰值。
- 单种子（SEED 42），不加种子。

### 基线读数（E0，`output_v2/exp-0923-mini-ep100/metrics.json`）

| | iter | `count_MAE_spike` | `matched_F1@8` | `spike_recall` | segm AP |
|:--|--:|--:|--:|--:|--:|
| 主读数（后半程计数最好的那次） | 8999 | **2.4648** | 0.6228 | 0.9511 | 77.40 |
| 全程峰值 | — | — | — | **0.9511** | **78.80** |

- E0 最终评估：偏差 +2.40、r 0.642（578 穗）。
- **不训练的参照线**：穗长直线，点数 = 8.924 + 0.0345 × 预测框对角线（mini train 拟合）。
  val 上 MAE 约 1.42（E0 最终评估的配对穗上 1.4187）。

### 主判据（批次统一）：主读数那次的 `count_MAE_spike`

| 条件 | 判定 |
|:--|:--|
| ≤ **1.28** | **计数成立** |
| ≥ **1.42**（穗长直线） | **判否** |
| 介于两者之间 | **歧义** |

### 护栏（批次统一，与父臂预注册是同一组线；任一不过则整体不成立）

- segm AP 全程峰值 ≥ **77.80**；
- `spike_recall` 全程峰值 ≥ **0.9311**；
- 主读数那次 `matched_F1@8` ≥ **0.6078**（E0 − 0.015）。

主判据过线但护栏未过，记「计数过线、护栏未过」，不记成立。

### 机制门（本臂）：计数头有没有学到穗长以外的信息

- 在最终评估的逐穗明细（`inference/spikelet_pair_counts.csv`）上，取同一批配对穗：
  **r(`pred_points`, `gt_points`) − r(`line_count`, `gt_points`) ≥ 0.03** → 机制启动；达不到 → 记「机制未启动」。
- 两个 r 都在取整后的整数计数上算：`pred_points` 是计数头的计数，`line_count` 是同一批穗只用直线的计数。口径同父臂。
- 参照：父臂 +0.127（0.7473 vs 0.6207，561 穗）。

### 归因（本臂，对父臂）

- **主判据过线（≤ 1.28）且三条护栏都过**，才记「父臂护栏失败来自计数损失回传，detach 修好了」。
- 下面两项只记录，不设判否线：
  - 后半程 13 次评估里，segm AP 与 E0 同 iter 的平均差（父臂 −2.05）；
  - 主读数 `count_MAE_spike` 相对父臂 1.1459 的变化。
- 不另设线的理由：
  - 「修好了」已由主判据加三条护栏判定，再加线等于给同一件事第二套标准。
  - segm AP 后半程平均差没有测过的噪声地板（单种子，只测过 `matched_F1@8` 的 0.015），定线只能拍脑袋。
- 结果分类（只给可能的结果起名，不改上面的线）：

| 主判据 | 护栏 | 记为 |
|:--|:--|:--|
| 过线 | 三条都过 | **成立**；归因成立 |
| 过线 | 未全过 | 计数过线、护栏未过（不记成立）。切断回传没有修好护栏：父臂的代价不只来自回传，或在单种子噪声内 |
| 歧义 / 判否 | 三条都过 | 不成立；记「护栏恢复、计数退化」。父臂的计数收益要靠计数损失塑造共享特征，与护栏代价是同一件事的两面 |
| 歧义 / 判否 | 未全过 | 不成立 |

### 次口径（批次统一，必报，不进判定）

- 在主读数 checkpoint 上 `--eval_only`，加 `MODEL.MaskDINO.TEST.SPIKELET_EVAL_SIGMAS '[8.0,16.0,32.0]'` 重评。
- 最终评估逐穗 CSV 上报：有符号偏差、Pearson r、预测计数标准差、E[GT | 预测] 斜率（= r × sd_GT / sd_pred），以及同批穗上的穗长直线 MAE。

参照（最终评估逐穗 CSV，10-22 用同一段代码复算；标准差按总体口径）：

| run | 配对穗 | MAE | 偏差 | r | 预测标准差（GT） | 斜率 | 同批穗直线 MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 父臂（计数头） | 561 | 1.198 | +0.018 | 0.747 | 1.99（2.40） | 0.90 | 1.417 |
| E0（存在性 > 0.35 的点数） | 578 | 2.599 | +2.405 | 0.642 | 1.54（2.41） | 1.00 | 1.419 |

### 噪声地板

- `matched_F1@8` 0.015。单种子，噪声内的差异记「歧义」。

### 观察量（只记录，不作判据）

- `loss_point_count`（每层）训练曲线的分段均值。对照父臂：iter 0~500 / 2000~2500 / 6000~6500 / 9000~9500 / 12000~12500
  依次为 1.46 / 1.38 / 1.10 / 0.90 / 0.77。detach 后计数头只能改自己，下降可能更慢。
- 末段其余训练损失（`loss_ce`、`loss_point_coord` 等），对照 E0 与父臂。归因成立时应回到 E0 水平。
- 最终 Δ 的分布（`pred_points` − `line_count`），以及 Δ 与直线残差（gt − 直线）的相关。对照父臂：69% 的穗非零、平均 |Δ| 1.01、相关 0.54。
- `out_points`（存在性阈值读法）的 MAE、偏差、r。对照父臂 2.383 / +2.119 / 0.600，E0 2.599 / +2.405 / 0.642。
- `count_MAE_img`。

## 已知限制

- **单种子。** 同代码同 SEED 的两次 run 到 iter 19 就会分叉（GPU 反向非确定性）。本臂与父臂、E0 的差异都是单次 run 对单次 run。
- **detach 后计数头只能用其余损失塑造出的特征。** 父臂的 Δ 可能部分依赖计数损失对特征的塑造，本臂的 r 可能比父臂低，计数可能退向直线。
- **全模型梯度裁剪仍把计数头算在内。** `CLIP_TYPE: full_model` 用全部参数的梯度总范数算裁剪系数 0.01/‖g‖，计数头自己的梯度也在里面。
  所以计数损失仍能通过这个系数，把共享参数的梯度统一缩小一点。
  - 实测（「实施前已完成的验证」第 4 项）：父臂权重下缩小 2.75%（AMP 2.81%），ImageNet 初值下 < 0.001%。
  - 这是对全部共享参数的统一缩放，方向不变。AdamW 对恒定的统一缩放不敏感，只在这个比例逐步变化时有一点影响。
  - 训练中不记录这个比例。
- 外层匈牙利匹配（哪个 query 配哪个 GT，进而哪些 Δ 受监督）仍由共享网络决定。训练锚用 GT 框、推理用预测框；直线系数固定不学。这三点同父臂。

## 成本

- 12500 iter，另加 25 次评估。
- 速度：
  - 冒烟 2.815 s/iter（与别人约 45 GB 的进程共卡）；
  - 父臂正式 run 纯训练 2.608 s/iter（xuyu GPU0 独占），25 次评估共 52 min，全程 9:59；
  - detach 不增加计算。按 2.6~3.8 s/iter 算，训练 9~13 h，评估另约 1 h。
- 峰值显存：
  - 冒烟 `max_mem` 22,989 MB，与父臂冒烟相同；父臂正式 run 24,701 MB（E0 24,195 MB）；
  - 本臂预计与父臂相同，约 24.7 GB（torch 口径）。nvidia-smi 口径约 27 GB，排卡按空闲显存 ≥ 28 GB 算。
- 卡由主 agent 排（本批用户临时放开 qiuhao、baiju、xuyu GPU0/1；4090 放不下 bs4）。

## 结论（2026-10-23）

**不成立：主判据歧义、护栏三条都过，按结果分类表记「护栏恢复、计数退化」；机制启动。**
详见 [exp-2026-10-22-mini-count-residual-detach](../experiments/exp-2026-10-22-mini-count-residual-detach.md)。

| 判据 | 要求 | 实测 | |
|:--|:--|:--|:--:|
| 主判据：读数那次（iter 10999）`count_MAE_spike` | ≤ 1.28 成立；≥ 1.42 判否 | **1.3104**（E0 iter 8999 为 2.4648，父臂 iter 9499 为 1.1459） | 歧义 |
| 护栏：segm AP 全程峰值 | ≥ 77.80 | 78.00 | ✅ |
| 护栏：`spike_recall` 全程峰值 | ≥ 0.9311 | 0.9478 | ✅ |
| 护栏：读数那次 `matched_F1@8` | ≥ 0.6078 | 0.6354 | ✅ |
| 机制门：r(头) − r(直线)，最终评估 578 穗 | ≥ +0.03 | **+0.070**（0.682 vs 0.612；父臂 +0.127） | ✅ |

- 归因记录量（只记录）：后半程 segm AP 与 E0 同 iter 的平均差 −0.73（父臂 −2.05）；主读数计数相对父臂 +0.165（1.3104 vs 1.1459）。
- 护栏回到 E0 水平：后半程 `matched_F1@8` +0.002、`spike_recall` −0.003；七项原有训练损失末段回到 E0 水平。
  这支持「父臂的护栏代价来自计数损失回传」。
- 计数退回去一截：训练集上的计数损失只从 1.46 降到 1.35（父臂降到 0.77），读数 1.31，落在歧义区间。
  它仍比同批穗的直线低 0.11，扣掉各自偏差后差距不变，收益来自 r。
- 父臂的计数收益与护栏代价是同一件事的两面：计数信息要靠计数损失塑造共享特征。
- 未验证的后续方向：只挡住进 backbone 的计数梯度；或不 detach、只降 `POINT_COUNT_WEIGHT`。
- 读数一律是默认读数，没有做阈值扫描。单种子。
