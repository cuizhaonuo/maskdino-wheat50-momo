# 想法：E 损失的梯度不进 backbone（K 臂 + backbone stop）

状态：**预注册（跑之前写死），未启动** ｜ 提出 2026-10-24（用户 10-24 决定开这一臂）｜ 分支 `mini/coral-ecount-bbstop`（从 `mini/coral-ecount`@`0fb4ad6` 开出；代码即 K 启动时的 `709059d`，之后只有文档提交；100 epoch）｜ 配置 `wheat_joint_single_stage_mini_coral_ecount_bbstop.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_coral_ecount.yaml`）｜ 输出 `output_v2/exp-<启动日>-mini-coral-ecount-bbstop`（占位，主 agent 定）

## 来源

- 父臂 K `mini/coral-ecount`：[ideas/coral-ecount.md](coral-ecount.md)、[exp-2026-10-23-mini-coral-ecount](../experiments/exp-2026-10-23-mini-coral-ecount.md)。
  CORAL 存在性头 + E = Σ_k σ(l_k) 的 L1 损失，计数读 round(E)，点按 K 输出。
  本臂只在它上面加一件事：E 损失（`loss_point_ecount*`）的梯度不进 `backbone.*`。
- mini 上四个计数臂的共同规律（都是单种子、默认读数）：

  | 臂 | 计数怎么来 | 主读数 count | r | 后半程 segm AP 平均差 | 护栏（旧线） |
  |:--|:--|--:|--:|--:|:--|
  | count-residual | 直线 + 残差头，L1 经 decoder 回传 | 1.1459 | 0.747 | 对 E0 −2.05 | 三条全挂 |
  | count-residual-detach（L） | 同上，头的输入 detach | 1.3104 | 0.682 | 对 E0 −0.73 | 全过 |
  | density-count（D） | mask_features 上的密度头 | 1.3239 | 0.668 | 对 E0 −1.99 | segm 挂 |
  | coral-ecount（K，父臂） | CORAL + E 的 L1，读 round(E) | 1.1897 | 0.748 | 对父臂 −2.84，对 E0 −3.29 | segm、F1@8 挂 |

  计数损失经共享特征回传，计数能压到 1.15~1.19，但 segm AP 掉 2~3；只训头（L）护栏恢复，计数退回 1.31。
- L 臂的零训练梯度诊断（该分支 `experiments-bank/ideas/count-residual-detach.md`）：count-residual 的计数梯度大头在 backbone，
  是其余损失在 backbone 上梯度的 3.2 倍，方向近正交，读作「份额挤占」。L 臂结论里列的未验证后续之一就是「只挡住进 backbone 的计数梯度」。
- **用户 10-24 的决定**：开这一臂（主 agent 转达）。
- **用户 10-24 的口径（原话）**：「我比较在意的数值是 count_MAE_spike,其实 segmAP 和 F1@8 基本保持就算可以,比如你的 K 臂在我认为算是不错的」。
  本臂的护栏按这句话放宽到 K 臂水平，见「护栏」。
- 批次说明 `output_v2/_batch-1022-count/AGENT_BRIEF.md` 第 2 节（仓库硬规则）与第 3 节（数据与 schedule）照旧适用；
  GPU 规则与判定线以本臂的派活说明为准。

## 现象：K 的收益与代价

| | E0 `exp-0923-mini-ep100` | K 的父臂 coral-exist `exp-0926-mini-coral-exist` | K `exp-1023-mini-coral-ecount` |
|:--|--:|--:|--:|
| 计数读法 | 点数 | round(E)（同口径重评，参照） | round(E) |
| 主读数 `count_MAE_spike` | 2.4648（iter 8999） | 1.3287（iter 11499） | **1.1897**（iter 11999） |
| 最终评估 r | 0.642 | 0.6716 | 0.7478 |
| segm AP 全程峰值 | 78.80 | 78.71 | **76.03** |
| 后半程 13 次 segm AP 与 E0 同 iter 的平均差 | 0 | −0.45 | **−3.29** |
| 主读数那次 `matched_F1@8` | 0.6228 | 0.6226 | 0.6033 |
| `spike_recall` 全程峰值 | 0.9511 | 0.9478 | 0.9315 |
| 末段（iter 11250~12499）五项分割 / 检测训练损失合计 | 15.836 | 15.674 | **17.922** |

- 五项 = `loss_ce`、`loss_mask`、`loss_dice`、`loss_bbox`、`loss_giou`，各 11 份（主输出、`_0`~`_8`、`_interm`，不含 dn），已乘权重，
  末段 63 行训练日志的均值。K 比 coral-exist 高 2.25（+14%）：训练集上分割 / 检测本身也拟合得更差，不只是 val 上。
- K 的 E 损失占总损失 13%~20%；末段主输出 `loss_point_ecount` 0.847，11 份合计 9.41。

## 零训练梯度诊断（实施前，2026-10-24）

做法仿 L 臂：本分支代码、本臂配置，固定 2 图 batch（mini train 前 2 张图经 LSJ mapper，85 个实例、6 个 wheat_2，与 L / K 的验证 batch 相同），
训练态前向一次，在同一张图上分别求 g_E（11 项 `loss_point_ecount*`）与 g_rest（其余 127 项）的梯度。
脚本 `output_v2/_smoke-1024-coral-ecount-bbstop/scripts/diag_bbstop.py`，产物在 `diag/`，汇总 `compare_diag.txt`。qiuhao A100 GPU1。

**K 的 `model_0011999.pth`（主读数 checkpoint），fp32：**

| 模块 | 可训练张量（E 到达的） | ‖g_E‖ | ‖g_rest‖ | ‖g_E‖ / ‖g_rest‖ | 余弦 |
|:--|--:|--:|--:|--:|--:|
| backbone | 53（53） | 300.8 | 99.4 | **3.03** | +0.13 |
| pixel decoder | 121（113） | 85.3 | 98.6 | 0.87 | −0.03 |
| transformer decoder | 208（202） | 25.0 | 54.1 | 0.46 | −0.01 |
| CORAL 头（打分 MLP + 25 个 bias） | 7（7） | 48.2 | 3.6 | 13.4 | −0.16 |
| 其余头（类别 / 框 / mask / query 等） | 21（6） | 5.0 | 127.7 | 0.04 | −0.00 |
| 非 backbone 合计 | 357（328） | 101.2 | 170.2 | 0.59 | −0.02 |
| 全部 | 410（381） | 317.4 | 197.1 | 1.61 | +0.06 |

- AMP（autocast fp16，梯度缩放 S = 1024，65536 / 8192 溢出）几乎相同：backbone 3.04（+0.14）、pixel decoder 0.86、decoder 0.43、全部 1.60。
- 换一份 batch（mini train 第 3、4 张图，12 个 wheat_2）：backbone 2.97（+0.16）、pixel decoder 1.04（+0.09）、decoder 0.47（+0.10）、全部 1.67。
- **随训练变大**（同一 batch，fp32，backbone / pixel decoder / decoder / 全部）：

  | 状态 | backbone | pixel decoder | decoder | 全部 | E 损失占总损失 |
  |:--|--:|--:|--:|--:|--:|
  | 第 0 步（ImageNet 初值） | 0.036（cos −0.29） | 0.026 | 0.060 | 0.042 | 4.3% |
  | K iter 5999 | 1.09（+0.06） | 0.26 | 0.16 | 0.36 | 21.2% |
  | K iter 11999 | **3.03**（+0.13） | 0.87 | 0.46 | 1.61 | 21.3% |

  其余损失的梯度随训练降了约 50 倍（backbone 5060 → 99），E 损失在 backbone 上的梯度反而从 184 升到 301。

**全模型梯度裁剪（`CLIP_TYPE full_model`，`CLIP_VALUE` 0.01，每步都生效）下，挡住 backbone 前后**（K iter 11999，fp32；括号里是第二份 batch）：

| | K（不挡） | bbstop | |
|:--|--:|--:|:--|
| 全模型梯度范数 | 383.1 | 220.3 | ×0.575（×0.582） |
| 裁剪系数 0.01/‖g‖ | | | ×1.74（×1.72） |
| backbone 占全模型更新（范数比） | 0.858 | 0.451 | |
| 其余损失在 backbone 上的那一块占全模型更新 | 0.259 | 0.451 | 裁剪后幅度 ×1.74 |
| backbone 梯度里其余损失的份额 ‖g_rest‖ / ‖g_E + g_rest‖ | 0.302 | 1 | |

- iter 5999：全模型范数 ×0.955，backbone 占比 0.392 → 0.270，K 的 backbone 梯度里其余损失份额 0.657。第 0 步：×1.006，几乎没有区别。
- **读法**：K 的后半程，backbone 拿到全模型更新的 86%，其中七成（按平方是九成）来自 E 损失，与 L 臂诊断的 count-residual（3.2 倍）同样严重。
  - 裁剪系数对全部参数是同一个倍数，AdamW 的 m/√v 对恒定的统一缩放不敏感，所以裁剪本身大体被抵消。
  - 真正起作用的是逐参数的二阶矩：K 的 backbone 上 √v 主要由 E 损失撑起，其余损失那一部分的归一化步长只剩约 0.30 倍（若 E 梯度逐步像噪声）；
    若 E 梯度逐步一致，它就把 backbone 推向自己的方向。挡住之后 backbone 的 AdamW 状态只来自其余损失。
  - pixel decoder 上 E 梯度仍是其余损失的 0.87~1.04 倍，本臂不动它。E 梯度到不了的 8 个 pixel decoder 张量正好是 1/4 分辨率的 mask 分支（`adapter_1`、`layer_1`、`mask_features`）；E 梯度进的是 deformable encoder，而它的 1/8 输出上采样后进 mask_features。K 的分割代价若有一部分出在这里，本臂去不掉。
- 只有 1 个 checkpoint、2 份 batch，只作参照，不进判定。

## 假设

1. K 的分割代价主要来自 E 损失进 backbone 的那部分梯度（份额挤占：AdamW 二阶矩被 E 梯度占满）。挡住之后，backbone 只由其余损失塑造，
   分割 / 检测的训练拟合与 val 上的 segm AP 都应向父臂 coral-exist 回升。
2. K 的计数收益有一部分来自 E 损失塑造 pixel decoder、decoder 与 CORAL 头。这些路径本臂都保留。
   - 若计数信息主要在这些模块里，计数接近 K（1.19），r 接近 0.75；
   - 若计数信息要靠 E 损失改 backbone，计数退向 L 臂那样的 1.31 与父臂的 1.33。
3. 两条都成立时，本臂同时保住计数、减小分割代价，即用户口径下比 K 更好的点。

## 修法

开关默认关闭；关着时训练循环就是 detectron2 的 `AMPTrainer`，与 K 逐位等价（见「实施前已完成的验证」）。

| 位置 | 改动 |
|:--|:--|
| `maskdino/config.py` | 新增 `MODEL.MaskDINO.POINT_ECOUNT_STOP_BACKBONE`（默认 False） |
| `train_net.py` | 新函数 `ecount_backbone_stop_backward` 与新类 `EcountBackboneStopAMPTrainer(AMPTrainer)`。`Trainer.__init__` 在开关打开时用它（并检查 `POINT_ECOUNT_WEIGHT > 0`、`SOLVER.AMP.ENABLED`、单卡，不满足直接报错）；关闭时照旧构造 `AMPTrainer`，代码路径不变 |
| 配置 | `wheat_joint_single_stage_mini_coral_ecount_bbstop.yaml`，`_BASE_` 为 K 的 yaml，只加 `POINT_ECOUNT_STOP_BACKBONE: True` |

`EcountBackboneStopAMPTrainer.run_step` 照搬 `detectron2/detectron2/engine/train_loop.py` 的 `AMPTrainer.run_step`（含本项目加的 `grad_accum_steps`：
只在累积周期开头 `zero_grad`、loss 除以 `grad_accum_steps`、累积满才 `grad_scaler.step` / `update`），只把一次 backward 拆成两次：

1. E 损失（`loss_dict` 里以 `loss_point_ecount` 开头的 11 项）：
   `torch.autograd.backward(scale(L_E / accum), inputs=非 backbone 的 357 个可训练张量, retain_graph=True)`。
   梯度只累加进这些参数的 `.grad`；autograd 只执行通往这些参数的那部分图（backbone 的反向不在其中）。
2. 其余 127 项：`scale(L_rest / accum).backward()`，照常到全部参数，释放计算图。

- 两份乘同一个 `GradScaler.scale`、都除以 `grad_accum_steps`。合并后 backbone 的 `.grad` 只有其余损失的梯度；
  非 backbone 的 `.grad` 是全部损失的梯度，与一次整体反传只差浮点求和顺序（在叶子上相加，而不是在中间节点先合并）。
- `grad_scaler.step` 里的 `unscale_`（inf / nan 检查、溢出跳步）、`FullModelGradientClippingOptimizer` 的全模型裁剪与 AdamW 照旧，都作用在合并后的梯度上。
- `loss_dict` 原样写进 metrics（`_write_metrics`），`total_loss` 的口径不变。
- 先反传 E 损失：这一遍只经过 decoder / pixel decoder，保留计算图时多出的只是这部分的中间梯度；先做整网反传再保留图，会在算 backbone 梯度时仍占着全部激活。两份相加与顺序无关。
- 不改：模型结构与初始化、前向、全部损失及权重、外层 / 内层匹配、DN、CoralHead 的 bias 学习率、推理与评估、读数（计数 round(E)、点按 K 输出）、schedule。
- 只实现了 AMP、单卡（本仓库的 mini 训练都是这样）。

## 与父臂的差异

- 解析后的 cfg（328 个叶子键）与 K 的 yaml 相比**只差 `POINT_ECOUNT_STOP_BACKBONE` 一个键**。
  与 K run 落盘的 `config.yaml` 相比另差 `OUTPUT_DIR`（`ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS` 在落盘文件里是 list、yaml 里是 tuple，数值相同）。
- 参数、state_dict、前向、138 项 loss、推理输出都与 K 相同；只有每步的反传不同。
- 训练循环换成 `EcountBackboneStopAMPTrainer`，每步多一次只经过 decoder / pixel decoder 的部分反传。

## 实施前已完成的验证

脚本与产物都在 `output_v2/_smoke-1024-coral-ecount-bbstop/`（`scripts/`、`diag/`、`equiv/`、`eval-off/`、`eval-on/`、`accum/`、`smoke_*`）。
父臂一侧用 `git archive 0fb4ad6` 的快照（`parent_0fb4ad6/`）运行，没有在 K 的 worktree 里执行任何东西。全部 GPU 任务在 qiuhao A100 GPU1。

**1. 配置**（CPU，`scripts/cfg_diff.py`，`cfg_diff.txt`）：见上一节。

**2. 开关关闭时与父臂逐位等价**（`scripts/equiv_test.py`，抄自 `_smoke-1022-coral-ecount/scripts/`，未改；`equiv/compare.txt`）

- 做法同 K：固定 random / numpy / torch 种子（建模型 0、mapper 1、前向 2），cudnn deterministic，ImageNet R-50；同一份 2 图 batch；训练态前向，fp32 一次、AMP 一次。
- 父臂快照（K 配置） vs 本 worktree（K 配置，开关取默认 False） vs 本 worktree（本臂配置，开关 True）：
  - **参数量同为 52,294,290，state_dict 733 个键逐张量哈希相同**，weight_dict 320 个键、`criterion.losses` 相同；
  - **138 项 loss 在 fp32 与 AMP 下都逐位相等**（fp32 总和 542.2360777556896，AMP 542.2805890738964），其中 `loss_point_ecount*` 11 项。
- 开关关闭时 `Trainer` 构造的就是 detectron2 的 `AMPTrainer`（冒烟日志里核对了类名），训练循环与 K 同一段代码。

**3. eval-only 核对**（`scripts/run_evals.sh`、`compare_evals.py`，`compare_evals.txt`）

K 的 `model_final.pth` 配它自己的 `config.yaml`，本 worktree 代码走 `train_net.main` 的 `--eval_only` 分支（`eval_dump.py` 只把 `verify_results` 换成完整精度 dump）：

| 设置 | 结果 |
|:--|:--|
| 不加覆盖（开关默认 False），`eval-off` | **40 项指标与 K 训练日志末次评估（iter 12500）完整精度逐项相同**（segm AP 76.03311132961342、`count_MAE_spike` 1.1943462897526502、`matched_F1@8` 0.6025866050808314 等）；`coco_instances_results.json`、`instances_predictions.pth` 与两份 spikelet CSV 和 K 最终评估逐字节相同 |
| 加 `MODEL.MaskDINO.POINT_ECOUNT_STOP_BACKBONE True`，`eval-on` | 与 `eval-off` 逐项相同，4 个文件逐字节相同。开关只改训练循环 |

**4. 开关打开的正确性**（`scripts/diag_bbstop.py`，`compare_diag.txt`）

在同一张图上先用 `torch.autograd.grad` 求参照，再调用 `train_net.ecount_backbone_stop_backward`（`run_step` 用的同一个函数）做两次反传，读 `.grad`。
参照：backbone 上 g_rest（只有其余损失），非 backbone 上 g_tot（全部损失）。地板：同一张图上把参照再求一次（MSDeformAttn 与 grid_sample 的反向用 atomicAdd，本来就不逐位确定）。

| 变体 | backbone：`.grad` vs g_rest（最大相对差 / 相对范数差） | 地板 | 非 backbone：`.grad` vs g_tot | 其中求和顺序本身（g_E + g_rest vs g_tot） | `.grad` vs g_E + g_rest | 对照：backbone `.grad` vs g_tot |
|:--|:--|:--|:--|:--|:--|:--|
| K 11999，fp32 | 5.2e-4 / 2.3e-4 | 7.3e-4 / 2.4e-4 | 3.4e-4 / 4.3e-5 | 3.4e-4 / 4.3e-5 | 5.4e-5 / 2.3e-6 | 1.2 / 0.91 |
| K 11999，AMP（S 1024） | 1.8e-3 / 4.7e-4 | 1.1e-3 / 4.6e-4 | 1.9e-3 / 2.3e-4 | 1.9e-3 / 2.3e-4 | 1.5e-4 / 1.8e-6 | 1.2 / 0.91 |
| K 11999，fp32，第二份 batch | 5.7e-4 / 2.7e-4 | 5.3e-4 / 2.8e-4 | 3.3e-4 / 4.4e-5 | 3.3e-4 / 4.4e-5 | 2.6e-5 / 1.4e-6 | 1.1 / 0.91 |
| K 5999，fp32 | 6.0e-4 / 2.1e-4 | 5.7e-4 / 2.1e-4 | 2.6e-4 / 1.3e-5 | 2.6e-4 / 1.3e-5 | 5.4e-5 / 5.9e-7 | 1.1 / 0.71 |
| 初值，fp32 | 1.1e-4 / 5.3e-5 | 1.3e-4 / 5.3e-5 | 4.2e-5 / 1.4e-5 | 4.2e-5 / 1.4e-5 | 8.8e-6 / 7.6e-7 | 0.06 / 0.037 |
| 初值，AMP（S 128） | 1.2e-3 / 1.6e-4 | 1.2e-3 / 1.9e-4 | 2.2e-3 / 3.0e-5 | 2.2e-3 / 3.0e-5 | 7.2e-4 / 3.1e-6 | 0.07 / 0.034 |

- **backbone 的梯度 == 只反传其余损失的梯度**：与 g_rest 的差和非确定性地板同一量级；与 g_tot 的相对范数差 0.91（训练后）—— E 损失那一大块确实被挡住了。
- **非 backbone 的梯度 == 反传全部损失的梯度**：相对范数差 1.3e-5~2.3e-4。这个差**整个**来自求和顺序：`.grad` 与「两份分开求再相加」只差地板（≤ 2.3e-6），
  而「两份相加」与整体反传之差和 `.grad` 与整体反传之差逐项相同。AMP 下大一些（2.3e-4），因为中间梯度是 fp16。
- **有梯度的可训练张量**：可训练 410 个，拆分反传后 410 个都有梯度、全部有限；训练后的权重 410 个非零，初值 405 个非零 + 5 个结构性为 0
  （`level_embed` 与框头末层零初始化挡住的 4 个，K 预注册里同样是这 5 个）。非零集合与整体反传相同；**fp32 与 AMP 的集合相同**，
  E 损失梯度非零的张量 fp32 / AMP 都是 381 个（初值 380 个），没有 autocast 在 no_grad 下缓存权重的问题（本改动也没有在前向里加任何调用）。

**5. `run_step` 整步核对，含 `grad_accum_steps` = 2**（`scripts/accum_test.py`，`accum/`）

真实的 `EcountBackboneStopAMPTrainer`：AdamW + 全模型裁剪（`Trainer.build_optimizer`）、`GradScaler(init_scale=256)`、EventStorage 写 metrics，K 的 `model_0011999.pth`。
数据由一个生成器给出，交出 batch 前设与参照相同的前向种子；`after_backward` 钩子拷下 `.grad`（已乘 scale，尚未 unscale）。
参照在 `run_step` 之前用 `autograd.grad` 求（backbone = g_rest、非 backbone = g_tot，各除以 accum）。

- accum = 1：`run_step` 写进 metrics 的 138 项 loss 与参照逐位相同；`.grad` 与参照：backbone 相对范数差 4.5e-4（地板 4.7e-4），
  非 backbone 2.3e-4（即第 4 条 AMP 下的求和顺序差）；410 个张量都有梯度、全部有限；step 后 410 个参数都更新。
- accum = 2（batch A = mini train 第 1、2 张图，B = 第 3、4 张图）：
  - 两个 micro-step 写进 metrics 的 loss 都与参照逐位相同；
  - 第 1 个 micro-step 前梯度全为 None（周期开头清零），之后**参数一个都没变**（不 step）；
  - 第 2 个 micro-step 前梯度不为 None（不清零），之后 **410 个参数都更新**；
  - 第 1 步后 `.grad` 等于 A/2：backbone 相对范数差 4.5e-4（地板 4.6e-4），非 backbone 2.3e-4；
    第 2 步后等于 A/2 + B/2：backbone 5.1e-4（地板 5.2e-4），非 backbone 2.5e-4。
- 附带发现：一个进程里的**第一次**前向，与之后同种子的前向在个别 loss 的末位上不同（第一版核对 `accum/v1_no_warmup/` 里 batch A 的 total 差 6e-8 相对），
  原因未查，与本改动无关（跨进程的第一次前向彼此逐位相同，第 2 条就是这样比的）。最终版先空跑一次前向再比。

**6. 冒烟**（`scripts/run_smoke.sh`、`smoke_train.py`、`summarize_smoke.py`；`smoke_off*`、`smoke_on*`、`smoke_summary.txt`）

10-24 19:47~20:12，qiuhao A100 GPU1，与别人 3 个进程共卡（约 46 GB、利用率 100%）。为了在共享卡上放得下，设了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（不影响 torch 口径的峰值）。

- 正式配置加 CLI 覆盖 `SOLVER.MAX_ITER 60`、`TEST.EVAL_PERIOD 0`，bs4；走 `train_net.setup` / `train_net.Trainer` 的完整训练循环（优化器、LR 调度、日志），
  只去掉 PeriodicCheckpointer，不存 `model_final.pth`，这不是一次训练 run。先关（K 配置，另去掉 EvalHook，只测速度与显存），再开（本臂配置，训练末跑一次完整 val 评估）。
- 内层 trainer：关时是 `detectron2.engine.train_loop.AMPTrainer`，开时是 `train_net.EcountBackboneStopAMPTrainer`。
  开时的日志：可训练张量 410 个，挡住 53 个（`backbone.*`），E 损失只反传到其余 357 个；挡住 backbone 的 loss 项 11 个（`loss_point_ecount`、`_0`~`_8`、`_interm`）。
- 两次 60 iter 都跑通，138 项 loss 全部有限。量级（每份，已乘权重；前 10 步均值 → 后 10 步均值）：

  | 损失 | 前 10 步 | 后 10 步 |
  |:--|:--|:--|
  | `loss_point_ecount` | 1.83~2.02（主输出 1.86，11 份合计 20.8） | 1.82~2.09（主输出 1.82，合计 20.7） |
  | `loss_point_cls`（CORAL BCE ×2） | 1.49~1.68（合计 17.4） | 1.51~1.69（合计 17.5） |
  | `loss_point_coord`（×5） | 1.74~1.78（合计 19.4） | 1.76~1.80（合计 19.5） |
  | `total_loss` | 582.7 | 548.4 |

  - `loss_point_ecount` 11 份合计占 `total_loss` 的 3.6% → 3.8%。开与关前 4 位有效数字相同：这 60 步 lr ≤ 1e-7（warmup 加上被 MAX_ITER 60 压短的 cosine），参数几乎不动，余下差异来自 GPU 非确定性。
- **速度**：关 7.74 s/iter（中位；`Overall training speed` 7.97 s/it），开 8.66 s/iter（8.83 s/it），**多 0.9 s/iter，约 +11%**。
  `data_time` 中位 0.05 s，这时是 GPU 受限（被别人的进程挤着），绝对值不代表正式 run。
- **同进程交替计时**（`scripts/bench_backward.py`，`bench_backward.json`：同一份 bs4 batch、K 的 `model_0011999.pth`、AMP，整体反传与拆分反传交替各 6 次）：
  前向中位 7.67 / 7.74 s；**反传 1.70 s → 2.85 s（×1.68，+1.15 s）**，整步 ×1.13。E 损失那一遍经过 pixel decoder（MSDeformAttn 的反向较贵），所以多出的不止 decoder。
- **峰值显存**：`max_mem` 关、开都是 **22,898 MB**（torch 口径，逐 MB 相同）；reserved 26,520 / 26,864 MB（expandable segments）。
  交替计时里两种反传的峰值也都是 21,543 MB：E 损失那一遍保留计算图没有抬高峰值（峰值在前向末 / 反传开头）。
- 训练末评估（开，未训练的模型）跑通，无报错：没有 score ≥ 0.4 的 wheat_2，`count_MAE_spike` 为 nan，`count_MAE_img` 48.735（= 每图平均 GT 点数，预测计数全为 0）。
  内置评估的读法与 K 相同，已由第 3 条在 K 的 checkpoint 上逐字节核对。

## 预期与判否线（跑之前写死）

### 读数规则（同上一批）

- 一律用默认读数，与 K 相同：内置评估，每穗计数 = clamp(round(E), 0, 25)（`TEST.POINT_CORAL_READOUT: expect`），小穗点输出前 K 个 slot（`POINT_THRESHOLD` 0.5）。
- **主读数**：后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估（排除 nan）；同一 checkpoint 报全部指标。护栏里的峰值用全程。
- 不扫阈值：不改 `POINT_THRESHOLD`，不跑任何 `--sweep`，不用 val 上拟合的校准值。单种子 SEED 42，不加种子。
- 如果中途换卡续跑，同一 iter 的评估以最后一次为准（K 就有一次续跑，没有重复的 iter）。

### 主判据（计数）

| 主读数 `count_MAE_spike` | 判定 |
|:--|:--|
| ≤ **1.28** | 计数成立 |
| ≥ **1.42**（穗长直线） | 判否 |
| 介于两者之间 | 歧义 |

### 护栏（按用户 10-24 的口径放宽到 K 臂水平；任一不过则整体不成立）

来源：用户 2026-10-24 原话「我比较在意的数值是 count_MAE_spike,其实 segmAP 和 F1@8 基本保持就算可以,比如你的 K 臂在我认为算是不错的」。

| 护栏 | 线 | 来历 | K 自己 |
|:--|:--|:--|--:|
| segm AP 全程峰值 | ≥ **76.0** | 放宽到 K 的水平（原线 77.80） | 76.03 |
| 主读数那次 `matched_F1@8` | ≥ **0.600** | 放宽到 K 的水平（原线 0.6078） | 0.6033 |
| `spike_recall` 全程峰值 | ≥ **0.9311** | 用户没提，保持原线 | 0.9315 |

- 主判据过线且三条护栏都过 → **成立**；主判据过线、护栏未全过 → 计数过线、护栏未过（不记成立）；主判据歧义 / 判否 → 不成立。

### 归因（对父臂 K）

两件事各给一组线。K 的数据全部取自 `output_v2/exp-1023-mini-coral-ecount/metrics.json`。
依据的统计在 `_smoke-1024-coral-ecount-bbstop/line_basis.txt`（`scripts/line_basis.py`，只读已有 run 的日志）。

**A1. 计数有没有保住**（主读数 `count_MAE_spike`）

| 主读数 | 记为 |
|:--|:--|
| ≤ **1.2297**（K 1.1897 + 0.04） | 保住 |
| ≥ **1.2592**（K + 0.0695，即丢掉 K 相对其父臂 coral-exist 同口径读数一半以上的收益） | 没保住 |
| 介于两者之间 | 歧义 |

- 容差 0.04 的来历（单种子臂之间的差异）：
  - count-residual 与 K 是两个「计数损失经共享特征回传」的单种子臂，最终评估 r 相同（0.747 / 0.748），主读数差 0.044（1.1459 vs 1.1897）；
  - K 自己后半程 12 次评估（去掉 iter 7499 的 1.3954）标准差 0.024、极差 0.068；
  - K 最终评估 566 穗上 MAE 的 val 抽样标准误 0.043（bootstrap 0.044）：单次评估的读数本身就有 0.04 量级的抽样误差。
- 换个说法：0.04 是 K 相对其父臂 coral-exist 同口径读数（1.3287）的收益 0.139 的 29%，「保住」即保住七成以上的收益。
- 「没保住」用收益的一半（0.0695）：一半以上的收益没了，说明它要靠 E 损失改 backbone。

**A2. 分割代价有没有减小**（后半程 13 次评估 iter 6499、6999 … 11999、12500，与 K 同 iter 的 segm AP 之差的平均）

| 平均差（本臂 − K） | 记为 |
|:--|:--|
| ≥ **+1.00** | 减小 |
| ≤ **+0.35** | 未减小 |
| 介于两者之间 | 歧义 |

- 依据：不经共享特征回传计数损失的 19 个 mini 100 epoch run（含 E0 自己，见 `line_basis.txt` 第 1 节），同一统计量对 E0 的值
  均值 −0.27、标准差 0.25、范围 −0.82 ~ +0.15。两两之差（171 对）的标准差 0.35、最大 0.97，|差| ≥ 0.35 的占 33%，≥ 1.0 的一对也没有。
  - +1.00 超过零效应臂之间的最大差；+0.35 是两两之差的一个标准差。
- 参照：K 的代价是对父臂 −2.84、对 E0 −3.29；L 臂切断全部回传后对其父臂恢复 +1.32。
- 另报末 4 次（10999、11499、11999、12500）的平均差，只记录。

### 机制门

两道门各管一边，分别记录，不合并成一个结论。

**M1. E 损失挡住 backbone 后仍让 E 更准**（计数一侧）

- 最终评估逐穗明细（`inference/spikelet_pair_counts.csv`）上 r(`pred_points`, `gt_points`)，`pred_points` 即 round(E)。
- r ≥ **0.7016**（K 的父臂 coral-exist 最终评估同口径 0.6716 + 0.03，与 K 的机制门同一条线）→ 启动；否则记「未启动：E 损失的作用要经过 backbone」。
- 同时记录与 K（0.7478）的差、E[GT | 预测] 斜率与预测计数标准差（K 1.000 / 1.78）。

**M2. backbone 归还给其余损失后，分割 / 检测的训练拟合回落**（分割一侧）

- 末段（iter 11250~12499，63 行训练日志）五项分割 / 检测损失合计（定义见「现象」）。
- ≤ **16.80**（K 17.922 与 K 的父臂 coral-exist 15.674 的中点）→ 启动；否则记「未启动：挡住 backbone 没有改善训练拟合」。
- 依据：与 K 同结构、不经共享特征回传计数损失的 13 个 mini run，这个量均值 15.65、标准差 0.16、范围 15.43~15.97；
  L 臂（全部切断）15.85，count-residual 16.97，density-count 17.16。中点比 K 低 1.12，约 7 个标准差。
- 不用「训练中 backbone 上的更新份额」当门：开关打开后 backbone 的梯度按构造只来自其余损失，份额必然变化，量它只是在核对实现
  （实现已由第 4、5 条核对）；变化的幅度已由上面的零训练诊断给出（K iter 11999 上 0.259 → 0.451）。

### 结果的记法（只给可能的组合起名，不改上面的线）

| A1 计数 | A2 分割 | 记为 |
|:--|:--|:--|
| 保住 | 减小 | K 的分割代价主要来自 E 损失进 backbone，计数收益不靠 backbone：挡 backbone 是对的方向 |
| 保住 | 未减小 | 计数不靠 backbone，但分割代价也不在 backbone（在 pixel decoder / decoder 的共享） |
| 没保住 | 减小 | 计数收益与分割代价都经 backbone：同 L 臂「同一件事的两面」，位置在 backbone |
| 没保住 | 未减小 | 挡 backbone 对两边都无益 |
| 任一项歧义 | | 记歧义，另一项照常写 |

整体判定（主判据 + 护栏）、归因（A1、A2）、机制门（M1、M2）分别写，互不覆盖。

### 次口径（必报，不进判定）

- 主读数 checkpoint 上 `--eval_only` 加 `MODEL.MaskDINO.TEST.SPIKELET_EVAL_SIGMAS '[8.0,16.0,32.0]'` 重评（σ 8 那一档应与训练日志相同）。
- 最终评估逐穗 CSV：有符号偏差、Pearson r、预测计数标准差、E[GT | 预测] 斜率（= r × sd_GT / sd_pred），
  以及同批穗上的穗长直线 MAE（`_smoke-1022-coral-ecount/scripts/line_on_pairs.py`：离线复现穗配对，按预测框套 round(8.924 + 0.0345 × 对角线)）。
  主读数 checkpoint 上同样报一遍（需要它的 `--eval_only` 逐穗明细，上一条的重评就有）。

### 诊断记录（只记录，不进判定，不用来挑读法或阈值；同 K 文档的「诊断记录」）

- 最终评估同一批穗：K 读数（`kcount`，即输出的点数）与 E（`ecount`，不取整）各自的 MAE、偏差、r、斜率；E − K 的均值；K ≥ 24 的穗的比例。
  K 的值：K 读数 6.300 / +6.26 / 0.659 / 0.428，E 1.222 / +0.06 / 0.753 / 1.029，E − K −6.20，K ≥ 24 占 57%。
- 自检：`out_points` 逐穗等于 `kcount`，`pred_points` 逐穗等于 clamp(round(`ecount`))。

### 观察量（只记录）

- 最终 25 个 CORAL bias：数值、跨度（K 1.10，父臂 1.94，初值 0.96）、相邻不增的对数。
- 训练曲线分段均值（每 1250 iter）：`loss_point_ecount`（11 份合计与主输出）、`loss_point_cls`、`loss_point_coord`、五项分割 / 检测损失、`total_loss`，与 K 同段对比。
- 后半程 13 次与 K、父臂、E0 同 iter 的 `count_MAE_spike`、`matched_F1@8`、segm AP、bbox AP、`spike_recall`、`n_pred_points`。
- 可选（主 agent 定要不要做）：
  - 在主读数 checkpoint 上用 `diag_bbstop.py`（同一份 batch）重做梯度诊断，看假如不挡，E 损失在 backbone 上的梯度会是多少（K iter 11999 是其余损失的 3.0 倍）；
  - 后半程 13 个 checkpoint 的 `--eval_only`，得到逐次的 r（K 文档做过）。

### 实施者的预测（写于启动前，不是判定线）

- **M2**：倾向启动。K 的后半程 backbone 拿全模型更新的 86%、其中七成来自 E 损失，挡住后这部分直接还给其余损失。
- **A2**：倾向「减小」（≥ +1.00），但恢复多半不完全：pixel decoder 上 E 梯度仍与其余损失相当（0.87~1.04 倍），mask_features 从这里出。
- **A1**：没有把握，倾向「歧义」（1.23~1.26）。E 损失最大的一块梯度在 backbone（300 vs pixel decoder 85、decoder 25）；
  L 臂切断全部共享回传后计数退回 1.31，本臂保留 pixel decoder 与 decoder 两条路，应介于两者之间。
- **主判据**：倾向过线（≤ 1.28）。
- **M1**：倾向启动（r 0.72~0.74）。
- **护栏**：segm AP 峰值 ≥ 76.0 倾向通过；`spike_recall` 峰值 ≥ 0.9311 倾向通过（K 0.9315，分割回升应带着 recall 回升）；
  `matched_F1@8` ≥ 0.600 没有把握：K 的 F1 低主要是 K 读数膨胀（刻度尺窄，E = y 时 K 顶到 24~25），本臂不动刻度尺，CORAL 头仍受 E 损失作用。
- 最可能的结果：分割代价明显减小、计数比 K 差一点，落在 A1 的歧义区或「保住」的边上。

## 已知限制

- **只挡 backbone。** E 损失仍塑造 pixel decoder（诊断里是其余损失的 0.87~1.04 倍）与 decoder；分割代价若有一部分在那里，本臂去不掉。
- **全模型裁剪仍计入 E 损失在非 backbone 上的梯度。** 裁剪系数对全部参数是同一个倍数，AdamW 对恒定的统一缩放不敏感，只在倍数逐步变化时有一点影响。
- **每步多一次部分反传**（只经过 decoder / pixel decoder），见「冒烟」与「成本」。
- **只实现了 AMP、单卡。** 开关打开而 AMP 关闭、或多卡时直接报错。
- **点输出仍是 K 读数**，刻度尺被学习率卡住（`coral-bias-lr-ceiling`），F1@8 受 K 膨胀的影响同 K。
- 外层匈牙利匹配仍由共享网络决定，哪些实例受 E 监督同 K。
- 零训练诊断只有一个主读数 checkpoint（另加 iter 5999 与初值）、两份 batch。
- 单种子（SEED 42）；同代码同种子的两次 run 到 iter 19 就会分叉（GPU 反向非确定性），本臂与 K 的差是单次 run 对单次 run。不补种子。

## 成本

- 12500 iter，另加 25 次评估。不增加参数。
- 显存：峰值与 K 相同（冒烟开 / 关都是 22,898 MB；K 正式 run 23,726 MB）。排卡同 K：不设 expandable segments 时按约 34 GB 空闲显存算；
  设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 时冒烟的 reserved 约 27 GB（是否设由主 agent 定，不影响数值）。
- 速度：每步多一次只到 pixel decoder 的反传，反传 ×1.68、整步约 ×1.1~1.15（共卡下 +0.9~1.15 s/iter）。
  - K 正式 run：qiuhao GPU1 中位 2.81 s/iter（共卡），xuyu GPU0 1.63 s/iter。按 ×1.15 估：约 3.2 s/iter（训练约 11 h）或约 1.9 s/iter（约 6.5 h），评估另约 1 h。
  - 若卡还像冒烟时这样被挤（关 7.7 s/iter、开 8.7 s/iter），12500 iter 要 30 h 左右。
