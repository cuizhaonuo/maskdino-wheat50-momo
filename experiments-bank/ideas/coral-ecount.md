# 想法：CORAL 存在性头 + 计数尺度损失（K 臂）

状态：**预注册（跑之前写死），未启动**，等主 agent 审核 ｜ 提出 2026-10-22 ｜ 分支 `mini/coral-ecount`（从 `mini/coral-exist`@`f029805` 开出，100 epoch）｜ 配置 `wheat_joint_single_stage_mini_coral_ecount.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_coral_exist.yaml`）｜ 输出 `output_v2/exp-<启动日>-mini-coral-ecount`（占位，主 agent 定）

## 来源

- 2026-10-22 计数改进三臂之一：L `mini/count-residual-detach`、D `mini/density-count`、K 本臂。
  三臂共用的说明与判定线在 `output_v2/_batch-1022-count/AGENT_BRIEF.md`。
- 父臂 `mini/coral-exist`：[ideas/coral-exist.md](coral-exist.md)、[exp-2026-09-26-mini-coral-exist](../experiments/exp-2026-09-26-mini-coral-exist.md)。
  本臂从它的分支头 `f029805` 开出，代码与配置都以它为基础。
- 完整集 `full/coral-exist`：同一个头，另给 `CoralHead.bias` ×10 学习率。
  见该分支的 `experiments-bank/experiments/exp-2026-10-09-full-coral-exist.md`。只读参照，没有合并任何代码。

## 现象（brief 给出的出发点）

1. 父臂 mini/coral-exist（`output_v2/exp-0926-mini-coral-exist`）：
   - 按本批读数规则，主读数在 iter 9999：`count_MAE_spike` 1.6655、`matched_F1@8` 0.6107。
   - 最终评估的 576 个配对穗：偏差 +1.44、r 0.620、预测计数标准差 2.70（GT 2.40）、E[GT | 预测] 斜率 0.551。读数**过度分散**。
   - CoralHead 的 25 个 bias 被学习率卡住（mini schedule 全程累积 lr 0.613），跨度只有 1.94。
2. 完整集上同一个头加了 bias ×10 学习率：bias 跨度到 7.78，读数反过来被**压缩**（标准差 1.25 vs GT 2.39，斜率 1.36）。
3. 两种情况都没有得到校准的计数尺度。

## 假设（brief 给出，未验证）

- 记 E = Σ_k σ(l_k)（25 个 slot 的存在性概率之和）、y = min(Ni, 25)。
- CORAL 损失对 25 个任务取均值。对一个实例，共享打分 g 上的梯度是 Σ_k (σ(l_k) − y_k) / 25 = (E − y) / 25。
  计数尺度的信号太弱，头学不出校准的尺度。
- 直接加 |E − y|：g 上的梯度变成 sign(E − y) · Σ_k σ'(l_k)，与误差大小无关，强得多。
- 假设成立时：默认读数 K 的尺度被校准（斜率接近 1），计数误差下降。

## 实施前的新发现：父臂的 E 已经校准，失真在读数 K

2026-10-22 验证时测得，不属于 brief 的预注册内容。
做法：本分支代码，父臂 `model_final.pth` 配它自己的 `config.yaml` 做 `--eval_only`，只打开诊断字段（见「修法」与「实施前已完成的验证」第 3 条）。
同一批 576 个配对穗：

| 读法 | MAE | 偏差 | r | 预测 sd（GT 2.40） | E[GT \| 预测] 斜率 |
|:--|--:|--:|--:|--:|--:|
| K = Σ_k 1[σ(l_k) > 0.5]（默认读数） | 2.024 | +1.44 | 0.620 | 2.70 | 0.551 |
| round(E) | **1.332** | −0.22 | 0.672 | 1.59 | **1.016** |
| E（不取整） | 1.359 | −0.21 | 0.683 | 1.55 | 1.059 |

- **每个穗都是 E ≤ K**，平均差 1.65，最大 6.86。
- 用最终的 25 个 bias，把每个穗的 E 反解成共享打分 g，再按 K = #{k : g + b_k > 0} 重算：576 个穗的 K 全部复现。
  所以读数完全由「一个共享打分 g + 一把固定刻度尺 b」决定。
- 这把刻度尺很窄：跨度 1.94，k ≥ 17 一段的间距只有 0.04~0.07。
  - 按它算，E 恰好等于 y 时 K 会多报：y = 10~17 时多 1~2 个；y ≥ 18 时 K 直接跳到 24~25，y = 18~22 时多 3~6 个。
  - 反过来，要 K = y，E 必须明显低于 y。例如 K = 18 要求 E ∈ [16.6, 16.9]。
- 对单个实例，CORAL BCE 在 g 上的梯度 (E − y)/25 恰好在 E = y 处为零。
  父臂的 BCE 本来就在把 E 推向 y，只是推得慢，而它已经推到了（斜率 1.016）。
- **结论**：父臂的计数尺度在 E 上已经校准。过度分散与 +1.44 的多报来自读数 K 与窄刻度尺的组合，不是 g 上的信号太弱。
- **对本臂的含义**：
  - 新损失作用于 E，默认读数 K 不变。
  - 在父臂的最终状态上（用上表这 576 个 val 配对穗估计），新损失对 25 个 bias 的平均梯度同号，只把刻度尺整体平移，不拉开它。
    拉开刻度尺只靠 CORAL BCE：它的下降方向是 k ≤ 15 往上、k ≥ 16 往下。而 bias 仍被学习率卡住。
  - 同一状态下，每个穗上新损失对 g 的梯度大小平均是 CORAL BCE（×2）的 47 倍。
    第 0 步打分 MLP 参数上的梯度范数比是 7.9 倍（见下文）。训练后 |E − y| 变小，BCE 的梯度跟着变小，而新损失的梯度大小与误差无关。
  - 所以预计 K 的失真不会因为新损失而消失，见下文「实施者的预测」。
- 按 brief，这一发现**不改**本臂的设计与判定线。是否仍按原设计启动，交主 agent 决定。
- 这只是父臂已有 checkpoint 上的一个诊断读数：没有选阈值，不进任何判定，也不拿来给任何臂挑读法。

## 修法

所有改动都由一个新键控制，默认关闭。关着时与 `mini/coral-exist` 逐位等价（见「实施前已完成的验证」）。

| 位置 | 改动 |
|:--|:--|
| `maskdino/config.py` | 新增 `MODEL.MaskDINO.POINT_ECOUNT_WEIGHT`，默认 0.0 = 关 |
| `maskdino/modeling/criterion.py` | 新的 loss 类型 `point_ecount`，产出 `loss_point_ecount`。对象是外层匹配上、标签为 wheat_2 且 Ni ≥ 1 的实例。E = Σ_k σ(l_k)，取 25 个 slot，logit 先转 fp32。损失 = Σ \|E − min(Ni, 25)\| ÷ 这类实例数，没有这类实例时为 0 且保持计算图。不做内层匹配，不消耗随机数。`loss_points`（CORAL BCE、前缀内层匹配、坐标 L1）一行不动 |
| `maskdino/maskdino.py` `from_config` | 权重 > 0 时才把 `loss_point_ecount` 放进 weight_dict，并把 `point_ecount` 接在 `points` 之后放进 losses。它随 `_interm` / `_{i}` 展开，共 11 项（主输出、9 份 aux、interm），与 CORAL BCE 用同一组输出 dict、同一份外层匹配。`_dn` 权重会生成，但 dn_losses 里没有它，不会被产生。权重 > 0 而 `POINT_CORAL_EXIST` 关着时直接报错 |
| `maskdino/maskdino.py` `instance_inference` | 权重 > 0 时给每个实例多存诊断字段 `pred_spikelet_ecount`：wheat_2 是 E，其余为 0。计数读法与点输出都不变 |
| `maskdino/evaluation/spikelet_evaluation.py` | 模型输出了这个字段时，`spikelet_pair_counts.csv` 末尾多一列 `ecount`，不进任何指标 |
| 配置 | `wheat_joint_single_stage_mini_coral_ecount.yaml`，`_BASE_` 为父臂 yaml，只加 `POINT_ECOUNT_WEIGHT: 1.0` |

不变的部分：
- CoralHead 的结构与初始化（bias 降序 1.00 → 0.04）；
- bias 的学习率（**没有** ×10）；
- CORAL BCE 及其权重 2.0、前缀内层匹配；
- 坐标头与 `POINT_COORD_WEIGHT` 5.0、外层匹配、DN；
- `TEST.POINT_THRESHOLD` 0.5 与推理读法：K = Σ_k 1[σ(l_k) > 0.5]，输出前 K 个 slot。

## 与父臂的差异

- 解析后的 cfg 与父臂 yaml 逐项比对，**只差一个键**：`POINT_ECOUNT_WEIGHT` 0.0 → 1.0。
- 一个参数都不加，仍是 52,294,290 个。state_dict 的 733 个键与父臂相同。
- 训练损失多 11 项 `loss_point_ecount*`，原有 127 项不变。
- 内置评估的指标与父臂同一口径、同一读法。逐穗明细多一列 `ecount`。

## 权重为什么取 1.0

brief 指定先取 1.0。量级上看不出明显不合适，保留。

- 第 0 步（固定 batch，fp32，已乘权重）：
  - `loss_point_ecount`：每份 1.50~2.78（主输出 1.87），11 份合计 23.5，占总损失 542.2 的 4.3%。
  - 同一 batch 上父臂已有的点损失：`loss_point_cls`（CORAL BCE ×2）每份 1.60~1.75，合计 18.4；`loss_point_coord`（×5）每份 1.75~1.84，合计 19.7。
- 冒烟 40 iter（bs4）上的量级见「实施前已完成的验证」第 4 条。
- 单位是「小穗个数」的 L1。`mini/count-residual` 的 `loss_point_count` 也取权重 1.0，第 0 步每份 1.78、合计占总损失 3.2%，与本臂同一量级。
- 第 0 步的梯度：
  - CoralHead 打分 MLP 上是 CORAL BCE 的 7.9 倍，bias 上 7.4 倍。这正是 brief 想加的计数尺度信号。
  - 共享 decoder 上是其余全部损失的 6.0%（CORAL BCE 是 0.8%）。新损失会经共享 decoder 回传，见「已知限制」。

## 实施前已完成的验证

脚本与产物都在 `output_v2/_smoke-1022-coral-ecount/`：`scripts/`、`equiv/`、`eval-off/`、`eval-diag/`、`smoke40/`。
父臂一侧用 `git archive f029805` 的快照（`parent_f029805/`）运行，没有在父臂的 worktree 里执行任何东西。

**1. 权重为 0 时与父臂逐位等价**

xuyu GPU0，15:00。脚本 `scripts/equiv_test.py` 与 `compare.py`，结果在 `equiv/compare.txt`。
全部改动定稿后（之后只改了注释）15:15 又整套重跑一次，结果与前两次逐位相同。

- 做法同 `mini/count-residual`：
  - 固定 random / numpy / torch 种子，cudnn deterministic，载 ImageNet R-50；
  - 同一份 batch：mini train 前 2 张图经 LSJ mapper，85 个实例，其中 6 个 wheat_2（Ni = 17 | 19、18、17、23、17），wheat 都没有点；
  - 训练态前向。
- **fp32 与 AMP（`torch.amp.autocast("cuda", fp16)`，与 AMPTrainer 相同）下，127 项 loss 都逐位相等。**
  fp32 总和 518.7550124824047。三次独立运行的结果也逐位相同。
- 参数量同为 52,294,290。state_dict 733 个键逐张量哈希相同，weight_dict 的 280 个键与 criterion.losses 也相同。

**2. 权重 1.0**

同一脚本加 `--on`，同一份 batch。

- 参数与 state_dict 不变。
- 原有 127 项 loss 在 fp32 与 AMP 下都与权重 0 时逐位相同（新损失不消耗随机数）。
- 多出 11 项：`loss_point_ecount`、`_0`~`_8`、`_interm`。
- **手算**：
  - 做法：包住 `loss_point_ecount`，记下每次调用的 logit、外层匹配与 targets，用 float64 numpy 独立重算。
  - 11 份与报告值的最大相对差：fp32 3.6e-7，AMP 2.4e-7。每份都计入 6 个 wheat_2、排除 79 个 wheat。
  - 挑一个实例：第 0 张图的 GT 0（Ni = 17）配到 query 18。
    25 个 σ(l_k) 从 0.771 降到 0.563（bias 初值 1.00 → 0.04，再加上 g）。E = 16.8186，|E − 17| = 0.1814。
- **梯度**（可训练张量共 410 个）：

| 只对这部分求梯度 | fp32 非零梯度张量 | AMP 非零梯度张量 | 说明 |
|:--|--:|--:|:--|
| `loss_point_ecount*`（11 项） | 380 | 380 | 打分 MLP 6/6、bias 1/1、decoder 202、pixel decoder 112、backbone 53、predictor 其余 6。另有 28 个不在图里（类别 / mask / 框 / 坐标头等），2 个第 0 步结构性为 0 |
| `loss_point_cls*`（CORAL BCE） | 380 | 380 | 与上一行是同一组张量 |
| 其余全部 | 405 | 405 | |
| 总损失 | 405 | 405 | 5 个第 0 步结构性为 0：框头末层零初始化挡住前两层，加上 level_embed。父臂相同 |

  - 四种情况下，fp32 与 AMP 有梯度的张量集合完全相同。没有踩到「no_grad 下经 autocast 预调输出头」的坑；本改动也没有在 no_grad 里调用任何模块。
  - AMP 下仿 GradScaler，从 scale 65536 往下退避：
    - 65536 与 4096 时 fp16 反向溢出，256 时有限；
    - 只算 CORAL BCE 时 4096 就有限，但它在 65536 也溢出；
    - 这是 GradScaler 头几步正常的退避，不是新损失特有的。
  - 梯度范数（fp32，第 0 步）：

    | 位置 | 新损失 | 对照 | 比值 |
    |:--|--:|:--|--:|
    | 打分 MLP | 162 | CORAL BCE 20.6 | 7.9× |
    | bias | 11.5 | CORAL BCE 1.57 | 7.4× |
    | decoder | 84 | 其余全部 1398 | 6.0% |
    | pixel decoder | 91 | 其余全部 3578 | 2.6% |
    | backbone | 184 | 其余全部 5060 | 3.6% |
- 显存（bs2，fp32 前向加 4 次求梯度）：14.2 GB。

**3. eval-only 核对**

xuyu GPU0，15:01~15:04。脚本 `scripts/run_evals.sh` 与 `compare_evals.py`，结果在 `compare_evals.txt`。

- 父臂 `model_final.pth` 配它自己的 `config.yaml`，本分支代码，走 `train_net.main` 的 `--eval_only` 分支。
  `eval_dump.py` 只把末尾的 `verify_results` 换成按完整精度 dump 结果。
- `eval-off`（权重取默认 0）：
  - **40 项指标与父臂训练日志的末次评估（iter 12500）完整精度逐项相同**。
    例如 `count_MAE_spike` 2.0243055555555554、`matched_F1@8` 0.6215089620675283、segm AP 78.5929841522224。
  - `coco_instances_results.json`、`instances_predictions.pth` 与两份 CSV，都与父臂最终评估逐字节相同。
- `eval-diag`（加 `POINT_ECOUNT_WEIGHT 1.0`，只为打开诊断字段）：
  - 指标与 eval-off 逐项相同；
  - `spikelet_pair_counts.csv` 前 4 列与父臂逐行相同，多一列 `ecount`；
  - 父臂模型上的诊断读数见上文「实施前的新发现」。
- eval-only 不算训练，不进 experiments-bank 的实验记录。

**4. 冒烟**

baiju GPU0，15:04~15:14。脚本 `scripts/smoke_train.py`，输出 `smoke40/`、`smoke40.json`、`smoke40_summary.txt`。

- 正式配置加 CLI 覆盖 `SOLVER.MAX_ITER 40`、`TEST.EVAL_PERIOD 0`，bs4。
- 走 `train_net.setup` / `train_net.Trainer` 的完整训练循环：AMPTrainer、优化器、LR 调度、日志、训练末评估。
  只去掉 PeriodicCheckpointer，不存 `model_final.pth`。这不是一次训练 run。
- 40 iter 与训练末的一次完整 val 评估都跑通，无报错，损失全部有限。
- 量级（每份，已乘权重；前 10 步均值 → 后 10 步均值）：

  | 损失 | 前 10 步 | 后 10 步 |
  |:--|:--|:--|
  | `loss_point_ecount` | 1.83~2.02（主输出 1.86，11 份合计 20.8） | 1.96~2.18（主输出 1.98，合计 22.2） |
  | `loss_point_cls`（CORAL BCE ×2） | 1.49~1.68 | 1.48~1.65 |
  | `loss_point_coord`（×5） | 1.74~1.78 | 1.76~1.80 |

  - `loss_point_ecount` 11 份合计占 `total_loss` 的 3.6% → 4.1%。
  - 这 40 步的 lr 都低于 1e-7（warmup，加上被 MAX_ITER 40 压短的 cosine），参数几乎没动，前后差别来自 batch 不同。
- 峰值显存 `max_mem` **22,935 MB**（torch 口径，与父臂的 23.0 GB 相同）。`max_memory_reserved` 32,508 MB，nvidia-smi 口径约 33 GB。
- 速度 6.68 s/iter（`Overall training speed: 38 iterations in 0:04:13`）。
  当时 baiju GPU0 上另有 3 个别人的进程，利用率 100%，这个数不代表独占速度。父臂同配置、与另一个 run 并发时是 2.48 s/iter。
- 训练末评估（未训练的模型）：
  - 没有 score ≥ 0.4 的配对 wheat_2，`count_MAE_spike` 为 nan，`count_MAE_img` 60.725。
  - 未训练的 CORAL 头 25 个概率都在 0.56~0.77，K 全是 25。
  - `spikelet_pair_counts.csv` 表头带上了 `ecount` 列。有配对穗时诊断列的输出，已由第 3 条的 eval-diag 验过。

## 预期与判否线（跑之前写死）

### 本批统一判定线（L、D、K 三臂相同，照 brief 第 4 节原样抄）

- **主对照 E0**：`output_v2/exp-0923-mini-ep100`（`mini/baseline`@`0f71161`）。
  - E0 读数（后半程 `count_MAE_spike` 最低那次，iter 8999）：count 2.4648、`matched_F1@8` 0.6228、spike_recall 0.9511、segm AP 77.40。
  - E0 全程峰值：segm AP 78.80、spike_recall 0.9511。最终评估：偏差 +2.40、r 0.642。
- **不训练参照线**：穗长直线，点数 = 8.924 + 0.0345 × 预测框对角线（mini train 拟合）。val 上 MAE 约 1.42
  （E0 最终评估的配对穗上 1.4187）。
- **读数**：后半程（iter ≥ 6250）`count_MAE_spike` 最低那次为主读数（排除 nan），同一 checkpoint 报全部指标。默认读数，不扫阈值。
- **主判据（三臂共用）**：主读数 `count_MAE_spike` ≤ **1.28** → 计数成立；≥ **1.42**（穗长直线）→ 判否；介于两者之间 → 歧义。
- **护栏（三臂共用，任一不过则整体不成立；与 `mini/count-residual` 的预注册是同一组线）**：
  - segm AP 全程峰值 ≥ **77.80**；
  - spike_recall 全程峰值 ≥ **0.9311**；
  - 主读数那次 `matched_F1@8` ≥ **0.6078**（E0 − 0.015）。
- **次口径（必报，不进判定）**：
  - 在主读数 checkpoint 上用 `--eval_only` 加 `MODEL.MaskDINO.TEST.SPIKELET_EVAL_SIGMAS '[8.0,16.0,32.0]'` 重评；
  - 最终评估逐穗 CSV 的有符号偏差、Pearson r、预测计数标准差、E[GT | 预测] 斜率（= r × sd_GT / sd_pred），以及同批穗上的穗长直线 MAE。
- **噪声地板**：`matched_F1@8` 0.015。单种子，噪声内的差异记「歧义」。

本臂的默认读数就是父臂的 K（`POINT_THRESHOLD` 0.5，前 K 个 slot），内置评估里的 `count_MAE_spike` 直接读它。

### 父臂在同一组线下的读数（参照，不是判据）

| | iter | `count_MAE_spike` | `matched_F1@8` | spike_recall | segm AP |
|:--|--:|--:|--:|--:|--:|
| 主读数（后半程计数最低） | 9999 | 1.6655 | 0.6107 | 0.9266 | 78.28 |
| 全程峰值 | — | — | — | 0.9478（iter 9499） | 78.71（iter 11499） |

- 父臂按统一主判据是「判否」（1.6655 ≥ 1.42），三条护栏都过。
- 父臂最终 checkpoint 的配对穗上，穗长直线（预测框）是 1.424。来源：`mini/count-residual` 预注册的表，10-07 主 agent 离线算的。

### 归因线（对父臂）

- 主读数 `count_MAE_spike` ≤ **1.50**（父臂主读数 1.6655 × 0.9）→「计数尺度损失有效」；否则记「无效」。

### 机制门（本臂的机制有没有启动）

- 取最终评估的逐穗明细（`inference/spikelet_pair_counts.csv`），按默认读数（`pred_points` 列，即 K）算
  E[GT | 预测] 斜率 = r × sd_GT / sd_pred。
- 斜率落在 **[0.85, 1.15]**（父臂 0.551）→「尺度被校准」，机制启动；否则记「机制未启动」。
- 同时记录预测计数标准差与 r（父臂 2.70 / 0.620）。

### 诊断读数（只记录，不进任何判定，不用来挑读法或阈值）

- 最终评估 `spikelet_pair_counts.csv` 的 `ecount` 列就是 E。在同一批穗上报 round(E) 的 MAE、偏差、r、预测 sd 与斜率。
- 父臂最终 checkpoint 上的同一读数：1.332 / −0.22 / 0.672 / 1.59 / 1.016。
- 用途：区分「新损失没有校准 E」与「E 校准了，但 K 读数没有跟上」。

### 观察量（只记录）

- 最终 25 个 bias：数值、跨度（父臂 1.94，初值 0.96）、相邻不增的对数、相对初值的位移。
- 最终评估：K ≥ 24 的配对穗数；E − K 的均值（父臂 −1.65）。
- `loss_point_ecount` 与 `loss_point_cls` 的训练曲线（分段均值）。
- 后半程 13 轮与父臂同 iter 的 segm AP、`matched_F1@8`、spike_recall 之差。
  新损失经共享 decoder 回传，`mini/count-residual` 在这里付过代价。

### 实施者的预测（写于启动前，不是判定线）

依据上文「实施前的新发现」：

- **机制门**：大概率不过，K 的斜率仍明显低于 0.85。
  - 父臂的 E 已经校准，K 的失真来自窄刻度尺。
  - 新损失只作用于 E，对 bias 只给同号的整体平移，而刻度尺仍被学习率卡住。
- **主判据**：倾向「判否」（≥ 1.42）。**归因线**：倾向「无效」。
  - K 读数的失真不会被新损失修好。
  - 新损失把 E 拉得更贴 y，g 的散度会变大；在窄刻度尺上，K 的散度跟着变大，读数可能更分散。
- **诊断读数** round(E)：应保持在 1.3 左右或略好，因为新损失直接拉 E。
- 若机制门反而通过，说明刻度尺被拉开了，最终 bias 跨度应明显大于父臂的 1.94。那时要回头查本节的推理哪里错了。

## 已知限制

- **bias 仍被学习率卡住。**
  - mini schedule 全程累积 lr 0.613，AdamW 下每个标量 bias 最多走约 0.6。
  - 本臂不改 bias 学习率、不改初始化，只验证这一项损失。
  - 完整集版本 `full/coral-exist` 给了 `CoralHead.bias` ×10 学习率，跨度到 7.78。
- **新损失作用于 E，读数却是 K。**
  K 与 E 的关系由刻度尺决定，刻度尺不拉开，E 校准不等于 K 校准（见「实施前的新发现」）。
- **新损失经共享 decoder 回传。**
  - 第 0 步它在 decoder 上的梯度是其余全部损失的 6%。
  - `mini/count-residual` 的 L1 计数损失与它同一量级，经共享 decoder 回传后 segm AP 平均 −2.05，三条护栏都没过。
  - 本臂不 detach；detach 是 L 臂 `mini/count-residual-detach` 要回答的问题。
- 训练时 E 取配上的 query 的全部 25 个 slot，与读数无关；推理时 K 只看 0.5 这一档阈值。
- 单种子（SEED 42），不补种子。噪声内的差异记「歧义」。

## 成本

- 12500 iter，另加 25 次评估。
- 新损失只是 11 次向量化的 sigmoid 加求和，不增加参数。
- 显存：冒烟 `max_mem` 22,935 MB（torch 口径），reserved 32.5 GB，与父臂（23.0 GB，nvidia-smi 约 32 GiB）相同。排卡按约 34 GB 空闲显存算。
- 速度：冒烟在一张被别人占满的卡上是 6.68 s/iter，不可比。预计与父臂相当：
  父臂全程 9:31，其中训练 8:36（2.48 s/iter，与另一个 run 并发），评估与存档 53 min。
