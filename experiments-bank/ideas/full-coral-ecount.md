# 想法（完整集）：CORAL 存在性头 + E 损失，计数读 round(E) · full/coral-ecount

状态：**预注册（跑之前写死），未启动** ｜ 提出 2026-10-24 ｜ 分支 `full/coral-ecount`（从 `full/coral-exist`@`a061dd1` 开出）｜
代码提交 `409378a` ｜ 配置 `wheat_joint_single_stage_full_coral_ecount.yaml`（`_BASE_` 为 `wheat_joint_single_stage_full_coral_exist.yaml`）｜
输出 `output_v2/exp-<启动日 MMDD>-full-coral-ecount`（占位，主 agent 定）｜ mini 对应臂：`mini/coral-ecount`（K 臂）

## 来源

- **mini K 臂** `mini/coral-ecount`：预注册 `ideas/coral-ecount.md`，结果 `exp-2026-10-23-mini-coral-ecount.md`（都在该分支上）。
  - 主读数 iter 11999 `count_MAE_spike` 1.1897，过了成立线 1.28；归因「有效」（≤ 1.1958，余量 0.006）；机制启动（r 0.748）。
  - 护栏两条没过：segm AP 全程峰值 76.03（E0 78.80，−2.77），读数那次 `matched_F1@8` 0.6033（E0 0.6228，−0.0195）。
- **用户 2026-10-24 对 mini K 的评价**（原话）：「我比较在意的数值是 count_MAE_spike,其实 segmAP 和 F1@8 基本保持就算可以,比如你的 K 臂在我认为算是不错的」。
  本臂在完整集上重做 K 臂，护栏按这句话放宽（见「护栏」）。
- **父臂 A** `full/coral-exist`：[ideas/full-coral-exist.md](full-coral-exist.md)、[exp-2026-10-09-full-coral-exist](../experiments/exp-2026-10-09-full-coral-exist.md)，
  run 在 `output_v2/exp-1009-full-coral-exist`。默认读数 K 下判定成立，后半程计数最低 1.3690（iter 16999）。
- **完整集 E 读数的零训练诊断**：`output_v2/_diag-1022-full-docs/e_readout_diag.py`、`ereadout-exp-1009-full-coral-exist.json`。
  A 的 model_final 上，K 读数 1.407（斜率 1.36，被压缩），round(E) 1.237（斜率 1.00）。
- **代码**：所有 full/* 分支共用 `full/combo` 的合并代码 `3bd2071`。本分支的代码提交 `409378a` 移植了 mini K 的两处实现：
  `47072eb`（E 损失 `POINT_ECOUNT_WEIGHT`）与 `709059d`（读数 `TEST.POINT_CORAL_READOUT expect`）。按完整集代码重写，没有 cherry-pick，没有合并任何分支。
- `full/combo-ecount`（另一个 agent）会从 `409378a` 开出，把 E 损失与读数扩展到 combo 的 group 层。本臂只开 A 的开关。

## 出发点

1. **A 的默认读数被压缩，E 本身已校准。** A 最终评估的 553 个配对穗上（GT 均值 15.98、标准差 2.39）：

   | 读法 | MAE | 偏差 | r | 预测 sd | E[GT \| 预测] 斜率 |
   |:--|--:|--:|--:|--:|--:|
   | K = Σ_k 1[σ(l_k) > 0.5]（A 的默认读数） | 1.407 | +0.46 | 0.713 | 1.25 | 1.36 |
   | round(E)，E = Σ_k σ(l_k) | **1.237** | −0.08 | 0.728 | 1.73 | **1.00** |
   | 穗长直线（同批穗） | 1.407 | −0.03 | 0.610 | 1.43 | 1.02 |

   - 压缩来自 bias 阶梯头部的平台：A 最终的 25 个 bias 里 k = 0~6 都是 2.468，打分 g 偏低的短穗，这 7 个任务的 σ 只有约 0.6，在 E 里各算约 0.6，在 K 里全部算满。
   - 其余的误差来自逐穗的相关性不够高（r 0.73）。
2. **mini K 的结果**：E 损失把父臂在同一读法下的计数从 1.3287 降到 1.1897（−10.5%，各取后半程最低），最终评估的 r 从 0.672 升到 0.748；代价是 segm AP、bbox AP、`matched_F1@8`、spike_recall 都降了。
   mini 的 bias 学习率被卡住、刻度尺窄，E 被推到 y 时 K 涨到 24~25，每穗多出 6.26 个点，F1 的下降主要来自这里。
3. **完整集与 mini 的差别**：完整集 A 的 CoralHead 根级 bias 有 ×10 学习率（`train_net.py` 的参数组，保持不动），刻度尺跨度 7.78（mini 父臂 1.94）。
   数据 2092 图 vs 500 图；schedule 26000 iter / 50 epoch vs 12500 iter / 100 epoch。

## 假设

- E 损失 |E − y| 直接压每穗的期望计数误差，梯度大小与误差无关；CORAL BCE 在共享打分 g 上的梯度只有 (E − y)/25。
- A 的 E 在均值上已经校准（斜率 1.00），逐穗还不够准（r 0.73）。若特征里有更多计数信息，E 损失能提高 r，round(E) 的 MAE 随之下降。
- 读数取 round(E)，与新损失的对象是同一个量，不经过 K 的阈值穿越。小穗点仍按 K 输出（做法同 mini K 的第二次修订）。

## 修法（代码提交 `409378a`）

所有改动由两个新键控制，默认值就是现有行为（与 `3bd2071` 逐位等价，见「实施前已完成的验证」）。

| 位置 | 改动 |
|:--|:--|
| `maskdino/config.py` | 新增 `MODEL.MaskDINO.POINT_ECOUNT_WEIGHT`（默认 0.0 = 关）与 `MODEL.MaskDINO.TEST.POINT_CORAL_READOUT`（默认 `threshold`，另一档 `expect`） |
| `maskdino/modeling/criterion.py` | 新 loss 类型 `point_ecount`，产出 `loss_point_ecount`。只作用于 slot 语义的输出（`pred_point_abs` 为假；A 的主输出、9 份 aux、interm 共 11 份）。对象是外层匹配上、标签为 wheat_2 且 Ni ≥ 1 的实例：E = Σ_k σ(l_k)（25 个 slot，logit 先转 fp32），损失 = Σ \|E − min(Ni, 25)\| ÷ 这类实例数；没有这类实例时为 0 且保持计算图。不做内层匹配，不消耗随机数。group 层输出（`pred_point_abs` 为真）直接跳过、不产生这个键。`loss_points`（CORAL BCE、前缀内层匹配、坐标 L1）一行不动 |
| `maskdino/maskdino.py` `from_config` | 权重 > 0 时才把 `loss_point_ecount` 放进 weight_dict（在 `_interm` / `_dn` / `_{i}` 展开之前），并把 `point_ecount` 接在 `points` 之后放进 losses。`_dn` 权重会生成，但 dn_losses 里没有它，不会被产生。守卫：两个新开关都需要 `POINT_CORAL_EXIST`（否则 ValueError）；与 `POINT_GROUP_DECODER` 同开时 NotImplementedError（combo 的 group 层留给 `full/combo-ecount`） |
| `maskdino/maskdino.py` `instance_inference` | CE 前缀路径（A 的路径）：小穗点两种读法下都输出前 K 个 slot。`expect` 下每穗计数 `pred_spikelet_counts` = clamp(round(E), 0, 25)，E 在 fp32 下只算一次，计数与诊断字段共用。诊断字段：`pred_spikelet_ecount`（E，wheat_2 之外为 0；权重 > 0 或 `expect` 时）、`pred_spikelet_kcount`（K；`expect` 时） |
| `maskdino/evaluation/spikelet_evaluation.py` | 计数指标（`count_MAE_spike`、`count_MAE_img` / RMSE、两份 CSV 的计数列）读 `pred_spikelet_counts`，没有时数点；现有各路径里它都等于输出的点数。点级 P/R/F1、`n_pred_points` 仍按实际输出的点算。有诊断字段时 `spikelet_pair_counts.csv` 末尾多 `ecount`，`expect` 下再多 `kcount`、`out_points`（实际输出的点数），都不进任何指标 |
| 配置 | `wheat_joint_single_stage_full_coral_ecount.yaml`，`_BASE_` 为 A 的 yaml，只加 `POINT_ECOUNT_WEIGHT: 1.0` 与 `TEST.POINT_CORAL_READOUT: expect` |

不变的部分：CoralHead 的结构与初始化（bias 降序 1.00 → 0.04）、bias 的 ×10 学习率、CORAL BCE 及其权重 2.0、前缀内层匹配、坐标头与 `POINT_COORD_WEIGHT` 5.0、
外层匹配、DN、`TEST.POINT_THRESHOLD` 0.5（`expect` 下它只决定输出哪些点，不参与计数）、schedule（26000 iter、bs4、lr 1e-4 cosine、EVAL_PERIOD 1000）、`SEED` 42。

## 与父臂 A 的差异

- 解析后的 cfg 与 A 的 yaml 逐键比对（340 个键），**只差两个键**：`POINT_ECOUNT_WEIGHT` 0.0 → 1.0，`TEST.POINT_CORAL_READOUT` threshold → expect。
  与 A run 落盘的 `config.yaml` 比，另外只差 `OUTPUT_DIR` 和一个 tuple / list 的序列化写法（值相同）。
- 一个参数都不加，仍是 52,294,290 个；state_dict 的 733 个键与 A 相同。
- 训练损失多 11 项 `loss_point_ecount*`，原有 127 项不变。
- 内置评估的计数换了读法：`count_MAE_spike`、`count_MAE_img` 读 round(E)。点级指标（`matched_F1@σ` 等）、segm / bbox AP、spike_recall、spike_precision 与 A 同一口径。

## 权重为什么取 1.0

沿用 mini K 的 1.0，不改。理由：单位都是「小穗个数」的 L1；第 0 步的量级和梯度比例与 mini K 几乎相同；改权重会让完整集与 mini 的对应关系多一个变量，
而用户认可的正是 mini K（权重 1.0）那样的取舍。

- 第 0 步（固定 batch，fp32，已乘权重；见「实施前已完成的验证」第 2 条）：
  - `loss_point_ecount` 每份 1.70~2.99（主输出 2.00），11 份合计 25.08，占总损失 583.8 的 **4.3%**（mini K 4.3%）；
  - 同一 batch 上 A 已有的 `loss_point_cls`（CORAL BCE ×2）每份 1.69~1.87、合计 19.57，`loss_point_coord`（×5）合计 20.85。
- 第 0 步的梯度范数比（fp32）：打分 MLP 上是 CORAL BCE 的 6.8 倍（mini 7.9），bias 上 6.65 倍（mini 7.4）；
  共享 decoder 上是其余全部损失的 5.1%（mini 6.0%），pixel decoder 2.7%，backbone 3.3%。
- mini K 训练中 E 损失占总损失 13%~20%，经共享 decoder 回传，是 segm AP 下降的主要嫌疑（未单独验证）。本臂不 detach，与 mini K 一致。

## 实施前已完成的验证

脚本与产物都在 `output_v2/_smoke-1024-full-coral-ecount/`（`scripts/` 下是全部脚本）。
父代码一侧用 `git archive a061dd1` 的快照（`parent_a061dd1/`，代码与 `3bd2071` 相同），没有在别的 worktree 里执行任何东西。
第 1~6 条在最终代码上做（工作区改动与之后提交的 `409378a` 逐字相同），第 7 条在 `409378a` 上做。环境 `MaskDINO-310`（py3.10 / torch 2.5.1）。

**1. 开关全关时与父代码逐位等价（A 与 combo 两个配置）**

xuyu GPU0，10-24 15:21~15:25。`scripts/run_equiv.sh`、`equiv_test.py`、`compare.py`，结果在 `equiv/compare.txt`。

- 固定 random / numpy / torch 种子，cudnn deterministic，载 ImageNet R-50。同一份 batch：完整集 train 第 6、7 张图经 LSJ mapper，64 个实例，其中 3 个 wheat_2（Ni = 17 | 20、19）。训练态前向。
- **A 配置**：fp32 与 AMP（`torch.amp.autocast("cuda", fp16)`，同 AMPTrainer）下 127 项 loss 逐位相等（fp32 总和 558.7459208071232）。
  参数量 52,294,290，state_dict 733 个张量逐个哈希相同，weight_dict 280 个键与 criterion.losses 相同。
- **combo 配置**（`wheat_joint_single_stage_full_combo.yaml`）：138 项 loss 在 fp32 与 AMP 下逐位相等（fp32 总和 592.8462981283665）。
  参数量 65,454,171，state_dict 1119 个张量逐个哈希相同，weight_dict 360 个键相同。

**2. E 损失打开（A 配置加 `POINT_ECOUNT_WEIGHT 1.0`，同一份 batch）**

- 参数与 state_dict 不变；原有 127 项 loss 在 fp32 与 AMP 下都与权重 0 时逐位相同（新损失不消耗随机数）；多出 11 项：`loss_point_ecount`、`_0`~`_8`、`_interm`。
- **手算**：包住 `loss_point_ecount`，记下每次调用的 logit、外层匹配与 targets，用 float64 numpy 独立重算。
  - 11 份与报告值的最大相对差：fp32 3.1e-7，AMP 5.0e-7。每份都计入 3 个 wheat_2、排除 61 个其他实例（wheat）。
  - 挑一个实例：第 0 张图的 GT 0（Ni = 17）配到 query 18。25 个 logit 从 1.2149 降到 0.2549（bias 初值 1.00 → 0.04 加上 g = 0.2149），
    σ 从 0.771 降到 0.563。E = 16.8186，|E − 17| = 0.1814。
- **梯度**（可训练张量共 410 个）：

  | 只对这部分求梯度 | fp32 非零梯度张量 | AMP 非零梯度张量 | 说明 |
  |:--|--:|--:|:--|
  | `loss_point_ecount*`（11 项） | 380 | 380 | 打分 MLP 6/6、bias 1/1、decoder 202、pixel decoder 112、backbone 53、predictor 其余 6；28 个不在图里（类别 / mask / 框 / 坐标头等），2 个第 0 步结构性为 0 |
  | `loss_point_cls*`（CORAL BCE） | 380 | 380 | 与上一行是同一组张量 |
  | 其余全部 | 405 | 405 | |
  | 总损失 | 405 | 405 | 5 个第 0 步结构性为 0（框头末层零初始化挡住前两层，加上 level_embed），A 相同 |

  - 四种情况下 fp32 与 AMP 有梯度的张量集合完全相同。没有踩到「no_grad 下经 autocast 预调输出头」的坑：本改动没有在 no_grad 里调用任何模块。
  - AMP 下仿 GradScaler 从 65536 往下退避：只算 E 损失时 65536 与 4096 溢出、256 有限；只算 CORAL BCE 时 4096 有限；其余全部要到 16；总损失 256 有限。这是 GradScaler 头几步正常的退避。
  - 显存（bs2，fp32 前向加 4 次求梯度）：14.7 GB。

**3. `--eval_only` 与训练日志逐项相同（默认读法、E 权重 0）**

10-24 15:23~15:28。`scripts/run_evals.sh`、`eval_dump.py`（走 `train_net.main` 的 `--eval_only` 分支，只把末尾的 `verify_results` 换成按完整精度 dump）、`compare_evals.py`，结果在 `compare_evals.txt`。

- A 的 `model_final.pth` 配它自己的 `config.yaml`（xuyu GPU0）：**40 项指标与 A 训练日志末次评估（iter 26000）完整精度逐项相同**，
  `coco_instances_results.json`、`instances_predictions.pth`、两份 CSV 与 A 的最终评估逐字节相同。
- combo 的 `model_final.pth` 配它自己的 `config.yaml`（qiuhao GPU1）：同样 40 项逐项相同、4 个产物逐字节相同。

**4. expect 读法复现零训练诊断**

A 的 `model_final.pth` 配它自己的 `config.yaml`，加 `MODEL.MaskDINO.TEST.POINT_CORAL_READOUT expect`（`eexp/A_expect_26000`）：

- `count_MAE_spike` **1.2368896925858952**，与 ereadout 诊断的 round(E) MAE 完全相同；
- 553 个配对穗逐穗对齐：K 相同、E 相同（CSV 4 位小数，最大差 4.99e-5）、计数 = rint(诊断 E)；
  配对（文件名、GT 点数、IoU）与 A 最终评估逐行相同，`kcount` = `out_points` = A 最终评估的点数，计数 = clamp(round(`ecount`))；
- 计数以外的 37 项指标与 A 训练日志末次评估逐项相同，`coco_instances_results.json`、`instances_predictions.pth` 逐字节相同（点没变）。

**5. A 的同口径轨迹**：见下文「A 同口径读数」。13 次评估里计数以外的 37 项指标都与 A 训练日志同 iter 的值逐项相同（checkpoint 与 iter 对得上，评估可复现）。

**6. 守卫**（CPU 上 `MaskDINO.from_config`，`guards_check.txt`）：本臂 yaml 下 losses 多 `point_ecount`、weight_dict 多 40 个 ecount 键；
combo + E 损失、combo + expect 报 NotImplementedError；没有 CORAL 头时两个开关报 ValueError；readout 写错报 ValueError。

**7. 冒烟**

xuyu GPU0，10-24 19:48~20:02。`scripts/run_smoke.sh`、`smoke_train.py`、`summarize_smoke.py`，输出 `smoke100/`、`smoke100.json`、`smoke100_summary.txt`、`smoke100.launch.log`。

- 本臂正式配置加 CLI 覆盖 `SOLVER.MAX_ITER 100`，bs4，完整集 train。走 `train_net.setup` / `train_net.Trainer` 的完整训练循环
  （AMPTrainer、优化器、LR 调度、日志）与训练末的一次完整 val 评估，只去掉 PeriodicCheckpointer，不存 checkpoint。这不是一次训练 run。
- 100 iter 与 val 评估都跑通，无报错，损失全部有限。
- 量级（每份，已乘权重；前 10 步均值 → 后 10 步均值）：

  | 损失 | 前 10 步 | 后 10 步 |
  |:--|:--|:--|
  | `loss_point_ecount` | 2.09~2.23（主输出 2.18，11 份合计 23.87） | 2.26~2.60（主输出 2.41，合计 26.76） |
  | `loss_point_cls`（CORAL BCE ×2） | 1.56~1.78（合计 18.30） | 1.58~1.81（合计 18.48） |
  | `loss_point_coord`（×5） | 1.78~1.81（合计 19.69） | 1.89~1.92（合计 20.92） |

  - `loss_point_ecount` 11 份合计占 `total_loss` 的 3.5% → 4.4%（`total_loss` 657.6 → 609.7）。
  - 这 100 步的 lr 都不超过 1e-7（warmup 1000 iter，cosine 被 MAX_ITER 100 压短），参数几乎没动，前后差别来自 batch 不同。
- 速度 4.50 s/iter（`Overall training speed: 98 iterations in 0:07:20`），训练末评估 200 图用了 6:11。
  当时 xuyu GPU0 上有另一个会话的正式 run 和别人的进程，利用率 100%，这两个数不代表独占速度（A 是 1.592 s/iter、每次评估约 2 min）。
- 峰值显存 `max_mem` **22,760 MB**（torch 口径；A 全程 23,790 MB）；`max_memory_reserved` 39,068 MB（含训练末评估）。
- 训练末评估（未训练的模型）：没有 score ≥ 0.4 的 wheat_2，`count_MAE_spike` 为 nan，`count_MAE_img` 48.735。
  `spikelet_pair_counts.csv` 的表头带上了 `ecount`、`kcount`、`out_points`。有配对穗时 expect 读法的输出由第 4、5 条在 A 的 13 个 checkpoint 上验过。

## 预期与判否线（跑之前写死）

### 读数

- **主读数**：后半程（iter ≥ 13000，共 13 次评估：13999 … 24999 每 1000 次，加 26000）`count_MAE_spike` 最低的那次（排除 nan），同一 checkpoint 报全部指标。
- 默认读数：每穗计数 round(E) = clamp(round(Σ_k σ(l_k)), 0, 25)；小穗点是前 K 个 slot（K = Σ_k 1[σ(l_k) > 0.5]，与 A 相同），`matched_F1@σ` 等点级指标按它算。
- 不扫阈值（不改 `POINT_THRESHOLD`，不跑任何 `--sweep`），不引入预注册之外的读法；单种子（`SEED` 42），不补种子。
- 噪声地板：`matched_F1@8` 0.015。单种子，噪声内的差异记「歧义」。

### 参照（写死；都从各 run 的 `metrics.json` 复核过）

| | 主读数（后半程 count 最低） | 全程峰值 segm AP | 后半程峰值 spike_recall | 最终评估（iter 26000） |
|:--|:--|--:|--:|:--|
| p30 `exp-0922-single-stage-p30` | iter 17999：count 2.4645、`matched_F1@8` 0.6302、segm AP 80.64 | 82.01（iter 24999） | 0.9560（iter 16999） | count 2.7679、F1@8 0.6477、segm 81.97 |
| A `exp-1009-full-coral-exist`（默认读数 K） | iter 16999：count 1.3690、F1@8 0.6228、segm AP 79.46 | 82.09（iter 26000） | 0.9347（iter 20999） | count 1.4069、F1@8 0.6340、segm 82.09 |
| A 同口径（round(E)，见下表） | iter 16999：count **1.2282**（561 穗）、F1@8 0.6228、segm AP 79.46 | — | — | count 1.2369、r 0.7276、斜率 1.004 |

- **完整集穗长直线**（不训练）：点数 = round(9.452 + 0.0329 × 预测框对角线)。系数是全量 train 5394 个带点 wheat_2 上用 GT 框对角线 polyfit 的结果
  （完整精度 9.45195 / 0.0328843，与 A 文档 `_diag-1022-full-docs/gen_tables.py` 同一做法）。
  A 最终评估的 553 个配对穗上 **1.4069**；A 后半程 13 次评估各自的配对穗上 1.380~1.440。

### A 同口径读数（同一读法，后半程 13 个 checkpoint；参照，不是 A 的结果）

- 计数列：A 已有的 checkpoint 配它自己的 `config.yaml`，加 `TEST.POINT_CORAL_READOUT expect` 做 `--eval_only`（10-24，启动前；`eexp/`，汇总在 `traj_A_expect.json`）。
  这是同一读法下的父臂轨迹，不是扫阈值：round(E) 没有可调参数。做法同 mini K 预注册的「父臂同口径读数」。
- 点级列：点与 A 相同，所以就是 A 训练日志同 iter 的 threshold 值（13 次都核对过逐项相同）。
- 按用户 10-22 的决定，round(E) 在本臂之外只作诊断记录：下表只用来定本臂的线，不是 A 的读数，A 的结论仍按它自己的默认读数。

| iter | 配对穗 | `count_MAE_spike`（round(E)） | 偏差 | r | sd（GT 2.39） | E[GT \| 预测] 斜率 | `count_MAE_img` | 同批穗直线 | `matched_F1@8`（点按 K） | segm AP | spike_recall | A 默认读数（K）的计数 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 13999 | 527 | 1.3169 | −0.30 | 0.684 | 1.75 | 0.915 | 12.90 | 1.3795 | 0.6135 | 78.96 | 0.8597 | 1.4175 |
| 14999 | 549 | 1.3206 | −0.29 | 0.673 | 1.58 | 1.016 | 13.79 | 1.3989 | 0.6127 | 80.61 | 0.8956 | 1.4936 |
| 15999 | 539 | 1.3655 | −0.50 | 0.671 | 1.58 | 0.994 | 12.63 | 1.3952 | 0.6148 | 80.77 | 0.8793 | 1.4249 |
| **16999** | 561 | **1.2282** | −0.22 | 0.722 | 1.71 | 1.001 | 15.13 | 1.3957 | 0.6228 | 79.46 | 0.9152 | 1.3690 |
| 17999 | 555 | 1.2577 | −0.21 | 0.713 | 1.73 | 0.984 | 14.02 | 1.4072 | 0.6262 | 80.57 | 0.9054 | 1.3982 |
| 18999 | 564 | 1.2819 | +0.14 | 0.713 | 1.68 | 1.010 | 15.20 | 1.4043 | 0.6246 | 79.59 | 0.9201 | 1.4433 |
| 19999 | 565 | 1.2602 | +0.00 | 0.718 | 1.70 | 1.004 | 15.14 | 1.4071 | 0.6309 | 81.17 | 0.9217 | 1.4195 |
| 20999 | 573 | 1.2600 | +0.08 | 0.721 | 1.69 | 1.013 | 16.23 | 1.4258 | 0.6311 | 81.71 | 0.9347 | 1.4433 |
| 21999 | 549 | 1.2295 | −0.15 | 0.729 | 1.72 | 1.010 | 12.53 | 1.4080 | 0.6331 | 81.71 | 0.8956 | 1.3953 |
| 22999 | 559 | 1.2326 | −0.07 | 0.728 | 1.72 | 1.015 | 14.01 | 1.4401 | 0.6334 | 82.06 | 0.9119 | 1.4043 |
| 23999 | 551 | 1.2523 | −0.16 | 0.724 | 1.72 | 1.010 | 13.29 | 1.4174 | 0.6310 | 81.92 | 0.8989 | 1.3938 |
| 24999 | 555 | 1.2306 | −0.03 | 0.732 | 1.74 | 1.008 | 14.05 | 1.4090 | 0.6348 | 81.94 | 0.9054 | 1.4108 |
| 26000（final） | 553 | 1.2369 | −0.08 | 0.728 | 1.73 | 1.004 | 13.78 | 1.4069 | 0.6340 | 82.09 | 0.9021 | 1.4069 |

- **A 同口径读数**：后半程最低在 iter 16999，**1.2282**（561 穗），r 0.722。
- round(E) 下 13 次的斜率都在 0.92~1.02，没有压缩；偏差 −0.50~+0.14。13 次都比同批穗的直线低 0.03~0.21，也都比 A 自己的 K 读数低 0.06~0.18。
- 末 9 次（iter ≥ 17999）均值 1.249、标准差 0.018，范围 1.2295~1.2819。

### 主判据（计数）

| 主读数 `count_MAE_spike` | 判定 |
|:--|:--|
| ≤ **1.18** | 计数**成立** |
| 介于 1.18 与 1.27 之间 | **歧义** |
| ≥ **1.27** | **判否** |

理由：

- mini K 的线是「成立 1.28 ≈ 直线 1.42 × 0.9，判否 = 直线」。完整集上照搬是：成立 ≤ 1.27（直线 1.4069 × 0.9 = 1.266），判否 ≥ 1.41。
  但 A 不加 E 损失、只换读法，后半程 13 次里已有 9 次低于 1.27（最低 1.2282）。照搬的成立线只靠读法就能过，回答不了「E 损失有没有用」，所以不照搬。
- **成立线 1.18** = A 同口径最低 1.2282 − 0.05。0.05 约是 A 末 9 次同口径评估标准差（0.018）的 3 倍。
  它相当于 A 同口径 × 0.96，与 mini K 的成立线相对其父臂同口径的位置相同（1.28 / 1.3287 = 0.963）。
  过了这条线，计数比 A 的默认读数（1.3690）低约 14%，比同批穗的直线（约 1.40）低约 16%。
- **判否线 1.27** = 直线 × 0.9，即 mini K 的成立线规则；也约等于 A 同口径最低 + 0.04（A 末 9 次标准差的 2 倍多）。
  本臂若连它都过不了，等于 E 损失让 round(E) 的计数比 A 自己同一读法下还差，记判否。
- 介于两者之间，与 A 同口径的差别在噪声量级内，记歧义。

### 护栏（任一不过则整体不成立）

**用户 2026-10-24 原话**：「我比较在意的数值是 count_MAE_spike,其实 segmAP 和 F1@8 基本保持就算可以,比如你的 K 臂在我认为算是不错的」

mini K 相对它的主对照 E0：segm AP 全程峰值 −2.77（76.03 vs 78.80），读数那次 `matched_F1@8` −0.0195（0.6033 vs 0.6228）。
完整集的线要让「mini K 那样的降幅」能过，参照换成 p30：

| 护栏 | 线 | 理由 |
|:--|:--|:--|
| segm AP 全程峰值 | ≥ **79.01**（p30 全程峰值 82.01 − 3.0） | mini K 的降幅 −2.77 能过，余量 0.23。A 的全程峰值 82.09 |
| 主读数那次 `matched_F1@8` | ≥ **0.6102**（p30 读数那次 0.6302 − 0.020） | mini K 的降幅 −0.0195 能过（余量只有 0.0005）；0.020 大于噪声地板 0.015，噪声内的下降一定能过。A 主读数那次 0.6228 |
| `spike_recall` 后半程峰值 | ≥ **0.9147**（A 后半程峰值 0.9347 − 0.02） | 见下 |

**spike_recall 为什么以 A 为参照**：

- 用户这句话没提 spike_recall，按以前的写法保留：「峰值 ≥ 参照峰值 − 0.02」（mini 批次与 `group-decoder`、`group-nms-match` 预注册都是这个写法）。
  以前的完整集预注册（A、kmo、group-nms-match、combo 等）没有 spike_recall 护栏，用的是 `spike_precision` 末值 ≥ 0.753，本臂把它降为观察量（见下）。
- 若按 p30 取参照，线是 0.9560 − 0.02 = 0.9360，**父臂 A 自己就过不了**（后半程峰值 0.9347）。
  p30 的 0.9560 是单次高点，它后半程 13 次在 0.830~0.956 之间摆动。父臂都过不了的线，判不出 E 损失有没有伤到检测，所以改以 A 为参照。
- mini K 的 spike_recall 峰值比它的父臂低 0.016（0.9315 vs 0.9478），按这条线能过。
- 后半程取峰值：完整集前半程有检出近零的退化评估。p30 与 A 的后半程峰值都等于全程峰值。

### 归因（对 A 同口径）

- 本臂主读数 `count_MAE_spike` ≤ **1.1054**（A 同口径读数 1.2282 × 0.9）→「E 损失在 E 读法上有效」；否则记「无效」。规则同 mini K（父臂同口径 × 0.9）。
- 同时报（不进判定）：后半程 13 次与 A 同 iter、同读法的逐次比较——本臂更低的次数、平均差、末 4 轮平均差。
  这一项不依赖「两边各取最低一次」，看的是整条后半程轨迹是否一致地低于 A。

### 机制门（E 损失有没有让 E 更准）

- 取最终评估的逐穗明细（`inference/spikelet_pair_counts.csv`），`pred_points` 列即 round(E)。
- r(round(E), GT) ≥ **0.7576**（A 同口径最终评估 553 穗上 0.7276，+0.03）→「E 损失让 E 更准」，机制启动；否则记「机制未启动」。
- 同时记录 E[GT | 预测] 斜率与预测计数标准差（A 同口径最终评估：1.004 / 1.73）。

### 次口径（必报，不进判定）

- 在主读数 checkpoint 上 `--eval_only` 加 `MODEL.MaskDINO.TEST.SPIKELET_EVAL_SIGMAS '[8.0,16.0,32.0]'` 重评，报 P / R / F1 / `matched_F1` 三档。
- 最终评估逐穗 CSV（round(E)）：有符号偏差、Pearson r、预测计数标准差、E[GT | 预测] 斜率（= r × sd_GT / sd_pred），以及同批穗上的穗长直线 MAE（同上文的直线与取整）。
  主读数 checkpoint 上同样报一遍（事后 `--eval_only` 取逐穗明细）。

### K / E 诊断（只记录，不进任何判定，不用来挑读法或阈值）

- 最终评估同一批穗上 K（`kcount`，即输出的点数）的 MAE、偏差、r、斜率（A：1.407 / +0.46 / 0.713 / 1.36），E（`ecount`，不取整）的同样统计（A：1.260 / −0.08 / 0.738 / 1.03）。
- **K 每穗多报多少**：K − GT 的均值（A +0.46）、E − K 的均值（A −0.54）、E ≤ K 的比例（A 82%）、K ≥ 24 的比例（A 0%，最大 K 21）。
- **bias 跨度**：最终 25 个 bias 的数值、跨度（A 7.784，初值 0.96）、头部平台（A k = 0~6 都是 2.468）、相邻不增的对数（A 24/24）、相对 A 的位移。
  A 的最终 bias：`2.468 ×7, 2.459, 2.451, 2.417, 2.354, 2.185, 1.880, 1.357, 0.754, −0.057, −0.808, −1.636, −2.363, −3.080, −3.691, −4.278, −4.738, −5.118, −5.317`。
- `out_points` 逐穗等于 `kcount`，`pred_points` 逐穗等于 clamp(round(`ecount`))（自检）。
- 按用户 10-22 的决定，这些读法只出现在诊断文件与实验文档里。

### 观察量（只记录）

- 后半程 13 次与 A 同 iter（同读法）的差：`count_MAE_spike`、r、`matched_F1@8`、segm AP、bbox AP、spike_recall、`n_pred_points`；也与 p30 同 iter 比。
- `spike_precision` 末值（以前完整集预注册的护栏线是 0.753 = p30 − 0.01；A 0.7628），bbox AP 全程峰值（A 81.42，p30 80.99）。
- `loss_point_ecount` 与 `loss_point_cls` 的训练曲线（分段均值），E 损失占总损失的比例（mini K 13%~20%）。

### 实施者的预测（写于启动前，不是判定线）

- **机制门**：倾向通过（r ≥ 0.7576）。mini K 最终评估的 r 比父臂同口径高 0.076；A 的 r 已到 0.73，提升空间可能小一些。
- **主判据**：倾向「成立」，也可能「歧义」。
  按 A 自己的轨迹，r 每升 0.01，同口径 MAE 约降 0.02（r 0.671 → 0.732 时 1.3655 → 1.2306）。r 到 0.76 时约 1.15~1.17，刚过成立线。
- **归因线**（≤ 1.1054）：倾向「无效」。要比 A 同口径好 10%，按上面的关系 r 大约要到 0.78~0.79。
- **护栏**：
  - segm AP 峰值 ≥ 79.01：倾向通过但余量不大。照搬 mini K 相对父臂的峰值降幅（−2.68），约 79.4。
  - `matched_F1@8` ≥ 0.6102：说不准。mini K 读数那次比父臂同 iter 低 0.020，照搬约 0.60~0.615。
    但完整集的刻度尺宽（×10 学习率，跨度 7.78），E 被推到 y 时 K 不必顶到 24~25：mini K 每穗多报 6.26 个点来自窄刻度尺，完整集的点膨胀应当小得多，F1 的降幅可能比 mini 小。
  - spike_recall 后半程峰值 ≥ 0.9147：倾向通过。
- **K / E**：K 的偏差从 A 的 +0.46 小幅上升；E 损失对 25 个 bias 的梯度同号，刻度尺跨度可能比 A 窄（mini K 1.10 vs 父臂 1.94）。
- 最可能的结果：计数「成立」或「歧义」、机制启动，整体成不成立看 `matched_F1@8` 与 segm AP 两条护栏。

## 已知限制

- **计数与点出自两个读数**：计数是 round(E)，点是前 K 个 slot。A 上平均 E − K = −0.54，输出的点比计数多。
  `count_MAE_spike`、`count_MAE_img` 读计数；点级 P/R/F1、`n_pred_points` 读点。E 损失会改变共享打分 g 与 bias，K 跟着变，所以 F1 的变化里也有 E 损失经 K 带来的间接影响。
- **新损失经共享 decoder 回传**。第 0 步它在 decoder 上的梯度是其余全部损失的 5.1%；mini K 的 segm AP 后半程平均比父臂低 2.84。本臂不 detach。
- **A 同口径读数来自 A 已有的 checkpoint**：同数据、同 schedule、同种子，但不是同一次训练；同种子的两次训练也会因 GPU 非确定性分叉。
- 主判据与归因比的是两边各自 13 次评估里的最低值，两边都有挑最低带来的偏低。逐次比较作为补充一并报告。
- 单种子（`SEED` 42），不补种子。噪声内的差异记「歧义」。
- P = 25：train 有 6 个实例 Ni > 25，E 损失的目标截到 25，与 CORAL levels 同口径（val 最大 22，不受影响）。

## 成本

- 26000 iter（bs4，约 50 epoch），另加 26 次评估（EVAL_PERIOD 1000）。新损失只是 11 次向量化的 sigmoid 求和，不增加参数。
- 参照 A：全程 12:35:05，其中纯训练 11:29:38（1.592 s/iter，xuyu A100），评估 26 次共 54 min（均 2:05）；峰值显存 `max_mem` 23,790 MB。
- 冒烟：4.50 s/iter、评估 6:11（卡被别人占满，不可比）；`max_mem` 22,760 MB，`max_memory_reserved` 39.1 GB。排卡按约 40 GB 空闲显存算。
- 预计与 A 相当：约 12~13 h（独占 A100）。卡上有别人的任务时会慢，冒烟时就是这种情况。

## 启动命令（正式 run 由主 agent 排卡启动）

```bash
cd /public/home/zncui/pycharm_project/maskdino-wheat50/.claude/worktrees/full-coral-ecount
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<卡号> \
/public/home/zncui/anaconda3/envs/MaskDINO-310/bin/python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_full_coral_ecount.yaml \
  OUTPUT_DIR /public/home/zncui/pycharm_project/maskdino-wheat50/output_v2/exp-<启动日 MMDD>-full-coral-ecount
```
