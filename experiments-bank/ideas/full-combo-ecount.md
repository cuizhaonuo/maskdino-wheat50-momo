# 想法（完整集）：三合一 combo 里把计数换成 K 臂方案（E 损失 + 计数读 round(E)）· full/combo-ecount

状态：**预注册（跑之前写死），未启动** ｜ 提出 2026-10-24（本文写于 10-25）｜ 分支 `full/combo-ecount`（从 `full/coral-ecount` 的代码提交 `409378a` 开出）｜
实现、配置与本文同一提交 ｜ 配置 `wheat_joint_single_stage_full_combo_ecount.yaml`（`_BASE_` 为 `wheat_joint_single_stage_full_combo.yaml`）｜
输出 `output_v2/exp-<启动日 MMDD>-full-combo-ecount`（占位，主 agent 定）｜ 父臂 `full/combo` ｜ 同批 `full/coral-ecount`

## 来源

- **用户 2026-10-24 原话**：「另外对于 full/combo 这里,这是三个创新点/分支合并后的模型(coral-exist, kmo-matcher, group-nms-match),其中计数的 coral-exist 然后你将 K 臂改进的方案替换之.」
  - 主 agent 按 K 臂的口径落实，已告诉用户：E 损失加在 CORAL 头上；每穗计数读 round(E)；
    每穗输出几个点仍由 K（σ > 0.5 的个数，即 combo 现有的 n_k）决定，选点照旧是 focal + 点 NMS 取 top-n_k。
- **用户 2026-10-24 对 mini K 的评价**（原话）：「我比较在意的数值是 count_MAE_spike,其实 segmAP 和 F1@8 基本保持就算可以,比如你的 K 臂在我认为算是不错的」。
- **父臂 combo** `full/combo`：预注册 `experiments-bank/ideas/full-combo.md`，结果 `experiments/exp-2026-10-09-full-combo.md`（都在 `full/combo` 分支上，头 `221a841`），
  run 在 `output_v2/exp-1009-full-combo`。按它自己的预注册判否（F1@8 后半程峰值 0.6242 ≤ 0.6477）；后半程计数最低 1.3393（iter 22999）。
- **K 臂**：
  - mini `mini/coral-ecount`（`exp-2026-10-23-mini-coral-ecount.md`）：主读数 1.1897 过线；归因「有效」（余量 0.006）；机制启动（r 0.748）；segm AP、F1@8 两条护栏未过。
  - 完整集 A 上的版本 `full/coral-ecount`（预注册 `ideas/full-coral-ecount.md`，在该分支 `8b1c112`；run `exp-1024-full-coral-ecount`，10-24 20:11 启动）。
    本臂与它同一批、同一套定线思路，便于同批比较。
- **n_k 与候选数差距的诊断**：`output_v2/_diag-1022-full-docs/nkgap-exp-1009-full-combo.json`（`nk_gap_diag.py`）。
  combo 的 model_final 上 95.1% 的预测穗候选比 n_k 多（平均截掉 2.53 个），0.4% 候选不足；输出点数几乎全由 coral 的 n_k 决定。
- **代码**：从 `409378a` 开出（`full/coral-ecount` 的代码提交，开关全关时与 combo 的合并代码 `3bd2071` 逐位等价）。
  本分支只按交接说明把 E 损失与读数扩展到 combo 的 group 层，没有 merge / cherry-pick / rebase 任何分支。

## 出发点

1. **combo 的计数通道**。CoralHead（`point_class_embed`，30 个任务：共享 1 维打分 + 30 个 bias）被 11 份输出共用：
   - slot 层（layer-0、interm）：30 个 slot logit，CE 前缀内层匹配，CORAL BCE 记在 `loss_point_cls_0`、`loss_point_cls_interm`；
   - group 层（主输出、第 1~8 层）：CoralHead 挂在实例 query 上输出 `pred_point_coral [B, Q, 30]`，CORAL BCE 记在 `loss_point_coral*`（9 份）；
     这些层的 `loss_point_cls` 是小穗 token 的 focal，与 CORAL 无关。
   - 推理：n_k = Σ_k 1[σ(c_k) > 0.5]（最终层 `pred_point_coral`）；候选 = token 分数 > 0.35 且推理 NMS（κ 0.05）存活；按分数取 top-n_k；
     计数 = 实际输出的点数 = min(n_k, 候选数)。
2. **combo 的默认读数被压缩，E 本身已校准**（与 A 同一现象）。combo 最终评估的 557 个配对穗（GT 均值 15.96、标准差 2.42）：

   | 读法 | MAE | 偏差 | r | 预测 sd | E[GT \| 预测] 斜率 |
   |:--|--:|--:|--:|--:|--:|
   | 输出的点数 = min(n_k, 候选)（combo 默认读数） | 1.370 | +0.38 | 0.722 | 1.20 | 1.45 |
   | K = n_k | 1.368 | +0.39 | 0.722 | 1.20 | 1.46 |
   | E = Σ_k σ(c_k)（不取整） | 1.211 | −0.16 | 0.757 | 1.70 | 1.08 |
   | **round(E)** | **1.190** | −0.16 | 0.745 | 1.72 | **1.04** |
   | 穗长直线（同批穗） | 1.442 | −0.01 | 0.610 | 1.41 | 1.05 |

   - 来自本分支代码在 combo model_final 上的 `--eval_only`（加 `TEST.POINT_CORAL_READOUT expect`、E 损失关），见「实施前已完成的验证」第 6 条。
   - 压缩来自 bias 阶梯头部的平台：combo 最终 30 个 bias 里 k = 0~6 都是 2.968（A 是 2.468），跨度 8.35。
3. **combo 的同口径（round(E)）后半程轨迹**比它自己的默认读数低 0.15~0.24（13 次里 12 次），最低 1.1705（iter 22999），
   已比 A 的同口径最低（1.2282）低 0.058。所以本臂的线必须以 combo 同口径轨迹为参照，不能照抄 `full/coral-ecount` 的数值。
4. **选点与 K 的耦合比 A 弱**：A 的点是前 K 个 slot，K 多 1 个就多输出 1 个点（mini K 的 K 被推到 24~25，每穗多报 6.26 个点）；
   combo 的点是候选里取 top-n_k，候选平均比 n_k 多约 2.4 个，n_k 再涨也最多把候选取完。

## 假设

- E 损失 |E − y| 直接压每穗的期望计数误差，梯度大小与误差无关；CORAL BCE 在共享打分 g 上的梯度只有 (E − y)/30。
- combo 的 E 在均值上已校准（round(E) 斜率 1.00~1.08），逐穗还不够准（r 0.71~0.75）。若特征里有更多计数信息，E 损失能提高 r，round(E) 的 MAE 随之下降。
- 读数取 round(E)，与新损失的对象是同一个量；小穗点仍按 n_k 取 top-n_k（同 K 臂：点仍按 K）。

## 修法（本提交）

所有改动由 `409378a` 已有的两个键控制，默认值就是现有行为（与 `409378a` / `3bd2071` 逐位等价，见「实施前已完成的验证」）。

| 位置 | 改动 |
|:--|:--|
| `maskdino/modeling/criterion.py` `loss_point_ecount` | group 层（`pred_point_abs` 为真：combo 的主输出与第 1~8 层 aux）原先直接 `return {}`，现在改取 `pred_point_coral[idx]`（[M, T]，decoder 已去掉 DN 段，与外层匹配下标对齐），E = Σ_k σ(c_k)（fp32），目标 min(Ni, T)，T = 30；valid 掩码（外层匹配上的 wheat_2 且 Ni ≥ 1）、分母、「没有这类实例时为 0 且保持计算图」都与 slot 层相同。slot 层路径（A 的 11 份、combo 的 layer-0 / interm）一行逻辑不改 |
| `maskdino/maskdino.py` `from_config` | 删掉 `409378a` 里两处「与 `POINT_GROUP_DECODER` 同开时 NotImplementedError」的守卫；没有 CORAL 头时的 ValueError 保留。weight_dict 的键与 A 相同（`loss_point_ecount` 随 `_interm` / `_dn` / `_{i}` 展开，只有 11 份会被产生） |
| `maskdino/maskdino.py` `instance_inference` | combo 分支（分组 decoder + CORAL）：`expect` 下 E 取最终层 `pred_point_coral`（fp32，只算一次），每穗计数 clamp(round(E), 0, T) 写进 `pred_spikelet_counts`；点不变（候选取 top-n_k）。诊断字段 `pred_spikelet_ecount` = E，`pred_spikelet_kcount` = n_k（候选不足 n_k 时大于输出的点数）。A 的前缀路径逐位不变 |
| `maskdino/evaluation/spikelet_evaluation.py` | 只改注释：`kcount` 是 CORAL 头的 K（combo 是 n_k），`out_points` 是实际输出的点数，A 上两者相同，combo 上候选不足时 `out_points` 更小 |
| `maskdino/config.py` | 只改两个键的注释 |
| 配置 | `wheat_joint_single_stage_full_combo_ecount.yaml`，`_BASE_` 为 combo 的 yaml，只加 `POINT_ECOUNT_WEIGHT: 1.0` 与 `TEST.POINT_CORAL_READOUT: expect` |

不变的部分：组合语义（slot 层 CE 前缀匹配 + CORAL BCE；group 层逐 token focal + 训练侧点 NMS 匹配 + `loss_point_coral`）、CoralHead 的结构与初始化（bias 降序 1.00 → 0.033）、
bias 的 ×10 学习率（`train_net.py` 的参数组）、KMO 辅助损失、100 query、`NUM_POINTS` 30、`TEST.POINT_THRESHOLD` 0.35、推理 NMS κ 0.05、
n_k 的 0.5 阈值、schedule（26000 iter、bs4、lr 1e-4 cosine、EVAL_PERIOD 1000）、`SEED` 42。

## 与父臂 combo 的差异

- 解析后的 cfg 与 combo 的 yaml 逐键比对（340 个键），**只差两个键**：`POINT_ECOUNT_WEIGHT` 0.0 → 1.0，`TEST.POINT_CORAL_READOUT` threshold → expect。
  与 combo run 落盘的 `config.yaml` 比，另外只差 `OUTPUT_DIR` 和一个嵌套 tuple / list 的序列化写法（值相同）。`cfg_diff.txt`。
- 一个参数都不加，仍是 65,454,171 个；state_dict 的 1119 个键与 combo 相同。
- 训练损失多 11 项 `loss_point_ecount*`（slot 层 2 份：`_0`、`_interm`；group 层 9 份：主输出、`_1`~`_8`），原有 138 项不变。
- 内置评估的计数换了读法：`count_MAE_spike`、`count_MAE_img` 读 round(E)。点级指标、segm / bbox AP、spike_recall、spike_precision 与 combo 同一口径。

## 权重为什么取 1.0

沿用 K 臂的 1.0（mini K 与 `full/coral-ecount` 都是 1.0），不改。理由：单位都是「小穗个数」的 L1；本臂与 `full/coral-ecount` 同批，E 损失的设置相同才好比；
用户认可的是 mini K（权重 1.0）那样的取舍；第 0 步的梯度比例与 `full/coral-ecount` 相当（见下）。

- **量级**（第 0 步，固定 batch，fp32，已乘权重；「实施前已完成的验证」第 3 条）：
  - `loss_point_ecount` 11 份合计 12.56（slot 层 2 份 2.09，group 层 9 份 10.47），占总损失 605.4 的 **2.1%**。
  - 同一 batch 上 `full/coral-ecount`（A 配置）是 25.08、**4.3%**。差别来自这个 batch：combo 有 30 个任务，初始 E ≈ 19.0，
    这 3 个 wheat_2 的 Ni = 17、19、20，|E − y| ≈ 2.0、0.1、0.9；A 只有 25 个任务，初始 E ≈ 16.8，|E − y| ≈ 0.2、3.2、2.2。
  - 冒烟 100 步（不同 batch，lr ≤ 1e-7、参数几乎不动；「实施前已完成的验证」第 9 条）：11 份合计每步均值 29.3（slot 层 5.6、group 层 23.7），占 total_loss 的 **5.1%**（中位 5.0%；前 10 步 5.1% → 后 10 步 6.0%），`full/coral-ecount` 的冒烟是 3.5% → 4.4%。combo 的初始 E ≈ 19，比 GT 均值（约 16）高，每份 |E − y| 约 2.7；A 的初始 E ≈ 16.8，每份约 2.2，所以占比略高。两臂在同一档，不改权重。E 与 CORAL BCE（11 份合计）之比 1.6。
  - 同一 batch 上已有的 CORAL BCE（slot 层 `loss_point_cls` 2 份 + group 层 `loss_point_coral` 9 份，×2）合计 19.24，group 层 focal 合计 6.72。
- **梯度范数比**（第 0 步，fp32）：
  - 共享部分上是其余全部损失的：decoder 层 6.0%（A 5.1%，A 那边的分组是整个 decoder）、pixel decoder 2.9%（A 2.7%）、backbone 4.0%（A 3.3%），与 A 相当。
  - CoralHead 上是 CORAL BCE 的：打分 MLP 1.75 倍（A 6.80）、bias 0.67 倍（A 6.65）。这个 batch 上 3 个穗的 E − y 有正有负（+2.0、+0.1、−0.9），
    E 损失对 30 个 bias 的梯度互相抵消了一部分；A 那边 3 个穗都是 E < y，没有抵消。这一项随 batch 变，不作为改权重的依据。
- **slot 层与 group 层**：两层都作用在同一个 CoralHead 上，11 份的结构与 A 相同（A 也是 11 份全在同一个 CoralHead 上）。
  slot 层的 E 梯度到不了 decoder 层（layer-0 的 query 是 detach 过的 encoder 输出，interm 只回传到 encoder 输出头与 pixel decoder），
  decoder 层只从 group 层的 9 份拿到 E 梯度。CORAL BCE 也是这样分布的（见第 3 条的张量集合）。

## 实施前已完成的验证

脚本与产物都在 `output_v2/_smoke-1024-full-combo-ecount/`（`scripts/` 下是全部脚本）。父代码一侧用 `git archive` 的快照
（`parent_409378a/`、`parent_3bd2071/`），没有在别的 worktree 里执行任何东西。第 1~8 条在最终代码上做：工作区的代码改动（`git diff -- maskdino train_net.py`）
sha256 为 `54e1ea67…`，各次运行的日志都记了这个值，与本提交的代码改动相同。环境 `MaskDINO-310`（py3.10 / torch 2.5.1）。

**1. 开关全关时逐位等价（A 与 combo 两个配置，对 `3bd2071` 与 `409378a`）**

baiju GPU0，10-24 20:17~20:19。`scripts/run_equiv.sh`、`equiv_test.py`、`compare.py`，结果 `equiv/compare.txt`。

- 固定 random / numpy / torch 种子，cudnn deterministic，载 ImageNet R-50。同一份 batch：完整集 train 第 6、7 张图经 LSJ mapper，64 个实例，其中 3 个 wheat_2（Ni = 17 | 20、19）。训练态前向。
- **combo 配置**：对两个快照，138 项 loss 在 fp32 与 AMP 下都逐位相等（fp32 总和 592.8462981283665）；参数量 65,454,171、state_dict 1119 个张量逐个哈希、weight_dict 360 个键、criterion.losses 都相同。
- **A 配置**：对两个快照，127 项 loss 在 fp32 与 AMP 下逐位相等（fp32 总和 558.7459208071232）；参数量 52,294,290、state_dict 733、weight_dict 280 个键都相同。

**2. A 配置 + `POINT_ECOUNT_WEIGHT 1.0`（`full/coral-ecount` 的训练路径）不受影响**

`409378a` 快照与本 worktree：138 项 loss（含 11 项 `loss_point_ecount*`）在 fp32 与 AMP 下逐位相等（fp32 总和 583.8210246860981）。本提交没有改 A 路径的数值。

**3. combo 配置加 `POINT_ECOUNT_WEIGHT 1.0`（同一份 batch）**

- 参数与 state_dict 不变；原有 138 项 loss 在 fp32 与 AMP 下都与权重 0 时逐位相同（新损失不消耗随机数）；多出 11 项：`loss_point_ecount`、`_0`~`_8`、`_interm`。
  按输出 dict 的 `pred_point_abs`：slot 层 = `_0`、`_interm`，group 层 = 主输出、`_1`~`_8`。
- **手算**：包住 `loss_point_ecount`，记下每次调用用的 logit（slot 层 `pred_point_logits[..., 0]`，group 层 `pred_point_coral`）、外层匹配与 targets，用 float64 numpy 独立重算。
  - 11 份与报告值的最大相对差：fp32 1.2e-6，AMP 9.3e-7。每份都计入 3 个 wheat_2、排除 61 个其他实例（wheat）。
  - group 层另用同样记下的 `pred_point_coral` 与外层匹配重算 `loss_point_coral`（CORAL BCE，wheat 权重 0.1）：9 份最大相对差 1.1e-7（fp32）/ 1.0e-7（AMP）。
    E 损失读的就是 `loss_point_coral` 用的那组 logit（同一批 query、已去掉 DN 段）。
  - **group 层样例**（主输出）：第 0 张图的 GT 0（Ni = 17）配到 query 52。30 个 logit 从 1.0470 降到 0.0803（bias 初值 1.000 → 0.033 加上 g = 0.0470），
    σ 从 0.7402 降到 0.5201。E = 19.0413，|E − 17| = 2.0413。
  - **slot 层样例**（layer-0）：同一个 GT 配到 query 15。30 个 logit 从 1.0755 降到 0.1089（g = 0.0755），σ 从 0.7456 降到 0.5272。E = 19.2355，|E − 17| = 2.2355。
- **梯度**（可训练张量共 742 个；`scripts/compare.py` 的分组）：

  | 只对这部分求梯度 | fp32 非零梯度张量 | AMP 非零梯度张量 | 说明 |
  |:--|--:|--:|:--|
  | `loss_point_ecount*`（11 项） | 389 | 389 | CoralHead 7/7、decoder 层 198、decoder 其他 4、小穗 token 分支 9、pixel decoder 112、backbone 53、predictor 其余 6 |
  | 其中 slot 层 2 项 | 178 | 178 | 到不了 decoder 层：CoralHead 7、pixel decoder 112、backbone 53、encoder 输出头与 decoder_norm 6 |
  | 其中 group 层 9 项 | 385 | 385 | 比全部少 encoder 输出头 4 个（只有 interm 用到） |
  | CORAL BCE（slot 层 `loss_point_cls` 2 项 + `loss_point_coral` 9 项） | 389 | 389 | 与 E 损失是**同一组**张量 |
  | group 层 focal（group 层 `loss_point_cls` 9 项） | 616 | 616 | 不经过 CoralHead |
  | E 以外全部 | 701 | 701 | |
  | 总损失 | 701 | 701 | 41 个第 0 步结构性为 0，与权重 0 时相同 |

  - 七种情况下 fp32 与 AMP 有梯度的张量集合完全相同；CoralHead 的 7 个张量在「只 group 层 E」「只 slot 层 E」「只 CORAL BCE」下 fp32 与 AMP 都有梯度。
    没有踩到「no_grad 下经 autocast 预调输出头」的坑：combo 训练侧的点 NMS（`_point_nms_keep`）在 no_grad 下只读已经算好的输出、不调用任何模块，本改动也没有在 no_grad 里调用模块。
  - AMP 下仿 GradScaler 从 65536 往下退避：只算 E 损失（及其两部分、CORAL BCE、focal）时 65536 溢出、4096 有限；E 以外全部与总损失要到 256。这是 GradScaler 头几步正常的退避。
  - 显存（bs2，fp32 前向加 7 次求梯度）：13.8 GB。

**4. `--eval_only` 与训练日志逐项相同（默认读法、E 权重 0）**

qiuhao GPU1，10-24 20:18~20:22。`scripts/run_evals.sh`、`eval_dump.py`（走 `train_net.main` 的 `--eval_only` 分支，只把末尾的 `verify_results` 换成按完整精度 dump）、`compare_evals.py`，结果 `compare_evals.txt`。

- combo 的 `model_final.pth` 配它自己的 `config.yaml`：**40 项指标与 combo 训练日志末次评估（iter 26000）完整精度逐项相同**，
  `coco_instances_results.json`、`instances_predictions.pth`、两份 CSV 与 combo 的最终评估逐字节相同。
- A（`exp-1009-full-coral-exist`）的 `model_final.pth` 配它自己的 `config.yaml`：同样 40 项逐项相同、4 个产物逐字节相同。

**5. 守卫**（CPU 上 `MaskDINO.from_config`，`guards_check.txt`）

- 本臂 yaml：losses 多 `point_ecount`，weight_dict 400 个键（多 40 个 ecount 键，只有 11 个会被产生），diag 开、readout expect。
- combo + 只开 E 损失、combo + 只开 expect 都能建；A 两个开关同开照常；没有 CORAL 头（GN 单点、基线）时两个开关报 ValueError；readout 写错报 ValueError。

**6. expect 读法在 combo 的 model_final 上（E 损失关，点不应变）**

combo 的 `model_final.pth` 配它自己的 `config.yaml`，加 `MODEL.MaskDINO.TEST.POINT_CORAL_READOUT expect`（`cexp/C_expect_26000`）：

- `count_MAE_spike` **1.1903052064631956**（combo 默认读数 1.3698），`count_MAE_img` 14.555（16.225）。
- 计数以外的 37 项指标与 combo 训练日志末次评估逐项相同；`coco_instances_results.json`、`instances_predictions.pth` 逐字节相同；逐图明细除计数列外逐行相同。
- 557 个配对穗逐行对齐：配对（文件名、GT 点数、IoU）与 combo 最终评估相同；`out_points` = combo 最终评估的点数；计数 = clamp(round(`ecount`), 0, 30)；
  `out_points` ≤ `kcount`，只有 2 穗候选不足（平均少 1.5 个）。
- 与 n_k 诊断（`nkgap-exp-1009-full-combo.json`，独立复算）按实例分数对齐 557/557：`kcount` = 诊断的 n_k，`out_points` = 诊断的 min(n_k, 候选数)，逐穗相同。

**7. combo 的同口径轨迹**：见下文「combo 同口径读数」。13 次评估里计数以外的 37 项指标都与 combo 训练日志同 iter 的值逐项相同（checkpoint 与 iter 对得上，评估可复现，点不变）。

**8. 配置**：`cfg_diff.txt`，见上文「与父臂 combo 的差异」。

**9. 冒烟**

baiju GPU0，10-25 14:56~15:33。`scripts/run_smoke.sh`、`smoke_train.py`、`summarize_smoke.py`，输出 `smoke100/`、`smoke100.json`、`smoke100_summary.txt`、`smoke100.launch.log`。

- 本臂正式配置加 CLI 覆盖 `SOLVER.MAX_ITER 100`，bs4，完整集 train。走 `train_net.setup` / `train_net.Trainer` 的完整训练循环
  （AMPTrainer、优化器、LR 调度、日志）与训练末的一次完整 val 评估，只去掉 PeriodicCheckpointer，不存 checkpoint。这不是一次训练 run。
  卡上空闲显存紧，加了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（只改分配器、不改数值）。
- 100 iter 与 val 评估都跑通，无报错，损失全部有限。
- 量级（已乘权重；前 10 步均值 → 后 10 步均值）：

  | 损失 | 前 10 步 | 后 10 步 |
  |:--|:--|:--|
  | `loss_point_ecount`（11 份） | 每份 2.23~2.92，合计 28.91（slot 层 2 份 5.76，group 层 9 份 23.15） | 每份 2.85~3.33，合计 34.66（6.55 / 28.11） |
  | `loss_point_coral`（group 层 CORAL BCE ×2，9 份） | 每份 1.53~1.67，合计 14.45 | 每份 1.54~1.65，合计 14.41 |
  | `loss_point_cls` slot 层（CORAL BCE ×2，2 份） | 每份 1.65，合计 3.30 | 每份 1.63，合计 3.27 |
  | `loss_point_cls` group 层（focal，9 份） | 每份 0.81~0.91，合计 7.70 | 每份 0.78~0.86，合计 7.34 |
  | `loss_point_coord`（×5，11 份） | 合计 124.0 | 合计 114.7 |

  - E 损失 11 份合计占 `total_loss`：全程均值 5.1%（中位 5.0%），前 10 步 5.1% → 后 10 步 6.0%（`total_loss` 579.3 → 572.4）。`full/coral-ecount` 的冒烟是 3.5% → 4.4%。
  - 这 100 步的 lr 都不超过 1e-7（warmup 1000 iter，cosine 被 MAX_ITER 100 压短），参数几乎没动，前后差别来自 batch 不同。
  - iter 80 的 batch 没有 Ni ≥ 1 的 wheat_2：E 损失 11 份都是 0（走「没有这类实例时为 0 且保持计算图」的分支，反传正常）。
    同一步 group 层 focal 9 份合计 325（其余步中位 7.6），来自 combo 原有的归一化（分母 max(num_pos, 1)，这一步 num_pos = 0），与本改动无关。
  - `point_nms/suppressed_frac` 均值 0.094，`starved_frac` 0.028。
- 速度 14.60 s/iter（`Overall training speed: 98 iterations in 0:23:50`），训练末评估 200 图用了 11:32。
  当时 baiju GPU0 上有别人的 3 个进程（共约 48 GB），利用率 100%，这两个数不代表独占速度（combo 是 3.609 s/iter、每次评估约 2 min）。
- 峰值显存 `max_mem` **25,428 MB**（torch 口径；combo 全程 26,962 MB）；`max_memory_reserved` 26,622 MB（expandable_segments，含训练末评估）。
- 训练末评估（未训练的模型）：没有 score ≥ 0.4 的 wheat_2，`count_MAE_spike` 为 nan，`count_MAE_img` 48.735。
  `spikelet_pair_counts.csv` 的表头带上了 `ecount`、`kcount`、`out_points`。有配对穗时 expect 读法的输出由第 6、7 条在 combo 的 13 个 checkpoint 上验过。

## 预期与判否线（跑之前写死）

### 读数

- **主读数**：后半程（iter ≥ 13000，共 13 次评估：13999 … 24999 每 1000 次，加 26000）`count_MAE_spike` 最低的那次（排除 nan），同一 checkpoint 报全部指标。
- 默认读数：每穗计数 round(E) = clamp(round(Σ_k σ(c_k)), 0, 30)，c 是最终层 `pred_point_coral`；小穗点是候选（token 分数 > 0.35 且推理 NMS 存活）里按分数取的 top-n_k，
  n_k = Σ_k 1[σ(c_k) > 0.5]（与 combo 相同），`matched_F1@σ` 等点级指标按实际输出的点算。
- 不扫阈值（不改 `POINT_THRESHOLD`，不改 n_k 的 0.5，不跑任何 `--sweep`），不引入预注册之外的读法；单种子（`SEED` 42），不补种子。
- 噪声地板：`matched_F1@8` 0.015。单种子，噪声内的差异记「歧义」。

### 参照（写死；都从各 run 的 `metrics.json` 或启动前的 `--eval_only` 复核过）

| | 主读数（后半程 count 最低） | 全程峰值 segm AP | 后半程峰值 spike_recall | 最终评估（iter 26000） |
|:--|:--|--:|--:|:--|
| p30 `exp-0922-single-stage-p30` | iter 17999：count 2.4645、`matched_F1@8` 0.6302、segm AP 80.64 | 82.01（iter 24999） | 0.9560（iter 16999） | count 2.7679、F1@8 0.6477、segm 81.97 |
| combo `exp-1009-full-combo`（默认读数） | iter 22999：count 1.3393、F1@8 0.6116、segm AP 82.36 | 82.36（iter 22999） | 0.9413（iter 16999） | count 1.3698、F1@8 0.6183、segm 81.96 |
| combo 同口径（round(E)，见下表） | iter 22999：count **1.1705**（563 穗）、F1@8 0.6116、segm AP 82.36 | — | — | count 1.1903、r 0.7451、斜率 1.044 |
| A 同口径（`full/coral-ecount` 的归因参照） | iter 16999：count 1.2282（561 穗）、F1@8 0.6228、segm AP 79.46 | — | — | count 1.2369、r 0.7276 |

- **完整集穗长直线**（不训练）：点数 = round(9.452 + 0.0329 × 预测框对角线)。系数是全量 train 5394 个带点 wheat_2 上用 GT 框对角线 polyfit 的结果
  （完整精度 9.45195 / 0.0328843，与 `full/coral-ecount` 同一做法）。combo 最终评估的 557 个配对穗上 **1.4417**；combo 后半程 13 次评估各自的配对穗上 1.394~1.452。
- `full/coral-ecount` 的线（同批，只作对照）：成立 ≤ 1.18、判否 ≥ 1.27、归因 ≤ 1.1054、机制门 r ≥ 0.7576；护栏 segm AP ≥ 79.01、F1@8 ≥ 0.6102、spike_recall ≥ 0.9147。

### combo 同口径读数（同一读法，后半程 13 个 checkpoint；参照，不是 combo 的结果）

- 计数列：combo 已有的 checkpoint 配它自己的 `config.yaml`，加 `TEST.POINT_CORAL_READOUT expect` 做 `--eval_only`（10-24，启动前；`cexp/`，汇总在 `traj_C_expect.json` / `.txt`）。
  这是同一读法下的父臂轨迹，不是扫阈值：round(E) 没有可调参数。做法同 `full/coral-ecount` 预注册的「A 同口径读数」。
- 点级列：点与 combo 相同，所以就是 combo 训练日志同 iter 的值（13 次都核对过逐项相同）。
- 按用户 10-22 的决定，round(E) 在本臂之外只作诊断记录：下表只用来定本臂的线，不是 combo 的读数，combo 的结论仍按它自己的默认读数。

| iter | 配对穗 | `count_MAE_spike`（round(E)） | 偏差 | r | sd（GT 2.42） | E[GT \| 预测] 斜率 | `count_MAE_img` | 同批穗直线 | `matched_F1@8` | segm AP | spike_recall | combo 默认读数的计数 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 13999 | 498 | 1.2229 | +0.13 | 0.707 | 1.52 | 1.072 | 12.69 | 1.3936 | 0.6006 | 79.62 | 0.8124 | 1.4618 |
| 14999 | 558 | 1.2151 | −0.01 | 0.740 | 1.68 | 1.062 | 16.87 | 1.4355 | 0.6071 | 80.48 | 0.9103 | 1.4552 |
| 15999 | 542 | 1.4483 | −0.95 | 0.724 | 1.64 | 1.062 | 11.94 | 1.4244 | 0.6112 | 80.18 | 0.8842 | 1.3801 |
| 16999 | 577 | 1.2374 | +0.06 | 0.741 | 1.66 | 1.081 | 16.41 | 1.4437 | 0.6088 | 80.86 | 0.9413 | 1.4558 |
| 17999 | 577 | 1.2132 | −0.32 | 0.741 | 1.70 | 1.051 | 15.82 | 1.4437 | 0.6153 | 79.92 | 0.9413 | 1.3674 |
| 18999 | 530 | 1.1981 | +0.03 | 0.740 | 1.73 | 1.032 | 13.72 | 1.4377 | 0.6242 | 80.08 | 0.8646 | 1.4132 |
| 19999 | 538 | 1.1840 | +0.08 | 0.744 | 1.75 | 1.024 | 13.65 | 1.4368 | 0.6151 | 81.23 | 0.8777 | 1.4089 |
| 20999 | 561 | 1.2264 | +0.01 | 0.726 | 1.66 | 1.048 | 14.98 | 1.4385 | 0.6240 | 81.42 | 0.9152 | 1.3993 |
| 21999 | 550 | 1.1727 | −0.07 | 0.743 | 1.78 | 1.001 | 13.97 | 1.4382 | 0.6177 | 81.12 | 0.8972 | 1.3636 |
| **22999** | 563 | **1.1705** | −0.17 | 0.753 | 1.74 | 1.041 | 15.19 | 1.4387 | 0.6116 | 82.36 | 0.9184 | 1.3393 |
| 23999 | 555 | 1.2072 | −0.23 | 0.743 | 1.70 | 1.059 | 14.30 | 1.4523 | 0.6144 | 82.05 | 0.9054 | 1.3604 |
| 24999 | 558 | 1.1918 | −0.15 | 0.748 | 1.74 | 1.038 | 14.98 | 1.4427 | 0.6212 | 82.05 | 0.9103 | 1.3728 |
| 26000（final） | 557 | 1.1903 | −0.16 | 0.745 | 1.72 | 1.044 | 14.56 | 1.4417 | 0.6183 | 81.96 | 0.9086 | 1.3698 |

- **combo 同口径读数**：后半程最低在 iter 22999，**1.1705**（563 穗），r 0.753。
- 末 9 次（iter ≥ 17999）均值 1.1949、标准差 0.018，范围 1.1705~1.2264（A 同口径末 9 次：均值 1.249、标准差 0.018）。
- 13 次里 12 次比 combo 自己的默认读数低 0.15~0.24、比同批穗的直线低 0.17~0.27。例外是 iter 15999：round(E) 偏差 −0.95，1.4483，比默认读数高 0.07、比直线高 0.02。
- round(E) 下 13 次的斜率都在 1.00~1.08，没有压缩；r 0.707~0.753。

### 主判据（计数）

| 主读数 `count_MAE_spike` | 判定 |
|:--|:--|
| ≤ **1.12** | 计数**成立** |
| 介于 1.12 与 1.21 之间 | **歧义** |
| ≥ **1.21** | **判否** |

理由（同 `full/coral-ecount` 的思路，参照换成 combo 同口径）：

- **成立线 1.12** = combo 同口径最低 1.1705 − 0.05。0.05 约是 combo 末 9 次同口径评估标准差（0.018）的 2.7 倍，
  相当于 combo 同口径 × 0.957，与 `full/coral-ecount`（1.18 / 1.2282 = 0.961）、mini K（1.28 / 1.3287 = 0.963）的成立线相对各自父臂同口径的位置相同。
  过了这条线，计数比 combo 的默认读数（1.3393）低约 16%，比同批穗的直线（约 1.44）低约 22%。
- **判否线 1.21** = combo 同口径最低 + 0.04（末 9 次标准差的 2 倍多）。本臂若连它都过不了，等于 E 损失让 round(E) 的计数比 combo 自己同一读法下还差，记判否。
  - `full/coral-ecount` 的判否线 1.27 同时满足两种写法：「直线 × 0.9」（1.4069 × 0.9 = 1.266）与「父臂同口径最低 + 0.04」（1.268）。
  - 在 combo 上这两种写法分开了：直线 × 0.9 = 1.4417 × 0.9 = **1.30**，父臂同口径最低 + 0.04 = **1.21**。原因是 combo 换读法后已经比直线低 0.17~0.27。
  - 用 1.30 作判否线时，一个比 combo 自己同一读法差 0.13 的结果只会记「歧义」，判不出 E 损失帮倒忙。所以取两者中更严的 1.21，语义与 `full/coral-ecount` 相同。
- 介于两者之间，与 combo 同口径的差别在噪声量级内，记歧义。

### 护栏（任一不过则整体不成立）

**用户 2026-10-24 原话**：「我比较在意的数值是 count_MAE_spike,其实 segmAP 和 F1@8 基本保持就算可以,比如你的 K 臂在我认为算是不错的」

mini K 相对它的主对照 E0：segm AP 全程峰值 −2.77（76.03 vs 78.80），读数那次 `matched_F1@8` −0.0195（0.6033 vs 0.6228）；相对父臂同口径读数那次（iter 11499）−0.0193（0.6033 vs 0.6226），与父臂同 iter（11999）比 −0.020。
`full/coral-ecount` 把线放宽到让「mini K 那样的降幅」能过。本臂沿用同一套，规则是：**父臂离批内统一线还有余量时用统一线；父臂自己就在线附近（余量小于噪声）时，改以父臂为参照、降幅不变**
（这正是 `full/coral-ecount` 对 spike_recall 的处理：「父臂都过不了的线，判不出 E 损失有没有伤到检测，所以改以 A 为参照」）。

| 护栏 | 线 | 理由 |
|:--|:--|:--|
| segm AP 全程峰值 | ≥ **79.01**（p30 全程峰值 82.01 − 3.0，与 `full/coral-ecount` 相同） | combo 的全程峰值 82.36，余量 3.35，统一线合适。mini K 的降幅 −2.77 能过 |
| 主读数那次 `matched_F1@8` | ≥ **0.5916**（combo 同口径读数那次 0.6116 − 0.020）；批内统一线 0.6102 只作观察量 | 见下 |
| `spike_recall` 后半程峰值 | ≥ **0.9213**（combo 后半程峰值 0.9413 − 0.02） | 以父臂为参照，写法同 `full/coral-ecount`。按 p30 − 0.02 = 0.9360，combo 自己就过不了 |

**F1@8 为什么不用批内统一线 0.6102（`full/coral-ecount` 的线 = p30 读数那次 0.6302 − 0.020）**：

- combo 自己同口径读数那次（iter 22999，也是它默认读数的读数那次）只有 0.6116，比 0.6102 高 0.0014，远小于噪声地板 0.015。
- combo 后半程 13 次的 F1@8 是 0.6006~0.6242（均值 0.6146、标准差 0.0068），其中 3 次低于 0.6102（iter 13999 0.6006、14999 0.6071、16999 0.6088）。
  即使 E 损失对点毫无影响，主读数落在哪一次也会让本臂约有四分之一的机会被 0.6102 判不过。这条线判不出 E 损失有没有伤到点。
- 改以父臂为参照，降幅仍是 0.020（与统一线相同，大于噪声地板）：mini K 相对父臂同口径读数那次的降幅 −0.0193 刚好能过（余量 0.0007），与 `full/coral-ecount` 设计时「mini K 的降幅 −0.0195 能过（余量 0.0005）」的位置相同。
- 0.6102 的结果照常报告（观察量，不进判定），供同批比较；结果文档里同时报主读数 F1@8 相对 p30 读数那次的差。
- **这一条换线需要主 agent 审**：若主 agent 认为本臂也应按 0.6102 判，在启动前改本节并在提交信息里注明。

**spike_recall**：combo 后半程峰值 0.9413（iter 16999、17999 两次）；全程峰值 0.9592 落在前半程（iter 9999），前半程有检出近零的退化评估，按 `full/coral-ecount` 的写法只看后半程。
mini K 的 spike_recall 峰值比它的父臂低 0.016，按这条线能过。以前完整集预注册的 `spike_precision` 末值护栏（≥ 0.753）在本臂降为观察量（同 `full/coral-ecount`）。

### 归因（对 combo 同口径）

- 本臂主读数 `count_MAE_spike` ≤ **1.0535**（combo 同口径读数 1.1705 × 0.9）→「E 损失在 E 读法上有效」；否则记「无效」。规则同 mini K、`full/coral-ecount`（父臂同口径 × 0.9）。
- 同时报（不进判定）：后半程 13 次与 combo 同 iter、同读法的逐次比较——本臂更低的次数、平均差、末 4 轮平均差。这一项不依赖「两边各取最低一次」。

### 机制门（E 损失有没有让 E 更准）

- 取最终评估的逐穗明细（`inference/spikelet_pair_counts.csv`），`pred_points` 列即 round(E)。
- r(round(E), GT) ≥ **0.7751**（combo 同口径最终评估 557 穗上 0.7451，+0.03）→「E 损失让 E 更准」，机制启动；否则记「机制未启动」。
- 同时记录 E[GT | 预测] 斜率与预测计数标准差（combo 同口径最终评估：1.044 / 1.72）。

### 次口径（必报，不进判定）

- 在主读数 checkpoint 上 `--eval_only` 加 `MODEL.MaskDINO.TEST.SPIKELET_EVAL_SIGMAS '[8.0,16.0,32.0]'` 重评，报 P / R / F1 / `matched_F1` 三档（combo 在 `model_0023999.pth` 上：`matched_F1@16 / @32` = 0.8963 / 0.9498）。
- 最终评估逐穗 CSV（round(E)）：有符号偏差、Pearson r、预测计数标准差、E[GT | 预测] 斜率（= r × sd_GT / sd_pred），以及同批穗上的穗长直线 MAE（同上文的直线与取整）。
  主读数 checkpoint 上同样报一遍（事后 `--eval_only` 取逐穗明细）。

### K / E 诊断（只记录，不进任何判定，不用来挑读法或阈值）

- 最终评估同一批穗上：K（`kcount` = n_k）、实际输出的点数（`out_points`）、E（`ecount`，不取整）各自的 MAE、偏差、r、斜率。
  combo 同口径最终评估：K 1.368 / +0.39 / 0.722 / 1.46；输出点数 1.370 / +0.38 / 0.722 / 1.45；E 1.211 / −0.16 / 0.757 / 1.08。
- **K 每穗多报多少**：K − GT 的均值（combo +0.39）、E − K 的均值（combo −0.55）、E ≤ K 的比例（combo 80%）、K ≥ 24 的比例（combo 0%，最大 K 20）、
  `out_points` < `kcount` 的穗数（combo 2/557）。
- **n_k 与候选数差距**：按 `_diag-1022-full-docs/nk_gap_diag.py` 的做法在最终 checkpoint 上复算（wheat_2、分数 ≥ 0.4 的预测穗）：候选多于 / 等于 / 少于 n_k 的比例、
  平均截掉与平均短缺（combo：95.1% / 4.5% / 0.4%，截掉 2.53，短缺 1.33）。本臂 expect 下 `pred_spikelet_counts` 是 round(E)，复算的输出点数改与 `pred_spikelet_points` 的长度核对。
- **bias 跨度**：最终 30 个 bias 的数值、跨度（combo 8.349，初值 0.967）、头部平台（combo k = 0~6 都是 2.968）、相邻不增的对数（combo 29/29）、相对 combo 的位移。
  combo 的最终 bias：`2.968 ×7, 2.961, 2.951, 2.918, 2.850, 2.676, 2.368, 1.837, 1.242, 0.438, −0.297, −1.109, −1.834, −2.538, −3.159, −3.761, −4.226, −4.612, −4.822, −4.996, −5.206, −5.258, −5.269, −5.381`。
- `pred_points` 逐穗等于 clamp(round(`ecount`), 0, 30)，`out_points` ≤ `kcount`（自检）。
- 按用户 10-22 的决定，这些读法只出现在诊断文件与实验文档里。

### 观察量（只记录）

- 后半程 13 次与 combo 同 iter（同读法）的差：`count_MAE_spike`、r、`matched_F1@8`、segm AP、bbox AP、spike_recall、`n_pred_points`；也与 p30 同 iter 比。
- 批内统一的 F1@8 线 0.6102 过没过；主读数 F1@8 相对 p30 读数那次（0.6302）的差。
- `spike_precision` 末值（combo 0.7537，以前完整集预注册的线 0.753）、bbox AP 全程峰值（combo 81.01，p30 80.99）。
- **与同批 `full/coral-ecount`（A + K 方案）、p30、combo 默认读数的比较**（只作参照，不下跨臂结论）：主读数的 count、F1@8、segm AP、spike_recall，最终评估的 r 与斜率。
- 损失曲线（分段均值）：`loss_point_ecount` 分 slot 层 / group 层、`loss_point_coral`、`loss_point_cls`（slot 层 CORAL BCE / group 层 focal 分开）；E 损失占总损失的比例
  （mini K 13%~20%）。`point_nms/suppressed_frac`、`starved_frac`（combo：前 500 iter 后 7.6%~9.7% / 0）。

### 实施者的预测（写于启动前，不是判定线）

- **机制门**（r ≥ 0.7751）：说不准，略倾向不通过。mini K 的 r 比父臂同口径高 0.076；combo 同口径的 r 已到 0.745（A 0.728），0.775 会是本项目最高的计数 r。
- **主判据**：「歧义」与「成立」各半。r 每升 0.01，同口径 MAE 约降 0.011（combo 自己的后半程轨迹，去掉 iter 15999）到 0.019（A 的轨迹）；r 升 0.03 时约 1.11~1.14，正好落在成立线附近。
- **归因线**（≤ 1.0535）：倾向「无效」，要比 combo 同口径好 10%。
- **护栏**：
  - segm AP 峰值 ≥ 79.01：倾向通过。照搬 mini K 相对父臂的峰值降幅（−2.68）约 79.7。
  - F1@8 ≥ 0.5916：倾向通过。combo 的点是候选里取 top-n_k，候选平均比 n_k 多约 2.4 个；E 损失把 n_k 推高时，每穗最多多出约 2.4 个点，
    不会像 mini K 那样多出 6 个。共享 decoder 上的梯度占比与 A 相当，定位的变化方向说不准。
  - spike_recall ≥ 0.9213：说不准。combo 后半程 spike_recall 在 0.81~0.94 间摆动，mini K 的峰值降了 0.016。
- **K / E**：E 损失对 30 个 bias 的梯度同号，刻度尺跨度可能比 combo 窄；K 的偏差从 +0.39 小幅上升；`out_points` < `kcount` 的穗会变多。
- 最可能的结果：计数「歧义」或「成立」、归因「无效」，整体成不成立看机制门以外的三条护栏，最可能卡在 spike_recall。

## 已知限制

- **计数与点出自两个读数**：计数是 round(E)，点是 top-n_k。combo 上平均 E − K = −0.55，输出的点比计数多。
  `count_MAE_spike`、`count_MAE_img` 读计数；点级 P/R/F1、`n_pred_points` 读点。E 损失会改变共享打分 g 与 bias，n_k 跟着变，F1 的变化里也有 E 损失经 n_k 带来的间接影响。
- **新损失经共享 decoder 回传**：第 0 步它在 decoder 层上的梯度是其余全部损失的 6.0%（A 5.1%）；mini K 的 segm AP 后半程平均比父臂低 2.84。本臂不 detach，与 K 臂一致。
- **combo 同口径读数来自 combo 已有的 checkpoint**：同数据、同 schedule、同种子，但不是同一次训练；同种子的两次训练也会因 GPU 非确定性分叉。
- 主判据与归因比的是两边各自 13 次评估里的最低值，两边都有挑最低带来的偏低。逐次比较作为补充一并报告。
- **F1@8 护栏换成了以父臂为参照的线**（见「护栏」），与 `full/coral-ecount` 的数值不同；两臂的 F1@8 护栏结论不能直接并排读，批内统一线另行报告。
- 单种子（`SEED` 42），不补种子。噪声内的差异记「歧义」。
- T = 30：train 只有 1 个实例 Ni > 30（最大 31），E 损失的目标截到 30，与 CORAL levels 同口径（val 最大 24，不受影响）。

## 成本

- 26000 iter（bs4，约 50 epoch），另加 26 次评估（EVAL_PERIOD 1000）。新损失只是 11 次向量化的 sigmoid 求和，不增加参数。
- 参照 combo：全程 27:03:28，其中纯训练 26:03:50（3.609 s/iter，qiuhao A100），评估 26 次共 53 min（均 2:03）；峰值显存 `max_mem` 26,962 MB。
- 冒烟：14.60 s/iter、评估 11:32（卡被别人占满，不可比）；`max_mem` 25,428 MB，`max_memory_reserved` 26,622 MB（expandable_segments）。
  排卡：加 `expandable_segments:True` 时按约 30 GB 空闲显存（combo 全程已分配峰值 26,962 MB，再加分配器余量与 CUDA context）；不加时按约 40 GB（`full/coral-ecount` 的冒烟在默认分配器下 22.8 GB 已分配对 39.1 GB 保留）。
- 预计与 combo 相当：独占 A100 约 27~28 h；卡上有别人的任务时会慢，冒烟时就是这种情况。

## 启动命令（正式 run 由主 agent 排卡启动）

```bash
cd /public/home/zncui/pycharm_project/maskdino-wheat50/.claude/worktrees/full-combo-ecount
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<卡号> \
/public/home/zncui/anaconda3/envs/MaskDINO-310/bin/python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_full_combo_ecount.yaml \
  OUTPUT_DIR /public/home/zncui/pycharm_project/maskdino-wheat50/output_v2/exp-<启动日 MMDD>-full-combo-ecount
```

卡紧时可加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（只改分配器、不改数值；冒烟就是这样跑的）。
