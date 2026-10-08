# 想法：分组 decoder 上，小穗 token 在 decoder 里逐层点 NMS 剔除（DDQ DETR 式，arm B）

状态：**预注册（未开跑）** ｜ 提出 2026-10-08 ｜ 分支 `mini/group-nms-prune`（从 arm A 的代码提交 C=`df90505` 开出，本分支的代码提交 `5f4d71b`）｜ 配置
`wheat_joint_single_stage_mini_group_nms_prune.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_group.yaml`，即 E3 的配置）｜ 输出 `output_v2/exp-<启动日期>-mini-group-nms-prune` ｜
启动提交：本文档所在的提交

**分叉点是有意的。** 用户指定从 `mini/group-decoder`（E3）出发，按 NMS 的作用范围各开一个分支：

- arm A `mini/group-nms-match`：从 `mini/group-decoder`@`00a7aaf` 开出，第一个提交 C=`df90505` 是两个 arm 共享的代码加 match 模式；
- arm B（本分支）：从 C 开出，只在其上加 prune 模式（`5f4d71b`）。没有 merge、cherry-pick 或 rebase。

主对照是 **E0**（`mini/baseline` 100 epoch，`output_v2/exp-0923-mini-ep100`，用户 2026-10-08 指定）。E3 与 arm A 只作归因参照。

## 来源

Zhang et al., *Dense Distinct Query for End-to-End Object Detection*（DDQ，CVPR 2023，arXiv 2303.12776）：

- 每个 refine 阶段之前做 class-agnostic NMS，loss 只算在留下的 query 上，训练与推理都做。
- 效应量：DETR 版（decoder 里已有 query 自注意力）只有 +0.6~0.7 AP；没有 query 交互的 FCN 版 +3 AP。
- 本 arm 对应 DETR 版：被压掉的 query 之后不再参与（这里用注意力屏蔽实现），不可逆。
- E3 的小穗 token 之间有实例内 SA，更接近 DETR 版，**预期效应小**。

## 现象

- **瓶颈在分数挑点，不在定位。**
  - E3 最终 checkpoint 上，8 px 内有 token 的 GT 小穗占 79.1%，分数完美挑选时 F1@8 的上限是 0.88。
  - 实际 F1@8 0.638，存在性 AUC 0.699（`output_v2/_diag-1008-group-nms/E3_plain.json`；与 `mini/group-quality` 分支 `ideas/group-quality.md`「现象」一节的表一致）。
- **用户的设想：匹配不稳。**
  - 几个 token 落在同一个小穗上时，一对一的内层匹配随机给一个判正、其余判负，而且每次迭代都可能换人。
  - 近乎相同的 token 之间于是有一正一负，分数的监督自相矛盾。
- E3 的跨层内层改判率（全 val、最终 checkpoint、不做 NMS）：相邻层 1→2 8.4%、2→3 4.0%、3→4 2.4%，之后 1.9%~0.7%，平均 **2.71%**。
- 本方案改的是分数的监督：去掉近重复之间的一正一负。它与已判否的 group-quality（把存在性目标换成定位质量）不是一回事。

## 假设

每层先剔除近重复，只让剩下的 token 去匹配，被剔除的从下一层起不再参与注意力：

1. 同一小穗上只剩一个 token 去配，跨层改判率下降（M1，用户的假设）；
2. 剩下的 token 的存在性分数不再被近重复之间的一正一负拉扯，挑点变好（M2：AUC_surv 上升）；
3. 默认阈值下 matched_F1@8 上升、计数改善（主判据、次判据）。

## 修法（arm B 的设计）

开关默认关；`POINT_NMS_MODE` 为 none / match 时与 C 逐位相同（见验证 1）。

**配置**（`maskdino/config.py`，yaml 里的值）：

- `MODEL.MaskDINO.POINT_NMS_MODE: "prune"`；
- `MODEL.MaskDINO.POINT_NMS_KAPPA`：剔除半径 κ，取配置值（本 arm 0.05，见下文「κ 的变更」）；
- `MODEL.MaskDINO.POINT_NMS_PRUNE_START_ITER: 2500`（warm-up，见下）；
- `MODEL.MaskDINO.TEST.POINT_NMS_KAPPA: 0.0`：推理直接用第 9 层的存活掩码，不再做最后一层 NMS。

**每层的剔除**（`dino_decoder.py` 的 `TransformerDecoder.forward`，第 l = 1..9 层）：

- 在本层框的细化之后，用本层输出（全部 detach）对每个实例的 K=30 个小穗 token 做贪心点 NMS：
  - 分数 = sigmoid(存在性头(本层 token))，与本层输出的存在性 logit 同一个头、同一个输入，fp32 计算；
  - 坐标 = 本层细化后的小穗点；
  - 半径 = κ·sqrt(w·h)，w、h 为本层细化后**预测框**的像素宽高。
    框用 decoder 内部的参考框（`new_reference_points`，即下一层的参考框）。它与该层输出的 `pred_boxes` 只差一个 `decoder_norm`，
    在 E3 最终 checkpoint 的 val 图上，wheat_2 实例两者 sqrt(w·h) 的对数差平均第 1 层 0.6%、之后 < 0.3%；
  - 在 iso 空间算：像素坐标 ÷ sqrt(w_px·h_px)，半径就是 κ；中心距严格小于半径才算重复，分数并列时下标小的优先。
- 只在上一层存活的 token 之间做：死 token 既不能留下，也不能压别人。alive_l = alive_{l-1} ∧ keep_l，alive_0 全为真。
- alive_l 用于两处：
  - 第 l 层的输出掩码：criterion 第 l 层的损失只算 alive_l；推理用第 9 层的；
  - 第 l+1 层起的注意力屏蔽：
    - 实例内 SA（实例 token + K 个 token）：`key_padding_mask` 屏蔽死 token；实例 token 永不屏蔽，不会整行全屏蔽；
    - 同 slot 跨实例 SA（批 = B·K，序列 = 100 个实例）：bool `attn_mask` 屏蔽死 token 作为 key，但保留自身对角，防止整行全屏蔽出 NaN；
    - 死 token 仍作为 query 计算，输出丢弃（DDQ 同款）；可变形 cross-attn 与 FFN 照常。
- 被剔除的 token 不进内层匹配，存在性与坐标损失都不算；留下但没配上的照常判负；损失分母（GT 小穗数）不变；不补回。
  这一部分是 C 里 criterion 的现成逻辑：输出带 `pred_point_keep` 时直接用它。
- 存在性头只作为 forward 参数传入、不注册成子模块：参数量与 state_dict 不变（65,322,300）。
- 批量 NMS 在 GPU 上向量化（`greedy_point_nms_batched`），规则与 C 的 `greedy_point_nms` 完全一致，不做 GPU→CPU 同步。

**warm-up：前 2500 iter 不剔除。**

- 开训时预测框近乎整图，κ·sqrt(w·h) 会大到一次压掉大半 token。实测（验证 3）：随机初始化或 ImageNet R-50 起点上一开训就剔除，
  第 1 层只剩 12%~13% 的 token，有点实例 100% 饿死。
- E3 在 iter 2499 时 segm AP 50.0，框才可用，所以取 2500。
- warm-up 期间 decoder 不剔除、输出不带 `pred_point_keep`，criterion 用所配 GT 框做 NMS，与 arm A **逐位相同**（验证 1）。
- 迭代号取自 `get_event_storage().iter`；推理总是剔除，所以训练中途（含 warm-up 期）的内置评估都是带剔除的。

**与 arm A 的唯一差别**：剔除是否延续到后续层的注意力、是否不可逆，以及推理与 warm-up 之后的训练用预测框作半径（A 训练用 GT 框、推理在最后一层用预测框做一次 NMS）。

## κ 的变更（开跑前）

- 原定 κ = 0.07（中位 val 穗上约 8 px）。
- 2026-10-08，开跑前，用户看完 E3 全量诊断后改为 **κ = 0.05**，两个 arm 相同：
  - κ = 0.07 时，E3 最终 checkpoint 上的推理 NMS 吃掉 5.40 个百分点的 8 px 覆盖，每层压掉 21%~23% 的 token；
  - κ = 0.05 时为 2.15 个百分点、12%~15%（`output_v2/_diag-1008-group-nms/E3_{plain,k005,k007}.json`）。
- 下文所有 κ 都取自配置（0.05），参照数字都按 0.05 重算。

## 与对照的差异

对 E3（`output_v2/exp-0925-mini-group-decoder`）：

| 项 | E3 | 本 arm |
|:--|:--|:--|
| `POINT_NMS_MODE` | none | **prune** |
| `POINT_NMS_KAPPA` | —（none 下不生效） | 0.05 |
| `POINT_NMS_PRUNE_START_ITER` | — | 2500 |
| `TEST.POINT_NMS_KAPPA` | 0 | 0（推理用第 9 层的存活掩码） |

对 arm A（`mini/group-nms-match`）：κ 相同；A 是 `POINT_NMS_MODE: match`、`TEST.POINT_NMS_KAPPA: 0.05`（最后一层 NMS）。

其余全部继承 E3：分组 decoder（100 个实例 query、K=30、iso、ρ c=1）、数据、schedule（12500 iter = 100 epoch，cosine，每 500 iter 评估一次，共 25 次）、`POINT_THRESHOLD` 0.35。
正式 run 除 `OUTPUT_DIR` 外不带任何 CLI 覆盖。

## 实施前已完成的验证

脚本与原始输出：`output_v2/_smoke-1008-group-nms/verify_prune/`（`verify_prune.py`，一次性脚本，不进 git）。GPU 部分都在 qiuhao GPU1（A100）上跑。

1. **逐位等价**（固定种子，mini train 前 2 张图经同一个 mapper，训练态前向 1 次；每个进程只比第一次前向）：
   - 三组比较，随机初始化与 E3 最终 checkpoint 各一次，共 6 组，**每组 127 项 loss（加权前、加权后）全部逐位相等**：
     - mode none：本分支 vs `00a7aaf`；
     - mode match（κ 0.05）：本分支 vs C（`df90505`，临时 detached worktree）；
     - mode prune、在 `EventStorage(2499)` 里（< start_iter 2500）：本分支 vs C 的 mode match。
   - 参数量同为 65,322,300，参数 sha256 相同（prune 不新增参数）。
   - prune 的 warm-up 期输出不带 `pred_point_keep`，`point_nms/*` 两个标量与 C 的 match 相同。
2. **批量 NMS 单测**：`greedy_point_nms_batched` 与「对 valid 子集逐组调 `greedy_point_nms`」逐个比对，CPU 与 CUDA 都测：
   - 1500 次随机用例（62,734 组、148.6 万个点；聚簇与均匀、量化分数与全同分数、25% 的点 valid=False、偶尔整组无效、标量与逐组半径、含 0 与负半径、float32 / float64）：**0 处不一致**；
   - 实际规模（400 组 × 30 点、40% 近重复）20 次、CPU 对 CUDA 300 次：0 处不一致；
   - 手工用例 16 个 × 2 种设备全部符合预期：距离恰好等于半径时不压、略小于时压，并列按下标，链式，中间点无效时两端都留，高分点无效时不压近邻，空输入。
3. **剔除生效**（κ 0.05，`POINT_NMS_PRUNE_START_ITER 0`）：
   - **前向 + 反向**（fp32 与 AMP 各一次）：随机初始化、ImageNet R-50 起点、E3 最终 checkpoint 上，损失全部有限，**735 个可训练参数全部有梯度、全部有限**。
   - **修掉一个 AMP 下的 bug**：最初的实现在 `no_grad` 里经 autocast 调存在性头算分数，autocast 会把权重的 fp16 副本缓存成无梯度版本，
     之后输出头复用这份缓存，AMP 反向时存在性头的 6 个参数**没有梯度**（fp32 下正常）。现在分数在关掉 autocast 的 fp32 下算，AMP 下 735 个参数都有梯度。
   - **开训时**（随机初始化 / R-50，mini train 4 张图）：第 1 层起只剩 13.4% / 12.2% 的 token，有点实例 100% 饿死。这是设置 warm-up 的依据。
   - **E3 最终 checkpoint，训练态**（mini train 前 16 张图经 LSJ，64 个有点实例；同一 batch、同一种子下与不剔除的 match 各前向一次，`PYTORCH_JIT=0`）：

     | 层 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
     |:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
     | 存活比例（全部 query） | 0.960 | 0.920 | 0.899 | 0.883 | 0.870 | 0.865 | 0.863 | 0.860 | 0.859 |
     | 匹配实例上留下的比例：prune | 0.963 | 0.935 | 0.916 | 0.904 | 0.891 | 0.886 | 0.883 | 0.880 | 0.879 |
     | 匹配实例上留下的比例：match | 0.962 | 0.960 | 0.957 | 0.950 | 0.947 | 0.948 | 0.947 | 0.946 | 0.946 |
     | 存在性损失相对 match | −0.0% | +2.4% | +7.1% | +9.3% | +13.3% | +23.8% | +26.7% | +29.9% | +37.0% |
     | 坐标损失相对 match | −0.3% | +1.8% | +2.1% | +3.7% | +2.7% | +1.6% | +2.9% | +3.8% | +4.2% |

     - 饿死比例（留下的 token 少于 Ni 的有点实例）：9 层全部 **0 / 64**。
     - 存活掩码逐层单调（alive_l ⊆ alive_{l-1}）。
     - 非点损失项的相对变化最大 0.03%（实例 token 经零初始化的门接收实例内 SA，E3 的门已不为 0，所以有微小变化）。
     - 点损失上升：E3 没在屏蔽下训练过，屏蔽改变了第 2 层起小穗 token 的特征。正式 run 在 iter 2500 打开剔除时会经历这样一次分布偏移。
     - EventStorage 里的 `point_nms/suppressed_frac` 0.095、`point_nms/starved_frac` 0。
   - **E3 最终 checkpoint，推理态**（val 前 10 张图）：
     - 存活比例（全部 query）第 1→9 层 0.962 → 0.877；输出里 score ≥ 0.4 的 wheat_2（51 个）0.888 → 0.766。
     - 推理输出的点 = 第 9 层存活 ∧ 分数 > 0.35：75 / 75 个 wheat_2 实例逐点一致。
     - 独立重算：按 decoder 的 iso 算术、逐实例调 C 的 `greedy_point_nms`、以上一层的存活为 valid，**9 层 27 万个 token 与 decoder 全部一致**。
       改用像素空间重算时有 69 个 token 不一致，全部在 sqrt(w_px·h_px) < 1 px 的退化背景框上（框缩成图像角上 0.014 px，半径小于 float32 坐标的分辨率的 10 倍）；
       框 ≥ 1 px 的 8,552 个实例层上 0 处不一致。
     - 同一批 wheat_2 输出 1096 个点；同一次前向不按存活过滤为 1130 个；不剔除的 E3 原样前向为 1276 个。
4. **诊断脚本冒烟**：`point_token_diag.py --sweep --run-dir <E3> --limit 5 MODEL.MaskDINO.POINT_NMS_MODE prune`（当时 κ 0.07）跑通，
   截到了逐层 `pred_point_keep`（`match_keep_source = pred_point_keep`），flip / auc_surv / coverage / readout 各字段正常。
   κ 0.05 的全量版就是下面的 E3 参照 1。
5. **冒烟训练**（代码 `5f4d71b`；正式 yaml + `SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 1000 MODEL.MaskDINO.POINT_NMS_PRUNE_START_ITER 20`；
   输出 `output_v2/_smoke-1008-group-nms/smoke_prune/`，汇总 `smoke_prune/smoke_summary.json`）：
   - 40 iter 与 iter 19 / 39 两次完整 val 评估都跑通，0 个 Traceback；`metrics.json` 两行训练记录里的 254 个 loss 值全部有限。
     未训练模型的评估指标全为 0（AP 0、matched_F1 0、预测点 0，没有配对穗所以 count_MAE 为 nan），与 E3 冒烟相同。
   - 前 20 iter 不剔除（warm-up），后 20 iter 剔除：

     | | iter 2~19（warm-up） | iter 20~39（剔除） |
     |:--|--:|--:|
     | s/iter 平均 / 中位 | 4.31 / 4.03 | 4.66 / 4.69 |
     | `point_nms/suppressed_frac` | 0.081（criterion 用 GT 框做的 NMS） | 0.888 |
     | `point_nms/starved_frac` | 0 | 1.0 |
     | `loss_point_cls`（加权后） | 0.623 | 0.074 |

     - 两段都与 arm A 的正式 run（`exp-1008-mini-group-nms-match`，17:09 起）同卡并发，前一段时它还在启动，
       所以后一段多出的 0.35~0.66 s/iter 是剔除开销的上界。
     - 剔除段 89% 的 token 被剔、有点实例全部饿死，存在性损失只剩存活 token 的那部分（分母不变），所以数值骤降，不是学到了东西。
       未训练模型的预测框近乎整图，这正是设 warm-up 的原因（验证 3 的开训统计同理）。
   - 峰值显存 max_mem 26,195 MB（warm-up 段 25,797 MB）；E3 冒烟为 25.9 GB，剔除约多 0.3 GB。nvidia-smi 上本进程占 33.6 GB。
   - 落盘的 `config.yaml` 与 E3 run 的逐行 diff：除 `OUTPUT_DIR` 与冒烟覆盖的 `MAX_ITER` / `EVAL_PERIOD` / `CHECKPOINT_PERIOD` 外，只多出四个键：
     `POINT_NMS_MODE: prune`、`POINT_NMS_KAPPA: 0.05`、`POINT_NMS_PRUNE_START_ITER: 20`（冒烟覆盖，yaml 里是 2500）、`TEST.POINT_NMS_KAPPA: 0.0`。
   - **第一次冒烟 OOM，与代码无关**：
     - 原因：16:54 启动时 GPU1 空闲 42.7 GB（当时的门槛是 40 GB）。随后 arm A 的 4 个评估进程与另一个 5.4 GB 的任务同时起来，
       iter 0 在 mask 损失的点采样处 OOM：本进程占 23 GB，卡上只剩 0.85 GB。
     - 处理：输出改名留作记录（`smoke_prune_oom_attempt1*`）。重跑等本 arm 的两份参照跑完、空闲显存连续 60 s ≥ 45 GB 才起；
       17:08 起跑时卡上空闲 81 GB，跑通（上面的数字都来自这次）。

## E3 的参照（零训练，κ 0.05，开跑前测得）

B 的推理过程是 decoder 里的逐层剔除，与 A 的最后一层 NMS 不同，所以 B 用自己的代码对 E3 权重开剔除，重算两份参照。

**参照 1：诊断**（E3 最终 checkpoint，全 val，`point_token_diag.py --sweep --run-dir <E3> MODEL.MaskDINO.POINT_NMS_MODE prune MODEL.MaskDINO.POINT_NMS_KAPPA 0.05`；
输出 `output_v2/_diag-1008-group-nms/E3_prune_k005.json`）。与 arm A 的三档并列（同一脚本、同一口径；σ = 8 px）：

| | 不做 NMS | 最后一层 NMS κ 0.05（A 的推理） | 最后一层 NMS κ 0.07 | **decoder 剔除 κ 0.05（B 的推理）** |
|:--|--:|--:|--:|--:|
| 文件 | `E3_plain.json` | `E3_k005.json` | `E3_k007.json` | `E3_prune_k005.json` |
| 逐层配对用的存活掩码 | 无 | criterion 用 GT 框做的 NMS | 同左 | decoder 截到的存活掩码 |
| 相邻层改判率平均（第 1→9 层） | 0.0271 | 0.0319 | 0.0342 | **0.0352** |
| 改判率 1→2 / 4→5 / 8→9 | 0.084 / 0.019 / 0.007 | 0.097 / 0.024 / 0.009 | 0.095 / 0.025 / 0.011 | 0.095 / 0.026 / 0.012 |
| 匹配实例上被压 / 剔除的比例（第 1 层 → 第 9 层） | — | 12.0% → 14.8% | 21.4% → 23.4% | 12.0% → 24.5%（累积） |
| 饿死比例（最大一层） | — | 0.17% | 0.52% | 0.34% |
| 存在性 AUC（全部 K 个 token） | 0.6989 | 0.6989 | 0.6989 | 0.7003 |
| **AUC_surv**（第 9 层留下的 token） | 0.6989 | 0.6905 | 0.6745 | **0.6664** |
| 留下的 token 比例（第 9 层） | 1.000 | 0.850 | 0.765 | 0.755 |
| 8 px 覆盖：全部 K / 留下的 | 0.7914 / 0.7914 | 0.7914 / 0.7699 | 0.7914 / 0.7374 | 0.7789 / 0.7301 |
| **覆盖损失**（百分点） | 0 | 2.15 | 5.40 | **4.88** |
| 读数：留下 ∧ 分数 > 0.35 的 `matched_F1@8` | 0.6378 | 0.6416 | 0.6404 | 0.5977 |
| 同一次前向只按分数 > 0.35 的 `matched_F1@8` | 0.6378 | 0.6378 | 0.6378 | 0.5993 |
| 读数的 `count_MAE_spike` | 2.764 | 2.597 | 2.449 | 1.620 |
| 偏差 / r / 去偏移 MAE | +2.53 / 0.552 / 1.61 | +2.27 / 0.551 / 1.57 | +2.04 / 0.536 / 1.58 | −0.23 / 0.580 / 1.64 |
| 预测点数（581 个配对穗） | 10734 | 10579 | 10449 | 9129 |

- 只看最终 checkpoint、配对上的 581 个穗；读数一列是诊断脚本按模型自己的读数规则重算的，前三列与 arm A 的结果相同。
- E3 的权重没在屏蔽下训练过：直接开剔除，第 2 层起小穗 token 的特征就变了。同一次前向不看存活、只按分数，F1@8 已从 0.638 降到 0.599，
  全部 K 个 token 的 8 px 覆盖也从 0.791 降到 0.779。所以这一列是「构造 + 分布偏移」的参照，不是「训练后的 B」的预期。
- 剔除是累积的：第 9 层的 token 已经过 9 次 NMS，留下的比例（0.755）低于单层做一次 κ 0.05 的 NMS（0.850）。

**参照 2：E3 + 剔除推理的读数**（E3 后半程 13 个 checkpoint 的 `--eval_only`，同样两个 opts；
输出 `output_v2/_eval-1008-group-decoder-prunek005/summary.json`，读数规则同下）：

| 读数 iter | count_MAE_spike | matched_F1@8 | spike_recall | segm/AP | 偏差 / r / 去偏移 MAE | 后半程峰值 segm/AP / spike_recall |
|--:|--:|--:|--:|--:|:--|:--|
| 10499 | **1.535** | **0.6007** | 0.9445 | 77.50 | −0.12 / 0.597 / 1.56 | 78.35（9999）/ 0.9657（7999） |

- 13 次逐个对比 E3 同 iter（不剔除）：
  - `matched_F1@8` 每次都更低，差 −0.026 ~ −0.055，平均 −0.043（剔除 0.579 ~ 0.607，E3 0.621 ~ 0.646）；
  - `count_MAE_spike` 每次都更低（1.54 ~ 2.00，E3 2.42 ~ 3.70），因为少出了约 15% 的点（13 次平均预测点数 13,798 对 16,215），偏差从 +2 以上降到 0 附近。
- 实例级指标几乎不动：`segm/AP`、`bbox/AP` 与 E3 同 iter 最大差 0.03 / 0.04，`spike_recall` 最大差 0.0016。
  剔除经实例内 SA 的门影响实例 token，所以不逐位相同（arm A 的最后一层 NMS 则逐位相同）。
- 末次（final）评估：count 1.620、偏差 −0.23、r 0.580，与参照 1 诊断脚本的重算逐位一致。
- 读法：在 E3 的权重上，剔除推理是「少出点、但挑得更差」，计数靠的是偏差变小，F1 是下降的。

## 预期与判否线（跑之前写死，2026-10-08）

**读数规则**（用户的硬性规定）：

- 一律用默认读数：训练内置评估，`POINT_THRESHOLD` 0.35。不做阈值扫描，单种子。
- 判据在同一次评估上读：取后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估，该次的全部指标都从这次读。
- 护栏（`segm/AP`、`spike_recall`）用全程峰值。
- 计数附有符号偏差、Pearson r、去偏移 MAE（由该次 `inference/spikelet_pair_counts.csv` 算）。
- 本 arm 的推理总是剔除，所以内置评估的点指标本身就是带剔除的读数。

### 主判据（对 E0）

E0：`output_v2/exp-0923-mini-ep100`，读数那次 iter 8999：`count_MAE_spike` 2.465、`matched_F1@8` 0.6228、`spike_recall` 0.9511、`segm/AP` 77.40；
全程峰值 `segm/AP` 78.80、`spike_recall` 0.9511。沿用本批对 E0 的线：

| 判据 | 线 |
|:--|:--|
| **主判据** 读数那次 `matched_F1@8` | ≥ **0.6428** 成立；≤ **0.6228** 判否；介于两者之间为歧义 |
| **次判据** 读数那次 `count_MAE_spike` | ≤ **2.21** 成立 |
| 护栏 `segm/AP` 全程峰值 | ≥ **77.80** |
| 护栏 `spike_recall` 全程峰值 | ≥ **0.9311** |

### 机制（最终 checkpoint，`point_token_diag.py`，口径同参照 1；必须传不带值的 `--sweep`，放在最前面）

本 arm 的诊断在 prune 模式下跑：逐层的配对用截到的存活掩码，AUC_surv 只看第 9 层存活的 token，覆盖损失 = 全部 K 个 token 的 8 px 覆盖 − 存活 token 的 8 px 覆盖。

- **M1（匹配稳定，用户的假设）**：第 1→9 层相邻层内层改判率的平均 ≤ **0.0190**（0.7 × E3 不做 NMS 的 0.0271，`E3_plain.json`）。
  不过就判「机制未启动」，端到端差异不归因于匹配变稳。
  同时报参照 1 里 E3 开剔除的改判率 0.0352（E3 不做 NMS 为 0.0271，最后一层 NMS κ 0.05 为 0.0319）：这是构造本身（剔除后再配）带来的部分，不是训练的效果。
- **M2（挑点）**：AUC_surv ≥ **0.7164**（参照 1 里 E3 开剔除的 AUC_surv 0.6664 + 0.05）。
- **护栏 G（覆盖）**：剔除造成的 8 px 覆盖损失 ≤ **4.88** 个百分点（参照 1 里 E3 开剔除的覆盖损失）。
  - 修订说明：原计划是绝对线 ≤ 1 个百分点。E3 自己在 κ 0.05 的最后一层 NMS 下就损失 2.15 个百分点、开剔除时 4.88 个百分点，
    绝对线等于事先注定不过。用户 2026-10-08（开跑前）改为相对线：不比 E3 套上同样的推理过程时更差。

### 归因（预注册，不作判据）

- **对「E3 + 剔除推理」（参照 2）**：读数那次 `matched_F1@8` ≥ 参照 2 + 0.02，或 `count_MAE_spike` ≤ 参照 2 × 0.9，记「训练侧有贡献」；
  否则记「收益来自推理侧的剔除」。具体的线：`matched_F1@8` ≥ **0.6207**（0.6007 + 0.02），或 `count_MAE_spike` ≤ **1.3819**（1.5354 × 0.9）。
- **A 对 B**：两 arm 读数那次的 `matched_F1@8` 差在噪声地板 0.015 以内，记「两种作用范围无差别」。
- 并列列出 arm A 的公共参照「E3 + 最后一层 NMS κ 0.05」（`output_v2/_eval-1008-group-decoder-testnms-k005/summary.json`，arm A 用 `df90505` 生成）：
  读数 iter 9499，count 2.312、F1@8 0.6348、spike_recall 0.9364、segm AP 77.40，偏差 +1.82、r 0.501、去偏移 MAE 1.61。
  它与本 arm 的推理过程不同，只作两个 arm 共同的参照，不进本 arm 的归因判定。

### 参照（不作判据）

| | 读数 iter | count_MAE_spike | matched_F1@8 | spike_recall | segm/AP | 来源 |
|:--|--:|--:|--:|--:|--:|:--|
| E0（主对照） | 8999 | 2.465 | 0.6228 | 0.9511 | 77.40 | `exp-0923-mini-ep100/metrics.json` |
| E3 | 9499 | 2.422 | 0.6326 | 0.9364 | 77.40 | `exp-0925-mini-group-decoder/metrics.json` |
| E3 + 最后一层 NMS κ 0.05（arm A 的推理） | 9499 | 2.312 | 0.6348 | 0.9364 | 77.40 | `_eval-1008-group-decoder-testnms-k005/summary.json` |
| E3 + decoder 剔除 κ 0.05（本 arm 的推理，参照 2） | 10499 | 1.535 | 0.6007 | 0.9445 | 77.50 | `_eval-1008-group-decoder-prunek005/summary.json` |

- 穗长直线（不训练）：每穗点数 MAE 1.43、r 0.635。截至 10-08 只有 `mini/count-residual` 的默认读数赢过它。

### 预期

- 按 DDQ 的 DETR 版结果，效应小。
- 计数大概率过次判据线，而且主要来自推理侧：E3 默认读数每穗多报 +2.53，开剔除本身就少出约 15% 的点；
  参照 2 里 E3 开剔除推理的计数读数已是 1.535（线是 2.21），训练侧要到 1.3819 以下才算有贡献。
- 主判据不乐观：参照 2 里 E3 开剔除推理的 F1@8 是 0.6007，已低于判否线 0.6228。要看训练能否把屏蔽带来的 F1 损失追回来，
  落在歧义区或判否的可能都不小。
- E3 权重直接开剔除时点损失上升（验证 3），所以 iter 2500 打开剔除后 F1 可能先降、再追回。判据只看后半程的读数。

## 归因限制

- 单种子，种子方差未测。
- warm-up 期（前 2500 iter）两个 arm 完全相同（逐位，见验证 1），差异从 iter 2500 起才出现。
- 推理侧的剔除与训练侧的改动绑在一起，靠参照 2（E3 + 剔除推理）拆开。
- 剔除改变前向：死 token 在注意力里被屏蔽，实例 token 经实例内 SA 也会受影响。所以 mask / 框 / 实例分数的变化不能全部归到「点的监督」上。
- 训练后期与推理的剔除半径来自预测框：框不准时半径跟着偏；A 在训练时用 GT 框。
- 不防饿死：E3 checkpoint 上实测为 0，但训练中途若出现，就按设计让多出的 GT 小穗那一层没有配对（每次迭代写进 `point_nms/starved_frac`）。

## 成本

- 12500 iter（100 epoch）。冒烟剔除段 4.66 s/iter（与 arm A 的正式 run 同卡并发）；按此全程约 16 h。
  E3 的 100 epoch 实测中位 3.80 s/iter（也与其它 run 并发），加上剔除开销，估计 14~16 h。
- 另加 25 次评估：冒烟里每次约 4.4 min（推理 2.6 min + COCO / 小穗点评估，同卡并发），共约 1.5~2 h。
- 峰值显存 max_mem 约 26.2 GB，nvidia-smi 约 34 GB，只能用 qiuhao 的 A100（GPU1）。
  冒烟时与 arm A 的正式 run（nvidia-smi 33.6 → 36.8 GB）合计约 70 GB。
- 每次存 checkpoint 785 MB × 25 个，约 20 GB 磁盘。
