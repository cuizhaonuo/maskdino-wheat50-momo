# 想法：每穗计数交给「密度图在预测 mask 内的积分」

状态：**预注册，待跑** ｜ 提出 2026-10-22 ｜ 分支 `mini/density-count`（从 `mini/baseline`@`0f71161` 开出，100 epoch）｜ 配置 `wheat_joint_single_stage_mini_density_count.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜ 输出 `output_v2/exp-<启动日>-mini-density-count`（启动日由主 agent 排卡时定）

> 本批「计数改进三臂」（2026-10-22，mini）之一：L `mini/count-residual-detach`、D `mini/density-count`（本臂）、K `mini/coral-ecount`。
> 三臂共用一组判定线：下文「统一判定线」一节照抄批次说明第 4 节（`output_v2/_batch-1022-count/AGENT_BRIEF.md`）。
> 机制门、归因、密度图自身的读数是本臂另写的。本臂的父臂就是 E0（`mini/baseline`）。

## 来源

- 批次说明（主 agent，`output_v2/_batch-1022-count/AGENT_BRIEF.md`）与主 agent 给本臂的设计。本文在设计上做了几处细化，理由见「与主 agent 设计的差异」。
- 现有计数都是从实例 query 的 256 维向量「回归一个数」：存在性头数点（E0）、CORAL（`mini/coral-exist`、`mini/coral-count`）、残差计数头（`mini/count-residual`）。
- 密度图计数换了一条路：每个小穗贡献一份质量，在实例区域内求和。参照是 `compare/` 里的 DM-Count（见「现象」第 3 条）。

## 现象

**1. 回归式计数的天花板。**

- 完整集（2092 图）2³ 消融里只有 CORAL 存在性头治计数：2.77 → 1.32~1.41，只和不训练的穗长直线打平（同批穗 1.41）。
  - 剩下的误差有两处：r 只有 0.71~0.74；读数被压缩，E[GT | 预测] 斜率 1.36（直线 1.02）。
- mini 上：E0 默认读数 2.4648（最终评估偏差 +2.40、r 0.642）；`mini/coral-exist` 1.6655；
  `mini/count-residual` 1.146、r 0.747、偏差 +0.02，是本项目计数最好的结果，但计数损失经共享 decoder 回传，segm AP 后半程平均 −2.05，三条护栏都没过。

**2. 不训练的穗长直线。**

- 点数 = 8.924 + 0.0345 × 框对角线（mini train 拟合）。E0 最终 checkpoint 的 578 个配对穗上（预测框）：
  MAE **1.4187**、偏差 −0.197、r 0.615（本分支 eval-dens0 的 `line_count` 列复算，与 `mini/count-residual` 的 eval-line 相同）。

**3. 密度图在 GT mask 内积分能数到 1.2~1.4。**

- `compare/experiments/DM-Count-foreground-rerun-20261022`：VGG19 密度模型，只用 wheat_2 点整图训练，val 上在 **GT mask** 内积分（不取整）。
  批次说明引的是 557 个 TP 穗上 1.26。该 run 末 12 次验证（epoch 28~50）：

  | 口径 | MAE 范围 | 最好一次（epoch 42） |
  |:--|--:|--:|
  | 557 个 TP 穗，GT mask 内积分 | 1.22~1.44 | 1.2216 |
  | 全部 613 个 wheat_2，GT mask 内积分 | 1.25~1.46 | 1.2549 |
  | 整图积分 vs 整图点数（200 图） | 2.67~3.46 | 2.669 |

- 它用的是 GT mask：检测漏检、mask 边界误差都不在里面，与本仓库「在预测实例里数」不等价，只作参照。

## 假设

1. pixel decoder 的 `mask_features`（1/4 分辨率、256 通道；mask logit 就是 query 嵌入与它的内积）里有足够的局部信息定位小穗。
   接一个小卷积头就能把它变成密度图 D（每个 1/4 像素上的小穗数）。
2. 每穗计数 = D 在该实例预测 mask 内的和。计数从「一个向量回归一个数」变成「局部质量求和」：
   误差来自局部密度误差和 mask 边界，不再有全局回归的压缩与偏差。
3. 因此每穗计数与 GT 的 r 应高于穗长直线（机制门 +0.03），主读数应 ≤ 1.28。

## 修法

所有开关默认关闭。关着时与 `mini/baseline` 逐位等价（见「实施前已完成的验证」）。

| 位置 | 改动 |
|:--|:--|
| `maskdino/config.py` | 新增 `POINT_DENSITY_HEAD`（False）、`POINT_DENSITY_SIGMA`（1.0，单位 1/4 像素）、`POINT_DENSITY_WEIGHT`（1.0）、`POINT_DENSITY_COUNT_WEIGHT`（1.0）、`POINT_DENSITY_COUNT_MIN_IOU`（0.5）、`POINT_DENSITY_DETACH`（False）、`POINT_COUNT_LINE`（[8.924, 0.0345]，只用于明细里的 `line_count` 列）、`TEST.POINT_COUNT_SOURCE`（`points` / `density`，默认 `points`） |
| decoder | `PointDensityHead`：3×3 conv 256→128 + GN + ReLU → 3×3 conv 128→64 + GN + ReLU → 1×1 conv → softplus（fp32）。读 `forward` 的 `mask_features`，输出 `pred_density`（[B, H/4, W/4]，与 `pred_masks` 同一网格），每图一份、不分层。建在 `__init__` 末尾并用 `torch.random.fork_rng` 包住。输出层 weight ~ N(0, 0.01)，bias 先验初始化为 softplus⁻¹(8.75e-4)。`POINT_DENSITY_DETACH=True` 时读 `mask_features.detach()`（本次不开） |
| LSJ mapper | 降级（`DEMOTE_BORDER_TRUNCATED`）的实例另记一个 bool 字段 `gt_demoted`。只多挂一个字段，不消耗随机数；密度头关着时没有代码读它 |
| `maskdino.py` `prepare_targets` | 密度头打开时给 targets 加：`density_points`（全部 GT wheat_2 小穗点，增广后像素坐标）、`point_counts`（每实例 Ni）、`demoted` |
| criterion | 新增两项损失，**只在最后一层算一次**（不进 `self.losses`，不随 interm / aux 复制；权重键也不带后缀）。见下 |
| 推理 | 在 1/4 分辨率、mask 上采样之前算每个 query 的 n̂_q = Σ D ⊙ 1[mask_logit_q > 0]，随 `instance_inference` 的 topk 选出。`density` 读法下 wheat_2 的 `pred_spikelet_counts` = round(n̂)，clamp ≥ 0；未取整的 n̂ 与穗长直线一并挂出。小穗点照旧按存在性 > 0.35 输出。`points` 读法一行不改。密度头打开时每图另输出整图 Σ D 与映射回原图尺寸的密度图（只给评估器） |
| 评估器 | `count_MAE_spike`、`count_MAE_img`（含 RMSE）、两份 CSV 的计数改读 `pred_spikelet_counts`；点级 P/R/F1、`n_pred_points`、`fp_point_*` 仍按实际输出的点算。`density` 读法下逐穗明细多 `out_points`（实际点数）、`line_count`（穗长直线，预测框，取整）、`density_raw`（未取整的密度计数）三列。有密度图时多三项指标 `density_MAE_img`、`density_gtmask_MAE_spike`、`density_gtmask_MAE_all`，逐穗明细多 `gtmask_density`、逐图明细多 `density_sum` 列（定义见「密度图自身的读数」）。改法参考 `mini/count-residual` 的评估器（`60d21a3`），在本分支重写，没有 cherry-pick |
| `train_net.py` | `_POINT_HEAD_KEYS` 加 `point_density_head`（只在 `FREEZE_EXCEPT_POINT_HEAD` 时用到，本臂不用） |
| `myscripts/infer/seg_infer_reg.py` | **不改**：`infer_image` 已优先读 `pred_spikelet_counts`，计数跟着配置的读法走 |

两项损失（`criterion.loss_density`）：

- **`loss_density`（稠密）**：Σ_像素 (D − T)²，每图求和、对 batch 取平均。
  - T：增广后图像里每个 GT wheat_2 小穗点在 1/4 网格上放一个 σ = 1 格的高斯。截断到 ±3 格，出图的格子截掉后按窗口总和重新归一，**每个点的质量恰为 1**，目标总质量 = 该图点数。
  - 坐标约定与 mask 上采样（bilinear、align_corners=False）一致：增广图连续坐标 x 落在网格坐标 x / 4，格子 j 的中心在 j + 0.5。
  - wheat 类穗按数据集定义没有点，目标为 0（DM-Count 对比模型也是这样整图训练的）。
  - LSJ 截断降级的穗：mapper 已清空它们的点，可见的小穗在目标里成了 0。按 `gt_demoted` 识别，在其 GT mask（1/4 网格上 max-pool，再外扩 1 格覆盖 σ 的溢出）内**不计损失**。
- **`loss_density_count`（实例计数）**：最后一层外层匹配上的 wheat_2（Ni ≥ 1）上 |n̂ − Ni|，n̂ = Σ D ⊙ σ(mask_logit.detach())。
  - mask_logit 取同一 query 最后一层 1/4 网格上的 logit，**detach**：只当积分区域，计数损失不给 mask 头、decoder 任何梯度。
  - 只算预测 mask（logit > 0）与 GT mask（1/4 网格上面积占比 ≥ 0.5）IoU ≥ 0.5 的对。
  - 除以 batch 里 wheat_2 且 Ni ≥ 1 的 GT 实例数（跨卡做法同 `num_masks`），所以 mask 还没学好时这一项自然很小，随 mask 变好逐步加大。
- 两项都默认回传进 pixel decoder 与 backbone（多任务）；只有 mask 那一路 detach。全程 fp32（D 本身 fp32，mask logit 转 fp32，只用逐元素运算）。

## 与主 agent 设计的差异（细化及理由）

照原设计未改的：σ ≈ 1 个 1/4 像素且截断后重新归一；softplus 非负输出；mask 一律 detach；训练用 σ(mask_logit) 的软积分、推理用硬 mask（logit > 0）、都在 1/4 分辨率；读法键 `TEST.POINT_COUNT_SOURCE`；`POINT_DENSITY_DETACH` 默认关；全部开关命名。

1. **计数损失加 IoU 筛选（`POINT_DENSITY_COUNT_MIN_IOU` 0.5）。**
   - 开训时 mask 是乱的。固定 batch 实测（ImageNet 初值）：6 个匹配上的 wheat_2，其 mask 覆盖全图 65,536 格里的 46,460~64,866 格，与 GT 的 IoU 只有 0.001~0.031。
     不筛的话软积分 n̂ ≈ 34~39（GT 17~23），计数损失会在 mask 学好之前（E0 的 segm AP 在 iter 999 才 5.96、iter 1499 才 30.2）一直把 D 往下压。
   - 评估器只数 mask IoU ≥ 0.5 的配对穗，筛选让训练与评估同一口径。
   - 分母用全部 wheat_2（Ni ≥ 1）的 GT 数，而不是过线的个数：这一项随 mask 变好自然加大，不会在早期由少数几个穗主导。
   - E0 的 `model_final.pth` 下同一 batch 6/6 过线（IoU 0.84~0.99）。
2. **计数损失只在最后一层算，interm / aux 不算。** D 每图只有一份；mask 已 detach，深监督不训练中间层，只会多出几份用较差 mask 积分的约束。推理也只用最后一层。
3. **建密度头时用 `fork_rng` 包住。** 建完后全局 CPU 随机数状态与开关关闭时相同（实测哈希相同）。DataLoader worker 的种子取自全局 CPU RNG，所以打开时的数据增广随机流与 E0 相同。GPU 反向的非确定性仍会让曲线分叉，这只是代码等价性的论据。
4. **输出层 bias 先验初始化。** softplus(0) = 0.69 / 格，比真实平均密度高约 790 倍。按批次的经验，标量 bias 在这个 schedule 下全程最多走约 0.6（累积 lr 0.613），所以 bias 必须一开始就放在先验上：
   softplus⁻¹(8.75e-4)，8.75e-4 = mini train 经 LSJ mapper（含降级）500 次采样的平均 57.4 点 ÷ 256² 格（`stats/lsj_stats.json`）。输出层 weight 用 N(0, 0.01) 而不是零：各层从第 0 步起都有梯度。
5. **稠密损失每图求和、batch 平均**（而不是全部像素平均）。量级约 10 / 图，权重取 O(1) 就够；「像素求和 = 小穗数」的含义也直接。
6. **隐层带 GN**（bs4，不用 BN），不依赖 `mask_features` 的尺度。

## 与对照的差异

- **主对照 E0**：`output_v2/exp-0923-mini-ep100`（`mini/baseline`@`0f71161`）。同数据、同起点（ImageNet R-50）、同 schedule、同 `SEED` 42。
  - 解析后的 cfg 与 `wheat_joint_single_stage_mini_ep100.yaml` 逐项比对（CPU，332 个叶子键），**只差三个键**：
    `POINT_DENSITY_HEAD` False → True、`POINT_DENSITY_WEIGHT` 1.0 → 2.0、`TEST.POINT_COUNT_SOURCE` points → density。
    配置里另写的 `POINT_DENSITY_SIGMA`、`POINT_DENSITY_COUNT_WEIGHT`、`POINT_DENSITY_COUNT_MIN_IOU`、`POINT_DENSITY_DETACH` 都等于默认值。
  - E0 run 落盘的 `config.yaml` 与 ep100 配置在本分支代码下解析，只差 `OUTPUT_DIR`，以及本分支新增、E0 落盘时还没有的 8 个键（都取默认值）。
- 新增参数 **369,089** 个（`point_density_head.*`，8 个张量）。
- 内置评估的 `count_MAE_spike` 读密度计数。`matched_F1@σ` 等点指标仍按存在性 > 0.35 输出的点算，与 E0 同一读法。
- 归因参照：父臂就是 E0，与主对照合一（见「归因」）。

**权重怎么定的**（批次说明：两项合计约占总损失 3%~5%，参照 `mini/count-residual` 计数损失的初始占比 3.2%）：

- 开训时计数项被 IoU 筛选整体挡掉，恰为 0（第 1 条差异）。两项合计的初始占比就是稠密项的占比。
- 稠密项未乘权重约 9.9 / 图：固定 2 图 batch 9.898；LSJ 500 次采样 ΣT² 均值 9.95 / 图（每点 0.174，小尺度下点挤在一起，高于孤立点的 1/(4π) = 0.080）。
- `POINT_DENSITY_WEIGHT` **2.0**：固定 batch 占 3.25%（19.80 / 608.80）；冒烟 bs4 两个窗口占 3.77% / 3.42%。落在 3%~5%。
- `POINT_DENSITY_COUNT_WEIGHT` **1.0**：每穗 |n̂ − Ni| 单位权重，只算一层。量级参照：E0 权重下（mask 已学好、密度头未训练、n̂ ≈ 0）同一 batch 为 18.17，约等于平均 Ni；训练后应降到 1~2。

## 实施前已完成的验证

脚本与产物都在 `output_v2/_smoke-1022-density-count/`（`scripts/`、`stats/`、`equiv/`、`eval-points/`、`eval-dens0/`、`smoke40/`、`bench_step.json`）。

**1. LSJ 统计与目标质量**（10-22，CPU，`scripts/lsj_stats.py`，`stats/lsj_stats.json`）

- mini train 经正式配置的 LSJ mapper 采样 500 次：每图 wheat_2 点数均值 57.4（中位 61.5、p90 97）；每图 3.67 个 wheat_2、0.95 个被降级，**降级占 wheat_2 的 20.6%**。
- 渲染出的目标总质量与点数的最大绝对差 1.5e-5（fp32 累加）。

**2. 开关关闭时与 E0 代码逐位等价**（10-22 15:12~15:14，xuyu A100 GPU1，`scripts/equiv_test.py`、`run_equiv.sh`、`compare.py`，结果见 `equiv/compare.txt`）

- 做法同 `mini/count-residual` 的 equiv_test：固定 random / numpy / torch 种子（建模型 0、mapper 1、前向 2），cudnn deterministic；载 ImageNet R-50；
  同一份 batch（mini train 前 2 张图经 LSJ mapper，85 个实例、6 个 wheat_2，Ni = 17 / 19、18、17、23、17，降级 3 + 1 个）；训练态前向 + 反向一次。
  `mini-baseline` worktree（`0f71161`）与本 worktree 各跑一次，ep100 配置。fp32 与 AMP（`torch.autocast` fp16）各一遍。
- **127 项 loss 逐位相等**（总和 fp32 588.9991957247257，AMP 589.0292835533619，两边相同；fp32 值与 `mini/count-residual` 当时的基点相同）。
- **参数量同为 52,300,433**；state_dict 732 个键逐张量哈希相同；建完模型后的全局 CPU RNG 状态相同。
- 有梯度（有限且非零）的可训练张量：fp32 与 AMP 都是 **404 / 409**，两边相同。没有梯度的 5 个是 `pixel_decoder.transformer.level_embed` 与 `decoder.bbox_embed.0.layers.{0,1}.{weight,bias}`：
  第 0 步它们下游是零初始化的层（MSDeformAttn 的 `sampling_offsets` / `attention_weights`、`bbox_embed` 末层），梯度恰为 0，基点同样如此。
  （AMP 下若把损失乘 2^10 以上再反向，fp16 会溢出出 10 个 inf / nan 张量，基点同样如此；表里的 AMP 数是不乘缩放的结果。）

**3. 开关打开**（同一脚本加 `--on`，同一份 batch，正式配置；10-22 19:54 重跑一遍确认，与 15:12 的结果只差权重缩放）

- 参数 52,669,522（+369,089）。**已有 732 个张量的初值与关闭时逐位相同**；建完模型后的全局 CPU RNG 状态也与关闭时相同（`fork_rng` 生效）。
- **原有 127 项 loss 逐位相同**，多出两项：`loss_density`、`loss_density_count`。
- D 形状 [2, 256, 256]、fp32，与 `pred_masks` 同一网格；第 0 步 D 在 7.8e-4~1.13e-3 之间，每图 Σ D 59.4 / 59.9（先验 57.4）。
- **稠密目标总质量 = 该图 GT wheat_2 点数**：17.000000 / 94.000000。与一份独立的 numpy float64 逐点实现逐像素比对，最大差 4.5e-8。
  降级穗的忽略区：图 0（3 个降级）1719 格、图 1（1 个）84 格。
- **手算核对**：`loss_density`（未乘权重）criterion 9.898037 = 手算 9.898037。
  `loss_density_count`：ImageNet 初值下 6 个穗 IoU 全在 0.031 以下、全部弃用，criterion = 手算 = 0；
  E0 `model_final.pth` 下 6/6 过线，拦截 criterion 实际用的最后一层匹配，float64 手算 **18.166580 = criterion 18.166580**，分母同为 6。
  该权重下软积分与硬 mask 积分几乎相同（如 1.554 vs 1.555、0.097 vs 0.097），训练与推理两种积分口径的差在这里 < 1%。
- **梯度隔离**（E0 权重，计数项非零）：
  - 只反传 `loss_density_count`：有梯度的只有密度头 8/8、pixel decoder 121/121、backbone 53/53；mask_embed 0/6、decoder 各层 0/208、其余输出头 0/21。
  - 只反传 `loss_density`：同样只有这三组有梯度。
  - ImageNet 初值下只反传 `loss_density`：密度头 8/8、pixel decoder 120/121、backbone 53/53，其余为 0；`loss_density_count` 为 0，没有任何梯度。
- **fp32 与 AMP**：有梯度的可训练张量都是 **412 / 417** = 关闭时的 404 + 密度头 8，没有梯度的仍是上面那 5 个，没有非有限梯度。
  AMP 下没有任何头在 no_grad 里预调用，autocast 缓存的问题不存在；密度头 8 个张量在 AMP 下都有梯度。
- **初始梯度量级**（只作参照）：
  - ImageNet 初值：稠密项在 pixel decoder 上的梯度范数 0.71，全部损失 3817；backbone 0.92 vs 4096。
  - E0 权重 + 未训练的密度头：两项密度损失的损失值占总损失的 50%（37.95 / 75.60），但在 pixel decoder 上的梯度范数只有 0.24（稠密）+ 0.28（计数），全部损失 77.7；backbone 0.15 + 0.13 vs 88.3。
  - 密度损失进共享特征的梯度要经过密度头的权重，开训时输出层很小，所以「损失值占比」高估了它对共享参数的作用；头的权重长大后这一路会变大，训练中不记录。

**4. eval_only 核对**（10-22 15:14~15:17，xuyu A100 GPU1，`scripts/run_evals.sh`、`eval_dump.py`、`compare_evals.py`，结果见 `evals_compare.txt`）

E0 的 `model_final.pth` 配它自己的 `config.yaml`，在本 worktree 代码上 `--eval_only`（`eval_dump.py` 直接调 `train_net.main`，与 `train_net.py --eval_only` 同一条路径，另把结果按全精度存成 JSON）：

| 设置 | 输出目录 | 结果 |
|:--|:--|:--|
| 不加覆盖（开关全关，`points` 读法） | `eval-points` | **40 个指标与 E0 `metrics.json` 末值（iter 12500）逐项相同**（`count_MAE_spike` 2.5986、segm AP 78.80 等）；`coco_instances_results.json` 与两份 spikelet CSV 和 E0 最终评估**逐字节相同** |
| 加 `POINT_DENSITY_HEAD True`、`TEST.POINT_COUNT_SOURCE density`（密度头随机初值） | `eval-dens0` | 37 项与 E0 末值逐项相同；只有 `count_MAE_spike`（15.55）、`count_MAE_img`、`count_RMSE_img` 三项计数不同；新增三项密度读数。点级指标、segm / bbox 全部不变 |

- `eval-dens0` 的逐穗明细（578 穗，与 E0 同一批）：`out_points` 列复现 E0 的点数读法（MAE 2.5986、偏差 +2.405、r 0.642）；
  `line_count` 列 MAE 1.4187、偏差 −0.197、r 0.615；`pred_points`（随机初值的头）MAE 15.55、偏差 −15.55，只验管线。
- 随机初值的密度头：`density_MAE_img` 25.28，`density_gtmask_MAE_all` 15.41，`density_gtmask_MAE_spike` 15.44（D ≈ 先验，积分 ≈ 0）。
- eval_only 不算训练，不进 experiments-bank 的实验记录。

**5. 冒烟**（10-22 15:16~15:24，baiju A100 GPU0，与别人的进程共卡；正式配置，CLI 只覆盖 `SOLVER.MAX_ITER 40`、`TEST.EVAL_PERIOD 40`，输出 `smoke40/`）

- 训练 40 iter 与一次完整 val 评估（iter 40）都跑通，无报错，损失全部有限。
- 损失（20 步窗口中位数，×权重）：

  | 窗口 | `total_loss` | `loss_density` | `loss_density_count` | 两项占比 |
  |:--|--:|--:|--:|--:|
  | iter 0~19 | 576.83 | 21.73 | 0.000 | 3.77% |
  | iter 20~39 | 542.54 | 18.54 | 0.000 | 3.42% |

  其余主项（iter 39）：`loss_ce` 20.50、`loss_bbox` 11.69、`loss_dice` 4.96、`loss_mask` 3.34、`loss_giou` 1.99、`loss_point_coord` 1.77、`loss_point_cls` 0.53。
  40 步全在 warmup 里（lr ≤ 5.3e-8），参数几乎没动，两个窗口的差来自 batch 不同。计数项为 0 是 IoU 筛选的预期行为。
- 评估：未训练的模型没有 score ≥ 0.4 的 wheat_2，`count_MAE_spike` 为 nan、`density_gtmask_MAE_spike` 为 nan（与 `mini/count-residual` 冒烟同样情况）；
  `density_MAE_img` 25.39、`density_gtmask_MAE_all` 15.43，评估内的密度读数路径照常执行。推理 2:32。
- 速度 4.85 s/iter（`38 iterations in 0:03:04`）：卡被别人的进程占满，不代表独占速度。
- 峰值显存 `max_mem` **23,722 MB**（torch 口径；`mini/count-residual` 冒烟 22,989 MB），密度头多约 0.7 GB。

**6. 单步开销**（10-22 19:55，xuyu A100 GPU1，`scripts/bench_step.py`，`bench_step.json`）

- 同一进程里交替跑「关」「开」两个模型的训练步（bs4，AMP 前向 + 反向 + AdamW 一步，固定 batch，各 8 次取中位）：
  关 2.957 s，开 3.050 s，**开销 +3.1%**。

## 统一判定线（批次说明第 4 节原文）

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

## 预期与判否线（跑之前写死）

### 读数规则

- 一律用默认读数：内置评估，`POINT_THRESHOLD` 取配置值 0.35；本臂每穗计数来自密度图（`POINT_COUNT_SOURCE: density`）。
- 不扫阈值：不改 `TEST.POINT_THRESHOLD` 去扫，不跑任何 `--sweep`，不用在 val 上拟合的校准值做对比。
- **主读数**：后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估（排除 nan）。该 checkpoint 的全部指标都从这次读。
- 护栏 segm AP、`spike_recall` 用全程峰值。单种子（SEED 42），不加种子。

### 基线读数（E0，`output_v2/exp-0923-mini-ep100/metrics.json`）

| | iter | `count_MAE_spike` | `matched_F1@8` | `spike_recall` | segm AP |
|:--|--:|--:|--:|--:|--:|
| 主读数（后半程计数最好的那次） | 8999 | **2.4648** | 0.6228 | 0.9511 | 77.40 |
| 全程峰值 | — | — | — | **0.9511** | **78.80** |

### 主判据：主读数那次的 `count_MAE_spike`

| 条件 | 判定 |
|:--|:--|
| ≤ **1.28** | **计数成立** |
| ≥ **1.42**（穗长直线） | **判否** |
| 介于两者之间 | **歧义** |

成立还要求三条护栏都满足。主判据过线但护栏未过，记「计数过线、护栏未过」，不记成立。

### 护栏（任一不过则整体不成立）

- segm AP 全程峰值 ≥ **77.80**；
- `spike_recall` 全程峰值 ≥ **0.9311**；
- 主读数那次 `matched_F1@8` ≥ **0.6078**（E0 − 0.015）。

### 机制门（本臂）：密度积分有没有给出穗长以外的信息

- 在最终评估的逐穗明细（`inference/spikelet_pair_counts.csv`）上，取同一批配对穗：
  **r(`pred_points`, `gt_points`) − r(`line_count`, `gt_points`) ≥ 0.03** → 机制启动；达不到 → 记「机制未启动」。
- 两个 r 都在取整后的整数计数上算：`pred_points` 是密度计数 round(n̂)，`line_count` 是同一批穗只用直线（预测框）的计数。口径同 `mini/count-residual`。
- 参照：E0 最终 checkpoint 的 578 穗上，直线的 r 是 0.615，存在性阈值读数的 r 是 0.642；`mini/count-residual` 是 r(头) − r(直线) = +0.127。
- 机制门只决定结果的叫法（见「归因」的表），不改主判据与护栏。

### 次口径（必报，不进判定）

- 同「统一判定线」：主读数 checkpoint 上 `--eval_only` 加 `SPIKELET_EVAL_SIGMAS '[8.0,16.0,32.0]'` 重评；最终评估逐穗 CSV 的有符号偏差、r、预测计数标准差、斜率、同批穗直线 MAE。
- 参照（E0 最终评估，578 穗，`mini/count-residual-detach` 预注册里的复算）：MAE 2.599、偏差 +2.405、r 0.642、预测标准差 1.54（GT 2.41）、斜率 1.00、同批穗直线 1.419。

### 密度图自身的读数（只记录，不进判定）

主读数那次与最终评估各报一次（都在 `metrics.json` 的 `spikelet/` 下，逐图 / 逐穗明细里有对应列）：

| 指标 | 定义 | 参照 |
|:--|:--|:--|
| `density_MAE_img` | 每图 \|整图 Σ D − 该图 GT wheat_2 点数\|，200 图平均 | DM-Count 整图 2.67~3.46；随机初值的头 25.3 |
| `density_gtmask_MAE_spike` | count_MAE_spike 的同一批配对穗上 \|GT mask 内 Σ D − Ni\|，不取整 | DM-Count 557 个 TP 穗 1.22~1.44（批次说明引 1.26）；随机初值的头 15.4 |
| `density_gtmask_MAE_all` | 全部 613 个 GT wheat_2 上的同一量，不依赖检测与配对 | DM-Count 全部 613 穗 1.25~1.46；随机初值的头 15.4 |

- GT mask 内积分与 `compare/` 的 DM-Count 评估同口径：原图 mask 在密度网格上取块内均值作权重（本仓库实现为把 D 最近邻放大回原图、按总量重新归一后在 GT mask 像素上求和，1024² 图上两者相同），不取整。
- 配对穗的集合与 DM-Count 的 557 个 TP 穗不同（各自的配对），只作参照。
- 这三项把「密度图本身准不准」与「预测 mask 准不准」分开：`density_gtmask_MAE_spike` 低而 `count_MAE_spike` 高，误差在 mask；两者都高，误差在密度图。

### 观察量（只记录，不作判据）

- `loss_density`、`loss_density_count` 的训练曲线（分段均值），及其占总损失的比例。
- 最终评估逐穗明细上：`density_raw`（未取整的密度计数）的 MAE、偏差、r；`out_points`（存在性阈值读法）的 MAE、偏差、r，对照 E0 2.599 / +2.405 / 0.642。
- `count_MAE_img`：密度计数也给误检的 wheat_2 计数，每图总数跟着变。

### 预期

- 不确定。DM-Count 用专门的 VGG19 在 GT mask 内数到 1.22~1.44；本臂的密度头只有两层卷积，读的是主要为 mask 训练的特征，积分区域还要换成预测 mask。
  读数大概率落在 1.2~1.5，跨着 1.28 与 1.42 两条线；护栏见下节。

## 归因（对父臂 = E0）

- 本臂与 E0 只差两处：
  1. 训练多两项密度损失，经 `mask_features` 回传 pixel decoder 与 backbone；
  2. 每穗计数的读法（`density`）。
- 点的输出读法与 E0 相同，计数损失里的 mask 已 detach。所以分割、检出、点（`matched_F1@8`、`out_points`）相对 E0 的变化，只能来自两项密度损失经共享特征的回传（含全模型梯度裁剪下的份额），外加单种子噪声。
- 读法本身的作用：同一 checkpoint、同一批配对穗上 `pred_points`（密度）vs `out_points`（存在性阈值，E0 的读法），只记录。
- 结果分类（只给可能的结果起名，不改上面的线）：

| 主判据 | 护栏 | 机制门 | 记为 |
|:--|:--|:--|:--|
| 过线 | 三条都过 | 启动 | **成立**；计数收益来自密度积分带来的穗长以外的信息 |
| 过线 | 三条都过 | 未启动 | **成立**，但相关性不比穗长直线强：收益主要来自偏差（标定），记「计数成立、机制未启动」 |
| 过线 | 未全过 | — | 计数过线、护栏未过（不记成立）。代价归到密度损失经共享特征的回传；后续候选是 `POINT_DENSITY_DETACH=True`（已实现，本次不开）或降 `POINT_DENSITY_WEIGHT` |
| 歧义 / 判否 | 三条都过 | — | 不成立；记「护栏无损、计数不够」。看 `density_gtmask_MAE_*` 分辨误差在密度图还是在 mask |
| 歧义 / 判否 | 未全过 | — | 不成立 |

- 不另设数值归因线的理由：父臂就是主对照 E0，归因的判定已由主判据加三条护栏给出；
  segm AP 后半程平均差等量没有测过的噪声地板（单种子，只测过 `matched_F1@8` 的 0.015），定线只能拍脑袋。

## 护栏风险（正面写）

- **前车之鉴**：`mini/count-residual` 的计数损失经共享 decoder 回传，后半程 segm AP 13/13 次低于 E0（平均 −2.05），`spike_recall`、`matched_F1@8` 也降，三条护栏都没过。
  它的计数损失占总损失从开训的 3.2% 涨到末段的 18%（8.6 / 47.5）。
- **本臂的回传离 mask 更近**：两项密度损失经 `mask_features` → pixel decoder → backbone 回传。mask logit 正是 query 嵌入与 `mask_features` 的内积，这一路直接改的是 mask 用的特征。
- **稠密项的占比大概率会涨**：E0 的总损失从开训约 480~580 降到末段约 37。σ = 1 格的高斯目标很尖，稠密误差有不小的下限；
  若末段稠密项还剩初值的 30%~70%（未乘权重 3~7 / 图），×2.0 后为 6~14，占末段总损失的 13%~27%，与 `mini/count-residual` 的 18% 同一量级。
  计数项末段约 1~2（占 3%~5%）。这是估计，不是测量。
- 缓和因素（不保证够）：
  - 计数项里 mask detach，mask_embed、decoder 拿不到密度损失的任何梯度（实测，第 3 项）；
  - 计数项只算最后一层、且有 IoU 筛选；
  - 密度头输出层初值很小，开训时密度损失进共享参数的梯度不到其余损失的 1%（实测，第 3 项）。头的权重长大后这一路会变大。
- **全模型梯度裁剪**（`CLIP_TYPE: full_model`，`CLIP_VALUE` 0.01，每步都生效）：密度头自己的梯度也计入总范数，会把共享参数的梯度统一缩小一点。训练中不记录这个比例。
- 若护栏未过：已实现的 `POINT_DENSITY_DETACH=True`（密度头读 `mask_features.detach()`，两项损失只训练密度头）是现成的后续臂。

## 已知限制

- **单种子。** 同代码同 SEED 的两次 run 到 iter 19 就会分叉（GPU 反向非确定性）。`fork_rng` 只保证数据增广随机流与 E0 相同，不保证曲线相同。
- **wheat 类穗的目标为 0。** 按数据集定义只有完整可计数的穗（wheat_2）有点；被遮挡、截断等穗（wheat）有小穗但目标为 0。
  密度头要学会「哪类穗才有质量」，这是语义判断。wheat 穗上残留的质量会进整图 Σ D（`density_MAE_img`），只有预测 mask 盖到相邻 wheat 穗时才进每穗计数。
- **原图边界截断的穗**（标注即为 wheat）无法与其它 wheat 区分，目标照旧是 0。LSJ 截断降级的穗能识别（`gt_demoted`），只在它们的 mask 内忽略。
- **训练软积分、推理硬 mask。** 训练用 σ(mask_logit)，推理用 logit > 0。E0 权重下同一 batch 两者差 < 1%；mask 学好之前差得多（初值下 34~39 vs 42~59），那段时间计数项被 IoU 筛掉。
- **积分区域是预测 mask。** mask 的系统误差（偏大、偏小、粘连相邻穗）直接进计数；训练用预测 mask，计数项会部分学会补偿。IoU 筛选在 1/4 网格上算，评估器在原图上算，口径略有不同。
- **1/4 分辨率。** LSJ 缩到 0.1 倍时一个穗只有几个格子，σ = 1 格的高斯会溢出 mask；val 原尺度下穗约 50 × 10 格，溢出小。
- 外层匈牙利匹配（哪个 query 配哪个 GT）仍由共享网络决定。

## 成本

- 12500 iter，另加 25 次评估。
- 速度：单步开销 +3.1%（第 6 项）。E0 / `mini/count-residual` 独占一张卡约 2.6 s/iter，按 2.7~3.9 s/iter 算，训练 9.5~13.5 h，评估另约 1 h（每次评估多一次整图密度图上采样与 GT mask 积分，量很小）。
- 峰值显存：冒烟 `max_mem` 23,722 MB；E0 正式 run 24,195 MB，`mini/count-residual` 24,701 MB。
  本臂预计约 24.9~25 GB（torch 口径）。nvidia-smi 口径约 27~28 GB，排卡按空闲显存 ≥ 29 GB 算。
- 卡由主 agent 排（本批用户临时放开 qiuhao、baiju、xuyu GPU0/1；4090 放不下 bs4）。
