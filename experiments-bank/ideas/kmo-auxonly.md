# 想法：KMO 只留辅助损失、去掉匹配项（归因实验）

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-20 ｜ 分支 `mini/kmo-auxonly`（从 `mini/kmo-dir`@`f7d1140` 开出，代码与 `mini/kmo-dir` 相同）｜
配置 `wheat_joint_single_stage_mini_kmo_auxonly.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_kmo_ep100.yaml`）｜ 输出 `output_v2/exp-1020-mini-kmo-auxonly`

**这是归因实验，不是改进实验**：它不为再提一个点，只回答 KMO 现有约 +0.015 的定位收益来自匹配项还是辅助损失。
KMO v2 同批（2026-10-20）三个臂：臂 A `mini/kmo-dir`（方向描述子）、臂 C `mini/kmo-detach`（头输入 detach）、本臂 D。
三个臂都从代码提交 `f7d1140` 开出，代码完全相同，只差配置，各自独立，不合并。
启动安排：A → xuyu GPU0，C → xuyu GPU1，D → baiju GPU0（用户 2026-10-20 为本批次临时放开这几张卡；baiju 与别人的任务共卡，会慢）。

## 为什么要单独量

KMO（[kmo-matcher](kmo-matcher.md)，CLTR Eq.3）在内层点匹配上做了两件事，一直绑在一起：

1. **匹配代价项** `COST_POINT_KNN_WEIGHT · |ŷ^k − y^k|`（权重 5），改变内层配对；
2. **辅助回归损失** `loss_point_knn`（`POINT_KNN_WEIGHT` 5，主输出、`_0`~`_8`、`_interm` 共 11 份）：KMO 头 `point_knn_embed`
   在配上的槽上回归 y^k，梯度经共享 decoder 回传。

KMO 最初的预注册在「归因限制」里写过：若结果成立，要把「配对改变」与「多任务正则」拆开，需要再跑消融臂
`COST_POINT_KNN_WEIGHT=0`（只保留辅助损失），「本次不跑」；它的结论把下一步排成
「两臂各补一个种子 → 消融 `COST_POINT_KNN_WEIGHT=0` → 计数阈值扫描 → …」。
100 epoch 复核（[kmo-matcher-ep100](kmo-matcher-ep100.md)）判「未决」：对 E0 末段仍领先约 +0.015，恰好等于噪声地板；
它的实验记录同样把「消融 `COST_POINT_KNN_WEIGHT=0`（只保留 `ŷ^k` 辅助损失、不改配对），同样 100 epoch」列为下一步。
本臂就是这个写过、但一直没跑的消融。

### B 诊断：KMO 头并不比点头更懂局部结构

扩展版 `myscripts/backup/vis_process/kmo_match_diag.py`（2026-10-20），现有 KMO 100 epoch run，train 100 batch，最后一层，
L1-only 配对，Ni ≥ 3，1586 个实例；JSON 在 `output_v2/exp-0924-mini-kmo-matcher-ep100/kmo_diag/model_{0008999,final}.v2.json`：

| | iter 8999 | final |
|:--|--:|--:|
| e_head（KMO 头） | 0.01769 | 0.01707 |
| e_proxy（由配上槽自己的预测点推算） | 0.01746 | 0.01721 |
| e_const（该实例 GT 中位数，偏乐观，只作参照） | 0.01575 | 0.01575 |
| e_head < e_proxy 的实例比例 | 52.7% | 55.6% |
| 穗内 r / 改判率 / 平局互换率 | 0.533 / 13.9% / 0.32% | 0.544 / 12.7% / 0.63% |

读法：KMO 头对局部结构的了解与「从预测点推算」基本一样，匹配项在实际中接近噪声级的平局打破；
KMO 约 +0.015 的定位收益，可能主要来自辅助回归损失对共享特征的塑形。这就是臂 C、D 要拆开的问题。

为什么「头只和推算一样好」就意味着匹配项帮不上忙——配对模拟（val 613 穗，预测点 = GT + E0 实测量级噪声，
沿轴 / 横轴中位 4.62 / 2.34 px，内层匈牙利把预测配回来源的比例）：

- 头完全准（描述子取真值）：L1 93.2% → 距离 95.7% → 方向 98.2%；
- 头只知道点头那么多（描述子由带噪预测点自己算）：L1 93.2% → 距离 93.1% → 方向 90.9%。

匹配项的收益全靠头比点头更懂局部结构。B 诊断里两者基本持平，所以预期匹配项本身的贡献很小。

### 辅助损失这条路上带着计数信号

KMO 的 y^k（k=3 平均近邻距离，框相对）：方差 57% 在穗内、43% 在穗间；按穗平均与 Ni 的相关 r = −0.81，是计数信号。
内层匹配只在一个穗内做，穗间那部分对同穗所有 GT 是常数，改不了任何配对；KMO 头却要经共享 decoder 学它（L1 ×5，11 层）。
本臂把这条路原样保留、只去掉匹配项，所以它也是计数副作用归因里的一个格子（配合臂 C 读）。

## 修法

只有配置，没有代码改动。代码是 `f7d1140`：两个新开关取默认值（`POINT_KMO_DESC` = dist、`POINT_KMO_DETACH` = False），
与 `mini/kmo-matcher` 逐位等价（见「实施前已完成的验证」）。

新配置 `configs/coco/instance-segmentation/wheat_joint_single_stage_mini_kmo_auxonly.yaml`：
`_BASE_: wheat_joint_single_stage_mini_kmo_ep100.yaml`，只设 `MODEL.MaskDINO.COST_POINT_KNN_WEIGHT: 0.0`。

- **内层匹配退回 Eq.2**：`C = 5·|p − p̂|₁ + 2·focal(ĉ)`，即纯 L1 + 存在性。`criterion._inner_point_match` 里 KMO 项是
  `C = C + point_cost_knn · cost_knn`；权重 0 时加的是 `0 · |ŷ^k − y^k|` = +0.0（ŷ^k 过 sigmoid，有限），
  代价矩阵与不开 KMO 时逐位相同，分配不变。
- **KMO 头与辅助损失原样保留**：`point_knn_embed`（3 层 MLP，138,009 个参数）、`loss_point_knn`（×5，11 份，经共享 decoder 回传）。
  唯一的连带变化：ŷ^k 的监督改在 L1 配对上算（KMO 原样时在 KMO 自己选的配对上算）。
- 数据、schedule（100 epoch，12500 iter）、初始权重、其余损失与匹配代价、INPUT 全部继承。推理不用 KMO 头，推理、评估代码不变。

## 与对照的差异

解析后的 cfg（`get_cfg()` + `add_deeplab_config` + `add_maskdino_config` + `merge_from_file`）逐键比对，各 331 个键：

| 比对 | 差异 |
|:--|:--|
| vs 父配置 `wheat_joint_single_stage_mini_kmo_ep100.yaml`（KMO） | **只差 `COST_POINT_KNN_WEIGHT` 5.0 → 0.0** |
| vs KMO run 实际用的 `output_v2/exp-0924-mini-kmo-matcher-ep100/config.yaml` | 另外只差 `OUTPUT_DIR`，和一项用不到的级联头权重 `ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS`（元组与列表的序列化差异） |
| vs E0 配置 `wheat_joint_single_stage_mini_ep100.yaml` | `POINT_KMO` False → True、`COST_POINT_KNN_WEIGHT` 5.0 → 0.0（E0 不开 KMO，这一键在 E0 里不生效） |

参数量 52,438,442、state_dict 738 个张量，与 KMO 相同；E0 是 52,300,433，差的 138,009 个就是 KMO 头。

## 预期与判读（跑之前写死）

以下照抄判据说明 `output_v2/_smoke-1020-kmo-v2/prereg_spec.md` 的「共同口径」与「臂 D」两节，没有改线。

### 共同口径

- 主对照 E0 = `output_v2/exp-0923-mini-ep100`（mini/baseline 100 epoch）；归因参照 KMO = `output_v2/exp-0924-mini-kmo-matcher-ep100`（`mini/kmo-matcher`）。
- 读数：后半程（iter ≥ 6250）内置评估里 `count_MAE_spike` 最低的那次，在同一次评估上报全部指标。只用默认读数（τ 0.35），不扫阈值；单种子，不补种子；`matched_F1@8` 噪声地板 0.015。
- E0 读数（iter 8999）：`count_MAE_spike` 2.4648；`matched_F1@3/5/8` 0.1504 / 0.3466 / 0.6228；`spike_recall` 0.9511；segm AP 77.40。全程峰值 segm AP 78.80、`spike_recall` 0.9511。
- KMO 读数（iter 8999）：`count_MAE_spike` 2.4878；`matched_F1@3/5/8` 0.1647 / 0.3579 / 0.6378；`spike_recall` 0.9331；segm AP 77.64。全程峰值 segm AP 78.60、`spike_recall` 0.9511。
- 本批（10-07）对 E0 的通用线：`matched_F1@8` ≥ 0.6428（+0.02）、segm AP 峰值 ≥ 77.80、`spike_recall` 峰值 ≥ 0.9311。
- 计数护栏（本组 KMO 臂专用：三个创新点要原理独立，KMO 不能伤计数）：读数那次 `count_MAE_spike` ≤ 2.4648（不比 E0 差）。KMO 自己是 2.4878（+0.023，单种子下在噪声内）。
- 机制门的数：扩展版 `myscripts/backup/vis_process/kmo_match_diag.py`，本 run 的 `model_final.pth`，train 集 `--batches 100`。

### 臂 D：`mini/kmo-auxonly`（归因实验，不是改进实验）

- 配置 `wheat_joint_single_stage_mini_kmo_auxonly.yaml`：`_BASE_: wheat_joint_single_stage_mini_kmo_ep100.yaml`；`COST_POINT_KNN_WEIGHT: 0.0`。
- 与 KMO 的差异只有匹配代价权重（5 → 0）：内层匹配退回纯 L1（+存在性），KMO 头与 `loss_point_knn`（经共享 decoder 回传）原样保留。KMO 最初预注册里写过、但没跑的消融。
- 假设（来自 B）：KMO 的定位收益来自辅助回归，不来自匹配。
- 机制检查：改判率必须为 0（权重 0 时 KMO 项不改分配，作为实现核对）；穗内 r 与 KMO 同量级（头照常学）。
- 判读（单种子，F1 小于 0.015 的差不下结论）：
  - F1@8 − E0（0.6228）≥ +0.015 且与 KMO（0.6378）之差绝对值 < 0.015 → 「收益来自辅助损失」；
  - F1@8 − E0 的绝对值 < 0.015 → 「辅助损失本身没有收益」（收益若存在则来自匹配，看臂 C）；
  - 计数：读数那次与 E0（2.4648）比，配合臂 C 判断计数副作用走的是哪条路。
- 2×2 归因（E0 = 都没有，C = 只有匹配，D = 只有辅助损失，KMO = 都有）：
  匹配主效应 = ((C − E0) + (KMO − D)) / 2；辅助损失主效应 = ((D − E0) + (KMO − C)) / 2。在 F1@8 与计数上各报。
- 护栏：segm AP 峰值 ≥ 77.80；`spike_recall` 峰值 ≥ 0.9311。

### 读数时的换算（只是算术，不改线）

- 诊断字段：改判率 = `M2_override_rate`，平局互换率 = `M2_tie_swap_rate`，穗内 r = `posthoc_within_inst_pearson_r`。KMO 最终的穗内 r 是 0.544。
- 两条判读换成读数那次 `matched_F1@8` 的区间：「收益来自辅助损失」⇔ [0.6378, 0.6528)；「辅助损失本身没有收益」⇔ (0.6078, 0.6378)。
  **≥ 0.6528（比 KMO 高出噪声地板以上）与 ≤ 0.6078（比 E0 低出噪声地板以上）两段，判据说明没有规定，须在启动前补定。**
- 2×2 的两个主效应之和恒等于 KMO − E0（`matched_F1@8` +0.0150，`count_MAE_spike` +0.0230），即这张表是在分摊 KMO 相对 E0 的差。

## 实施前已完成的验证

### 代码提交 `f7d1140` 的验证（`mini/kmo-dir`，见该提交信息）

- 默认开关下与 `mini/kmo-matcher` 逐位等价（mini train 前 2 张图、训练态 fp32 前向一次）：**138 项 loss 全部逐位相等**，
  参数量 **52,438,442**，**738 个 state_dict 张量哈希相同**。
- CPU：dist 路径 3000 次内层匹配、60 个 batch 的 `loss_points` 与 `mini/kmo-matcher` 逐位相同。
- 诊断回归：dist 下 KMO `model_final` 的原有字段与 09-24 的旧 JSON 一致（同 1586 实例 / 24766 点，r 0.5441 vs 0.5444，
  推翻率 12.67% vs 12.61%，差异来自 4090 vs A100 的 fp16 数值）。

### 本臂的核对

脚本在 `output_v2/_smoke-1020-kmo-v2/scripts/auxonly_*`，产物在 `output_v2/_smoke-1020-kmo-v2/auxonly/`，不进仓库。

**1. cfg 核对**（`auxonly_cfg_check.py`、`auxonly_params_cpu.py`）：结果见上「与对照的差异」。冒烟 checkpoint 的张量名与形状和按新配置建出的模型一致。

**2. CPU 单元核对：权重 0 时 KMO 项不改变内层分配**（`auxonly_unit_cpu.py`）。
点头参数取自解析后的新配置（P = 25、代价 5 / 2、`POINT_FOCAL_ALPHA` 0.5、K = 3、dist）。同一组随机输入，比
`point_cost_knn=0.0` 的 `SetCriterion._inner_point_match` 与不开 KMO（L1 + 存在性）的结果；代价矩阵在传给
`linear_sum_assignment` 处截获，按 float32 位模式比。Ni ∈ {0, 1, 2, 3, 4, 7, 15, 24, 25}，每族 1080 例：

| 输入族 | 分配逐位相同 | 代价矩阵逐位相同（Ni > 0） | 有精确 L1 平局的例数（Ni ≥ 2） | 对照：权重 5 时分配不同（Ni ≥ 2） |
|:--|--:|--:|--:|--:|
| 连续坐标，ŷ^k = sigmoid(N(0,1)) | 1080/1080 | 960/960 | 66/840 | 604/840 |
| 坐标在 1/16 网格、logit 全同（制造平局） | 1080/1080 | 960/960 | 686/840 | 788/840 |
| ŷ^k 取极端有限值（0、1、1e-30、±1e30、±3e38） | 1080/1080 | 960/960 | 57/840 | 568/840（另 244 例 5 × 3e38 溢出，求解器报不可行） |
| fp16 输入（模拟 AMP） | 1080/1080 | 960/960 | 106/840 | 621/840 |

- 平局一族最严：多个最优解里取哪个，取决于 C 的最后一位。686 例有精确平局，分配仍逐位相同。
- 最后一列说明这个比对有检测力：同样输入下 KMO 原样（权重 5）在连续、平局、fp16 三族里有 72% ~ 94% 的例子换了分配。
- **前提：ŷ^k 必须有限。** ŷ^k 含 inf 或 NaN 时 0 × inf = NaN，代价矩阵被污染，求解器直接报错（`matrix contains invalid numeric entries`）。
  训练里 ŷ^k = sigmoid(MLP(·))，MLP 输出 ±inf 也会被压到 1 / 0（fp16、fp32 都核过）；只有头内部出现 NaN 才会触发，那时训练本身已经发散。
- `loss_points` 端到端（40 组，含 Ni = 0 的 wheat 实例与 Ni = 1，一半用 fp16 预测）：`loss_point_coord` / `loss_point_cls` 与不开 KMO **逐位相同**；
  `loss_point_knn` 照常产生，等于在 L1 配对上独立重算的 Σ|ŷ^k − y^k| / num_pos（最大相对误差 1.0e-7）。

**3. 冒烟**（baiju A100 GPU0，与别人的训练任务共卡：约 34 GB、利用率 99%）。2026-10-20 12:00 启动，新配置原样（`IMS_PER_BATCH` 4），
CLI 只加 `SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 0 SOLVER.CHECKPOINT_PERIOD 40`，输出 `output_v2/_smoke-1020-kmo-v2/smoke40-kmo-auxonly`：

| 项 | 结果 |
|:--|:--|
| 报错 | 0（训练 rc 0；detectron2 每 iter 检查 total_loss 有限） |
| 损失 | 138 项全部有限（iter 19 / 39 两次记录），含 11 份 `loss_point_knn*`（主输出、`_0`~`_8`、`_interm`） |
| 评估 | 训练结束后的 val 评估完整跑通，用时 4:23。40 iter 时 segm AP 0、点指标 0 / nan，`count_MAE_img` 48.735，与以往未训练模型的值相同，正常 |
| 速度 | **5.32 s/iter**（metrics.json 里 iter 39 的 `time`，即 iter 20~39 的中位数）；38 iter 总均 5.16 s/iter |
| 峰值显存（log.txt 的 max_mem） | **22,875 MB**（iter 19 时 21,845 MB）。KMO 100 epoch 全程是 24.5G |
| 进程显存（nvidia-smi，含缓存） | 前 20 iter 升到 **45,960 MiB**，与共卡任务的 34,192 MiB 合计 80,152 / 81,920 MiB；iter 20 前后回落到 29,370 MiB，一直到评估结束。没有设 `PYTORCH_CUDA_ALLOC_CONF`，日志无 OOM，推测是缓存分配器在显存申请失败后释放缓存、重试成功 |

**4. 诊断**（`kmo_match_diag.py --batches 5`，冒烟的 `model_final.pth`，baiju GPU0）：

| 项 | 结果 |
|:--|:--|
| `M2_override_rate`（改判率） | **0.0** |
| `M2_tie_swap_rate`（平局互换率） | **0.0** |
| `err_under_KMO_match` 与 `M1_err_under_L1_match` | 都是 0.344379，完全相同（配对相同的直接体现） |
| 实例 / 点 | 79 / 1276 |

40 iter 时 LR 不到 1e-7，头还没开始学（穗内 r 0.003），这些数只说明管线跑通，不说明头学得怎样。

## 已知限制

- **单种子**，与所有 mini 实验一致，不补种子。F1 差小于 0.015 不下结论；同代码同种子的两次训练也不逐位相同，臂间差里本身含 run 间噪声。
- **2×2 不是严格的析因设计**，两个主效应各自混有一点交互：
  - 臂 C 的头读 detach 后的特征，匹配项用到的 ŷ^k 与 KMO 的不是同一个头；
  - 本臂的辅助损失在 L1 配对上算，KMO 在 KMO 自己选的配对上算（选择效应）。
  四个格子也都各只有一个种子。
- **计数读数混有标定**：默认读数（τ 0.35）下 E0 每穗系统多报约 +2.4，`count_MAE_spike` 主要量的是这个偏差。不扫阈值，计数上的差与主效应都按这个口径读。
- **与别人共卡**（baiju GPU0）：速度取决于对方的任务。冒烟时整卡显存曾到 80,152 / 81,920 MiB；对方任务若再涨显存，两边都可能 OOM。

## 成本

- 12500 iter。按冒烟在共卡时的 5.32 s/iter，训练约 **18.5 h**；另加 25 次评估：冒烟的最终评估 4:23（未训练模型，结果文件 1 GB，偏慢），
  KMO 100 epoch 平均每次 2:14，估计 1 ~ 2 h。合计约 **20 h**，比 KMO 100 epoch 的 11:34:34（xuyu A100，3.07 s/iter）慢约 8 h。
  共卡的任务结束后会变快（baiju 单独跑时 `mini/point-scale-gate` 全程 1.92 s/iter）。
- 机制诊断（train 100 batch）约 10 min。
- 峰值显存 max_mem 约 23 ~ 25 GB（KMO 100 epoch 全程 24.5G）；nvidia-smi 口径的进程占用更高，见「已知限制」。
