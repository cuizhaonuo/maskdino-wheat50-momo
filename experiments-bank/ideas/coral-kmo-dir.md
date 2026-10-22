# 想法：把方向版 KMO 放进 coral-exist 的前缀内层匹配（方案 A）

状态：**预注册（判定线在跑之前写死）** ｜ 提出 2026-10-22 ｜ 分支 `mini/coral-kmo-dir`（从 `mini/coral-exist`@`f029805` 开出，合并 `mini/kmo-dir` 的代码提交 `f7d1140`，用户批准）｜ 配置
`wheat_joint_single_stage_mini_coral_kmo_dir.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_coral_exist.yaml`）｜ 输出 `output_v2/exp-<启动日期>-mini-coral-kmo-dir`

> 合并（2026-10-22，用户明确批准这一次）：`git merge --no-ff f7d1140`，合并提交 `dfb8cce`。只合 `f7d1140` 这个提交（连同它的祖先，即
> `mini/kmo-matcher` 的 KMO 实现与文档），不含 `mini/kmo-dir` 之后的配置与文档提交。所以下文提到的 kmo-dir / kmo-detach / kmo-auxonly
> 三臂文档都不在本分支上，引用时写明所在分支。

## 动机

### 1. 三个创新点要放进同一个模型

`coral-exist`、`kmo-matcher`、`group-nms-match` 是待选的创新点，最终会放进同一个模型，所以要求三者**在原理上相互独立**：
叠在一起时，每一个都还得有自己的贡献。本实验回答其中一格：KMO 叠在 coral-exist 上，还有没有独立贡献。

### 2. KMO v2 三臂的结论

读数口径与本实验相同（后半程 `count_MAE_spike` 最低的那次），数都从各 run 的 metrics.json 复核过：

| 臂 | 分支 | 描述子 | 读数 iter | `count_MAE_spike` | `matched_F1@8` |
|:--|:--|:--|--:|--:|--:|
| E0 | `mini/baseline` | — | 8999 | 2.4648 | 0.6228 |
| KMO | `mini/kmo-matcher` | dist，K=3 | 8999 | 2.4878 | 0.6378 |
| A | `mini/kmo-dir` | dir，K=2 | 9999 | 2.8467 | 0.6255 |
| C（头读 detach 后的特征） | `mini/kmo-detach` | dist，K=3 | 9999 | 2.5487 | 0.6443 |
| D（匹配代价权重置 0，只留辅助损失） | `mini/kmo-auxonly` | dist，K=3 | 8999 | 2.2465 | 0.6192 |

- **2×2 拆分**（E0 都没有、C 只有匹配、D 只有辅助损失、KMO 都有；见 `mini/kmo-detach` 分支的 `exp-2026-10-20-mini-kmo-detach.md`）：
  `matched_F1@8` 的**匹配主效应 +0.0201**（C − E0 +0.0216、KMO − D +0.0186），**辅助损失主效应 −0.0051**。
  KMO 的定位收益来自匹配项，不来自辅助损失。
- **KMO 改变的配对里约 85% 是「哪些槽被选为正样本」变了**，只有约 15% 是正样本槽之间换了配对
  （`mini/kmo-dir` 分支的 `exp-2026-10-20-mini-kmo-dir.md`；方向项、距离项都是约 85%）。
  被选槽集合变了，存在性的监督也跟着变。
- **全量数据**（26000 iter，同一读数口径，从各 run 的 metrics.json 复核）：KMO 的 `matched_F1@8` 增益
  单独用时 +0.021（`exp-1009-full-kmo-matcher` 对 `exp-0922-single-stage-p30`，+0.0205）；
  叠 coral-exist 只有 +0.003（`exp-1011-full-coral-kmo` 对 `exp-1009-full-coral-exist`，+0.0028）；
  三者组合 −0.006（`exp-1009-full-combo` 对 `exp-1011-full-coral-gn`，−0.0061）。
  **注意**：`full/coral-kmo` 的实现里，前缀分支在加 KMO 代价之前就 return 了，KMO 只剩 dist 的辅助损失
  （核对过该 run 的 config：`POINT_CORAL_EXIST` true、`POINT_KMO` true、`POINT_GROUP_DECODER` false）。
  所以那个 +0.003 量的是「前缀里的辅助损失」，**KMO 的匹配项在前缀里还没有被测过**。
- **方向头学到了真结构**（A 的机制门）：八成实例上头比「由预测点推算」更准，‖c‖ ≥ 0.3 的点上与 c 的余弦 0.746 对 0.635
  （e_head / e_proxy / e_const = 0.0398 / 0.0515 / 0.0584）。但单独用时改判率 34.9%，多半在换正样本槽
  （槽集合改变的实例 30.2%，KMO 为 11.3%），`matched_F1@8` 0.6255 反而低于 KMO 的 0.6378，计数也更差。

### 3. 本臂

coral-exist 的前缀内层匹配把正样本槽固定为前 Ni 个，「哪些槽是正样本」由前缀决定，KMO 换不动。
把 KMO 放进前缀，它就只能决定「正样本槽配哪个小穗」，即上面那约 15% 的排列变化。

**假设**：KMO 的匹配项若有独立于「换正样本槽」的贡献，本臂的 `matched_F1@8` 应高于 coral-exist。
正样本槽集合被前缀固定、CORAL 的 levels 不变，计数不应变差。

## 修法

代码来自合并提交 `dfb8cce`：`f7d1140` 的 KMO（`POINT_KMO_DESC` / `POINT_KMO_DIR_SCALE` / `POINT_KMO_DETACH` 三个开关）
与 coral-exist 的前缀匹配 + CORAL 存在性头拼在一起。两边各自的开关关掉时，与原分支逐位一致（见「实施前已完成的验证」）。

- **内层匹配**（`criterion._inner_point_match` 的前缀分支）：先按 coral-exist 只取前 m = min(Ni, 25) 个 slot，
  再把 KMO 项加在这个已经限制到前缀的 [m, Ni] 代价上：
  C = ‖p − p̂‖₁ + (COST_POINT_KNN_WEIGHT / COST_POINT_COORD_WEIGHT)·s·‖ĉ − c‖₁，本臂的相对权重为 5 / 5 = 1，s = 0.13044。
  - 行仍只有前 m 个 slot，**被选中的行集合恒为前缀**，`tgt_exist` 仍恰好是 CORAL 的 levels，存在性的监督不受 KMO 影响。
  - 单独用 KMO 时的代价是 5·L1 + 2·focal + 5·s·KMO。存在性项在前缀内对所有分配是同一个常数，省去后与上式只差整体倍数，
    **最优分配相同**。KMO 与坐标的相对权重和 A 臂一样。
  - 坐标项不乘回 5，是因为 L1 代价常有精确平局，整体乘 5 的舍入会改变平局的打破方式；这样 `COST_POINT_KNN_WEIGHT` 置 0 时，
    代价与 coral-exist 逐位相同。
- **损失**：`loss_point_knn` = s·Σ_配上的槽 ‖ĉ − c‖₁ / num_pos，再乘 5（`POINT_KNN_WEIGHT`）；主层、`_0`~`_8`、`_interm` 共 11 项，与 A 相同。
  存在性仍是 coral-exist 的 CORAL BCE，坐标 L1 不变。
- **头**：`point_knn_embed`（MLP(256→256→256→50)，过 tanh）仍建在 decoder `__init__` 末尾；CoralHead 留在原 `point_class_embed` 的位置。
  参数 52,294,290 → 52,438,724（+144,434）。这个头只在训练期用，推理不变：仍按 coral-exist 取前 K 个 slot，`POINT_THRESHOLD` 0.5。
- `POINT_KMO_DETACH` 为 False：头读未 detach 的 decoder 输出，与 A 相同。

## 与对照的差异（cfg 核对）

- **对 coral-exist（归因参照、主判据的基准）：只差 KMO**。KMO 的取值与 A 臂完全相同，所以本臂与 A 之间只差「有没有 coral-exist」。
- **对 A（`mini/kmo-dir`）**：只差 coral-exist 的两项（`POINT_CORAL_EXIST`、`TEST.POINT_THRESHOLD`）。
- **对 E0**：两者的差相加。

用 `get_cfg()` + `add_deeplab_config` + `add_maskdino_config` + `merge_from_file` 解析，
全部 332 个键逐项比对（tuple / list 统一后比较）：

| 比较对象 | 取值不同的键（对照 → 本臂） |
|:--|:--|
| `wheat_joint_single_stage_mini_coral_exist.yaml` | `POINT_KMO` False → True、`POINT_KMO_DESC` dist → dir、`POINT_KMO_K` 3 → 2 |
| coral-exist run 落盘的 `exp-0926-mini-coral-exist/config.yaml` | 同上三个键，另有 `OUTPUT_DIR` |
| `mini/kmo-dir` 的 `wheat_joint_single_stage_mini_kmo_dir.yaml` | `POINT_CORAL_EXIST` False → True、`TEST.POINT_THRESHOLD` 0.35 → 0.5 |
| A run 落盘的 `exp-1020-mini-kmo-dir/config.yaml` | 同上两个键，另有 `OUTPUT_DIR` |
| E0 run 落盘的 `exp-0923-mini-ep100/config.yaml` | 上面五个键，另有 `OUTPUT_DIR` |

- 配置里还显式写了 `POINT_KMO_DIR_SCALE` 0.13044、`COST_POINT_KNN_WEIGHT` 5.0、`POINT_KNN_WEIGHT` 5.0、`POINT_KMO_DETACH` False，
  都等于 `maskdino/config.py` 的默认值，所以不出现在差异里，写出来只为对照。
- 其余关键项相同：`IMS_PER_BATCH` 4、`MAX_ITER` 12500、`EVAL_PERIOD` 500、`SEED` 42、
  数据 `wheat_cat2_pts_mini_coco_train / val`、ImageNet R-50 起点。

## 预期与判定线（跑之前写死）

### 共同口径

- 单种子，不补种子；`matched_F1@8` 的噪声地板 **0.015**。
- 只用默认读数（`POINT_THRESHOLD` 0.5，与 coral-exist 相同），**不扫阈值**（score / count / point threshold 都不扫）。
- **读数**：后半程（iter ≥ 6250）13 次内置评估里 `count_MAE_spike` 最低的那一次，全部指标都报同一次评估的值。
- 参照的读数，都从各自的 metrics.json 复核过：

| | 读数 iter | `count_MAE_spike` | `matched_F1@3 / @5 / @8` | `spike_recall` | segm AP | 全程峰值：segm AP / `spike_recall` |
|:--|--:|--:|--:|--:|--:|--:|
| E0（`exp-0923-mini-ep100`） | 8999 | 2.4648 | 0.1504 / 0.3466 / 0.6228 | 0.9511 | 77.40 | 78.80 / 0.9511 |
| coral-exist（`exp-0926-mini-coral-exist`） | 9999 | 1.6655 | 0.1478 / 0.3315 / 0.6107 | 0.9266 | 78.28 | 78.71 / 0.9478 |
| A（`exp-1020-mini-kmo-dir`，只作描述） | 9999 | 2.8467 | 0.1571 / 0.3603 / 0.6255 | 0.9364 | 77.36 | 78.26 / 0.9527 |

主对照按惯例是 E0；本实验的问题是对 coral-exist 的增量，所以主判据用增量。

### 主判据：对 coral-exist 的增量

ΔF1@8 = 本臂读数那次的 `matched_F1@8` − 0.6107：

| 条件 | 判定 |
|:--|:--|
| ΔF1@8 ≥ **+0.015**（即 ≥ 0.6257），且机制门、计数护栏都过 | **成立**：KMO 的匹配项在前缀里有独立贡献 |
| ΔF1@8 ≥ +0.015，但机制门不过 | **主判据过线、机制未启动**（不记成立） |
| ΔF1@8 ≥ +0.015，机制门过、计数护栏不过 | **主判据过线、护栏未过**（不记成立） |
| \|ΔF1@8\| < 0.015 | **未决** |
| ΔF1@8 ≤ **−0.015**（即 ≤ 0.5957） | **判否** |

同时报 `matched_F1@3` / `@5` 对 coral-exist 的增量（基准 0.1478 / 0.3315）。

### 对 E0 的通用线（照报，不作主判据）

沿用 10-07 那批的预注册：`matched_F1@8` ≥ **0.6428**（+0.02）、segm AP 峰值 ≥ **77.80**、`spike_recall` 峰值 ≥ **0.9311**。
三条都照报是否过线，不进上表的判定。

### 计数护栏：KMO 不能伤 coral-exist 的计数

- 读数那次的 `count_MAE_spike` ≤ **1.6655**（coral-exist 的读数）。
- 另报后半程 13 次评估同 iter 的平均计数差（本臂 − coral-exist），只作描述。coral-exist 后半程 13 次的均值是 2.2343。
- 两边的读数都是「13 次里取最低」，有选择效应。若严格线不过、但同 iter 平均差的绝对值 < 0.1，判定里注明「可能是读数选择效应」。

### 机制门

工具：`myscripts/backup/vis_process/kmo_match_diag.py`，本 run 的 `model_final.pth`，train 集 `--batches 100`，seed 0，参数与 A 相同。
在本臂里，诊断脚本把 `point_kmo` 置 False 得到的「L1-only 配对」走的是 coral-exist 的前缀路径，所以下面的比较都在前缀内。

- **M1**：e_head < e_proxy，且 e_head < e_const（JSON 的 `e_head` / `e_proxy` / `e_const`，点加权均值，两分量 L1 再乘 s；
  e_const 是 0 向量）。A 是 0.0398 / 0.0515 / 0.0584。
- **M2**：前缀内的改判率 ≥ **1%**（JSON 的 `M2_override_rate`）。脚本的判据是「KMO 配对的 Eq.2 代价比 L1-only 配对高出 1e-5 以上」；
  两种配对的行相同，存在性项相消，于是改判即 KMO 配对的 5·L1 比前缀内 L1 最优高出 1e-5 以上。
  只换了 L1 平局的计入 `M2_tie_swap_rate`，照报，不算改判。
  A 单独用时是 34.9%（多半在换正样本槽）。
- **实现核对**：KMO 配对与 L1-only 配对的被选行集合变化率必须**恰为 0**（做法同 `output_v2/_smoke-1020-kmo-v2/scripts/rowset_diag.py`：
  包一层 `_inner_point_match`，同一实例的两次调用比行集合）。不是 0 就是实现错误，先查实现，不按上表判定。

### 另报（只作描述）

- 对 A 的差（单独用 vs 放进前缀）：`matched_F1@3 / @5 / @8`、`count_MAE_spike`、M1 的三个数、M2 改判率。
- 训练结束后只写内置评估的结果，不跑任何阈值扫描。

## 实施前已完成的验证（CPU，2026-10-22）

脚本与产物在 `output_v2/_smoke-1022-coral-kmo-dir/`（`scripts/`、`dumps/`），不提交。全部在 CPU 上跑（`CUDA_VISIBLE_DEVICES=""`），
参照分支的 worktree 只读（`python -B`，不写字节码）。

- **KMO 关时与 `mini/coral-exist` 逐位一致**：两个 worktree 各 dump 一次后比对。
  - 内层匹配：前缀 / 非前缀各 3000 个随机实例（Ni ∈ 0..30，含 Ni > 25，三分之一坐标量化制造 L1 平局），42,894 / 43,478 个配对，下标逐项相同。
  - `loss_points`：前缀 / 非前缀各 60 个随机 batch，每项损失的比特相同。
  - 同一 seed 在 CPU 上建模型：733 个张量的键、形状、哈希全同，参数 52,294,290。
- **coral-exist 关、KMO 开时与 `f7d1140` 逐位一致**（对 `mini/kmo-dir` 的 worktree，代码即 `f7d1140`）：
  dir（K=2）、dist（K=3）各 3000 个实例、60 个 batch，内层匹配下标与三项损失的比特全同。
- **两者都开**（dir：K=2、s=0.13044；dist：K=3），每种描述子 6000 个随机实例，含约 1000 个 Ni > 25：
  - 被选中的行恒为 0..m−1（权重 5、权重 0、KMO 关三种设置都是），列是 m 个互不相同的合法 GT 下标；
  - 分配在独立 float64 实现的 5·L1 + 5·s·L1(ĉ, c)（限制到前缀）上最优，与 scipy 最优代价的最大差 6e-7；
  - `COST_POINT_KNN_WEIGHT` 0 时分配与 KMO 关逐项相同（0 处不同）；
  - 权重 5 对权重 0，前缀内的排列确实会变（合成输入，只说明机制在起作用，不预示训练里的比例）：
    dir 在全随机 / 近 GT 两种输入上 88.4% / 54.3% 的实例排列变了（坐标 L1 严格变差的改判占全部实例的 80.7% / 52.3%，其余是平局互换）；
    dist 为 74.2% / 30.0%（改判 53.6% / 25.6%）；
  - 对照：同样的输入不加前缀时，KMO 改变被选行集合的比例为 54%~63%，说明上面「行恒为前缀」的检查是测得出漏洞的；
  - `loss_points`：每个实例的 CORAL levels 恰为 [1]·min(Ni,25) + [0]·(25 − min(Ni,25))（包住 `coral_loss` 取实参核对）；
    `loss_point_cls` 在权重 5 / 权重 0 / KMO 关下逐位相同；`loss_point_knn`、`loss_point_coord` 与按同一配对独立重算的值一致
    （最大相对误差 2e-7）。
- **建造顺序**：KMO 开时只多 `point_knn_embed.*` 6 个张量（+144,434，最后一层输出 50 = 25×2），
  其余 733 个张量与 KMO 关逐位相同，KMO 头没有打乱 coral-exist 已有模块的初始化。按新配置建的模型与「coral-exist 配置 + KMO 各键」完全相同。
- **cfg 核对**：见上文「与对照的差异」。

**待做（GPU，等主 agent 定卡后再做，结果补在本节）**：KMO 关时整模型前向的 loss 与 coral-exist 逐位等价；
本配置原样 40 iter 冒烟（速度、显存、11 个 `loss_point_knn*` 是否齐全）；诊断脚本与行集合核对在冒烟 checkpoint 上跑通。

## 已知限制

- 单种子。
- **coral-exist 自身的实现缺陷**：CORAL 存在性头的 bias 被学习率卡住、未收敛（见
  [exp-2026-09-26-mini-coral-exist](../experiments/exp-2026-09-26-mini-coral-exist.md) 的「实现缺陷」）。本臂沿用同一实现、同一 schedule，
  两臂一致，不影响「把 KMO 加进前缀」的归因；但两臂的计数都建立在一个没收敛的 CORAL 头上。
- **能起作用的空间小**：单独用时 KMO 改变的配对里只有约 15% 是排列变化，放进前缀后只剩这一类可做，效应可能落在噪声内。
- 前缀内 L1 平局常见，KMO 打破平局的那部分不算改判（计入平局互换率），M2 只看严格改判。
- 全量数据上「叠 coral-exist 只有 +0.003」的那个 run 里 KMO 的匹配项没有生效（见动机），不能用来预测本臂。

## 成本

12500 iter，约 **11.5 h**。A 臂实测 12:26（3.29 s/iter），coral-exist 实测 9:31（2.48 s/iter，与 `mini/coral-count` 并发）。
前缀匹配比 A 便宜，KMO 头与 `loss_point_knn` 比 coral-exist 多一些，估计介于两者之间、偏 A。
另加结束后的机制门诊断与行集合核对（各 100 batch），约 10 min。用卡按 CLAUDE.md，由主 agent 决定。
