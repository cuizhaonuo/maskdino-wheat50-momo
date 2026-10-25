# 想法：把方向版 KMO 放进 coral-exist 的前缀内层匹配（方案 A）

状态：**预注册判定：判否**（ΔF1@8 = −0.0181 ≤ −0.015；机制门 M1、M2 过，实现核对的行集合变化率恰为 0；计数严格线未过，1.7500 vs ≤ 1.6655，同 iter 平均差 −0.0335，可能是读数选择效应；segm AP、`spike_recall` 护栏过）｜ 验证于 [exp-2026-10-24-mini-coral-kmo-dir](../experiments/exp-2026-10-24-mini-coral-kmo-dir.md) ｜ 提出 2026-10-22 ｜ 分支 `mini/coral-kmo-dir`（从 `mini/coral-exist`@`f029805` 开出，合并 `mini/kmo-dir` 的代码提交 `f7d1140`，用户批准）｜ 配置
`wheat_joint_single_stage_mini_coral_kmo_dir.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_coral_exist.yaml`）｜ 输出 `output_v2/exp-1024-mini-coral-kmo-dir`（2026-10-24 15:12:58 从 `600206e` 启动）

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
| ΔF1@8 ≥ **+0.015**（即 ≥ 0.6257），且机制门、全部护栏（下文三条）都过 | **成立**：KMO 的匹配项在前缀里有独立贡献 |
| ΔF1@8 ≥ +0.015，但机制门不过 | **主判据过线、机制未启动**（不记成立） |
| ΔF1@8 ≥ +0.015，机制门过、任一护栏不过 | **主判据过线、护栏未过**（不记成立） |
| \|ΔF1@8\| < 0.015 | **未决** |
| ΔF1@8 ≤ **−0.015**（即 ≤ 0.5957） | **判否** |

同时报 `matched_F1@3` / `@5` 对 coral-exist 的增量（基准 0.1478 / 0.3315）。
〔修订第 1 条：「护栏」由只有计数改为下文三条〕

### 对 E0 的通用线

沿用 10-07 那批的预注册：`matched_F1@8` ≥ **0.6428**（+0.02）、segm AP 峰值 ≥ **77.80**、`spike_recall` 峰值 ≥ **0.9311**。
`matched_F1@8` 这条照报是否过线，不进上表的判定；segm AP、`spike_recall` 两条是护栏，进判定，见下。〔修订第 1 条〕

### 护栏（三条都要过）〔修订第 1 条〕

- **计数**：KMO 不能伤 coral-exist 的计数，见下一小节。
- **segm AP 峰值** ≥ **77.80**（E0 78.80 − 1.0）。
- **`spike_recall` 峰值** ≥ **0.9311**。

后两条与 KMO v2 三臂（A、C、D）的护栏一致（coral-exist 当时只设了 segm AP 一条）；峰值取全程 25 次评估。

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

## 预注册修订（2026-10-24，启动前）

修订时正式训练还没有启动，没有任何正式结果。判据线的数值一条没动；上文受影响处加了〔修订第 N 条〕的注。

**1. segm AP、`spike_recall` 两条进入判定，作为护栏。**
原文把 segm AP 峰值 ≥ 77.80、`spike_recall` 峰值 ≥ 0.9311 放在「对 E0 的通用线（照报，不作主判据）」里，判定只看计数护栏。
现改为三条护栏（计数、segm AP 峰值、`spike_recall` 峰值）都过才记「成立」，任一不过记「主判据过线、护栏未过（不记成立）」。
`matched_F1@8` ≥ 0.6428 仍只对 E0 照报，不进判定。
理由：与 KMO v2 三臂的护栏一致；本臂要能进组合模型，就不能以分割或召回为代价，只看计数会漏掉这两种退步。

**2. 实现核对的后果保留。** 被选行集合变化率不为 0 → 实现错误，先查实现，不按判定表判定。
冒烟 checkpoint 上已核对为恰好 0（见下节 GPU 第 3 项）。

**3. 前缀内代价的写法保留**：C = L1 + (COST_POINT_KNN_WEIGHT / COST_POINT_COORD_WEIGHT)·s·‖ĉ − c‖₁。理由：
- 最优分配与 A 臂的代价（5·L1 + 2·focal + 5·s·KMO）限制到前缀后相同：存在性项在前缀内是常数；CPU 上 12,000 例在独立 float64 代价上都是最优；
- 权重 0 时与 coral-exist 逐位一致（CPU 上 0 处不同）。坐标项若整体乘回 5，L1 平局的打破方式会变，权重 0 时最多约 49% 的实例与 coral-exist 配对不同；
- GPU 上同一 batch KMO 开 / 关只有 11 份 `loss_point_coord*` 不同，CORAL 的 `loss_point_cls*` 逐位相同（见下节 GPU 第 1 项）。

**4. 「实施前已完成的验证」补上 GPU 三项**：逐位等价、40 iter 冒烟、冒烟 checkpoint 上的诊断与行集合核对。

**5. 用卡与显存封顶。**
- **卡不写死**：正式训练的卡由主 agent 在启动时按空闲显存选定。用户 2026-10-24 放开了用卡，原话「baiju 和 qiuhao 全部,以及 xuyu 的 gpu 0 和 1 都可以用」。
  冒烟与 GPU 验证都在 qiuhao A100（GPU1）上做，卡上常驻别人的两个训练进程（约 27.7 GB、利用率约 100%）。
- **显存封顶**：候选的卡上常有别人的进程（10-24 时 qiuhao A100、xuyu GPU0 上都有别人利用率 100% 的进程），正式训练与冒烟一样用 `train_capped.py` 包装：

```bash
ssh <机器> "cd <本 worktree> && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<卡号> PYTHONPATH=. \
  CUDA_MEM_FRACTION=0.40 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <MaskDINO-310 python> <主仓库>/output_v2/_smoke-1020-kmo-v2/scripts/train_capped.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_mini_coral_kmo_dir.yaml \
  OUTPUT_DIR <主仓库>/output_v2/exp-<启动日期>-mini-coral-kmo-dir"
```

  `OUTPUT_DIR` 必须用绝对路径指向主仓库的 `output_v2/`，不能落在 worktree 里。`train_capped.py` 在进程开头调
  `torch.cuda.set_per_process_memory_fraction(0.40)`（80 GB 的卡上限 31.7 GiB），再用 `runpy` 原样执行 `train_net.py`，
  只改显存分配方式，不改训练逻辑和数值。
- **耗时**按冒烟的 s/iter 重估，并注明共卡会变慢，见「成本」。

## 实施前已完成的验证

脚本与产物在 `output_v2/_smoke-1022-coral-kmo-dir/`（`scripts/`、`dumps/`、`equiv/`、`smoke40/`、`diag/`），不提交。
参照分支的 worktree 一律只读（`python -B`、`PYTHONDONTWRITEBYTECODE=1`，跑完 `git status` 无改动）。

### CPU（2026-10-22，`CUDA_VISIBLE_DEVICES=""`）

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

### GPU（2026-10-24，qiuhao A100 GPU1，与别人的进程共卡）〔修订第 4 条〕

一律 `ssh qiuhao`，`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`。卡上常驻别人的两个训练进程，合计约 27.7 GB、利用率约 100%。

**1. 逐位等价**（`scripts/equiv_test.py`、`compare_equiv.py`，照 `_smoke-1020-kmo-v2` 的 equiv 做法）：
固定种子建模型并载入 ImageNet R-50，mini train 前 2 张图过同一个 LSJ mapper（同种子），训练态 fp32 前向一次。
batch 两边相同：2 张图，6 个 wheat_2，Ni = 17 / 19 / 18 / 17 / 23 / 17。

| 比较 | 结果 |
|:--|:--|
| KMO 关（`wheat_joint_single_stage_mini_coral_exist.yaml`）：本分支 vs `mini/coral-exist` 的 worktree | **127 项 loss 逐位相等**（float 比特），含 11 份 CORAL 的 `loss_point_cls*`；state_dict **733 个张量哈希全同**；参数 52,294,290；前向峰值显存都是 11,767 MiB |
| 本分支 + 新配置（KMO 开）| 138 项 loss 全部有限，比 KMO 关多出的恰好是 11 份 `loss_point_knn*`（主层、`_0`~`_8`、`_interm`），`loss_point_knn`（×5 后）0.2845；`pred_point_knn` 在主层 / interm / 9 份 aux 都是 [2, 300, 25, 2]（DN 的 63 个 query 已切掉），主层与 interm 的取值在 [−0.29, 0.36] ⊂ [−1, 1]；参数 +144,434，只多 `point_knn_embed.*` 6 个张量，其余张量哈希与 KMO 关相同 |
| 同一 batch 上 KMO 开 vs 关 | 127 项共有的 loss 里，比特不同的**只有 11 份 `loss_point_coord*`**（前缀内的配对变了，如主层 1.7820 → 1.7893）；**11 份 `loss_point_cls*` 逐位相同**，其余 105 项也逐位相同。这是「KMO 只改变前缀内的排列、不动存在性监督」在真实前向上的直接体现 |

**2. 冒烟**（`scripts/run_smoke40.sh`）：2026-10-24 14:55 启动，新配置原样（`IMS_PER_BATCH` 4），CLI 只加
`SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 0 SOLVER.CHECKPOINT_PERIOD 40`，用显存封顶包装
（`CUDA_MEM_FRACTION=0.40 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True train_capped.py`，上限 31.7 GiB），
输出 `output_v2/_smoke-1022-coral-kmo-dir/smoke40`：

| 项 | 结果 |
|:--|:--|
| 报错 | 0（训练 rc 0；detectron2 每 iter 检查 total_loss 有限） |
| 损失 | iter 19 / 39 两次记录各 138 项，**全部有限**；11 份 `loss_point_knn*` 与 11 份 CORAL 的 `loss_point_cls*`（主层、`_0`~`_8`、`_interm`）都在。主层 `loss_point_knn`（×5 后）0.282 / 0.273，`loss_point_cls`（×2 后）1.574 / 1.581，`loss_point_coord` 1.797 / 1.781 |
| 速度 | **7.01 s/iter**（38 iter 总均，log 的 Overall training speed）；metrics.json 里 iter 39 的 `time`（iter 20~39 的中位数）6.86 s/iter |
| 峰值显存（log.txt 的 max_mem） | **22,943 MiB**（iter 19 时 22,339 MiB），在封顶之内 |
| 进程显存（nvidia-smi，含 CUDA 上下文与缓存） | 本进程峰值 26,902 MiB；整卡峰值 54,576 / 81,920 MiB（别人的进程约 27.7 GB） |
| 评估 | 训练结束后的 val 评估（200 图）完整跑通，用时 5:45。40 iter 全在 warmup 里，未训练模型 segm AP 0、点指标 0 / nan；CORAL 头的 bias 初始化全为正，100 个 wheat_2 预测（都在同一张图上）各读出 K = 25（`n_pred_points` 2500 = 100 × 25，K 的上限就是 25），`count_MAE_img` 60.725。未训练的值，不作判读 |

**3. 冒烟 checkpoint 上的诊断与行集合核对**（`scripts/run_diag_smoke.sh`，同样封顶；train 集 `--batches 5`）：

| 项 | 结果 |
|:--|:--|
| `kmo_match_diag.py` 字段 | 34 个，与 `mini/kmo-dir` 冒烟诊断的字段逐项相同，没有缺项、没有 None；`kmo_desc` dir、`kmo_k` 2、`kmo_dir_scale` 0.13044 |
| 实例 / 点 | 67 / 1,090（Ni ≥ 2 的最后一层实例；全部 Ni < 25） |
| e_head / e_proxy / e_const | 0.0679 / 0.1313 / 0.0600；e_head < e_proxy 的实例 100% |
| 改判率 / 平局互换率 / 穗内 r | 85.1% / 14.9% / −0.056 |
| **行集合核对**（`rowset_diag.py`） | 67 个实例，**被选行集合变化率恰为 0**（0 / 67，`n_swapped_in_slots` 0）；67 个实例全是「行相同、排列不同」，没有一例配对完全相同 |
| 一致性 | 行集合脚本内部顺带跑的那份诊断，与单独跑的诊断 33 个字段逐项相同（同 seed、同数据，可复现） |

40 iter 时 LR 不到 1e-7，KMO 头没开始学，坐标预测也几乎是常数，前缀内的 L1 代价近乎平局，于是 KMO 项决定了全部排列
（改判 + 平局互换 = 100%，与 A 冒烟时的 98.7% + 1.3% 是同一现象）。这些数只说明管线跑通、前缀守住了，不作机制判读。

## 已知限制

- 单种子。
- **coral-exist 自身的实现缺陷**：CORAL 存在性头的 bias 被学习率卡住、未收敛（见
  [exp-2026-09-26-mini-coral-exist](../experiments/exp-2026-09-26-mini-coral-exist.md) 的「实现缺陷」）。本臂沿用同一实现、同一 schedule，
  两臂一致，不影响「把 KMO 加进前缀」的归因；但两臂的计数都建立在一个没收敛的 CORAL 头上。
- **能起作用的空间小**：单独用时 KMO 改变的配对里只有约 15% 是排列变化，放进前缀后只剩这一类可做，效应可能落在噪声内。
- 前缀内 L1 平局常见，KMO 打破平局的那部分不算改判（计入平局互换率），M2 只看严格改判。
- 全量数据上「叠 coral-exist 只有 +0.003」的那个 run 里 KMO 的匹配项没有生效（见动机），不能用来预测本臂。
- **与别人共卡**〔修订第 5 条〕：速度取决于对方任务的负载（冒烟 7.0 s/iter，约为 A 臂的 2 倍）。本进程的显存封顶在 31.7 GiB，
  但对方若再涨显存，整卡仍可能被挤满，届时任何一方都可能 OOM。

## 成本〔修订第 5 条〕

- **卡**：正式训练的卡由主 agent 在启动时按空闲显存选定，用显存封顶包装（`train_capped.py`，见修订第 5 条）。
- **训练**：12500 iter。按冒烟的 **7.0 s/iter**（qiuhao A100，与别人两个利用率约 100% 的训练进程共卡）估算，约 **24 h**。
  共卡会变慢，实际速度取决于所选卡上别人的负载。同类 run 在负载较轻时：A 臂 3.29 s/iter（12:26）、coral-exist 2.48 s/iter（9:31），
  按这个量级训练约 9 ~ 12 h。修订前「约 11.5 h」是按不共卡估的。
- **评估**：25 次内置评估。冒烟的最终评估 5:45（未训练模型，每个 wheat_2 读出 25 个点，偏慢）；以往 mini run 每次约 2 ~ 2.5 min，
  共卡会更慢，估计合计 1 ~ 2.5 h。
- **合计**：共卡负载如冒烟时约 **25 ~ 27 h**；卡上负载轻时约 10 ~ 14 h。
- **显存**：max_mem 约 23 GB（冒烟 22,943 MiB；A 臂全程 24.2 GB），在 31.7 GiB 的封顶之内；nvidia-smi 口径本进程约 27 GB。
- 另加结束后的机制门诊断与行集合核对（各 100 batch）：冒烟上 5 batch 各约 45 s（含建模型），估计合计 10 ~ 15 min。

## 结论（2026-10-25）

**预注册判定：「判否」。** 详见 [exp-2026-10-24-mini-coral-kmo-dir](../experiments/exp-2026-10-24-mini-coral-kmo-dir.md)。
run 在 xuyu A100 GPU0 上（本批临时用卡，用户原话「我觉得, baiju 和 qiuhao 全部,以及 xuyu 的 gpu 0 和 1 都可以用」），
与别人的进程共卡、显存封顶 0.40，12500/12500 iter，19:14，5.19 s/iter，0 报错。未做阈值扫描。

| | 本跑 | coral-exist | 线 |
|:--|--:|--:|:--|
| 实现核对：被选行集合变化率（最终） | **0**（0 / 1582） | — | 恰为 0 ✅ |
| M1：e_head / e_proxy / e_const（最终，dir 量纲） | **0.0397** / 0.0548 / 0.0590 | — | e_head 最小 ✅ |
| M2：前缀内改判率（最终） | **20.0%**（另有平局互换 7.1%） | — | ≥ 1% ✅ |
| `matched_F1@8`（读数 iter 9999） | **0.5926** | 0.6107 | ΔF1@8 = **−0.0181** ≤ −0.015 → **判否** |
| ΔF1@3 / ΔF1@5 | −0.0097 / −0.0060 | | 同时报 |
| `count_MAE_spike`（读数那次） | 1.7500 | 1.6655 | 护栏 ≤ 1.6655 ❌；同 iter 平均差 −0.0335 → 可能是读数选择效应 |
| segm AP 峰值 | 78.29 | 78.71 | 护栏 ≥ 77.80 ✅ |
| `spike_recall` 峰值 | 0.9445 | 0.9478 | 护栏 ≥ 0.9311 ✅ |
| 对 E0 的通用线：`matched_F1@8` | 0.5926 | | ≥ 0.6428 ❌（只照报） |

- **判否是稳的**：后半程 13 次同 iter 的 `matched_F1@8` 全部低于 coral-exist（均值 −0.0157），从 iter 2499 起可比的 21 次也全部更低。
- **机制按设计运转**：前缀守住了（被选槽集合一次也没变）；方向头学得和单独用时一样好（e_head 0.0397 vs A 的 0.0398，穗内 r 0.722 vs 0.716）；
  KMO 在前缀内换了 27.1% 实例的排列。但只换排列没有带来定位收益，反而略差。
- **计数与 coral-exist 持平**（最终评估偏差 +1.52 vs +1.44、r 0.623 vs 0.620），A 臂单独用时的计数副作用没有带进来。
- **对 A（只作描述）**：读数那次 F1@8 −0.033、计数 −1.10。
- **对「原理独立」**：在 coral-exist 的前缀里，方向版 KMO 的匹配项没有独立的正贡献。结合 2×2（单独用时 KMO 改变的配对约 85% 在换正样本槽），
  KMO 的定位收益更可能来自「换哪些槽当正样本」，而这正是 coral-exist 的前缀固定住的东西。这一条是非预注册的推断。
- 单种子，不补种子。
