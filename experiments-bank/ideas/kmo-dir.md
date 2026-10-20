# 想法：KMO 的近邻描述子从「平均距离」换成「近邻方向」

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-20 ｜ 分支 `mini/kmo-dir`（从 `mini/kmo-matcher`@`3319798` 开出；代码 `f7d1140`）｜ 配置 `wheat_joint_single_stage_mini_kmo_dir.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_kmo_ep100.yaml`）｜ 输出 `output_v2/exp-1020-mini-kmo-dir`

> 本实验是 KMO v2 三臂中的臂 A。另外两臂是 `mini/kmo-detach`（臂 C：KMO 头读 detach 后的特征）和
> `mini/kmo-auxonly`（臂 D：匹配代价权重置 0，只留辅助损失）。三臂都从 `mini/kmo-dir`@`f7d1140` 开出，
> 代码完全相同、只差配置，各写各的预注册，互不合并。
> 共同口径和判据线来自三臂共用的说明 `output_v2/_smoke-1020-kmo-v2/prereg_spec.md`，下文照抄，没有改线。

## 动机

用户 2026-10-20 补充了背景：`coral-exist`、`kmo-matcher`、`group-nms-match` 是待选的创新点，最终会放进同一个模型，
所以要求三者**在原理上相互独立**。现状是 KMO 对 `count_MAE_spike` 没有贡献，甚至有副作用；定位虽有提升，作用也有限。
因此继续改 KMO。

按本批的读数口径（见下文「共同口径」），KMO 100 epoch
（[exp-2026-09-24-mini-kmo-matcher-ep100](../experiments/exp-2026-09-24-mini-kmo-matcher-ep100.md)）对 E0 的差：
`matched_F1@8` +0.015（0.6378 vs 0.6228），`count_MAE_spike` +0.023（2.4878 vs 2.4648，单种子下在噪声内）。

### 1. 距离描述子 y^k 带着计数信号，而这部分对内层匹配没用

KMO 的 y^k 是 k=3 个近邻的平均距离（框相对）。2026-10-20 只读标注、在 CPU 上统计：

- 方差 **57% 在穗内、43% 在穗间**。按穗平均的 y^k 与小穗数 Ni 的相关 **r = −0.81**：同样大的框里小穗越多，间距越小，
  这就是计数信号。
- 内层匹配只在一个穗内做。穗间那部分对同穗的所有 GT 是同一个常数，只给代价矩阵每一列加同一个数，**改不了任何配对**。
  但 KMO 头仍要经共享 decoder 学它：`loss_point_knn` 是 L1 ×5，主层、`_0`~`_8`、`_interm` 共 11 层。
  KMO 要和 coral-exist（每穗计数）原理独立，辅助损失就不该往共享特征里灌计数信号。

### 2. 方向描述子不含计数信息

方向描述子 c_i 取指向 k=2 个最近邻的单位向量的平均：

- 单位向量已经把距离除掉，描述子**与尺度无关**，疏密不直接进入描述子。实测按穗平均的模长与 Ni 的 **r ≈ +0.01**。
  回归目标里没有计数信息，`loss_point_knn` 经共享 decoder 回传的梯度也就不带计数信号。coral-exist 用秩一致存在性头 +
  前缀内层匹配，管的是「这个穗有几个小穗」；方向描述子在原理上与它独立。
- 方向描述子编码「近邻在哪一侧」。**56% 的小穗两侧都有近邻（模长 < 0.3）**，方向接近 0；
  方向信息主要来自穗端和单侧点（近邻都在一侧时模长趋近 1）。

### 3. 配对模拟：方向的上限更高，但收益全靠头比点头更懂局部结构

在 val 的 613 个穗上做配对模拟。预测点 = GT + E0 实测量级的噪声（沿轴 / 横轴中位 4.62 / 2.34 px），
统计内层匈牙利把预测配回其来源 GT 的比例：

| 描述子从哪来 | L1（Eq.2） | 距离（k=3） | 方向（k=2） |
|:--|--:|--:|--:|
| 头完全准（取真值） | 93.2% | 95.7% | **98.2%** |
| 头只知道点头那么多（由带噪预测点自己算） | 93.2% | 93.1% | **90.9%** |

如果头比「从预测点推算」更懂局部结构，方向描述子的潜在收益比距离大。
如果头只学到点头已有的信息，方向反而比纯 L1 差。收益全靠头比点头更懂局部结构，所以机制门 M1 设为 **e_head < e_proxy**。

### 4. B 诊断：现有 KMO 头并不比「从预测点推算」更懂局部结构

用扩展版 `kmo_match_diag.py` 诊断现有的 KMO 100 epoch：train 集 100 个 batch，最后一层，L1-only 配对，Ni ≥ 3，共 1586 个实例。
JSON 在 `output_v2/exp-0924-mini-kmo-matcher-ep100/kmo_diag/model_{0008999,final}.v2.json`。

| | iter 8999 | final |
|:--|--:|--:|
| e_head（KMO 头） | 0.01769 | 0.01707 |
| e_proxy（由配上槽自己的预测点推算） | 0.01746 | 0.01721 |
| e_const（该实例 GT 中位数，偏乐观，只作参照） | 0.01575 | 0.01575 |
| e_head < e_proxy 的实例比例 | 52.7% | 55.6% |
| 穗内 r / 改判率 / 平局互换率 | 0.533 / 13.9% / 0.32% | 0.544 / 12.7% / 0.63% |

读法：
- KMO 头对局部结构的了解与「从预测点推算」基本一样，匹配项实际上接近噪声级的平局打破。
- KMO 约 +0.015 的定位收益，可能主要来自辅助回归损失对共享特征的塑形。臂 C、D 负责把这一点拆开。
- 本臂问另一个问题：换成不含计数信息、上限更高的方向描述子后，头能不能学得比点头更好。

**本臂的假设**：如果方向头能比「由预测点推算」更准（M1），内层配对就更对，`matched_F1@8` 应当上升。
方向描述子不含计数信息，所以计数不应变差。

## 修法

这一步不加新代码。`f7d1140` 在 KMO 上加了三个开关 `POINT_KMO_DESC` / `POINT_KMO_DIR_SCALE` / `POINT_KMO_DETACH`，
默认值下与 `mini/kmo-matcher` 逐位一致。本臂的配置只设三项：`POINT_KMO_DESC: "dir"`、`POINT_KMO_K: 2`、`POINT_KMO_DIR_SCALE: 0.13044`。

- **描述子**（`criterion.knn_resultant_direction`）：在框相对坐标 `gt_rel` 上计算
  c_i = (1/k_eff)·Σ_{j∈kNN(i)} (p_j − p_i) / max(‖p_j − p_i‖, 1e-12)，其中 k_eff = min(K, Ni−1)，K = 2，不含自身；Ni ≤ 1 时为 0。
  模长 ∈ [0, 1]：两侧各有一个近邻（穗中段）时趋近 0，两个近邻都在一侧（穗端、单侧点）时趋近 1。
- **头**：沿用 `point_knn_embed`（3 层 MLP，隐层 256），每个 query 输出 P·2 = 50 维（P = 25），过 **tanh**，每个分量 ∈ [−1, 1]。
  它与 dist 的头只差最后一层的输出维（25 → 50），参数量从 52,438,442 变为 52,444,867（+6,425）。
  这个头只在训练期用（内层匹配的代价和 `loss_point_knn`），推理不用。
- **代价**（内层匹配，P 个点槽 × Ni 个 GT 小穗）：
  C = 5·‖p − p̂‖₁ + 2·focal(存在性) + 5·s·‖ĉ − c‖₁。
  KMO 项从 `5·|ŷ^k − y^k|` 换成两分量的 L1 之和再乘 s。
- **损失**：`loss_point_knn` = s·Σ_匹配点 ‖ĉ − c‖₁ / num_pos，两分量求和，分母不变，再乘 5（`POINT_KNN_WEIGHT`）。
  主层、`_0`~`_8`、`_interm` 共 11 项，与 KMO 相同。
- **尺度 s = 0.13044**：取 mini train 全部 wheat_2 GT 小穗（Ni ≥ 2，按原始标注复核共 43024 点）到其 2 个最近邻的平均距离（框相对），
  再取中位数。s 把单位方向换算成与坐标同一量纲的长度：近邻都在一侧时，s·c 约等于指向近邻的平均偏移。
  这样代价与损失的权重 5 仍然是「与坐标项同权」，与 KMO 的含义一致（官方实现把 (x, y, d) 放进同一个 L1）。
- `POINT_KMO_DETACH` 保持 False：头读未 detach 的 decoder 输出，与 KMO 相同。detach 是臂 C 的变量。

## 与对照的差异

- **对 KMO（归因参照，`mini/kmo-matcher`，`output_v2/exp-0924-mini-kmo-matcher-ep100`）：只差描述子**，
  即 dist、k=3 换成 dir、k=2。代价与损失的权重、DETACH、schedule、数据和其余旋钮全部相同。
  `f7d1140` 在默认开关下与 `mini/kmo-matcher` 逐位等价（见下文「实施前已完成的验证」），所以代码上也只有描述子这一条路径不同。
- **对 E0（主对照，`mini/baseline` 100 epoch，`output_v2/exp-0923-mini-ep100`）**：多开了 `POINT_KMO`（dir 描述子）。

**cfg 核对**：用 `get_cfg()` + `add_deeplab_config` + `add_maskdino_config` + `merge_from_file` 解析，
全部 331 个键逐项比对。

| 比较对象 | 取值不同的键 |
|:--|:--|
| `wheat_joint_single_stage_mini_kmo_ep100.yaml` | 只有 `POINT_KMO_DESC`（`dir` vs `dist`）和 `POINT_KMO_K`（2 vs 3） |
| KMO run 落盘的 `exp-0924-mini-kmo-matcher-ep100/config.yaml` | 同上两个键。另有 `OUTPUT_DIR`，以及 `BBOX_REG_WEIGHTS` 的 tuple / list 写法差（值相同，落盘格式所致） |
| E0 run 落盘的 `exp-0923-mini-ep100/config.yaml` | 上面两个键之外，再多 `POINT_KMO`（True vs False）。`OUTPUT_DIR` 与 tuple / list 写法差同上 |

- `POINT_KMO_DIR_SCALE` 两边都是 0.13044，所以不出现在差异里。它等于 `maskdino/config.py` 的默认值，dist 下不用；
  新配置里显式写出，只为对照。
- 其余关键项两边相同：`POINT_KMO_DETACH` False，`COST_POINT_KNN_WEIGHT` / `POINT_KNN_WEIGHT` 5.0，
  `IMS_PER_BATCH` 4、`MAX_ITER` 12500、`EVAL_PERIOD` 500、`SEED` 42。

## 预期与判否线（跑之前写死）

判据线照抄三臂共用的 `prereg_spec.md`，没有改线。

### 共同口径

- 主对照 E0 = `output_v2/exp-0923-mini-ep100`（`mini/baseline` 100 epoch）；
  归因参照 KMO = `output_v2/exp-0924-mini-kmo-matcher-ep100`（`mini/kmo-matcher`）。
- **读数**：取后半程（iter ≥ 6250）内置评估里 `count_MAE_spike` 最低的那一次，全部指标都报同一次评估的值。
  只用默认读数（τ 0.35），**不扫阈值**；**单种子**，不补种子；`matched_F1@8` 的噪声地板为 **0.015**。
- E0 与 KMO 的读数都落在 iter 8999，已对照两个 run 的 metrics.json 复核：

| | `count_MAE_spike` | `matched_F1@3 / @5 / @8` | `spike_recall` | segm AP | 全程峰值：segm AP / `spike_recall` |
|:--|--:|--:|--:|--:|--:|
| E0 | 2.4648 | 0.1504 / 0.3466 / 0.6228 | 0.9511 | 77.40 | 78.80 / 0.9511 |
| KMO | 2.4878 | 0.1647 / 0.3579 / 0.6378 | 0.9331 | 77.64 | 78.60 / 0.9511 |

- 本批（10-07）对 E0 的通用线：`matched_F1@8` ≥ **0.6428**（+0.02）、segm AP 峰值 ≥ **77.80**、`spike_recall` 峰值 ≥ **0.9311**。
- 计数护栏是本组 KMO 臂专用的：三个创新点要原理独立，KMO 不能伤计数。
  读数那次的 `count_MAE_spike` ≤ **2.4648**，即不比 E0 差。KMO 自己是 2.4878（+0.023，单种子下在噪声内）。
- 机制门的数：用扩展版 `myscripts/backup/vis_process/kmo_match_diag.py` 在本 run 的 `model_final.pth` 上跑，train 集 `--batches 100`。

### 机制门

- **M1**：e_head < e_proxy，且 e_head < e_const。
  - 取诊断 JSON 的 `e_head` / `e_proxy` / `e_const`，都是点加权均值；dir 时都是两分量 L1 再乘 s。
  - dir 的 e_const 是 0 向量。它不知道真值，不像 dist 的实例中位数那样偏乐观。
- **M2**：最后一层的改判率（对 L1-only）≥ **1%**，即 JSON 的 `M2_override_rate`。

### 主判据：读数那次的 `matched_F1@8`

| 条件 | 判定 |
|:--|:--|
| ≥ **0.6428** | **成立**（M1、M2、全部护栏也要过） |
| 落在 (0.6228, 0.6428) | **歧义** |
| ≤ **0.6228** | **判否** |

### 归因（对 KMO）

| F1@8 − 0.6378 | 结论 |
|:--|:--|
| ≥ +0.015 | 方向优于距离 |
| 绝对值 < 0.015 | 未决 |
| ≤ −0.015 | 方向更差 |

同时报 `matched_F1@3` / `@5` 对 KMO 的差。

### 护栏

- 计数：读数那次的 `count_MAE_spike` ≤ **2.4648**。
- segm AP 峰值 ≥ **77.80**。
- `spike_recall` 峰值 ≥ **0.9311**。

训练结束后只写内置评估的结果，不跑任何阈值扫描（score / count / point threshold）。

## 实施前已完成的验证

### 代码提交 `f7d1140`

脚本与产物在 `output_v2/_smoke-1020-kmo-v2/`。

- **单元测试**（CPU）：
  - `knn_resultant_direction` 对照暴力实现，共 525 组，最大误差 1.2e-7。
  - dir 时 ĉ 取常数，3000 例的分配代价都等于 Eq.2 最优（最大差 4e-7）。
  - dist 路径在两个 worktree 的 dump 逐位相同（3000 次内层匹配、60 个 batch 的 `loss_points`）。
- **逐位等价**：默认开关下与 `mini/kmo-matcher` 的 138 项 loss 逐位相等，参数 52,438,442，738 个张量哈希相同。
- **DETACH**：打开时，`loss_point_knn` 对头以外 409 个参数的梯度全为 None。本臂不开 DETACH，这一条只核对开关本身。
- **4090 上的 dir + detach 冒烟**：bs2、1.37 s/iter、峰值 12.3 GB（K 仍是 3）。扩展后的诊断在该 checkpoint 上跑通了 dir 分支。
- **诊断回归**：dist 下，KMO `model_final` 的原有字段与 09-24 的旧 JSON 一致。
  两边同为 1586 实例 / 24766 点；r 0.5441 vs 0.5444，推翻率 12.67% vs 12.61%。

### 本配置的 cfg 核对

见上文「与对照的差异」：与 KMO 的 ep100 配置相比，只差 `POINT_KMO_DESC`、`POINT_KMO_K` 两个键。

### 本配置的冒烟（xuyu GPU0，A100 80GB，2026-10-20 12:01）

新配置原样（`IMS_PER_BATCH` 4），只加 `SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 0 SOLVER.CHECKPOINT_PERIOD 40`。
输出在 `output_v2/_smoke-1020-kmo-v2/smoke40-kmo-dir`，启动脚本 `scripts/run_smoke40_kmo_dir.sh`。

- **loss**：iter 19 / 39 两次记录各 138 项，全部有限。11 个 `loss_point_knn*`（主层、`_0`~`_8`、`_interm`）都在，
  `loss_point_knn` 约 0.25（×5 后）。
- **速度**：第 20 iter 之后稳定在 **2.72 s/iter**（iter 20~39 的均值；同一窗口的中位数为 2.67）。
- **显存**：峰值 **22,974 MiB**（日志的 `max_mem`，PyTorch 分配峰值）。nvidia-smi 看到的进程占用峰值为 51,966 MiB，含缓存分配器的预留。
- **评估**：训练结束后的 val 评估（200 图）完整跑通，用时 4:37。40 iter 全在 warmup 里（lr ≤ 5e-8），
  未训练模型各指标为 0 / nan。`count_MAE_img` 为 48.735，与 KMO 冒烟时未训练模型的 48.74 相同。
- **诊断**：在冒烟的 `model_final.pth` 上跑 `kmo_match_diag.py --batches 5`，输出 `diag_smoke40_kmo_dir.json`。
  dir 分支运行正常，字段齐全：
  - 配置字段：`kmo_desc` dir、`kmo_k` 2、`kmo_dir_scale` 0.13044。
  - e_head / e_proxy / e_const = 0.0643 / 0.1302 / 0.0563，并有 e_head < e_proxy 的实例比例。
  - 改判率 98.7%、平局互换率 1.3%、穗内 r 0.027。
  - ‖c‖ ≥ 0.3 的点占 41.8%，并报了这些点上 ĉ、y_proxy 与 c 的余弦。

  这是未训练的头，数值只说明管线通了，不作机制判读。改判率接近 100%，与 [kmo-matcher](kmo-matcher.md)「已知风险」里
  60 iter 时的 99% 是同一现象。

## 已知限制

- B 诊断显示，现有距离头对局部结构的了解不比由预测点推算多。方向头能否学得更好，正是本实验的未知数，先验不高。
- 56% 的小穗两侧都有近邻，方向接近 0；距离恰好相等时，方向取决于 topk 选中哪几个邻居。
- 单种子。

## 成本

在 xuyu A100 GPU0 上跑，估计 10.4 ~ 11.5 h。

- 按冒烟的 2.72 s/iter：训练 12500 iter 约 9.4 h，加 25 次内置评估（参照 KMO 100 epoch 的 55:43），约 10.4 h。
- KMO 100 epoch 实测 11:34，当时是 3.07 s/iter，与别的 run 共卡并发。本 run 独占 GPU0，但与臂 C（GPU1）及别人的任务同机；
  如果被拖到同样的速度，约 11.5 h。
- 另加结束后的机制门诊断（100 batch），约 3 min。
