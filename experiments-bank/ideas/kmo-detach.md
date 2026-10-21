# 想法：KMO 头的输入 detach —— 让 KMO 只通过匹配起作用

状态：**预注册判定：判否**（detach 没消掉副作用：读数那次计数 2.5487，线 ≤ 2.4648；机制门、`matched_F1@8` 0.6443 与护栏都过）｜ 验证于 [exp-2026-10-20-mini-kmo-detach](../experiments/exp-2026-10-20-mini-kmo-detach.md) ｜ 提出 2026-10-20（判定线在跑之前写死，启动前修订过一次，见「预注册修订」）｜ 分支 `mini/kmo-detach`（从 `mini/kmo-dir`@`f7d1140` 开出，代码与 `mini/kmo-dir` 相同）｜ 配置 `configs/coco/instance-segmentation/wheat_joint_single_stage_mini_kmo_detach.yaml` ｜ 输出 `output_v2/exp-1020-mini-kmo-detach`

> KMO v2 三臂之一（臂 C）。三臂都从 `mini/kmo-dir`@`f7d1140` 开出，代码完全相同，只差配置。`f7d1140` 是代码提交，
> 在 `mini/kmo-matcher`@`3319798` 上给 KMO 加了两个开关，默认值下与 `mini/kmo-matcher` 逐位一致。
> 三臂分工：臂 A `mini/kmo-dir` 换方向描述子；臂 C（本分支）只开 detach；臂 D `mini/kmo-auxonly` 只把匹配代价权重置 0。

## 动机

### 三个创新点要原理独立

三个创新点将放进同一个模型：coral-exist（秩一致存在性头，管计数）、kmo-matcher（KMO 内层点匹配，管穗内配对）、
group-nms-match（匹配前的逐层点 NMS）。这要求它们原理独立，各自只动自己那一环，叠在一起时不互相抵消。
落到 KMO 上，就是 KMO 不能碰计数。

### KMO 的 y^k 里有计数信号，KMO 头却经共享 decoder 学它

数据来自 2026-10-20 的离线统计（只读标注，CPU）：

- KMO 的 y^k 是 k=3 的平均近邻距离（框相对）。它的方差 57% 在穗内、43% 在穗间。
  按穗平均后，y^k 与 Ni 的相关 r = −0.81，所以 y^k 带着计数信号。
- 内层匹配只在一个穗内做。穗间那部分对同穗所有 GT 是常数，改不了任何配对。
- KMO 头却要经共享 decoder 学完整的 y^k（L1 ×5；共 11 个 `loss_point_knn*`，来自 interm、初始预测和 9 层 decoder）。
  所以对配对没用的穗间部分，也照样通过 `loss_point_knn` 回传进共享特征。这可能就是 KMO 计数副作用的来源。

已有的计数证据：读数那次（iter 8999），KMO 的计数是 2.4878，E0 是 2.4648，高 0.023，单种子下在噪声内。

所以副作用是否真的存在还说不准。本臂要回答的是：去掉这条回传路径以后，KMO 是否不伤计数。

### B 诊断：KMO 头不比「由预测点推算」更懂局部结构

诊断设置：2026-10-20，扩展版 `kmo_match_diag.py`，在现有 KMO 100 epoch 上跑；train 100 batch，看最后一层，
L1-only 配对，只取 Ni ≥ 3 的实例，共 1586 个。
JSON 在 `output_v2/exp-0924-mini-kmo-matcher-ep100/kmo_diag/model_{0008999,final}.v2.json`。

| | iter 8999 | final |
|:--|--:|--:|
| e_head（KMO 头） | 0.01769 | 0.01707 |
| e_proxy（由配上槽自己的预测点推算） | 0.01746 | 0.01721 |
| e_const（该实例 GT 中位数，偏乐观，只作参照） | 0.01575 | 0.01575 |
| e_head < e_proxy 的实例比例 | 52.7% | 55.6% |
| 穗内 r / 改判率 / 平局互换率 | 0.533 / 13.9% / 0.32% | 0.544 / 12.7% / 0.63% |

**读法：**

- KMO 头对局部结构的了解，与「从预测点推算」基本一样。
- 所以匹配项在实际中接近噪声级的平局打破。
- KMO 约 +0.015 的定位收益（读数那次 `matched_F1@8` 0.6378 vs E0 0.6228），可能主要来自辅助回归损失对共享特征的塑形，
  而不是来自匹配。

### 本臂与臂 D 合起来做 2×2 归因

KMO 由两部分绑在一起：匹配代价，加上经共享 decoder 回传的辅助回归损失。
本臂去掉后者的回传、保留匹配；臂 D 去掉匹配（`COST_POINT_KNN_WEIGHT` 0）、保留辅助损失。连同 E0 与 KMO，正好凑成 2×2：

| | 辅助损失不回传到共享特征 | 辅助损失回传到共享特征 |
|:--|:--|:--|
| 不带 KMO 匹配 | E0（`exp-0923-mini-ep100`） | 臂 D `mini/kmo-auxonly` |
| 带 KMO 匹配 | **臂 C（本分支）** | KMO（`exp-0924-mini-kmo-matcher-ep100`） |

2×2 同时回答两个问题：

1. KMO 的计数副作用（若存在）走的是哪条路。
2. KMO 的定位收益来自匹配，还是来自辅助损失。

## 修法

没有代码改动。开关随 `f7d1140` 已在代码里（`MODEL.MaskDINO.POINT_KMO_DETACH`，默认 False）。
本分支只新增配置 `wheat_joint_single_stage_mini_kmo_detach.yaml`：`_BASE_` 是 `wheat_joint_single_stage_mini_kmo_ep100.yaml`（KMO 100 epoch），
只设 `POINT_KMO_DETACH: True`。

**detach 的位置：** `maskdino/modeling/transformer_decoder/maskdino_decoder.py` 的 `forward_prediction_heads`，
在 `decoder_norm` 之后、KMO 头 `point_knn_embed`（3 层 MLP）之前：

```python
knn_in = decoder_output.detach() if self.point_kmo_detach else decoder_output
outputs_point_knn = self.point_knn_embed(knn_in).reshape(b, q, self.num_points, self.point_kmo_dim)
```

interm、初始预测和 9 层 decoder 的输出都走这个函数，所以 11 个 `loss_point_knn*` 的输入全部被 detach。

**含义：**

- **前向不变。** ŷ^k 的数值与不 detach 时逐位相同。内层匹配照常用它，
  代价仍是 Eq.3：C = 5·|p − p̂|₁ + 5·|y^k − ŷ^k| + 2·focal(ĉ)。
- **反传被切断。** `loss_point_knn`（×5）只训练 KMO 头自己的参数，不再回传到 `decoder_norm` 及其上游
  （decoder 各层、像素解码器、主干）。
- **KMO 只剩一条路影响模型其余部分：** ŷ^k 改变内层配对 → 坐标 / 存在性损失的目标跟着变 → 这些损失的梯度去塑形共享特征。
- **推理不受影响。** KMO 只用于训练期的内层匹配。

## 与对照的差异

- **对 KMO**（`mini/kmo-matcher`，`exp-0924-mini-kmo-matcher-ep100`）：只差 detach。
  以下全部相同：描述子（dist，k=3）、匹配代价权重 5、损失权重 5、数据、schedule（12500 iter = 100 epoch，bs 4）、
  初始权重（ImageNet R-50）、`SEED` 42。
- **对 E0**（`mini/baseline`，`exp-0923-mini-ep100`）：差 `POINT_KMO`（KMO 匹配与 KMO 头）和 detach 两处。

**cfg 核对。** 方法：`get_cfg()` + `add_deeplab_config` + `add_maskdino_config` + `merge_from_file`，在 CPU 上展平后，
逐项比对全部叶子键。

| 比对 | 叶子键 | 取值不同的键 |
|:--|--:|:--|
| 新配置 vs `wheat_joint_single_stage_mini_kmo_ep100.yaml` | 331 / 331 | **只有** `MODEL.MaskDINO.POINT_KMO_DETACH`：False → True |
| 新配置 vs KMO run 落盘的 `exp-0924-mini-kmo-matcher-ep100/config.yaml` | 331 / 331 | `POINT_KMO_DETACH`，外加 `OUTPUT_DIR`，以及 `ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS` 的 list / tuple 序列化差别（值相同，MaskDINO 不用） |
| 新配置 vs E0 落盘的 `exp-0923-mini-ep100/config.yaml` | 331 / 331 | `POINT_KMO`、`POINT_KMO_DETACH`，外加同上两项 |

第二行说明：本分支上的 KMO 配置能复现 KMO run 实际用的 cfg，所以「对 KMO 只差 detach」对那个 run 本身也成立。

## 预期与判否线（跑之前写死）

### 共同口径（KMO v2 三臂共用）

- **对照。** 主对照 E0 = `output_v2/exp-0923-mini-ep100`（mini/baseline 100 epoch）；
  归因参照 KMO = `output_v2/exp-0924-mini-kmo-matcher-ep100`（`mini/kmo-matcher`）。
- **读数。** 取后半程（iter ≥ 6250）内置评估里 `count_MAE_spike` 最低的那次，在同一次评估上报全部指标。
  - 只用默认读数（τ 0.35），不扫阈值。
  - 单种子，不补种子。
  - `matched_F1@8` 噪声地板 0.015。
- **E0 读数（iter 8999）：**
  - `count_MAE_spike` 2.4648；`matched_F1@3/5/8` 0.1504 / 0.3466 / 0.6228；`spike_recall` 0.9511；segm AP 77.40。
  - 全程峰值：segm AP 78.80、`spike_recall` 0.9511。
- **KMO 读数（iter 8999）：**
  - `count_MAE_spike` 2.4878；`matched_F1@3/5/8` 0.1647 / 0.3579 / 0.6378；`spike_recall` 0.9331；segm AP 77.64。
  - 全程峰值：segm AP 78.60、`spike_recall` 0.9511。
- **对 E0 的通用线**（沿用 10-07 那批的预注册）：`matched_F1@8` ≥ 0.6428（+0.02）、segm AP 峰值 ≥ 77.80、`spike_recall` 峰值 ≥ 0.9311。
- **计数护栏**（本组 KMO 臂专用：三个创新点要原理独立，KMO 不能伤计数）：读数那次 `count_MAE_spike` ≤ 2.4648，即不比 E0 差。
  KMO 自己是 2.4878（+0.023，单种子下在噪声内）。
- **机制门的数**：用扩展版 `myscripts/backup/vis_process/kmo_match_diag.py`，在本 run 的 `model_final.pth` 上跑，train 集 `--batches 100`。

以上 E0 / KMO 读数已按两个 run 的 `metrics.json` 复核。

### 机制门

M1′ 在启动前换过线，理由见文末「预注册修订」。两项都在最终 checkpoint 上读。

- **M1′（头仍学得到穗内结构）：** 穗内 Pearson r ≥ 0.3，即诊断的 `posthoc_within_inst_pearson_r`。
- **M2（不是空操作）：** 改判率 ≥ 1%，即诊断的 `M2_override_rate`。

参照值来自 B 诊断（KMO，未 detach，final）：穗内 r 0.544、改判率 12.7%。
e_head / e_proxy / e_const（KMO 为 0.01707 / 0.01721 / 0.01575）照常报告，只作参照，不作判据。

### 主判据

两条：

- (i) 读数那次计数 ≤ 2.4648；
- (ii) 读数那次 `matched_F1@8` ≥ 0.6428。

先看机制门，再看主判据和护栏。前两行的判定名是启动前补全的，见文末「预注册修订」。

| 条件 | 判定 |
|:--|:--|
| 机制门不过（M1′ 或 M2） | **「机制未启动」**：主判据照常读数并报告，但不能归因为「只靠匹配的 KMO」 |
| 机制门过；(i)、(ii) 都过，护栏不过 | **「主判据过线、护栏未过（不记成立）」** |
| 机制门过；(i)、(ii) 都过，护栏都过 | **成立** |
| 机制门过；(i) 过、(ii) 不过 | **「不伤计数，但定位收益未过线」** |
| 机制门过；(i) 不过 | **判否**（detach 没消掉副作用） |

### 归因（对 KMO）

报计数与 `matched_F1@8` 之差（C − KMO）；在噪声内不下结论。

### 2×2 归因（与臂 D 合报）

E0 = 都没有，C = 只有匹配，D = 只有辅助损失，KMO = 都有。

- 匹配主效应 = ((C − E0) + (KMO − D)) / 2
- 辅助损失主效应 = ((D − E0) + (KMO − C)) / 2

在 `matched_F1@8` 与计数上各报，都用读数那次。单种子，F1 小于 0.015 的差不下结论。

### 护栏

- segm AP 峰值 ≥ 77.80；
- `spike_recall` 峰值 ≥ 0.9311。

## 实施前已完成的验证

### 代码提交 `f7d1140` 的验证

产物在 `output_v2/_smoke-1020-kmo-v2/`，汇总见 `equiv/compare.txt`。

- **默认开关下与 `mini/kmo-matcher` 逐位等价**（默认开关即 `POINT_KMO_DESC=dist`、`POINT_KMO_DETACH=False`）。
  - 设置：mini train 前 2 张图，训练态 fp32 前向一次。
  - 结果：138 项 loss 全部逐位相等，参数量 52,438,442，738 个 state_dict 张量的哈希相同。
- **DETACH 检查**（本臂的组合 dist + DETACH；1 张图，只对 11 个 `loss_point_knn*` 求和后反传）。
  - 开时，KMO 头以外 409 个参数的梯度全为 None，decoder 0/208 个参数有梯度；KMO 头的各参数都有非零梯度。
  - 关时，decoder 202/208 个参数有非零梯度，主干也有。
  - 同一描述子下，开与关的前向 loss 逐位相同。

### 本分支

- **cfg 核对：** 见上「与对照的差异」，新配置相对 KMO 100 epoch 只差 `POINT_KMO_DETACH`。
- **目标卡冒烟**：2026-10-20 14:44–14:52，xuyu A100 GPU1，与别人约 16 GB 的进程共卡；
  从 `931419c` 启动，脚本是 `output_v2/_smoke-1020-kmo-v2/scripts/detach_run_smoke.sh`。
  - 设置：新配置原样（`IMS_PER_BATCH` 4），加 `SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 0 SOLVER.CHECKPOINT_PERIOD 40`。
    输出在 `output_v2/_smoke-1020-kmo-v2/smoke40-kmo-detach`，启动日志是 `output_v2/_smoke-1020-kmo-v2/smoke40-kmo-detach.launch.log`。
  - 运行：train rc=0，0 个 Traceback。训练结束时的 val 评估跑通（未训练模型，各指标为 0，`count_MAE_img` 48.74）。
  - 损失：138 项 loss 全部有限；11 个 `loss_point_knn*` 都在，约 1.60–1.66。
  - 速度与显存：3.48 s/iter（38 iter 用时 2:12），第 20–39 iter 的中位 iter 时间 2.98 s；`max_mem` 22,877 MiB。
  - 诊断：在冒烟的 `model_final.pth` 上跑 `--batches 5`，rc=0，字段齐全
    （含 e_head / e_proxy / e_const 与 `posthoc_within_inst_pearson_r`）。头未训练，数值无意义。产物在 `output_v2/_smoke-1020-kmo-v2/detach/`。

## 已知限制

- **只能证明「不比 E0 差」。** KMO 自己的计数差（+0.023）在单种子噪声内。
  即使 detach 真的消掉了副作用，读数上也只能看到「≤ 2.4648」，看不出「比 KMO 好」。
  反过来，(i) 是不带余量的硬线，单种子噪声也可能让本臂在没有副作用时不过线。
- **detach 后，KMO 头只能依赖别的损失塑形的特征。**
  - 头不能再让共享特征迁就自己，ŷ^k 可能学得更差（e_head 变大），匹配项也随之更接近噪声。
  - 因此本臂里的「匹配」与 KMO 里的「匹配」不完全是同一个东西，2×2 的匹配主效应是这两种头的平均。
  - 交互项 KMO − C − D + E0 若明显不为零，两个主效应都要打折看。
- **两个主效应之和恒等于 KMO − E0**（`matched_F1@8` +0.0150、计数 +0.0230）。
  总量本身就在噪声地板上，拆开后各自更小，2×2 很可能落在噪声内，按「不下结论」处理。
- **单种子。** 与所有 mini 实验一致，不补种子。

## 成本

- **卡：** xuyu A100 GPU1，与别人的进程（约 16 GB）共卡。
  - 这是本批的临时例外。CLAUDE.md 只许用 qiuhao A100 GPU1。
  - 用户 2026-10-20 原话：「这次你可以在 ssh qiuhao 的卡 0, ssh xuyu 的卡 0 和 1 和 ssh baiju 的卡 0 上同时进行」。
- **训练：** 12500 iter，按冒烟实测 3.48 s/iter 估约 12.1 h。
- **评估：** 25 次约 1 h（KMO 100 epoch 的 25 次评估共 55:43）。
- **合计：** 约 13 h。可能偏保守：冒烟只有 38 iter，均值受个别慢 iter 影响，第 20–39 iter 的中位是 2.98 s；
  KMO 100 epoch 在同一张卡上全程实测 3.07 s/iter（11:34:34）。
- **诊断：** `model_final` 上的机制诊断（100 batch）约 3 min。

## 预注册修订（2026-10-20，启动前）

正式训练启动前改了以下几处。除 M1′ 换线外，判据线的数值一条没动：主判据 (i) ≤ 2.4648 与 (ii) ≥ 0.6428、
M2 的 1%、护栏 77.80 / 0.9311，都与初稿（`931419c`）相同。

1. **M1′ 换线：e_head < e_const → 穗内 Pearson r ≥ 0.3**（最终 checkpoint，`posthoc_within_inst_pearson_r`）。
   - 原线不可用。dist 的 e_const 是知道每穗 GT 中位数的常数，偏乐观，B 诊断表里也标为「只作参照」。
     连没有 detach 的 KMO 自己的头都不过（final 0.01707 vs 0.01575），按原线本臂几乎必然不过，机制门失去区分力。
   - 新线取自 KMO 100 epoch 原预注册（[kmo-matcher-ep100](kmo-matcher-ep100.md)）的 M1′，KMO 实测 0.544。
     它直接检验头在同一穗内各槽之间有没有信号，这正是 KMO 改配对所需要的。
   - e_head / e_proxy / e_const 照常报告，只作参照。
2. **补全判定名。** 初稿只列了机制门和主判据，没写下面两种情况怎么判：
   - 机制门（M1′ 或 M2）不过：主判据照常读数，但判定写「机制未启动」。
     这时 detach 后的头没学到穗内结构，或者不改配对，端到端差异不能归因为「只靠匹配的 KMO」。
   - (i)、(ii) 都过，但护栏不过：写「主判据过线、护栏未过（不记成立）」。
3. **删去动机里 KMO 末值计数那句**（2.82 vs 2.60，偏差 +2.68 vs +2.41）。
   对比一律用后半程 `count_MAE_spike` 最低那次的读数，不用 model_final 末值。
4. **成本改用本配置在目标卡上的冒烟实测**（3.48 s/iter），并写明用卡是本批的临时例外，附用户原话。
   初稿借用的是 KMO 100 epoch 的 3.07 s/iter。
5. **冒烟从「待补」改为实测结果**，见「实施前已完成的验证」。

## 结论（2026-10-21）

**预注册判定：「判否」（detach 没消掉副作用）。** 详见 [exp-2026-10-20-mini-kmo-detach](../experiments/exp-2026-10-20-mini-kmo-detach.md)。

| | 臂 C（detach） | KMO | E0 | 线 |
|:--|--:|--:|--:|--:|
| M1′ 穗内 Pearson r（终） | **0.508** | 0.544 | — | ≥ 0.3 ✅ |
| M2 改判率（终） | **13.2%** | 12.7% | — | ≥ 1% ✅ |
| (i) 读数那次 `count_MAE_spike` | **2.5487**（iter 9999） | 2.4878 | 2.4648 | ≤ 2.4648 ❌ |
| (ii) 读数那次 `matched_F1@8` | **0.6443** | 0.6378 | 0.6228 | ≥ 0.6428 ✅ |
| 护栏 segm AP 峰值 | 78.79 | 78.60 | 78.80 | ≥ 77.80 ✅ |
| 护栏 `spike_recall` 峰值 | 0.9657 | 0.9511 | 0.9511 | ≥ 0.9311 ✅ |

- **机制门通过。** detach 后的头仍学到穗内结构，也照样改配对，所以测到的是「只靠匹配的 KMO」。
  参照项里头的误差变大：e_head 0.01847，高于由预测点推算的 0.01706。
- **计数没过线，按判定表判否。** 读数那次比 E0 高 0.084。切断 `loss_point_knn` 经共享 decoder 的回传以后，KMO 的计数劣势仍在：
  同 iter 比，后半程 13 次评估里 C 有 11 次、KMO 有 12 次计数高于 E0。
- **定位过线，但对 KMO 的差都在噪声内。** `matched_F1@8` 0.6443，比 E0 高 0.0216。
  对 KMO：`matched_F1@8` +0.0065（`@3` +0.0009、`@5` +0.0116），计数 +0.061，都不下结论。
- **计数的形态变了。** 最终评估上，C 的计数偏差回到 E0 水平（+2.40；KMO +2.68），但逐穗相关最弱（r 0.611；E0 0.642、KMO 0.658）。
  这只是末值上的一次评估，不改判定。
- **2×2 归因（与臂 D `mini/kmo-auxonly` 合报，单种子）：KMO 的定位收益来自匹配项，不来自辅助损失。**
  - `matched_F1@8` 的匹配主效应 +0.0201，过噪声地板 0.015；辅助损失主效应 −0.0051，在噪声内。
  - 计数上匹配项 +0.163（变差）、辅助损失 −0.140（变好），大致抵消。交互 +0.157 与主效应同量级，只作描述、不下结论。
  - 详见结果文档的「2×2 归因」一节。判定（判否）不变。
- 未做阈值扫描。
