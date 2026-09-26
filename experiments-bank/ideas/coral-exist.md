# 想法：秩一致存在性头 + 前缀内层匹配（B 臂）

状态：**成立**（`count_MAE_spike` 峰值 1.67 / 末值 2.02，基线 2.46 / 2.60；`matched_F1@8` 峰值 0.6301，末 4 轮比基线低 0.0097，在噪声内）｜ 验证于 [exp-2026-09-26-mini-coral-exist](../experiments/exp-2026-09-26-mini-coral-exist.md) ｜ 2026-09-26 15:11:56 启动于 `eb3e466`，qiuhao A100 GPU1，与 A 臂并发 ｜ 预注册提交于 `eb3e466`（跑之前）｜ 提出 2026-09-26 ｜ 分支 `mini/coral-exist`（从 `mini/baseline`@`0f71161` 开出，100 epoch）｜ 配置
`wheat_joint_single_stage_mini_coral_exist.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜ 输出 `output_v2/exp-0926-mini-coral-exist`

来源：Cao, Mirjalili, Raschka, *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*
（Pattern Recognition Letters 2020，arXiv 1901.07884），本地源码 `/public/home/zncui/pycharm_project/coral-cnn`。

同一批还有两条 CORAL 分支，都从 `mini/baseline`@`0f71161` 分出，彼此独立、不合并：
`mini/coral-count`（A：保留原点头，另挂一个 CORAL 计数头）与 `mini/coral-img`（C：图级 wheat_2 穗数）。

## 现象

1. **每穗计数带系统偏差。** ep100 基线（`output_v2/exp-0923-mini-ep100`）末值：配上的 578 个穗
   **每穗平均多报 2.40 个点**，`count_MAE_spike` 2.599。计数是「25 个无序槽里存在性 > 0.35 的个数」，从未被直接监督。
2. **有序前缀这个目标，计数上已经被证明有效。** `exp/point-canonical-order` 分支的
   `experiments-bank/experiments/exp-2026-08-25-point-canonical-order.md`（overfit10 训练，81 图 val 上检验）：
   把 GT 小穗沿主轴排序、恒等分配到 slot 0..Ni−1 后，val 计数 MAE 从 8.60 降到 **2.38（−72%）**，是那一批唯一泛化下来的收益。
   但它有两个问题：
   - **定位变差**：恒等分配要求 slot k 命中特定那个小穗，`loss_point_coord` 收敛值是基线的 5.3 倍，mF1 最低。
     当时的诊断是「前缀目标」与「坐标的 slot→位置语义」被绑在了一起。
   - **前缀不完全一致**：独立的 25 个 logit 学到的前缀一致性只有 0.762，没有任何结构保证「slot k 亮则 slot k−1 也亮」。

## 假设

「slot k 亮 ⟺ `Ni > k`」本身就是 CORAL 的 25 个二分类任务。把存在性头换成 CORAL 结构
（`logit_k = g(h) + b_k`，共享 1 维打分、各自一个 bias），并把内层匹配限制在前缀上：

1. **计数**：拿到 canonical-order 的前缀目标收益，同时由 CORAL 结构保证单调（定理 1：最优点上 bias 有序，
   `P(Ni>k)` 随 k 单调不增），读出时按 CORAL 的求和规则取 `K = Σ_k 1[P(Ni>k) > 0.5]`。
2. **定位**：前缀内部仍做匈牙利（代价只用坐标 L1），GT 小穗可以按距离配给前 Ni 个 slot 中的任何一个，
   **不引入** canonical-order 那种 slot→位置语义。这就是 canonical 文档「下一步 3」想要的拆分：
   前缀目标保留，坐标一致性约束不加。
   代价是坐标头的配对自由度从「25 选 Ni 再排列」降到「前 Ni 个的排列」，定位可能仍有损失，这正是主判据第二维要测的。

## 修法

新增 `cfg.MODEL.MaskDINO.POINT_CORAL_EXIST`（默认 False，关着时模型与基线逐位相同）。一个开关同时改三处，作为方案整体检验：

1. **头**：`point_class_embed = CoralHead(256, 256, 25, 3)`，输出 `[B,Q,25]`，reshape 成 `[B,Q,25,1]` 后与原头同形，下游不改。
   参数名仍以 `point_class_embed` 开头（被 `_POINT_HEAD_KEYS` 匹配）。打开后 `POINT_CLASS_MLP` 不起作用。
2. **内层匹配**（`criterion._inner_point_match`）：只让前 `m = min(Ni, 25)` 个 slot 参与匈牙利，代价只用坐标 L1
   （前缀内全是正样本，存在性代价对所有分配相同）。Ni ≤ 25 时是 [Ni, Ni] 方阵；Ni > 25（train 2 个穗）时 25 个 slot 按坐标挑 25 个 GT。
   于是「slot k 被配上 ⟺ k < min(Ni, 25)」，`tgt_exist` 恰好就是 CORAL 的 levels。
3. **存在性损失**：focal 换成 CORAL 的 BCE：每实例对 25 个任务取均值，实例间按权重加权平均
   （Ni=0 的 wheat 仍取 `POINT_EMPTY_CLS_WEIGHT`=0.1）。
   **口径与基线不同**：基线是 focal 对 25 个槽求和后除以本 batch 的 GT 点总数。这是方案的一部分，不单独消融。

推理：`K = Σ_k 1[σ(l_k) > POINT_THRESHOLD]`，输出**前** K 个 slot 的坐标。按求和计数、取前缀，不要求 bias 已经排好序。
配置把 `POINT_THRESHOLD` 从 0.35 改为 **0.5**（CORAL 的决策规则，`afad-coral.py:358`）。0.35 是为 focal 标定的存在性概率挑的档，
对 BCE 标定的 `P(Ni>k)` 没有意义。

不变的：坐标头、坐标 L1、`POINT_COORD_WEIGHT` 5.0、`POINT_CLASS_WEIGHT` 2.0、外层匹配、DN。
`POINT_FOCAL_ALPHA` 与 `COST_POINT_CLASS_WEIGHT` 打开后不再参与点头。

`coral_loss × 25` 与论文 `cost_fn` 逐值一致（CPU 单测）。CPU 单测也核对了：Ni=7 时匹配行恰为 0..6，
Ni=0 / 25 / 30 三种边界；`loss_point_cls` 与手算的 `coral_loss(前缀 levels, 权重 1/0.1/1)` 一致；读法取前缀。

## 与基线的差异

两项配置：`POINT_CORAL_EXIST: True`、`TEST.POINT_THRESHOLD: 0.5`（解析后的 cfg 逐项比对，只差这两个键）。
参数：`point_class_embed` 从 MLP(256→256→256→25) 换成 MLP(256→256→256→1) + 25 个 bias，净减 6,143 个。
开关关闭时 state_dict 的 732 个键与形状与 ep100 checkpoint 完全一致；`loss_points` 与内层匹配在开关关闭时
对同一合成输入的输出与 `mini/baseline` 逐位相同。

## 预期与判否线（跑之前写死）

对照 ep100 基线（`output_v2/exp-0923-mini-ep100`，`mini/baseline`@`0f71161`）：

| 指标 | 峰值 | 末值 | 末 4 轮摆幅 |
|:--|--:|--:|--:|
| `count_MAE_spike` | **2.46**（iter 8999） | **2.60** | 0.07 |
| 每穗有符号偏差（预测 − GT） | — | **+2.40** | — |
| `matched_F1@8` | **0.6331**（iter 11499） | 0.6312 | 0.004 |
| `segm/AP` | 78.80 | 78.80 | 0.70 |
| `loss_point_coord`（×5 后，末 20 条训练记录均值） | — | 0.3433 | — |

噪声地板沿用 mini 实验族的 **0.015**（`matched_F1@8`）。

### 主判据：二维（计数 × 定位），看训练期评估曲线的峰值

| 条件 | 判定 |
|:--|:--|
| `count_MAE_spike` 峰值 ≤ **2.21**（基线 ×0.9）且 `matched_F1@8` 峰值 ≥ **0.6181**（基线 − 0.015） | **成立**：拿到前缀计数的收益，没有付定位的代价 |
| `count_MAE_spike` 峰值 ≤ 2.21，但 `matched_F1@8` 峰值 < 0.6181 | **计数成立、定位有代价**（重演 canonical-order），记下代价大小 |
| `count_MAE_spike` 峰值 ≥ **2.46** | **判否** |
| 其余 | **未决**（单种子，不补种子） |

### 护栏

- `segm/AP` 峰值 ≥ **77.80**（基线 − 1.0）。点头改动不应伤到分割。

### 观察量（只记录，不作判据）

- 末值 `loss_point_coord` 与基线 0.3433 之比（canonical-order 当时是 5.3×），用来判断定位代价是否来自前缀约束。
- 最终 25 个 bias 相邻有序（`b_k ≥ b_{k+1}`）的比例：检验定理 1 在这个任务上是否兑现。
- 末值每穗有符号偏差，对照基线 +2.40。
- 阈值扫描：`model_final` 上 `POINT_THRESHOLD` ∈ {0.3, 0.4, 0.5, 0.6, 0.7} 各 `--eval_only` 一次，
  看 0.5 是否落在最优附近（CORAL 的 0.5 是理论值，基线的 0.35 是经验值）。

## 已知限制

- 单种子，与之前所有 mini 实验一致。
- 三处改动绑在一起检验。若判「计数成立、定位有代价」，要拆开归因（结构 / 前缀匹配 / 损失口径）需要额外的消融臂。
- 训练期评估用的是 0.5 这一档，基线是 0.35。两者都是各自方案的「默认决策点」，不是各自的最优档。
- slot 越靠后，得到坐标监督的比例越低（slot k 只在 `Ni > k` 的穗上有坐标损失；k ≥ 22 在 mini train 里不到 1%）。
  这些 slot 只有在计数头报出很大 K 时才会被读出，影响有限。

## 成本

12500 iter。存在性头参数略减，内层匹配的代价矩阵从 [25, Ni] 缩到 [Ni, Ni]，预计与基线同速，约 11~14 h。
另加阈值扫描 5 次 `--eval_only`（约 10 min）。GPU 冒烟（bs2，30 step）的实测显存与速度见下节。

## 冒烟

**GPU**（qiuhao A100 GPU1，2026-09-26 15:03，跑之前）：独立脚本 import `train_net.Trainer`（不经 `train_net.py` 的 main，
不算一次训练 run），ImageNet R-50 起点，bs2 × 30 step（AMP），再在 20 张 val 上推理 + `SpikeletPointEvaluator`。
（s/iter 不可比：第一个跑的臂要冷读 NFS 上的图，且当时别人的 vLLM 占着 GPU 的 100% 算力。）

| 项 | 结果 |
|:--|:--|
| 30 步损失 | 全部有限；`loss_point_cls`（CORAL，×2 后）1.500 → 1.134 |
| 每步产生的 `loss_point_cls*` | 11 个（主输出 + 9 份 aux + interm） |
| 第 0 步梯度 | `point_class_embed.*`（CoralHead）全部非零 |
| 峰值显存（bs2，torch 口径） | 11.97 GiB |

**CPU**（`CUDA_VISIBLE_DEVICES=""`，一点不占 GPU；15:14~15:19）：把 ep100 基线的
`output_v2/exp-0923-mini-ep100/model_final.pth` 载入本臂模型，与 `mini/baseline` 同权重的输出在 12 张 val
（top-100 里 94 个 wheat_2 条目）上逐实例比对。补上 GPU 冒烟没走到的读法分支（30 步的模型还不出 wheat_2）。
（存在性头的形状变了，ep100 的 `point_class_embed.*` 载不进来，本臂的存在性头是随机初始化；坐标头照常载入。）
另跑一次基线的 `POINT_THRESHOLD -1`，让每个 wheat_2 按 slot 顺序输出全部 25 个点，作为「前 K 个 slot」的参照：
- 实例（类别、分数）与基线逐项相同；
- 94 个 wheat_2 的输出**逐位等于**基线全 slot 输出的前 K 行（存在性头未训练，K ∈ [23, 25]）。

**正式 run 的早期实测**（与 A 臂并发）：iter 180 前后稳定 2.50 s/iter，比 A 臂快约三成——前缀匹配的代价矩阵更小、
不算 focal 代价，存在性损失也从逐实例 focal 换成了一次批量的 CORAL。`max_mem` 23.0 GB（torch 口径），
nvidia-smi 占用约 32 GiB。按这个速度训练约 8.7 h，另加 25 次评估。
