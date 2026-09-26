# 想法：每个实例 query 挂一个 CORAL 序数计数头（A 臂）

状态：**待验证（预注册，跑之前写死）** ｜ 提出 2026-09-26 ｜ 分支 `mini/coral-count`（从 `mini/baseline`@`0f71161` 开出，100 epoch）｜ 配置
`wheat_joint_single_stage_mini_coral_count.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜ 输出 `output_v2/exp-<启动日期>-mini-coral-count`

来源：Cao, Mirjalili, Raschka, *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*
（Pattern Recognition Letters 2020，arXiv 1901.07884），本地源码 `/public/home/zncui/pycharm_project/coral-cnn`。

同一批还有两条 CORAL 分支，都从 `mini/baseline`@`0f71161` 分出，彼此独立、不合并：
`mini/coral-exist`（B：存在性头本身改成 CORAL 结构 + 前缀匹配）与 `mini/coral-img`（C：图级 wheat_2 穗数）。

## 现象

每穗报几个点，现在是「25 个无序点槽里存在性 sigmoid > 0.35 的个数」。这个数**没有被直接监督**：
存在性头的目标是「这个槽有没有被内层匈牙利配上」，槽是无序的，25 个 logit 彼此独立。
计数只是阈值化后的副产品。

ep100 基线（`output_v2/exp-0923-mini-ep100`）末值：配上的 578 个穗里，**每穗平均多报 2.40 个点**，
`count_MAE_spike` 2.599 主要就是这个系统偏差。KMO 的 100 epoch 记录里也写过「两臂都系统性多数」。
08-25 的 slot 诊断还发现存在性头大体只学到了每个槽的边际频率（固定 slot 后跨实例 AUC 0.721，冻结 trunk 时测得）。
一个只学到边际频率的头，靠阈值读出来的计数自然带着系统偏差。

## 假设

每穗小穗数 `Ni`（mini train 7~31、均值 15.6；val 9~24、均值 15.9）是一个**序数量**。
按 CORAL 的做法直接监督它：拆成 25 个二分类任务 `Ni > k`（k = 0..24），共享一个 1 维打分 `g(h)`、各自一个 bias。

1. **计数**：计数头直接对 `Ni` 做监督，没有阈值标定的问题，应当去掉 +2.4 的系统偏差。
   存在性头退回到只负责「哪几个槽更像真点」的排序，不再决定个数。
2. **表征**：多一项与每穗计数直接相关的监督信号，可能反过来让 query 表征更好地编码「这穗有多长、有几节」。
   这一项效应不确定，靠同一 checkpoint 的 threshold 读法与基线对比来量。

## 修法

新增 `cfg.MODEL.MaskDINO.POINT_COUNT_HEAD`（默认 False，关着时一个参数都不建，模型逐位不变），
`POINT_COUNT_WEIGHT=2.0`（与 `POINT_CLASS_WEIGHT` 同值），`TEST.POINT_COUNT_READOUT ∈ {threshold, coral}`。

- `maskdino/modeling/coral.py`（三条 CORAL 分支逐字相同）：`CoralHead`、`coral_levels`、`coral_loss`、`coral_predict`。
- decoder：`point_count_embed = CoralHead(256, 256, 25, 3)`，输出 `pred_point_count_logits [B,Q,25]`。
  interm、initial_pred、9 层 decoder 都输出，经 `dn_post_process` 切掉 DN 部分，进 `aux_outputs`。
- criterion：在 `loss_points` 的同一个实例循环里，对每个外层匹配上的实例取目标 `min(Ni, 25)`，
  权重在 Ni>0 时为 1，Ni=0 时取 `POINT_EMPTY_CLS_WEIGHT`（0.1，与存在性损失同一口径），
  产出 `loss_point_count`，随 `_interm` / `_{i}` 深监督带上。**坐标、存在性、内层匹配一行不动。**
- 推理：`coral` 读法下 `K = Σ_k 1[σ(l_k) > 0.5]`，再按存在性分数取 top-K 个槽；`threshold` 读法与基线逐位相同。

| 与论文代码的对应 | 论文（`model-code/afad-coral.py`） | 本仓库 |
|:--|:--|:--|
| 打分 | `fc(512→1, bias=False)`（:275） | 3 层 MLP（256→256→256→1），与其余预测头同构 |
| bias | `linear_1_bias`，全零初始化（:276） | 同结构；按 coral_pytorch 的 `preinit_bias` 初始化为降序 `T/T … 1/T` |
| 目标 | `levels = [1]*y + [0]*(K-1-y)`（:167） | 同；`y` 先截到 `[0, 25]`（train 有 2 个穗 Ni>25） |
| 损失 | 对 K−1 个任务**求和**、batch 取均值（:335-338） | 对任务**取均值**（差常数 1/25，深监督 11 份输出下量级可控），实例间按权重加权平均 |
| 预测 | `sum(probas > 0.5)`（:358-359） | 同 |
| task importance | 可选，默认全 1 | 不用 |

`coral_loss × 25` 与论文 `cost_fn` 逐值一致（CPU 单测，随机 64×25：35.845757 vs 35.845757）。

## 与基线的差异

只有两项配置：`POINT_COUNT_HEAD: True`、`TEST.POINT_COUNT_READOUT: coral`（解析后的 cfg 与
`wheat_joint_single_stage_mini_ep100.yaml` 逐项比对，只差这两个键）。
新增参数 131,866 个（`point_count_embed.*`）。开关关闭时 state_dict 的 732 个键与形状与 ep100 checkpoint 完全一致；
`loss_points` 在开关关闭时对同一合成输入的输出与 `mini/baseline` 逐位相同。

## 预期与判否线（跑之前写死）

对照 ep100 基线（`output_v2/exp-0923-mini-ep100`，`mini/baseline`@`0f71161`）：

| 指标 | 峰值 | 末值 | 末 4 轮摆幅 |
|:--|--:|--:|--:|
| `count_MAE_spike` | **2.46**（iter 8999） | **2.60** | 0.07 |
| 每穗有符号偏差（预测 − GT） | — | **+2.40** | — |
| `matched_F1@8` | 0.6331（iter 11499） | 0.6312 | 0.004 |
| `segm/AP` | 78.80 | 78.80 | 0.70 |

噪声地板沿用 mini 实验族的 **0.015**（`matched_F1@8`）。`count_MAE_spike` 的判据用相对量（×0.9），
因为它的末 4 轮摆幅只有 0.07，而 +2.4 的系统偏差若被去掉，改善量级应远大于 10%。

### 机制门（先判）

`model_final.pth` 用 `--eval_only` 各评一次（权重相同，只改读法）：

```bash
python train_net.py --eval_only --config-file <out>/config.yaml MODEL.WEIGHTS <out>/model_final.pth \
  OUTPUT_DIR <out>/eval_threshold MODEL.MaskDINO.TEST.POINT_COUNT_READOUT threshold
```

- **M**：coral 读法的 `count_MAE_spike` **低于** threshold 读法 → 通过。
  不低于 → 判**「机制未启动」**：计数头给出的个数不比数阈值更准，CORAL 在这里没有带来东西。

### 主判据：`count_MAE_spike`（coral 读法，训练期评估曲线）

| 条件 | 判定 |
|:--|:--|
| 峰值 ≤ **2.21** 且末值 ≤ **2.34**（基线 ×0.9），且护栏满足 | **成立** |
| 峰值 ≥ **2.46** | **判否** |
| 其余 | **未决**（单种子，不补种子，见 no-extra-seeds） |

### 护栏

- `matched_F1@8` 峰值 ≥ **0.6181**（基线峰值 − 0.015）。coral 读法会改每穗报几个点，从而改点级 P/R 的平衡，这一项不能明显倒退。
- `segm/AP` 峰值 ≥ **77.80**（基线 − 1.0；基线末 4 轮摆幅 0.70）。

### 观察量（只记录，不作判据）

- 末值每穗有符号偏差（`spikelet_pair_counts.csv` 算 mean(pred − gt)），对照基线 +2.40。
- 本臂的 threshold 读法 vs 基线：两者读法相同，差异只来自辅助损失对表征的影响。
- `loss_point_count` 的训练曲线；最终 25 个 bias 相邻有序的比例（定理 1 说最优点上应当有序）。
- `count_MAE_img` 与过检倍数：本臂不针对过检，预期不变，记下来以免误读。

## 已知限制

- 单种子，与之前所有 mini 实验一致。
- coral 读法只决定「报几个」，选哪几个仍靠存在性分数排序。若存在性排序本身很差，计数对了 `matched_F1@8` 也未必好。
- `POINT_COUNT_WEIGHT=2.0` 未调。损失口径是「任务均值」，初始值约 0.69×2，与存在性 focal 同一量级。
- 训练期评估用的是 coral 读法；与基线训练曲线对比时，读法和表征两个因素是混在一起的，拆分只能靠机制门那一次 `--eval_only`。

## 成本

12500 iter。ep100 基线 10:49（前 3000 iter 与 p30 并发）。计数头每层只多一个 3 层 MLP，预计与基线同速，约 11~14 h。
另加 1 次 `--eval_only`（约 2 min）。GPU 冒烟（bs2，30 step）的实测显存与速度见下节。

## 冒烟

（待 qiuhao A100 空出后补记：bs2 峰值显存、s/iter、新参数梯度非零、两种读法在 20 张 val 上都能出点。）
