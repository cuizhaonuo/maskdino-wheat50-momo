# 想法：点损失的尺度门控——LSJ 把穗缩到小穗分辨不出来时，这个实例不算点损失

状态：**判否**（读数那次 iter 8999：计数 2.520 > 2.21，定位 0.6153 < 0.6428，两条都不过；护栏都过）｜
验证于 [exp-2026-10-07-mini-point-scale-gate](../experiments/exp-2026-10-07-mini-point-scale-gate.md) ｜
2026-10-07 20:38 启动于 `ca4891b`，baiju A100 GPU0（用户为本批次临时放开）｜ 预注册提交于 `ca4891b`（跑之前）｜
提出 2026-10-07 ｜ 分支 `mini/point-scale-gate`（从 `mini/baseline`@`0f71161` 开出，100 epoch）｜
配置 `wheat_joint_single_stage_mini_point_scale_gate.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜
输出 `output_v2/exp-1007-mini-point-scale-gate`

同批（2026-10-07，用户批准，一个方案一个分支）另有 `mini/count-residual`、`mini/group-quality`，各自独立，不合并。

## 来源

项目内分析（2026-10-07）。依据有三条：测试时的小穗间距、训练时 LSJ 的缩放范围、APGCC 三臂的教训。
思路与 SNIP（Singh & Davis, *An Analysis of Scale Invariance in Object Detection – SNIP*, CVPR 2018）相同：
某个缩放下尺度不合适的目标，不回传它的梯度。本臂只把这条用在点损失上。

## 现象

**测试时，小穗间距只比最细特征的格宽大一倍。**

- 图像全是 1024²，测试时不缩放（`MIN/MAX_SIZE_TEST 1024`）。
- val 有 613 个 wheat_2。同穗小穗的最近邻距离先按实例取中位数，再取中位数：**17.0 px**（逐点汇总是 16.7 px）。
- 沿穗轴相邻小穗的间距（投影到 GT 点的主轴上），中位数 **9.5 px**。
- 解码器 cross-attn 用 4 层特征：res3 / res4 / res5，加一层下采样。最细的是 **stride 8**（res3）。
  stride 4 的 `mask_features` 只用于 mask（`maskdino_encoder.py` 的 forward：FPN 层排在 4 层之后，不进 `multi_scale_features`）。

**训练时，LSJ 把相当一部分穗缩到格宽以下。**

- `ResizeScale` 取 s ~ U[0.1, 2.0]。图像本身是 1024²，缩放因子就是 s。
- mini train 有 2754 个 Ni ≥ 2 的 wheat_2。按实例对 s 解析求平均：
  - 缩放后最近邻间距中位数 d̄ < 8 px：**20.6%**；< 4 px：**7.7%**。
  - 沿轴间距 < 8 px：**41.8%**。
- 放大（s > 1）时裁剪会丢掉一部分实例，所以 batch 里的实际比例更高。
  用真实 mapper（LSJ + 裁剪 + 贴边降级）跑 4 个 epoch（2000 个样本），d̄ < 8 px 的实测占 **27.9%**：

| s | 图数 | Ni≥2 实例 | d̄ < 8 px 的比例 |
|:--|--:|--:|--:|
| 0.1 – 0.3 | 195 | 1027 | 100% |
| 0.3 – 0.4 | 110 | 604 | 94.9% |
| 0.4 – 0.5 | 109 | 609 | 62.7% |
| 0.5 – 0.6 | 107 | 574 | 24.4% |
| 0.6 – 0.8 | 208 | 1104 | 3.7% |
| 0.8 – 1.0 | 201 | 1122 | 0.2% |
| 1.0 – 2.0 | 1070 | 2716 | 0 |
| **合计** | 2000 | 7756 | **27.9%** |

**这些样本为什么有害。**

- d̄ 低于格宽时，相邻小穗落在同一个特征格里。点的定位和内层匹配都近乎随机。
- 坐标损失是框相对坐标的 L1，大小实例同权。同样 0.05 的框相对误差，在 30 px 的穗上是 1.5 px，在 300 px 的穗上是 15 px。
  小实例上要的精度，特征给不了。等于给点头喂噪声。
- APGCC 三臂的教训（`mini/apg-refine` 分支的 `experiments-bank/ideas/apg-refine.md`，结论一节）：
  APG 的正负半径约 (2, 2~8) px，低于 1/4 特征的 4 px 格宽。存在性 BCE 停在 ln 2 附近，AUC 从基线 0.667 降到 0.630。
  监督低于特征格宽时，学不出存在性。

## 假设

只在小穗分辨得出的尺度上监督点。

1. 存在性头会更依赖局部线索，少做实例级的「猜点数」。
2. 默认读数下，多报和定位都会改善。基线末值每穗多报 **+2.40** 个点（`exp-0923-mini-ep100/inference/spikelet_pair_counts.csv`，578 个配对穗，r 0.64）。

预期收益是同批三个方向里最小的。但它最便宜、因素最干净：不加参数，不改推理。

## 修法

开关 `MODEL.MaskDINO.POINT_MIN_NN_PX`，默认 0.0 = 关。关着时与 `mini/baseline` 逐位等价（见下文验证）。

1. **配置**（`maskdino/config.py`）：新增 `POINT_MIN_NN_PX`。
2. **targets 带图像尺寸**（`maskdino/maskdino.py`）：`prepare_targets` 与 `prepare_targets_detr` 给 targets 加 `image_wh`，
   即增广后图像的像素宽高 (W, H)。写法照 `mini/group-decoder` 的同一处改动，本分支重写，没有 cherry-pick。
3. **门控**（`maskdino/modeling/criterion.py`）：
   - `forward` 算损失前调用一次 `_mark_point_scale_skip`。对每个 Ni ≥ 2 的 GT：小穗坐标 × `image_wh` 换回像素，
     算每点的最近邻距离，取中位数 d̄（偶数个点取两个中值的均值，同 `numpy.median`）。d̄ < 阈值记为忽略。
   - `loss_points` 遇到被忽略的配对直接跳过：**不算坐标损失，不算存在性损失，不计入归一化分母 `num_pos`**。
     这是忽略，不是当负样本。
   - 标记只取决于 GT，每次前向算一次。主输出、9 层 aux、interm 的点损失共用。
   - Ni = 1 与 wheat（Ni = 0）照旧。检测、mask、DN 损失不受影响。
   - 日志：`metrics.json` 记 `point_scale_gate_skip_ratio`（每 batch 的比例，20 iter 中位数）和
     `point_scale_gate_skip_ratio_cum`（全程累计）。分母是 batch 里 Ni ≥ 2 的 GT 实例数。
     外层匹配给每个 GT 都配了 query（300 个 query；mini train 每图最多 121 个 GT），所以它也是点损失里被跳过的比例。
4. **正式配置**：`POINT_MIN_NN_PX: 8.0`，即 res3 的格宽。像素指增广后的图像像素。阈值先验取定，没有调。
5. **诊断脚本**：`myscripts/backup/vis_process/point_token_diag.py`，从 `mini/group-decoder` 逐字带入
   （该文件引入于 `c3666e0`，到分支头 `00a7aaf` 没改过），只在文件头加了来源注释，`git add -f` 提交。
   它只依赖 `instance_inference` 的签名、`point_threshold` / `spikelet_class_id` 属性和评估器，不依赖分组 decoder。

## 与对照的差异

- 对照：`mini/baseline` 100 epoch，`output_v2/exp-0923-mini-ep100`。
- 解析后的 cfg 与 `wheat_joint_single_stage_mini_ep100.yaml` 逐键比对（325 个键）：只差 `POINT_MIN_NN_PX` 0.0 → 8.0。
  与基线 run 的 `config.yaml` 比，另外只差 `OUTPUT_DIR`，和一项用不到的级联头权重（列表与元组的序列化差异）。
- 参数量 52,300,433，与基线相同。推理、评估代码不变。

## 实施前已完成的验证

脚本与产物在 `output_v2/_smoke-1007-point-scale-gate/checks/`，不进仓库。

**1. CPU 单测**（`gate_unit_test.py`，合成数据）：全部通过。

| 项 | 结果 |
|:--|:--|
| d̄ 与 `numpy.median` 一致：140 组（Ni 2~31），阈值放在 d̄ ± 1e-3 两侧，标记都翻转 | 通过 |
| Ni < 2 永不忽略；阈值 1e9 时 Ni ≥ 2 全忽略；阈值 1e-9 时全不忽略 | 通过 |
| 阈值 8：点损失与「门控关闭、从 indices 里删去被忽略配对」的参考**逐位相同**（12 个 Ni≥2 实例忽略 6 个） | 通过 |
| 阈值 1e-6（什么都不忽略）：点损失与门控关闭逐位相同 | 通过 |
| 被忽略实例对应的 query 收不到任何梯度 | 通过 |
| `EventStorage` 里两个标量的值正确 | 通过 |

**2. 等价性**（`equiv_check.py`，baiju A100 GPU0）。
mini train 按 DatasetCatalog 顺序取前 16 张，2 张一组共 8 个 batch。固定 random / numpy / torch 种子，经同一个 mapper，
训练态前向一次（AMP 同训练，cudnn deterministic），ImageNet R-50 起点。分别在本分支和 `mini/baseline` 的 worktree 里跑：

| 项 | 本分支（开关关） vs `mini/baseline` |
|:--|:--|
| 全部损失：8 batch × 127 项 = 1016 个值（按 `float.hex` 比） | **逐位相同** |
| 参数量 / state_dict 张量数 | 52,300,433 / 732，相同 |
| 初始权重 sha256 | 相同 |

**3. 开关打开**（正式配置，同 8 个 batch，前向 + 反传）。

- 非点损失（每 batch 105 项）与开关关闭**逐位相同**。点损失（每 batch 22 项）只在有实例被忽略的 5 个 batch 里变化。
- 损失、梯度全部有限。
- 忽略 26 / 65 = 40%（只有 16 张图，样本小）。
- **极端情况**：batch 5 的 7 个 Ni ≥ 2 实例全被忽略。`num_pos` = 0，分母取 1。
  wheat 负样本的存在性损失没有了分母：`loss_point_cls` 从 0.60 升到 42.4，点头梯度范数从约 11 升到 495
  （全模型梯度范数 9141。全模型裁剪到 0.01 后，点头约占这一步梯度的 5%，平时约 0.2%）。见「归因限制」第 1 条。

**4. 真实 mapper 模拟**（`skip_ratio_sim.py`，CPU，不建模型）。
mini train 跑 4 个 epoch，共 2000 个样本。逐样本把 mapper 输出交给 criterion 的同一个函数判定：

- 忽略比例 **27.9%**（7756 个 Ni≥2 实例里 2165 个），分 s 的明细见「现象」一节。
- 按 4 张一组（`IMS_PER_BATCH`）：500 组里没有一组的分母为 0。分母均值 242 → 175，逐组比值的中位数 0.755。
- wheat 负样本存在性损失的相对权重（∝ wheat 实例数 / 分母）：中位数 ×1.37，均值 ×1.66；p99 从 1.33 升到 5.3。

**5. 冒烟**（baiju A100 GPU0，2026-10-07 20:25，正式配置，CLI 只覆盖 `SOLVER.MAX_ITER 40`、`TEST.EVAL_PERIOD 20`，
输出 `output_v2/_smoke-1007-point-scale-gate`，.pth 已删）：

| 项 | 结果 |
|:--|:--|
| 报错 | 0 |
| 损失 | 全部有限（含 11 份 `loss_point_*`） |
| 评估 | 两次都走通（iter 19 与训练结束）。40 iter 时 segm AP 0、点指标 nan，正常：基线 iter 499 的 segm AP 也只有 2.23 |
| 速度 | 2.09 s/iter（38 iter 用 1:19，baiju 上单独跑） |
| 峰值显存（log.txt 的 max_mem） | 23,100 MB（基线同阶段 23,065 MB） |
| 40 iter（160 张图）的忽略比例 | 累计 32.0%；20 iter 中位数 0.30 / 0.25 |

**6. 诊断脚本**：在 CPU 上对基线 `model_final.pth` 跑通
`point_token_diag.py --sweep --run-dir output_v2/exp-0923-mini-ep100 --out … --limit 3 MODEL.DEVICE cpu`
（3 张 val 图，配上 8 / 10 个穗，`sweep` 为空，不扫阈值）。
注意 `--sweep` 是 nargs="*"，带 opts 时要放在 `--run-dir` 之前。

## 预期与判否线（跑之前写死）

**读数**（用户的硬性规定）：

- 一律用默认读数：内置评估，`POINT_THRESHOLD` 取配置里的 0.35。不做阈值扫描。不用扫描出的最优档，也不用在 val 上拟合的校准值。
- 判据在同一次评估上读：**后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估**。同一 iter 的全部指标都从这次读。
- 护栏（segm AP、spike_recall）用全程峰值。
- 单种子。

**基线读数**（`output_v2/exp-0923-mini-ep100/metrics.json`）：

| | iter | `count_MAE_spike` | `matched_F1@8` | `spike_recall` | segm AP |
|:--|--:|--:|--:|--:|--:|
| 后半程计数最好的那次 | 8999 | **2.465** | **0.6228** | 0.9511 | 77.40 |
| 全程峰值 | — | — | — | **0.9511** | **78.80** |

诊断 AUC 0.6671（`output_v2/_diag-0926/exp-0923-mini-ep100.json`，`model_final.pth`）。

**判据**：

| 判据 | 线 | 过线 |
|:--|:--|:--|
| 计数 | 读数那次的 `count_MAE_spike` ≤ **2.21**（基线 2.46 × 0.9，与 `mini/coral-count` 同一条线） | **计数成立** |
| 定位 | 读数那次的 `matched_F1@8` ≥ **0.6428**（基线 0.6228 + 0.02） | **定位成立** |
| — | 两条都不过 | **判否** |

**护栏**（全程峰值）：

- segm AP 峰值 ≥ **77.80**（基线 78.80 − 1.0）。
- `spike_recall` 峰值 ≥ **0.9311**（基线 0.9511 − 0.02）。
- 护栏任一不过时，过线的判据只记「过线但护栏不过」，不算成立（同 `mini/coral-count`：成立以护栏满足为前提）。

**机制只记录，不作判据**：诊断 AUC 与基线 0.6671 的差。
工具 `point_token_diag.py`，val，σ = 8，阈值 0.35，读 `model_final.pth`（与基线同口径），`--sweep` 不带值。

**观察量**（只记录）：

- `point_scale_gate_skip_ratio_cum` 的末值。预期在模拟的 27.9% 附近。
- 末次评估 `spikelet_pair_counts.csv` 的每穗有符号偏差与 r，对照基线 +2.40 / 0.64。
  用来解读计数的变化是偏差变了，还是相关变了。

## 归因限制

1. **分母效应。** 方案规定被忽略实例的 Ni 不计入 `num_pos`，wheat 负样本（Ni = 0，权重 0.1）的存在性损失却照算。
   分母变小，这部分负样本的相对权重就变大：按 4 张一组模拟，中位数 ×1.37、均值 ×1.66，极端时分母取 1（见验证第 3、4 条）。
   它可能把存在性头整体往「没有点」推。所以若多报减少，一部分可能来自这个效应，而不是监督更干净。
   读数时对照诊断 AUC：纯粹的偏移不改变排序，AUC 不变；局部判别变好，AUC 才会升。
2. **监督量减少。** 约 28% 的 wheat_2 实例不再有点监督。若结果变差，分不清是「少了噪声」还是「少了数据」。
   本臂没有「随机丢掉同样比例」的对照臂。
3. **默认读数混有标定。** τ = 0.35 下基线每穗多报 +2.40，`count_MAE_spike` 主要量的是这个偏差。
   不扫阈值，所以计数判据过线也可能只是偏差变了。上面的偏差与 r 只用来解读，不进判据。
4. **阈值只看最近邻。** 8 px 先验取定，没有调。门控按 d̄（原生尺度约 17 px）判。
   沿轴间距更小（原生 9.5 px），s 在约 0.47 ~ 0.84 之间时沿轴已低于 8 px，但这些实例大多不被门控。
5. 单种子。差异落在噪声里时记「歧义 / 未决」。

## 成本

- 12500 iter。冒烟在 baiju 上单独跑 2.09 s/iter，约 7.3 h，另加 25 次评估（基线共 49:51）。合计约 8 h。
- 峰值显存约 23 GB（冒烟 max_mem 23,100 MB），与基线相同。
- 门控每 iter 只多二十来个小张量运算，不加参数。

## 结论（2026-10-08）

**判否。** 计数、定位两条都不过，护栏都过。详见 [exp-2026-10-07-mini-point-scale-gate](../experiments/exp-2026-10-07-mini-point-scale-gate.md)。

| | 本臂 | 基线 | 线 |
|:--|--:|--:|--:|
| 读数那次（后半程 `count_MAE_spike` 最低） | iter 8999 | iter 8999 | — |
| `count_MAE_spike` | **2.520** | 2.465 | ≤ 2.21 ❌ |
| `matched_F1@8` | **0.6153** | 0.6228 | ≥ 0.6428 ❌ |
| segm AP 峰值 | 78.04 | 78.80 | ≥ 77.80 ✅ |
| `spike_recall` 峰值 | 0.9494 | 0.9511 | ≥ 0.9311 ✅ |
| 诊断 AUC（只记录） | 0.6558 | 0.6671 | — |
| 忽略比例（按实例累计） | 28.9% | — | 模拟 27.9% |

- 门控按设计生效，比例与模拟一致。
- 多报没有减少：最终评估每穗偏差 +2.66（基线 +2.40），r 0.611（基线 0.642）。与假设 2 相反。
- 存在性没有变成更好的局部判别：AUC 低 0.011。
- 分母混杂（归因限制第 1 条）预计让计数看起来变好；实测计数变差，判否不受它影响。
- 未预注册的弱线索：后半程 `matched_F1@8` 有 11/13 次高于基线同 iter，平均 +0.007，多在噪声地板 0.015 内；读数那次是 −0.0074。
