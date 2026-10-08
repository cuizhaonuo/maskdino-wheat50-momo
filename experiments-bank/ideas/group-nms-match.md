# 想法：小穗 token 逐层点 NMS 后再做内层匹配 · arm A（match：只在匹配与损失侧做）

状态：**预注册（未开跑）** ｜ 提出 2026-10-08 ｜ 分支 `mini/group-nms-match`（从 `mini/group-decoder`@`00a7aaf` 开出，100 epoch）｜ 配置
`wheat_joint_single_stage_mini_group_nms_match.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_group.yaml`，即 E3 的配置）｜ 输出 `output_v2/exp-<启动日期>-mini-group-nms-match`
｜ 代码提交 `df90505`（共享 NMS 工具 + match 模式）｜ 启动提交：（主 agent 填）｜ 启动时间：（主 agent 填）

**分叉点是有意的（用户指定）。** mini 实验的约定是新 arm 一律从 `mini/baseline` 开。本 arm 要改的是分组 decoder 的小穗 token 的匹配，
离开分组 decoder 就不存在，所以从 `mini/group-decoder`@`00a7aaf` 开出，在 E3 上续做。
姊妹 arm B（`mini/group-nms-prune`，在 decoder 里逐层剔除）从本分支的代码提交 `df90505` 开出，两者共用 NMS 工具与 `pred_point_keep` 入口。
按约定（mini arm 的主对照一律是 E0），**主对照是 E0**（`mini/baseline` 100 epoch，`output_v2/exp-0923-mini-ep100`），
E3（`output_v2/exp-0925-mini-group-decoder`）只作归因参照。

## 来源

Zhang et al., *Dense Distinct Query for End-to-End Object Detection*（DDQ，CVPR 2023，arXiv 2303.12776）：

- 每个 refine 阶段之前做 class-agnostic NMS，只让留下的 query 去匹配，损失只算在留下的 query 上；训练和推理都做。
- 动机：几乎相同的 query 之间被一对一匹配判成一正一负，监督自相矛盾。
- 效应量：DETR 版（decoder 里已有 query 自注意力）只有 +0.6~0.7 AP；没有 query 交互的 FCN 版 +3 AP。
- E3 的小穗 token 之间已有实例内自注意力，更接近 DETR 版，**预期效应小**。

## 现象

**瓶颈在分数挑点，不在定位。** E3 最终 checkpoint 上，79.1% 的 GT 小穗在 8 px 内有 token，分数完美挑选时 F1@8 上限 0.88，
实际 0.638；存在性 AUC 0.70（见 [group-decoder](group-decoder.md) 与 `mini/group-quality` 的诊断）。

**用户的设想：匹配不稳。** 几个 token 落在同一个小穗上时，一对一匹配随机给一个判正、其余判负，而且每次迭代都可能换人。
本方案改的是分数的监督：去掉近乎相同的 token 之间的一正一负。它与已判否的 QFL（`mini/group-quality`，改存在性的目标值）不是一回事。

E3 上的实测（最终 checkpoint，全 val，下节「E3 诊断」）：

- 内层匹配的跨层改判率（相邻层同一 GT 小穗配到的 token 下标不同的比例）均值 0.0271，主要在前几层（1→2 0.084，8→9 0.007）。
- 匹配前若按 κ=0.05 做 NMS，每层 12~15% 的 token 是近重复。

## 假设

内层匹配前先去掉近重复（按存在性分数贪心 NMS），只让留下的 token 去匹配，被压掉的不算任何损失：

1. 近乎相同的 token 之间不再一正一负，内层匹配跨层更稳（机制 M1）；
2. 存在性分数在留下的 token 上更会挑点（机制 M2：AUC_surv）；
3. 默认阈值下 matched_F1@8 与计数改善（主判据、次判据）。

## 修法

代码提交 `df90505`（开关默认关，关着时与 `mini/group-decoder`@`00a7aaf` 逐位等价，见「实施前已完成的验证」1）。

**配置**（`maskdino/config.py`；本 arm 的 yaml 三项全开）：

- `MODEL.MaskDINO.POINT_NMS_MODE`：`"none"`（默认）/ `"match"`（本 arm）；arm B 另加 `"prune"`。
- `MODEL.MaskDINO.POINT_NMS_KAPPA`：κ，训练侧半径。本 arm **0.05**。
- `MODEL.MaskDINO.TEST.POINT_NMS_KAPPA`：推理侧半径，默认 0 = 关。本 arm **0.05**（yaml 里两项引用同一个 `&kappa`）。

**训练侧**（`maskdino/modeling/criterion.py` 的 `loss_points`，只作用于第 1~9 层小穗 token 的绝对坐标输出）：

- 每个外层匹配上的实例、每层，按存在性分数对 K=30 个 token 做贪心 NMS（分数降序，并列按下标）：
  中心距 < κ·√(w·h) 算重复，w、h 是所配 **GT 框**的像素宽高。在 iso 空间做（坐标除以 √(w·h)），坐标与分数都 detach。
- 被压掉的 token 不进内层匹配，存在性与坐标损失都不算（梯度恰为 0）。
- 留下但没配上的 token 照常判负，所以超出半径的远重复仍会被学着压低。坐标损失只算配上的（与 E3 相同）。
- 损失分母（总 GT 小穗数）不变。对 Ni=0 的 wheat 实例同样做。
- **不加「防饿死」补回**：留下的 token 少于 Ni 时，多出的 GT 小穗本层没有配对（见「归因限制」里的起步瞬态）。
- 前向不变：某层被压的 token 下一层照常参与注意力，还能回来。layer-0 与 interm 的 MLP 点头不受影响。
- 每次迭代写进 `metrics.json`：`point_nms/suppressed_frac`（被压 token 占比）、`point_nms/starved_frac`（留下的 token 少于 Ni 的实例占比）。
- 实现：每张图的 NMS 一次算完（一次 GPU→CPU 同步），每个实例仍各自做；与逐实例调用逐位相同（验证 3）。

**推理侧**（`maskdino/maskdino.py` 的 `instance_inference`）：最后一层对每个 wheat_2 实例的全部 30 个点做同样的 NMS，
半径用**预测框**的 κ·√(w·h)（像素），再按 `POINT_THRESHOLD` 0.35 过滤（先 NMS 后过滤与先过滤后 NMS 结果相同）。
**内置评估的点指标就是带 NMS 的。**

**诊断**（`myscripts/backup/vis_process/point_token_diag.py`，与训练共用 `criterion._point_nms_keep` / `_inner_point_match`）：
用 forward hook 截第 1~9 层的 token，按 criterion 的规则逐层配对（GT 框取 GT mask 外接框），报跨层改判率、AUC_surv、覆盖、两组读数。

### κ 的选择：0.07 → 0.05

计划里定的是 κ=0.07。**用户在 E3 全 val 诊断之后、任何 run 开跑之前改成 0.05（2026-10-08）**，arm A、B 同改。

- 数据依据（mini 标注、原生尺度，val 613 穗 / 9747 点）：GT 小穗最近邻距离 p1 8.8 / p5 10.9 / 中位 16.7 px；穗的 √(w·h) 中位 113.9 px。
  - κ=0.07：中位穗上半径 7.97 px，**0.77%** 的 GT 小穗在半径内有真邻居；
  - κ=0.05：中位穗上半径 5.69 px，**0.04%**。
  - LSJ 训练把图缩放 0.1~2.0 倍，像素半径不能写死，所以按框尺度归一化。
- 改动的理由（E3 最终 checkpoint，全 val，下表）：推理 NMS 造成的 8 px 覆盖损失 κ=0.07 为 5.40 个百分点、κ=0.05 为 2.15；
  匹配前 NMS 每层压掉的 token 21~23% 对 12~15%；E3 + 推理 NMS 的读数 matched_F1@8 0.6404 对 0.6416，计数 2.449 对 2.597。

## E3 参照（零训练，2026-10-08）

### E3 诊断（最终 checkpoint，全 val）

`point_token_diag.py`，代码 `df90505`，qiuhao GPU1；输出 `output_v2/_diag-1008-group-nms/E3_{plain,k005,k007}.json`。
三档设置：plain = 匹配前与推理侧都不做 NMS；κ=0.05 / 0.07 = 两侧都用该 κ。581 个配上的穗，9262 个 GT 小穗，σ = 8。
自检：plain 的旧口径 `check_matched_F1` = 0.6378275655131025，与 E3 `metrics.json` 末次 matched_F1@8 逐位相同。

| | plain | κ=0.05 | κ=0.07 |
|:--|--:|--:|--:|
| 跨层改判率（相邻 8 对的均值） | **0.0271** | 0.0319 | 0.0342 |
| AUC_all（全部 30 个 token） | 0.6989 | 0.6989 | 0.6989 |
| AUC_surv（推理 NMS 留下的 token） | 0.6989 | **0.6905** | 0.6745 |
| 推理 NMS 留下的 token 占比 | 100% | 85.0% | 76.5% |
| 8 px 覆盖率：全部 token / 留下的 token | 0.7914 / 0.7914 | 0.7914 / 0.7699 | 0.7914 / 0.7374 |
| 覆盖损失（百分点） | 0 | **2.15** | 5.40 |
| 匹配前 NMS 每层压掉的 token（第 1~9 层） | — | 12.0~14.8% | 20.9~23.4% |
| 匹配前 NMS 饿死的穗（每层） | — | 0.17% | 0.17~0.52% |
| 读数（模型规则，阈值 0.35）matched_F1@8 | 0.6378 | 0.6416 | 0.6404 |
| 读数 count_MAE_spike | 2.764 | 2.597 | 2.449 |
| 读数 计数偏差（pred − gt） | +2.534 | +2.267 | +2.043 |
| 读数 Pearson r | 0.552 | 0.551 | 0.537 |
| 读数 去偏移 MAE | 1.614 | 1.574 | 1.576 |

- 逐对改判率（1→2 … 8→9）：
  - plain：0.0836 / 0.0404 / 0.0244 / 0.0192 / 0.0168 / 0.0168 / 0.0089 / 0.0067
  - κ=0.05：0.0972 / 0.0492 / 0.0262 / 0.0243 / 0.0181 / 0.0205 / 0.0110 / 0.0086
  - κ=0.07：0.0948 / 0.0548 / 0.0330 / 0.0246 / 0.0208 / 0.0206 / 0.0138 / 0.0113
- **构造本身就让改判率上升**：在 E3 上只加匹配前 NMS，改判率从 0.0271 升到 0.0319（κ=0.05），因为各层的 NMS 决定本身也在变。
- AUC_surv 的正样本：留下的 token 与 GT 做纯距离一对一匈牙利，配上且 ≤ 8 px。若改用全部 30 个 token 匹配得到的标签，κ=0.05 为 0.6835。
- 覆盖损失：全部 30 个 token、只用留下的 token，分别与 GT 做纯距离一对一匈牙利，≤ 8 px 的 GT 比例之差。
  沿穗轴相邻小穗约 9.5 px，κ·√(w·h) 内的两个 token 可能各盖一个真小穗，压掉一个就丢一个。
- AUC_all 0.6989 是 GPU 上的数；早先 CPU 上的诊断是 0.6991（`output_v2/_diag-0926/`）。本 arm 的诊断同样在 GPU 上跑。

### E3 + 推理 NMS（后半程 13 个 checkpoint）

`train_net.py --eval_only`，E3 自己的 `config.yaml` + `MODEL.MaskDINO.TEST.POINT_NMS_KAPPA 0.05`，其余不变；
iter 6499 … 11999 与 final（12500）。输出 `output_v2/_eval-1008-group-decoder-testnms-k005/`（`summary.json`）。
同样跑了 κ=0.07，留作附加信息：`output_v2/_eval-1008-group-decoder-testnms-k007/`。

| | 读数 iter | count_MAE_spike | matched_F1@8 | spike_recall | segm/AP | 偏差 | r | 去偏移 MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| E3（无 NMS） | 9499 | 2.422 | 0.6326 | 0.9364 | 77.40 | — | — | — |
| **E3 + 推理 NMS κ=0.05** | 9499 | **2.312** | **0.6348** | 0.9364 | 77.40 | +1.82 | 0.501 | 1.610 |
| E3 + 推理 NMS κ=0.07 | 9499 | 2.225 | 0.6339 | 0.9364 | 77.40 | +1.63 | 0.479 | 1.673 |

- 读数规则同下文（iter ≥ 6250 里 count_MAE_spike 最低的那次）。偏差 / r / 去偏移 MAE 由该次 `inference/spikelet_pair_counts.csv` 算
  （574 个配上的穗）。E3 无 NMS 在 iter 9499 没有逐穗明细；它的最终评估为 +2.534 / 0.552 / 1.614。
- 推理侧 NMS 不改 mask、框与实例分数：13 次评估的 segm/AP、bbox/AP、spike_recall、spike_precision 与 E3 同 iter 逐项相同。
  所以护栏的全程峰值就是 E3 的（segm/AP 78.35 @ 9999，spike_recall 0.9657 @ 7999）。
- 13 个 iter 上，推理 NMS κ=0.05 让 count_MAE_spike 降 0.09~0.34，matched_F1@8 升 0.001~0.004。
- 最终 checkpoint 一行（0.6416 / 2.597 / +2.267 / 0.551）与上面诊断 κ=0.05 的读数逐位相同。

## 与对照的差异

单因素对照 **E3**（归因参照）：

| 项 | E3 | 本 arm |
|:--|:--|:--|
| `POINT_NMS_MODE` | none（默认） | **match** |
| `POINT_NMS_KAPPA` | —（none 下不生效） | **0.05** |
| `TEST.POINT_NMS_KAPPA` | 0（默认，关） | **0.05** |

其余全部继承 E3：分组 decoder（100 个实例 query、K=30、iso、ρ c=1、`POINT_EMPTY_CLS_WEIGHT` 0.1、`POINT_FOCAL_ALPHA` 0.5）、
数据、schedule（12500 iter = 100 epoch，cosine，每 500 iter 评估一次，共 25 次）、`POINT_THRESHOLD` 0.35。
冒烟落盘的 `config.yaml` 与 E3 run 的逐行 diff：除 `OUTPUT_DIR` 和冒烟覆盖的三项（`MAX_ITER` / `EVAL_PERIOD` / `CHECKPOINT_PERIOD`）外，只多出上表三个键。
正式 run 除 `OUTPUT_DIR` 外不带任何 CLI 覆盖。

## 实施前已完成的验证

脚本与原始输出：`output_v2/_smoke-1008-group-nms/verify/`（`verify_group_nms.py`，一次性脚本，不进 git；
最终代码的结果在 `verify/final/`，κ=0.05 的重测在 `verify/k005/`）。GPU 部分都在 qiuhao GPU1（A100）上跑。

1. **等价性（开关关着）**：固定 random / numpy / torch 种子，mini train 前 2 张图经同一个 mapper，训练态前向 1 次，
   在本 worktree 与基点 worktree（`mini/group-decoder`@`00a7aaf`）各跑一次：
   - 随机初始化与载入 E3 最终 checkpoint：**127 项 loss（加权前、加权后）逐位相等**；参数量同为 65,322,300，参数 sha256 相同。
   - 用本分支代码对 E3 `model_final.pth` 跑 `--eval_only`（推理 NMS 关）：四个预测文件（`coco_instances_results.json`、
     两个 spikelet CSV、`instances_predictions.pth`）与 E3 训练期末次评估**逐字节相同**；matched_F1@8 0.6378、
     count_MAE_spike 2.7642、segm AP 78.13，与 `metrics.json` 末条一致。
2. **NMS 单测**（CPU 与 CUDA，全部 0 处不一致）：
   - 3000 组随机点集（含聚簇、并列分数、P 0~31、半径含 0 与负数）与暴力实现逐个比对；
   - 距离恰好等于半径时不压；并列分数按下标；链式（被压的点不再压别人）；
   - 先 NMS 后过滤 ≡ 先过滤后 NMS（3000 组）；批量调用 ≡ 逐组调用（300 组）。
   - `loss_points` 合成数据单测（20 个种子）：被压 token 的坐标与 logit 梯度恰为 0；损失与「把被压 token 物理删掉、按原路径算」的参照一致；
     κ=0 的 match、注入全 True 的 `pred_point_keep` 都与原路径逐位相同；MLP 点头输出不受影响。
3. **match 打开**（mini train 前 16 张图经 LSJ，训练态；κ=0.05）：
   - E3 最终 checkpoint：每层被压 3.8~5.4%（全部 4.3%），**饿死 0**；第 1~9 层点损失变化：存在性 −2.2%~+1.0%、坐标 +0.7%~+3.0%。
     κ=0 的 match 与 none 127 项逐位相同；非点损失项与 MLP 点头的损失不变；逐图批量的 keep 与逐实例的 keep 逐位相同（6831 个实例·层）；
     EventStorage 里两项 `point_nms/*` 正常写出。
   - ImageNet R-50 起点（正式 run 的起点）：前向 + 反向跑通，**735 个可训练参数张量都有梯度，全部有限**。
     每层被压约 10%，饿死 6.3~10.9%（64 个有点实例里 4~7 个），本层没配上的 GT 小穗占 2.0%。见「归因限制」的起步瞬态。
   - 随机初始化下原代码本身就不完全可复现：同一进程里第一次前向与之后的前向在最终层 mask / dice / bbox 损失上有差（基点代码同样如此、
     数值相同），与 NMS 无关；等价性检查因此只比各进程的第一次前向。
   - κ=0.07（改 κ 之前）：E3 final 每层被压 7.5~9.5%、饿死 0；R-50 起点饿死 11~14%（4.6% 的 GT 小穗没配上）。
4. **推理侧**：E3 final、val 前 5 张图，`instance_inference` 带 NMS 的输出与离线 NMS 逐点相同（33/33 个实例），
   先 NMS 后过滤 ≡ 先过滤后 NMS，κ=0 时与原阈值规则逐位相同。
   诊断脚本的 `readout_model` 与内置评估器在同 5 张图上 F1@3/5/8 与计数逐位相同（κ=0.07 时测）；
   κ=0.05 的全 val 对照见上文「最终 checkpoint 一行」。
5. **冒烟**（κ=0.05 的最终 yaml + `SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 1000`，
   输出 `output_v2/_smoke-1008-group-nms/smoke_match_k005`；训练段独占 GPU1）：
   - 40 iter 与 iter 19 / 39 两次完整 val 评估都跑通，0 报错；`metrics.json` 两条记录的 127 项 loss 全部有限，
     total_loss 520.2 → 507.6，`loss_point_cls`（加权后）0.62 → 0.60。
   - `point_nms/suppressed_frac` 0.081 → 0.072，`point_nms/starved_frac` 0 → 0（20 iter 窗口的中位数）。
   - **3.42 s/iter**（38 iter / 2:09）；峰值显存 max_mem 25,888 MB，与 E3 冒烟相同，NMS 不增显存。
   - 未训练模型的评估指标全为 0（AP 0、matched_F1 0、预测点 0），与 E3 冒烟相同。
   - κ=0.07 的冒烟（`smoke_match`，与 4 个评估并发）：同样 0 报错、loss 全部有限，4.55 s/iter（受并发拖慢），
     `point_nms/starved_frac` 0.075 → 0.038。
   - criterion 的额外耗时（E3 checkpoint、4 图 batch、只前向，κ=0.07 时测）：none 1.37 s → match 1.71 s。

## 预期与判否线（跑之前写死，2026-10-08）

**读数规则**（用户的硬性规定）：

- 一律用默认读数：训练内置评估，`POINT_THRESHOLD` = 0.35（配置值），推理侧 NMS κ=0.05（配置值）。
  不做阈值扫描，不用扫描最优档或在 val 上拟合的校准值做对比。
- 判据在同一次评估上读：取后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估，该 iter 的全部指标都从这次读。
- 护栏（`segm/AP`、`spike_recall`）用全程峰值。
- 单种子。

**E0 的读数**（`output_v2/exp-0923-mini-ep100/metrics.json`，按上面的规则重算）：

- 读数那次在 iter 8999：`count_MAE_spike` 2.465，`matched_F1@8` 0.6228，`spike_recall` 0.9511，`segm/AP` 77.40。
- 峰值：`segm/AP` 78.80（final），`spike_recall` 0.9511（iter 8999）。
- 诊断（最终 checkpoint）：存在性 AUC 0.6671，重复点 0.08%，8 px 覆盖率 75.5%；最终评估计数偏差 +2.40、r 0.642。

### 机制（先判）

用 `point_token_diag.py`，本 arm 最终 checkpoint，按 run 自己的配置（匹配前与推理侧都是 κ=0.05），σ = 8。
必须传不带值的 `--sweep` 关掉扫描，且放在最前面（它是 `nargs="*"`，放在后面会吞掉后续参数）：

```
python myscripts/backup/vis_process/point_token_diag.py --sweep --run-dir output_v2/exp-<启动日期>-mini-group-nms-match --out <json>
```

- **M1（匹配稳定，用户的假设）**：第 1→9 层相邻层的内层改判率均值 ≤ **0.0190**（E3 plain 0.0271 × 0.7）。
  不过 → 判「**机制未启动**」，端到端差异不归因于匹配变稳。
  这条线很严：在 E3 上只加匹配前 NMS，改判率就从 0.0271 升到 0.0319（κ=0.05；κ=0.07 为 0.0342），
  构造本身是让改判率上升的。过线需要训练把它压到 E3 + NMS 的 60% 以下。报告时并列 E3 plain 与 E3 + κ0.05 NMS 两个数。
- **M2（挑点）**：AUC_surv ≥ **0.7405**（E3 + κ0.05 推理 NMS 的 0.6905 + 0.05）。
- **护栏 G（覆盖，已修订）**：NMS 造成的 8 px 覆盖损失 ≤ **2.15 个百分点**，即不比 E3 + 同样推理 NMS（κ=0.05）的覆盖损失大。
  - 修订：计划里原写「≤ 1 个百分点」（绝对线）。E3 自己在两档 κ 下都过不了（κ=0.05 为 2.15、κ=0.07 为 5.40）：
    损失主要来自推理 NMS 这个构造本身，绝对线检验的就不是训练了。
  - 用户 2026-10-08（看过 E3 全 val 诊断、开跑之前）改为相对线：本 arm 的覆盖损失 ≤ E3 + 同 κ 推理 NMS 的覆盖损失。

### 主判据：读数那次的 `matched_F1@8`（对 E0）

| 条件 | 判定 |
|:--|:--|
| ≥ **0.6428**（E0 + 0.02） | **成立** |
| ≤ **0.6228**（E0） | **判否** |
| 介于两者之间 | **歧义** |

+0.02 高于 mini 实验族的噪声地板 0.015（见 [group-decoder](group-decoder.md)）。

### 次判据：计数（单独判）

读数那次的 `count_MAE_spike` ≤ **2.21**（E0 2.465 × 0.9）→ 计数成立。

### 护栏

- `segm/AP` 峰值 ≥ **77.80**（E0 峰值 78.80 − 1.0）；
- `spike_recall` 峰值 ≥ **0.9311**（E0 峰值 0.9511 − 0.02）。

### 归因（预注册，不作判据）

- **训练侧有没有贡献**：对照「E3 + 推理 NMS κ=0.05」的读数（iter 9499：计数 2.312，matched_F1@8 0.6348，同一读数规则）。
  本 arm 读数那次 matched_F1@8 ≥ **0.6548**（+0.02）或 `count_MAE_spike` ≤ **2.081**（× 0.9）→ 记「**训练侧有贡献**」；
  否则记「**收益来自推理 NMS**」。
- **A 对 B**：两 arm 读数那次 matched_F1@8 之差在噪声地板 0.015 内 → 记「**两种作用范围无差别**」。

### 参照（不作判据）

- E3 读数（iter 9499）：`count_MAE_spike` 2.422 / `matched_F1@8` 0.6326 / `spike_recall` 0.9364 / `segm/AP` 77.40；
  峰值 `segm/AP` 78.35、`spike_recall` 0.9657。
- 穗长直线（不训练，每穗点数按 GT mask 长边一条直线估）：val MAE 1.43、r 0.635。
- 计数一律附偏差 / Pearson r / 去偏移 MAE（读数那次的 `inference/spikelet_pair_counts.csv`），说明差异是偏移还是相关。

### 预期

按 DDQ 的 DETR 版结果，效应小：主判据大概率落在歧义区；计数改善可能主要来自推理 NMS
（E3 只加推理 NMS，读数那次计数 2.422 → 2.312，F1@8 +0.002）。

## 归因限制

- **推理 NMS 混在内置评估里。** 本 arm 的读数带推理 NMS，与 E0 / E3 的读数不同规则。「E3 + 推理 NMS」一行把只靠推理侧能拿到的部分单独量出来，
  归因只看本 arm 比它多出的部分。
- **起步阶段的饿死是已知瞬态，不补回**（用户 2026-10-08 决定，不加补回）。
  - κ=0.05 下，ImageNet R-50 起点每层有 6.3~10.9% 的有点实例饿死（2.0% 的 GT 小穗本层没配上）。
  - 沿 E3 自己的训练轨迹（无 NMS）测同一统计：iter 499 每层 ≤ 1.6%（0.77% 的 GT 小穗没配上），iter 999 ≤ 1.6%（0.07%），
    iter 2499 与最终 checkpoint 为 0。本 arm 的轨迹可能不同。
  - 每次迭代的 `point_nms/starved_frac` 写在 `metrics.json` 里，跑完一并报告。
- **覆盖与 NMS 半径的取舍。** κ·√(w·h) 内的两个 token 可能各盖一个真小穗（沿穗轴相邻约 9.5 px），推理 NMS 会丢掉一部分覆盖；
  护栏 G 因此改成相对线（见上）。
- **M1 的方向。** 构造本身让改判率上升（E3：0.0271 → 0.0319），M1 不过时，不能说明「匹配没有变稳」之外的别的事。
- **结论只对分组 decoder 成立。** NMS 作用于小穗 token；MLP 点头（E0 / E2）上是否有用，这里不检验。
- 单种子。

## 成本

- 12500 iter（100 epoch）。冒烟独占 A100 时 3.42 s/iter，约 12 h；与 arm B 同卡并发时按 E3 100 epoch 实测的 3.71~3.82 s/iter
  再加 match 的额外耗时估，约 13~15 h。另加 25 次评估，每次约 2.5~3 min，共约 1 h。
- 峰值显存约 26~27 GB（冒烟 max_mem 25.9 G），只能用 qiuhao 的 A100（GPU1）。
- 每次存 checkpoint 785 MB × 25 个，约 20 GB 磁盘。
