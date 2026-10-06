# 想法：APGCC 的 IFI 精修头 + APG 辅助点（形式 A）

状态：**训练中（2026-10-06 启动）** ｜ 提出 2026-09-26 ｜ 分支 `mini/apg-refine`（从 `mini/baseline`@`0f71161` 开出）｜
配置 `wheat_joint_single_stage_mini_apg_refine.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜
输出 `output_v2/exp-<启动日期>-mini-apg-refine`

同组的另外两个形式：`mini/ifi-refine`（本形式去掉 APG 的归因对照臂，代码逐字相同）、`mini/apg-dense`（APGCC 原样的稠密锚点头）。

## 来源

Chen et al., *Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance*（APGCC, ECCV 2024,
arXiv 2405.10589），本地源码 `/public/home/zncui/pycharm_project/APGCC`。两件事：

- **APG**（§4.1, Eq.5-7，只在训练时）：每个 GT 点周围撒 k_pos 个正辅助点（每轴 U(±n_pos)）和 k_neg 个负辅助点
  （每轴 ±U(n_pos, n_neg)），**不经匈牙利匹配**，直接送进同一个预测头：正点 conf→1、预测位置→GT；负点 conf→0、Δ→0。
  总损失 L_point + 0.2·L_APG。论文取 (k_pos, k_neg) = (2, 2)、(n_pos, n_neg) = (2, 8) px（stride 8）。
  Table 7（SHHA MAE）：只用 Matcher 54.04 → Matcher + APG **48.84**；只用 APG 58.46。
  **官方仓库只放了推理代码，APG 分支是 `raise NotImplemented`**，本实现照论文公式写。
- **IFI**（§4.2, Eq.8）：任意位置取最近 4 个特征格 Z_i，连同相对偏移 δ_i 和位置编码 φ(δ_i) 过 MLP，按对角面积 S_i/S 加权求和。
  Table 10：最近邻 53.16 → 双线性 51.25 → IFI **48.84**；F1@8 76.27 → 78.46。
  官方实现的局部加权循环写成 `x = ret + conte * area`（ret 从不累加，实际只用了最后一个邻格），本实现按论文累加 4 个邻格。

## 现象

baseline 的点头（`maskdino_decoder.py` 的 `point_coord_embed` / `point_class_embed`）是纯 query MLP：
25 个槽的框相对坐标和存在性全部从实例向量里吐出来，**点从不看自己所在位置的图像特征**。
mini/baseline 100 epoch 的诊断（`output_v2/_diag-0926/exp-0923-mini-ep100.json`，`point_token_diag.py`，val，阈值 0.35）：

| 量 | 值 | 读法 |
|:--|--:|:--|
| 存在性 AUC（全部 K 个点对 GT 做匈牙利，「配上且 ≤ 8 px」为正） | **0.667** | 存在性头分不太清哪些点落在真小穗上 |
| 全部配对的中位误差：沿穗轴 / 横轴 | **4.62 / 2.34 px** | 小穗最近邻间距 16.7 px，沿轴约 9.5 px |
| `matched_F1@3 / @5 / @8` | **0.158 / 0.350 / 0.631** | 放宽容差就涨，**瓶颈是定位精度** |
| 重复点率（同穗内 < 3 px 的点对） | 0.08% | 几乎没有重复 —— APG 负点「去重」的作用在这里用不上 |

旁证：
- 08-25 的 slot 诊断（冻结 trunk）里存在性头只学到每槽的边际频率，固定槽跨实例 AUC 0.721。
- `mini/group-decoder` 给了每个小穗 token 自己的 2-D 参考点和可变形采样，100 epoch 的 AUC 0.699、沿轴误差 4.49 px，
  预注册的机制门「AUC ≥ 基线 + 0.05」没过：**有了局部特征，存在性也没有自动变成局部判别**。
- APGCC 的核心动机「匹配不稳定」在这里不成立：08-25 在冻结 trunk 下测过内层匹配 Jaccard 0.998。本实验不把它当机制。

## 假设

APG 的价值在于**显式的、位置对比式的局部监督**：GT 附近 ±0.125·d̄ 为正、0.125~0.5·d̄ 之外为负，外加「从附近吸附到最近 GT」的偏移监督。
配上 IFI（让头真的看到该位置的特征），存在性应变成「此处是不是小穗中心」的局部判别（AUC ↑），
偏移应把粗槽的 4~5 px 误差吸到 GT 上（沿轴误差 ↓、`matched_F1@3/5` ↑），最终抬高 `matched_F1@8`。

## 修法

开关 `POINT_REFINE: True` + `POINT_APG: True`（默认都关；关着时不建任何参数，模型与 baseline 逐位一致）。
新模块 `maskdino/modeling/point_ifi.py`；decoder / criterion / maskdino.py 各有一处接线。

**精修头**（只接在最后一层 decoder 之后，以实例 query 为条件）：

1. u = 预测框（detach）∘ 25 槽粗坐标（detach），padded 图像归一化坐标。
2. 在 u 处按 IFI 取 1/4（`mask_features`，256 维）与 1/8（多尺度特征第一层，256 维）两层特征：
   各自 1×1 投影到 64 维 → 4 邻格各拼 φ(2δ)（34 维，频率 2^linspace(0,6,8) 可学习）→ f_θ（2 层 128）→ 双线性权重累加；
   再拼实例 query 的线性投影（128），过 2 层 256 的 MLP，分出存在性 logit 和偏移 Δ（以 1/4 格 = 4 px 为单位，零初始化）。
   精修头共 289,299 个参数。
3. r = u + Δ(u)；**存在性在 r.detach() 处再求一遍**。推理输出 r 与 σ(logit(r))（`TEST.POINT_SOURCE` auto → refine）。

**为什么存在性在 r 处求、不在 u 处求**：baseline 粗槽到 GT 的中位误差约 5 px ≈ 0.3·d̄，正好落在 APG 负点区间 [0.125, 0.5]·d̄ 里。
若在 u 处求存在性，主损失说「配上了，存在」，APG 对几乎相同的输入说「不存在」，二者直接矛盾，矛盾正落在要测的 AUC 机制上。

**损失**（都只在最后一层算，不进 `_0.._8` / `_interm` / `_dn`）：

| 项 | 权重 | 内容 |
|:--|--:|:--|
| `loss_point_ref_coord` | 5 | r 换到 GT 框相对系，与 `loss_point_coord` 同口径：内层匈牙利（5·L1 + 2·focal）后 L1 |
| `loss_point_ref_cls` | 2 | 在 r 处的存在性，与 `loss_point_cls` 同口径：focal（α=0.5）覆盖全部 25 槽，Ni=0 乘 0.1 |
| `loss_point_apg_cls` | 0.4 | 辅助点存在性 BCE：正→1、负→0，正负各取均值后相加（同官方 `loss_auxiliary`） |
| `loss_point_apg_loc` | 1.0 | 辅助点定位 L1（GT 框相对系）：正负点都要求 a + Δ(a) 吸附到**最近的** GT |

- 两项主损失**只算预测框与 GT 框 IoU ≥ 0.5 的配对**（`POINT_REFINE_MIN_IOU`）。u 由预测框换算，开训时框很差，u 离 GT 几十个框宽，
  局部头无从修正。冒烟测试第 0 iter 不设门控时 `loss_point_ref_coord` 为 137.6（粗点的 76 倍），把 backbone 梯度范数从 4637 抬到 6990，
  在全模型梯度裁剪（0.01）下会压低所有其他损失的有效学习率。评估器也只评 mask IoU ≥ 0.5 的穗配对。APG 围着 GT 撒点，不受门控。
- 25 槽头及其全部原损失**原样保留**（粗定位 + 深监督），数值与 baseline 同口径。
- 单元测试：Δ ≡ 0 且 u 用 GT 框换算时，两项主损失逐值等于 `loss_point_coord` / `loss_point_cls`（差 1.5e-8 / 0）。

**APG 辅助点**：

- 每个带点实例（Ni ≥ 2 且 d̄ ≥ 4 px）的每个 GT 小穗撒 2 正 2 负。d̄ = 该穗 GT 小穗的平均最近邻距离（像素）。
  **半径按 d̄ 自适应**：n_pos = 0.125·d̄、n_neg = 0.5·d̄，原生尺度（d̄ ≈ 16.7 px）约 (2, 8) px，与论文一致；
  LSJ 缩放范围 0.1~2.0，固定像素的半径在小尺度下会跨到相邻小穗上。
- 负点若离本穗任一 GT 近于 0.125·d̄ 即丢弃；落到图外的辅助点丢弃。
- 采样用专用 CPU `torch.Generator`（`POINT_APG_SEED + rank`），不扰动 DN 用的全局 RNG。
- 辅助点以所在配对的实例 query 为条件，与主项的第二遍合并成一次头调用。

**与论文的差异**（都写在配置和代码注释里）：

| 项 | 论文 | 本实现 | 理由 |
|:--|:--|:--|:--|
| 预测头 | 稠密锚点，每个锚点一个提议 | 25 槽粗定位之后的精修头，以实例 query 为条件 | 保留 baseline 的实例-点结构；稠密形式见 `mini/apg-dense` |
| 负点的定位目标 | Δ → 0（不许动） | 吸附到最近 GT | 存在性在吸附之后判，「从 2~8 px 外吸过来」正是精修头的本职 |
| 存在性 | 在锚点处 | 在吸附后的 r 处 | 见上文「为什么存在性在 r 处求」 |
| 损失形式 | CE + λ·L2²（px，λ=2e-4） | BCE + L1（GT 框相对系） | 与 baseline 点损失同口径，权重按 λ5 = 0.2 × 主点损失权重 |
| 半径 | 固定 (2, 8) px | (0.125, 0.5)·d̄ | LSJ 尺度变化大 |
| IFI 加权 | 官方代码只用了最后一个邻格 | 按论文 Eq.8 累加 4 个邻格 | 官方实现的 bug |

## 与对照的差异

对照 mini/baseline（`output_v2/exp-0923-mini-ep100`）：同数据、同 schedule（100 epoch）、同样从 ImageNet 起步，
七项原损失、权重、匹配代价、点头五个旋钮、INPUT 全部继承不动。**开关打开时，baseline 原有的 732 个参数张量的随机初始化与 baseline 逐位相同**
（精修头放在 decoder `__init__` 最后建，不插队消耗 RNG）；多出的只有精修头的 24 个参数张量与四项新损失。

已知混杂：全模型梯度裁剪 `CLIP_VALUE 0.01`，新损失会改变其他损失的有效学习率。冒烟测试里记录各部分裁剪前的梯度范数（见下文）。

## 预期与判否线（跑之前写死）

对照 mini/baseline 100 epoch（`output_v2/exp-0923-mini-ep100`）：

| 指标 | 峰值 | 末值 | 末 4 轮均值 |
|:--|--:|--:|--:|
| `matched_F1@8` | **0.6331**（iter 11499） | **0.6312** | 0.6310 |
| `matched_F1@3 / @5`（iter 11499） | 0.161 / 0.349 | | |
| `segm/AP` | 78.80 | 78.80 | |
| `count_MAE_spike` | 2.46 | 2.60 | |

噪声地板沿用 mini 实验族的 **0.015**。单种子（用户 2026-09-24 决定不加种子）。

### 机制门（先判；决定主判据能不能归因到 APG + IFI）

工具：`myscripts/backup/vis_process/point_token_diag.py`（从 `mini/group-decoder`@`d1f428c` 按文件带入，口径与 group-decoder 相同），
val，σ=8，阈值 0.35，读 `model_final.pth`。baseline 数值取自 `output_v2/_diag-0926/exp-0923-mini-ep100.json`。

- **M1（存在性变成局部判别）**：AUC **≥ 0.7171**（baseline 0.6671 + 0.05，与 group-decoder 同线）。
  不通过 → 判**「机制未启动」**，端到端差异不归因于局部存在性。
- **M2（精修真的吸附了）**：全部配对的沿轴中位误差 **≤ 4.16 px**（baseline 4.62 的 −10%；iso-norm 4.43、group-decoder 4.49 都没到）。
  不通过 → 判**「精修未起作用」**。

### 主判据：`matched_F1@8`（与 kmo-matcher-ep100 同一套线）

| 条件 | 判定 |
|:--|:--|
| 峰值 ≥ **0.6481** 且末值 ≥ **0.6462**（都是基线 +0.015），且护栏满足 | **成立** |
| 峰值 ≤ **0.6331** | **判否** |
| 其余 | **歧义 / 未决**（不补种子） |

### 护栏

- `segm/AP` 峰值 ≥ **77.80**（基线 78.80 − 1.0）。精修头的梯度经 IFI 回到 pixel decoder 与 backbone。
- `count_MAE_spike` 按**校准后最优档**判：用 `point_token_diag.py --sweep 0.1 0.15 … 0.9`（步长 0.05，与逐档 `--eval_only` 等价）
  在 val 上扫 `POINT_THRESHOLD`，取最优档 ≤ **1.61**（基线最优档 1.467 @ τ 0.5，× 1.1；该值来自 0.35~0.7 的扫描，判定时用同一脚本、同一网格在基线上重扫核对）。
  在该档补一次 `--eval_only`，由 `spikelet_pair_counts.csv` 报 Pearson r、有符号偏差，并与「全部预测常数 16」的 MAE（1.86）并列。
- 诊断的重复点率 ≤ **1%**（基线 0.08%）。精修后两个槽可能吸到同一个小穗上，而主损失只让其中一个为正。

### 归因（与 `mini/ifi-refine` 对比）

- A（本臂）与 B（`mini/ifi-refine`）末 4 轮 `matched_F1@8` 均值之差 ≥ 0.015 → **APG 有独立贡献**；否则记「APG 增量未决」。
- 同时比较两臂的 M1：APG 的直接作用对象是存在性，AUC 之差是它的机制读数。

### 观察量（只记录，不作判据）

- `matched_F1@3 / @5`：精修最应该抬的是这两档。
- 同一 checkpoint 用 `MODEL.MaskDINO.TEST.POINT_SOURCE coarse` 再评一次：分开「trunk 变了」与「头换了」。
- 诊断阈值扫描的最优档（精修头的存在性校准与 0.35 可能不同）。
- 训练日志里 `loss_point_ref_coord` 与 `loss_point_coord` 之比：< 1 说明精修确实比粗点准（同一 GT 框相对口径）。

## 预注册修订（2026-10-06，启动前）

计数护栏原写「`count_MAE_spike` 峰值 ≤ 2.71（默认阈值 0.35，2.46 × 1.1）」，启动前改为按校准后最优档判（见上文护栏）。
理由来自 CORAL 三臂的事后对照（`mini/coral-count` 分支 `experiments-bank/experiments/exp-2026-09-26-mini-coral-count.md`，
`output_v2/_diag-0927-coral/`）：默认阈值下的 `count_MAE_spike` 主要量的是系统多报（基线 τ 0.35 时偏差 +2.40），
只调阈值的基线就从 2.60 降到 1.467，全部预测常数 16 也有 1.86。本组换了出点的头（存在性校准与 25 槽头不同），
默认阈值的计数读数更不可比。主判据、机制门、其余护栏不变。

## 实施前已完成的验证

CPU 单元测试（脚本不进仓库，与 KMO 相同做法）：

| 项 | 结果 |
|:--|:--|
| IFI 权重非负且和为 1；`sample_linear` 与 `F.grid_sample(bilinear, border, align_corners=False)` | 通过（最大误差 2.1e-6） |
| 格中心：一个权重为 1、δ=0；四格中点：权重各 0.25、\|δ\|=0.5；越界退化为边界格 | 通过 |
| SpatialEncoding 维度 34、φ(0)、频率 2^linspace(0,6,8) | 通过 |
| d̄：10 px 等距直线 = 10；Ni=1 返回 None；重复点夹到 1 px | 通过 |
| APG 采样（200 次）：正点离 GT ≤ √2·n_pos；负点每轴 \|偏移\| ∈ [n_pos, n_neg]、离任一 GT ≥ 0.125·d̄；都在图内；最近 GT 下标正确；不动全局 RNG | 通过 |
| 精修主损失在 Δ ≡ 0、u 用 GT 框时逐值等于 baseline 点损失（含 Ni=0 实例） | 通过（1.5e-8 / 0） |
| 头调用次数：两遍 = M·P 与 M·P + 辅助点数；IoU 门控去掉一个配对恰好少 P 个位置 | 通过 |
| APG 开：四项损失有限，梯度到达精修头 | 通过 |
| **开关全关时与 baseline 逐位一致**（CPU eval 前向，随机初始化，256² 输入）：state_dict 732 个张量、全部输出、`instance_inference` 签名 | 通过 |
| 开关打开时，baseline 原有 732 个参数张量的初始化与 baseline 相同 | 通过 |
| 精修源在 Δ ≡ 0 时输出的像素坐标与 coarse 源逐值相同（0.0 px） | 通过 |

GPU 冒烟（qiuhao A100 GPU1，09-27 05:14 起；与另外两个训练共卡；产物在 `output_v2/_smoke-0926/`，.pth 已删）。

第一轮：从 ImageNet 起步，与正式训练相同的配置。

| 项 | 结果 |
|:--|:--|
| 开关全关，训练集第一个 batch（AMP、固定种子）：127 项损失与 baseline 逐值相同（总损失 628.5845） | 通过（梯度范数 1e-5 级的差异来自 MSDeformAttn 反传的原子加法，baseline 自己重跑也有） |
| **开关打开**，第一个 batch：baseline 原有 127 项损失与 baseline 逐值相同 | 通过（初始化与 RNG 都没被新头扰动） |
| 同上：新损失（已加权）`apg_cls` 0.555、`apg_loc` 0.100；`ref_coord` / `ref_cls` 为 0 | 符合预期：开训时没有预测框与 GT 的 IoU ≥ 0.5，精修主损失尚未生效（见下面第二轮） |
| 同上：梯度范数 精修头 0.04 / 全模型 7253.41（baseline 7253.47） | 新损失在开训时不扰动全模型裁剪 |
| 50 iter 训练：新损失键都在 metrics.json 里且有限；训练末自带的 val 评估走通 | 通过（50 iter 时 segm AP 为 0，baseline iter 499 也只有 1.9，点指标无从读） |
| 峰值显存（训练日志 max_mem）| 23471 MB（baseline 同阶段 23065 MB，+0.4 GB） |
| 速度 | 4.00~4.13 s/iter（与两个训练共卡）；baseline 同阶段 3.79~3.82（当时与 p30 共卡），约 +5% |

第二轮：载入 mini/baseline 100 epoch 的 `model_final.pth`（开关打开，精修头保持初始化，Δ ≡ 0），在真实预测上走一遍。

| 项 | 结果 |
|:--|:--|
| `--eval_only`，`POINT_SOURCE coarse`：segm AP 78.8017、`matched_F1@3/5/8` 0.1580 / 0.3502 / 0.6312、`count_MAE_spike` 2.5986、预测 15665 点 | **与 baseline 末值逐值相同** |
| `--eval_only`，默认源（refine）：`matched_F1@8` 0.5881、`count_MAE_spike` 9.07、预测 21575 点 | 走通。点的位置与 coarse 相同（Δ ≡ 0），存在性头未训练（σ≈0.5，几乎全部 25 槽都过 0.35），所以多数 |
| 诊断 `--limit 20`：配对 84/90 穗，每穗 K = 25（2100 个 token），AUC 0.480（未训练，约等于随机），沿轴中位误差 4.27 px | 走通 |
| 训练集第一个 batch：`loss_point_ref_coord` 0.4452 vs `loss_point_coord` 0.3802（都已 ×5）；`ref_cls` 0.4367 vs 0.1611 | 框准时精修主损失生效。Δ ≡ 0 时两者之比 **1.17** 就是「预测框 vs GT 框」的框系误差，这是精修要消掉的量 |
| 同上：梯度范数 精修头 0.41 / 全模型 240.4 | 训练中后期新头的梯度量级也不大 |

据此补一条观察量的参照：训练日志里 `loss_point_ref_coord / loss_point_coord` 在 Δ ≡ 0 时是 1.17；训练后 < 1.17 说明精修在消框系误差，< 1 说明精修点比粗点（在 GT 框系下）还准。

## 用法

```bash
# 训练（qiuhao A100 GPU1）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_mini_apg_refine.yaml \
  OUTPUT_DIR /public/home/zncui/pycharm_project/maskdino-wheat50/output_v2/exp-<MMDD>-mini-apg-refine
# 机制门
python myscripts/backup/vis_process/point_token_diag.py --run-dir output_v2/exp-<MMDD>-mini-apg-refine --out <json>
# coarse 源消融
python myscripts/backup/vis_process/point_token_diag.py --run-dir output_v2/exp-<MMDD>-mini-apg-refine --out <json> \
  MODEL.MaskDINO.TEST.POINT_SOURCE coarse
```

## 成本

12500 iter。baseline 独占 A100 约 2.3 s/iter、10:49；精修头每 batch 多约 1 万个位置的小 MLP，预计增加不到 5%。
