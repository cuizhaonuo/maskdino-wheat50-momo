# 想法：APGCC 原样的稠密锚点小穗头 + APG（形式 C）

状态：**预注册判定：机制未启动（M1 未过）；主判据判否；计数护栏不过** ｜ 验证于 [exp-2026-10-07-mini-apg-dense](../experiments/exp-2026-10-07-mini-apg-dense.md) ｜ 提出 2026-09-26 ｜ 分支 `mini/apg-dense`（从 `mini/baseline`@`0f71161` 开出）｜
配置 `wheat_joint_single_stage_mini_apg_dense.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜
输出 `output_v2/exp-<启动日期>-mini-apg-dense`

同组另外两个形式：`mini/apg-refine`（A：25 槽粗定位 + IFI 精修头 + APG）、`mini/ifi-refine`（B：A 去掉 APG）。
共享模块 `maskdino/modeling/point_ifi.py` 按文件从 `mini/apg-refine`@`2c77bb2` 带入（逐字相同）。
APGCC 论文与官方代码的要点、baseline 点头的现象与诊断数字，见 `mini/apg-refine` 分支的 `experiments-bank/ideas/apg-refine.md`；
这里抄下本臂需要的部分。

## 来源

APGCC（ECCV 2024, arXiv 2405.10589）：**点提议 = 稠密锚点**（stride 8，每格 2×2 个参考点，每个提议预测存在性与偏移），
提议与 GT 点用匈牙利一对一匹配（代价 τ·‖p−p̂‖₂ − ĉ，τ = 0.05/px），配上为正、其余为负。
- APG：每个 GT 点周围撒正/负辅助点送进同一个预测头（正 conf→1、位置→GT；负 conf→0、Δ→0）。Table 7：Matcher 54.04 → Matcher + APG 48.84。
- IFI：任意位置的特征由最近 4 格插值（论文 Eq.8；官方代码只用了最后一个邻格，本实现按论文累加）。Table 10：最近邻 53.16 → IFI 48.84。

## 现象（与形式 A 相同，要点）

baseline 的小穗点来自「实例 query → MLP → 25 个框相对坐标 + 25 个存在性」，点不看自己位置的图像特征。
mini/baseline 100 epoch 诊断（`output_v2/_diag-0926/exp-0923-mini-ep100.json`）：存在性 AUC **0.667**；
全部配对中位误差沿轴 / 横轴 **4.62 / 2.34 px**；`matched_F1@3/5/8` = **0.158 / 0.350 / 0.631**；重复点 0.08%。
08-25 诊断：存在性头按槽的边际频率输出（固定槽跨实例 AUC 0.721）——这是「固定 25 个槽」这种表示本身的毛病。

## 假设

把点的表示从「每个实例 25 个槽」换成 APGCC 的「全图稠密锚点」：
1. 存在性变成每个位置的局部判别（此处是不是小穗中心），不再受「第 k 个槽平均多常被选中」支配；APG 的位置对比监督直接作用于它。
2. 定位是锚点（至多离 GT 2.8 px）加一个局部偏移，而不是从实例向量里回归 25 个坐标。
3. 没有 25 个槽的上限（每穗 Ni 最多 31）。

与形式 A 的区别：A 保留 25 槽作粗定位、只在槽的位置上做局部判别与吸附；C 完全放弃槽，点与实例的归属改由预测掩码决定。
代价：点与实例的关联从「结构上就属于这个 query」变成「落在谁的掩码里」，穗相互遮挡处可能归错。

## 修法

开关 `POINT_DENSE: True` + `POINT_APG: True`（默认都关；关着时不建任何参数，模型与 baseline 逐位一致）。

**稠密头**（类别无关、不看实例 query；decoder `__init__` 最后建，开关打开时 baseline 原有 732 个参数张量的初始化不变）：

- 锚点：每个 1/4 格中心一个，1024² 输入下 **65,536** 个，与 APGCC「stride 8、每格 2×2 锚点（8i+4±2 px）」是同一组像素位置。
- 头：1/4（`mask_features`）与 1/8 两层特征，各 1×1 投影到 64 维 → 4 邻格各拼 φ(2δ) → f_θ（2 层 128）→ 双线性权重累加 →
  2 层 256 的 MLP → 存在性 logit（先验 π = 0.01）与偏移 Δ（以 1/4 格 = 4 px 为单位，零初始化）。共 223,635 个参数。
  提议位置 p = a + Δ(a)。

**训练**（每次前向算一次，不进 `_0.._8` / `_interm` / `_dn`）：

| 步骤 | 内容 |
|:--|:--|
| 有效 wheat_2 穗 | 带点且 d̄ ≥ 4 px（d̄ = 该穗 GT 小穗平均最近邻距离；Ni = 1 时取 0.1·max(框宽, 框高)） |
| 区域与权重 | 有效穗的 GT 掩码在 1/4 上覆盖 ≥ 0.5 的格，膨胀 0.5·d̄，权重 **1**；wheat 类 GT 掩码内权重 **0**（忽略）；d̄ 过小的 wheat_2 为 0；其余背景 **0.1**（`POINT_EMPTY_CLS_WEIGHT`） |
| 选锚点 | 每图至多 16384 个：先 GT 小穗 0.5·d̄ 以内的，再 wheat_2 区域其余的，再随机背景补足；权重 0 的不选 |
| 匹配 | 逐穗匈牙利：候选 = 该穗区域内、尚未被别的穗配走的选中锚点；代价 ‖p−g‖₂/d̄ − σ(conf)（APGCC Eq.1 的形式） |
| `loss_point_dense_cls` ×2 | focal（α=0.5, γ=2）× 锚点权重，全部选中锚点求和 / ΣNi |
| `loss_point_dense_coord` ×5 | 配上锚点的 L1（GT 框相对系，与 `loss_point_coord` 同口径）/ ΣNi |
| `loss_point_apg_cls` ×0.4 | APG 存在性 BCE：正→1、负→0，正负各取均值后相加 |
| `loss_point_apg_loc` ×1.0 | APG 定位 L1（GT 框相对系）：正点吸附到最近 GT；负点 **Δ→0（论文 Eq.6）** |

APG 半径 (0.125, 0.5)·d̄（原生尺度约 (2, 8) px，与论文一致），每个 GT 小穗 2 正 2 负；负点离本穗任一 GT 近于 0.125·d̄ 即丢弃。
采样用专用 CPU `torch.Generator`，不扰动 DN 用的全局 RNG。

**25 槽头及其全部原损失保留**，只作训练期辅助监督：监督上唯一的变化是多了稠密损失与 APG，推理上唯一的变化是点改由稠密头产出。

**推理**（`TEST.POINT_SOURCE` auto → dense，`MaskDINO._dense_to_query_candidates`）：
全部 65,536 个提议在 1/4 掩码上采样得到每个 query 的掩码概率 M →
提议归给「wheat_2 类分数 ≥ 0.3 且 M 最大」的 query（要求 M ≥ 0.5）→ 每个实例从自己拥有的提议里按存在性取 64 个候选。
输出与 25 槽头同形（`[Q,64,2]` 坐标 + `[Q,64,1]` logit），`instance_inference`、评估器、诊断脚本都不用改。

**与论文的差异**：

| 项 | 论文 | 本实现 | 理由 |
|:--|:--|:--|:--|
| 每格 2×2 锚点的实现 | 同一特征位置的输出通道 | 每个锚点位置单独求值（IFI） | APG 需要能在任意位置求值的头 |
| 分类损失 | CE，负样本权 λ1 = 0.5，对全部提议取均值 | focal（α=0.5）/ ΣNi，背景权 0.1 | 本任务锚点比 GT 多两个数量级（6.5 万 vs 约 50），与 baseline 点存在性同口径 |
| 定位损失 | λ·L2²（px，λ=2e-4） | L1（GT 框相对系）×5 | 与 baseline 点坐标同口径 |
| 匹配范围 | 整张图一次匈牙利 | 逐穗，只在该穗区域内 | 点要归到实例；整图 2 万锚点 × 400 点的匈牙利太慢 |
| 半径 | 固定 (2, 8) px | (0.125, 0.5)·d̄ | LSJ 缩放 0.1~2.0 |
| wheat 类区域 | — | 忽略 | 这些穗的小穗没标注 |
| IFI 加权 | 官方代码只用最后一个邻格 | 按 Eq.8 累加 4 个邻格 | 官方实现的 bug |

## 与对照的差异

对照 mini/baseline（`output_v2/exp-0923-mini-ep100`）：同数据、同 schedule（100 epoch）、同样从 ImageNet 起步，
七项原损失、权重、匹配代价、点头五个旋钮、INPUT 全部继承不动；baseline 原有参数的初始化逐位相同。
多出的只有稠密头的 22 个参数张量与四项新损失。

已知混杂：全模型梯度裁剪 `CLIP_VALUE 0.01`，新损失会改变其他损失的有效学习率；冒烟测试里记录各部分裁剪前的梯度范数。

## 预期与判否线（跑之前写死）

对照 mini/baseline 100 epoch：`matched_F1@8` 峰值 **0.6331**（iter 11499）、末值 **0.6312**、末 4 轮均值 0.6310；
`matched_F1@3/5`（iter 11499）0.161 / 0.349；`segm/AP` 峰值 78.80；`count_MAE_spike` 峰值 2.46。噪声地板 **0.015**，单种子。

### 机制门（先判；`point_token_diag.py`，val，σ=8，阈值 0.35，读 `model_final.pth`；baseline 取自 `output_v2/_diag-0926/exp-0923-mini-ep100.json`）

- **M1（存在性是局部判别）**：AUC **≥ 0.7171**（baseline 0.6671 + 0.05）。
  口径说明：本臂每个实例的候选是它自己拥有的锚点（掩码内）里存在性最高的 64 个，baseline 是 25 个槽，都属于该实例；
  候选池大小与正样本率不同，AUC 只看排序，照常比较，但在记录里注明（见下文「冒烟暴露并修掉的一个口径问题」）。
  不通过 → 判**「机制未启动」**。
- **M2（定位变准）**：全部配对的沿轴中位误差 **≤ 4.16 px**（baseline 4.62 的 −10%）。不通过 → 判**「定位未改善」**。

### 主判据：`matched_F1@8`（与形式 A、B 同一套线）

| 条件 | 判定 |
|:--|:--|
| 峰值 ≥ **0.6481** 且末值 ≥ **0.6462**，且护栏满足 | **成立** |
| 峰值 ≤ **0.6331** | **判否** |
| 其余 | **歧义 / 未决**（不补种子） |

### 护栏

- `segm/AP` 峰值 ≥ **77.80**（稠密头的梯度经 IFI 回到 pixel decoder 与 backbone）。
- `count_MAE_spike` 按**校准后最优档**判：用 `point_token_diag.py --sweep 0.1 0.15 … 0.9`（步长 0.05，与逐档 `--eval_only` 等价）
  在 val 上扫 `POINT_THRESHOLD`，取最优档 ≤ **1.61**（基线最优档 1.467 @ τ 0.5，× 1.1；该值来自 0.35~0.7 的扫描，判定时用同一脚本、同一网格在基线上重扫核对）。
  在该档补一次 `--eval_only`，由 `spikelet_pair_counts.csv` 报 Pearson r、有符号偏差，并与「全部预测常数 16」的 MAE（1.86）并列。
  稠密头的计数由阈值直接决定，先验 0.01 + focal 下分数整体偏低，默认 0.35 尤其不可比。
- 诊断的重复点率 ≤ **1%**（基线 0.08%）。稠密提议没有 NMS，去重全靠一对一匹配训练出来的存在性。

### 观察量（只记录，不作判据）

- `matched_F1@3 / @5`。
- 同一 checkpoint 用 `MODEL.MaskDINO.TEST.POINT_SOURCE coarse` 再评一次（25 槽头，与 baseline 同形），分开「trunk 变了」与「头换了」。
- 与形式 A 对比：同样有 APG 与 IFI，差别只在点的表示（稠密锚点 vs 25 槽 + 精修）。
- 归属损失：GT 穗配对上、但该穗的 GT 点落在别的预测掩码里被归错的比例（从诊断的逐穗 CSV 事后算）。

## 预注册修订（2026-10-06，启动前）

计数护栏原写「`count_MAE_spike` 峰值 ≤ 2.71（默认阈值 0.35，2.46 × 1.1）」，启动前改为按校准后最优档判（见上文护栏）。
理由来自 CORAL 三臂的事后对照（`mini/coral-count` 分支 `experiments-bank/experiments/exp-2026-09-26-mini-coral-count.md`，
`output_v2/_diag-0927-coral/`）：默认阈值下的 `count_MAE_spike` 主要量的是系统多报（基线 τ 0.35 时偏差 +2.40），
只调阈值的基线就从 2.60 降到 1.467，全部预测常数 16 也有 1.86。本组换了出点的头（存在性校准与 25 槽头不同），
默认阈值的计数读数更不可比。主判据、机制门、其余护栏不变。

## 启动记录

2026-10-06 10:15:25 与 `mini/apg-refine`、`mini/ifi-refine` 一起在 qiuhao A100（GPU1）上启动，第 0 iter 就 OOM：
另两个进程的缓存分配器各自预留了 51 GB / 28 GB（实际分配峰值都只有 23.5 GB），剩下的不够第三个。该次输出目录已删，不计入任何结果。
改为排队（`output_v2/_queue-apg-dense/run.sh`）：GPU1 空闲显存连续 3 次（约 6 分钟）≥ 32 GB 后，从本分支 HEAD 自动启动，
输出按实际启动日命名为 `output_v2/exp-<MMDD>-mini-apg-dense`，启动提交记在其 `stdout.log` 首行附近与 `_queue-apg-dense/queue.log`。

## 实施前已完成的验证

CPU 单元测试（脚本不进仓库）：

| 项 | 结果 |
|:--|:--|
| `point_ifi.py` 全部单测（IFI 权重与 `grid_sample` 一致、锚点像素位置 4k+2、锚点处 IFI 取到该格特征、d̄、APG 采样、不动全局 RNG） | 通过（与形式 A 同一文件、同一组测试） |
| 选锚点：wheat 类区域一个不选；权重 wheat_2 区域 1、背景 0.1；无重复；预算 300 时先取 n_neg 以内的锚点 | 通过 |
| 两个互相重叠的 wheat_2 穗：正样本总数 = ΣNi（16），没有锚点被配两次 | 通过 |
| 四项损失有限；梯度到达稠密头；先验 0.01 下初始 `loss_point_dense_cls` 为 2.25（未加权） | 通过 |
| `_dense_to_query_candidates`：形状 `[B,Q,K,2]` / `[B,Q,K,1]`；重叠处归给掩码概率更高的合格 query；不合格 query 按掩码不排他取点；归属的 logit = 稠密头 logit、补位为 −20；大掩码下 64 个候选全在掩码内；重叠的合格实例不共用候选 | 通过 |
| **开关全关时与 baseline 逐位一致**（CPU eval 前向，随机初始化，256² 输入）：state_dict 732 个张量、全部输出、`instance_inference` 签名 | 通过 |
| 开关打开时，baseline 原有 732 个参数张量的初始化与 baseline 相同 | 通过 |
| CPU eval 前向（开关打开，阈值 −1）：100 个实例、每个 64 个候选点，坐标落在图内 | 通过 |

GPU 冒烟（qiuhao A100 GPU1，09-27 05:40 起；与另外两个训练共卡；产物在 `output_v2/_smoke-0926/apg-dense/`，.pth 已删）。

第一轮：从 ImageNet 起步，与正式训练相同的配置。

| 项 | 结果 |
|:--|:--|
| 开关打开，训练集第一个 batch：baseline 原有 127 项损失与 baseline 逐值相同 | 通过 |
| 同上：新损失（已加权）`dense_cls` 4.498、`dense_coord` 0.106、`apg_cls` 1.839、`apg_loc` 0.018 | 有限。`apg_cls` 大是因为先验 0.01 下正辅助点的 BCE ≈ −log 0.01，这是论文的设计 |
| 同上：梯度范数 稠密头 1.73 / 全模型 7253.47（baseline 7253.47） | 不扰动全模型裁剪 |
| 50 iter 训练：新损失键在 metrics.json 里且有限；峰值显存 23481 MB（baseline 同阶段 23065 MB，+0.4 GB）；3.86~3.95 s/iter（共卡；baseline 同阶段 3.79~3.82） | 通过 |

第二轮：载入 mini/baseline 100 epoch 的 `model_final.pth`（开关打开，稠密头保持初始化）。

| 项 | 结果 |
|:--|:--|
| `--eval_only`，`POINT_SOURCE coarse`：segm AP 78.8017、`matched_F1@3/5/8` 0.1580 / 0.3502 / 0.6312、`count_MAE_spike` 2.5986、15665 点 | **与 baseline 末值逐值相同** |
| `--eval_only`，默认源（dense）：0 个点过 0.35 | 符合预期：稠密头未训练，先验 0.01 |
| 诊断 `--limit 20`：配对 84/90 穗，每穗 K = 64（5376 个 token），AUC **0.479**（未训练 ≈ 随机），正样本率 0.124 | 走通 |
| 训练集第一个 batch：稠密头梯度 1.81 / 全模型 240.4 | 训练中后期新头的梯度量级也不大 |

**冒烟暴露并修掉的一个口径问题**：第一版推理从全图存在性前 4000 的提议里给每个穗挑 64 个候选，一个穗拥有的不足 64 个时，
用掩码外、分数为 0 的提议补位。这些补位是平凡的负样本，未训练的头也能测出 AUC 0.906，M1 会被虚抬、与 25 槽头不可比。
现在每个穗从**自己拥有的全部锚点**（掩码内、归属于它）里按存在性取 64 个，只有拥有的不足 64 个时才补位（原生尺度下一个穗约 800 个锚点，
实际不会触发）；`TEST.POINT_DENSE_PRE_TOPN` 随之删除。修后未训练头的 AUC 为 0.479。对应单测：
大掩码下 64 个候选全在掩码内且全部是自己的；两个重叠的合格实例不共用候选。

M1 的口径说明随之更新：候选都在实例掩码内，与 25 槽头「每个槽都属于这个实例」同口径；
仍有的差别是候选池大小（64 vs 25）与正样本率（未训练时 0.12 vs 0.48），AUC 只看排序，照常比较并在记录里注明。

## 用法

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_mini_apg_dense.yaml \
  OUTPUT_DIR /public/home/zncui/pycharm_project/maskdino-wheat50/output_v2/exp-<MMDD>-mini-apg-dense
python myscripts/backup/vis_process/point_token_diag.py --run-dir output_v2/exp-<MMDD>-mini-apg-dense --out <json>
```

## 成本

12500 iter。每 iter 多约 6.5 万个锚点的小 MLP 前后向（约 0.15 TFLOP）与每图几个穗的匈牙利；eval 时全网格 6.5 万锚点分块求值。
预计比 baseline 慢 10% 以内，冒烟测试实测后更新。

## 结论（2026-10-07）

**预注册判定：机制未启动；主判据判否；计数护栏不过。** 详见 [exp-2026-10-07-mini-apg-dense](../experiments/exp-2026-10-07-mini-apg-dense.md)。

| | 本臂 | baseline | 线 |
|:--|--:|--:|--:|
| M1 存在性 AUC | **0.564** | 0.667 | ≥ 0.7171 ❌ |
| M2 沿轴中位误差 | **3.53 px** | 4.62 | ≤ 4.16 ✅ |
| `matched_F1@8` 峰值 / 末值 | 0.5137 / 0.5128 | 0.6331 / 0.6312 | ≤ 0.6331 → 判否 |
| `segm/AP` 峰值 | 78.23 | 78.80 | ≥ 77.80 ✅ |
| `count_MAE_spike`（默认 τ 0.35，后半程最低） | 6.28 @ iter 9999 | 2.46 @ iter 8999 | ≤ 2.71 ❌ |
| 重复点率 | 0.00% | 0.08% | ≤ 1% ✅ |

计数按默认读数判（2026-10-07 起不用阈值扫描的结果对比；启动前改成「校准后最优档」的那次修订不再使用），两种读法都不过。

- 定位最准，但稠密头的分数挤在 0.25~0.45 的窄带里，相邻锚点（间距 4 px）分不开，计数与 GT 不相关。
- 同一 checkpoint 的 25 槽头（推理不用）0.6439，是三臂最高；三臂共有「25 槽头比 baseline 高约 +0.01」的线索，未经预注册。
