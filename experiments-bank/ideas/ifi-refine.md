# 想法：只加 IFI 精修头，不加 APG（形式 B，APGCC 组的归因对照臂）

状态：**预注册判定：主判据判否；M2 通过；计数护栏不过** ｜ 验证于 [exp-2026-10-06-mini-ifi-refine](../experiments/exp-2026-10-06-mini-ifi-refine.md) ｜ 提出 2026-09-26 ｜ 分支 `mini/ifi-refine`（从 `mini/baseline`@`0f71161` 开出）｜
配置 `wheat_joint_single_stage_mini_ifi_refine.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_ep100.yaml`）｜
输出 `output_v2/exp-1006-mini-ifi-refine`

## 与形式 A 的关系

代码与 `mini/apg-refine`@`2c77bb2` **逐字相同**（按文件 `git checkout 2c77bb2 -- <文件>` 带入，不是 merge / cherry-pick）。
来源、现象、结构、损失、与论文的差异、单元测试，全部见 `mini/apg-refine` 分支的 `experiments-bank/ideas/apg-refine.md`，
这里只写本臂自己的部分。关键数字抄在下面，免得来回翻。

**本臂与 A 的唯一差别：`POINT_APG: False`**。精修头、两项精修主损失（`loss_point_ref_coord` ×5、`loss_point_ref_cls` ×2，
只算预测框 IoU ≥ 0.5 的配对）、存在性在吸附后的 r 处求，这些都与 A 相同；没有 `loss_point_apg_*`，也不采样辅助点。

- **A − B = APG 的贡献**：APG 的直接作用对象是存在性（正负辅助点）与偏移（吸附监督）。
- **B − baseline = IFI 精修头本身的贡献**：局部特征 + 在匹配上的槽上学偏移。

结构一句话：25 槽 query 头及其全部原损失保留作粗定位；最后一层 decoder 之后接 IFI 精修头（1/4 + 1/8 两层特征、4 邻格插值、
拼实例 query），u = 预测框 ∘ 粗坐标（均 detach）→ r = u + Δ(u) → 在 r 处求存在性。推理输出 r 与 σ(logit(r))。

## 假设

IFI 让点头第一次看到自己位置的图像特征。只靠主损失（匹配上的槽：吸到 GT；全部槽：存在性），
偏移应能把粗槽 4~5 px 的误差吸掉一部分（沿轴误差 ↓），但存在性的监督仍只来自「这 25 个槽里哪些被匈牙利选中」，
与 baseline 同一种监督，**预期 AUC 不会明显上升**（group-decoder 的先例：局部特征 + 同样的监督，AUC 没过 +0.05）。
这正是 A 与 B 的对照要检验的：存在性的局部判别是否需要 APG 的位置对比监督。

## 预期与判否线（跑之前写死）

对照 mini/baseline 100 epoch（`output_v2/exp-0923-mini-ep100`）：`matched_F1@8` 峰值 **0.6331**（iter 11499）、末值 **0.6312**、
末 4 轮均值 0.6310；`segm/AP` 峰值 78.80；`count_MAE_spike` 峰值 2.46。噪声地板 **0.015**，单种子。

### 机制门（工具 `point_token_diag.py`，val，σ=8，阈值 0.35，读 `model_final.pth`；baseline 取自 `output_v2/_diag-0926/exp-0923-mini-ep100.json`）

- **M2（精修真的吸附了）**：全部配对的沿轴中位误差 **≤ 4.16 px**（baseline 4.62 的 −10%）。不通过 → 判**「精修未起作用」**。
- M1 存在性 AUC（baseline 0.6671）：**只记录**。它与 A 的差是 APG 对存在性的贡献。

### 主判据：`matched_F1@8`（与 A 同一套线）

| 条件 | 判定 |
|:--|:--|
| 峰值 ≥ **0.6481** 且末值 ≥ **0.6462**，且护栏满足 | **成立** |
| 峰值 ≤ **0.6331** | **判否** |
| 其余 | **歧义 / 未决**（不补种子） |

### 护栏

- `segm/AP` 峰值 ≥ **77.80**；诊断的重复点率 ≤ **1%**（基线 0.08%）。
- `count_MAE_spike` 按**校准后最优档**判：用 `point_token_diag.py --sweep 0.1 0.15 … 0.9`（步长 0.05，与逐档 `--eval_only` 等价）
  在 val 上扫 `POINT_THRESHOLD`，取最优档 ≤ **1.61**（基线最优档 1.467 @ τ 0.5，× 1.1；该值来自 0.35~0.7 的扫描，判定时用同一脚本、同一网格在基线上重扫核对）。
  在该档补一次 `--eval_only`，由 `spikelet_pair_counts.csv` 报 Pearson r、有符号偏差，并与「全部预测常数 16」的 MAE（1.86）并列。

### 归因（与 A 对比，结论写在两边的实验记录里）

- A 与 B 末 4 轮 `matched_F1@8` 均值之差 ≥ 0.015 → APG 有独立贡献；否则记「APG 增量未决」。
- A 与 B 的 AUC 之差：APG 对存在性局部判别的机制读数。

### 观察量

`matched_F1@3 / @5`；同一 checkpoint 用 `TEST.POINT_SOURCE coarse` 再评一次；诊断阈值扫描的最优档；
训练日志里 `loss_point_ref_coord` 与 `loss_point_coord` 之比。

## 预注册修订（2026-10-06，启动前）

计数护栏原写「`count_MAE_spike` 峰值 ≤ 2.71（默认阈值 0.35，2.46 × 1.1）」，启动前改为按校准后最优档判（见上文护栏）。
理由来自 CORAL 三臂的事后对照（`mini/coral-count` 分支 `experiments-bank/experiments/exp-2026-09-26-mini-coral-count.md`，
`output_v2/_diag-0927-coral/`）：默认阈值下的 `count_MAE_spike` 主要量的是系统多报（基线 τ 0.35 时偏差 +2.40），
只调阈值的基线就从 2.60 降到 1.467，全部预测常数 16 也有 1.86。本组换了出点的头（存在性校准与 25 槽头不同），
默认阈值的计数读数更不可比。主判据、机制门、其余护栏不变。

## 启动记录

2026-10-06 10:14:25 在 qiuhao A100（GPU1）上从 `7abdb68` 启动（预注册修订之后、代码与 `2c77bb2` 逐字相同），输出 `output_v2/exp-1006-mini-ifi-refine`，与 `mini/apg-refine`（10:13:24 启动）共卡并发。
同时启动的 `mini/apg-dense` 在开训第 0 iter OOM（两个进程的缓存分配器各自预留了 51 GB / 28 GB，实际分配峰值都只有 23.5 GB），
其输出目录已删，改为排队到 GPU1 空出 32 GB 再自动启动，不影响本臂。

## 实施前已完成的验证

代码与 A 逐字相同，A 的单元测试与 GPU 首 batch 检查（开关全关逐位一致、Δ≡0 时精修损失等于 baseline 点损失、
开关打开时 baseline 原有参数初始化不变）同样适用于本臂。本臂自己的：

GPU 冒烟（qiuhao A100 GPU1，09-27 05:27 起；与另外两个训练共卡；产物在 `output_v2/_smoke-0926/ifi-refine/`，.pth 已删）：

| 项 | 结果 |
|:--|:--|
| 开关打开（本臂配置），训练集第一个 batch：baseline 原有 127 项损失、总损失 628.5845 与 baseline 逐值相同 | 通过 |
| 同上：`ref_coord` / `ref_cls` 为 0（开训时没有 IoU ≥ 0.5 的预测框）；精修头梯度 0 | 符合预期 |
| 50 iter 训练：新损失键在 metrics.json 里且有限；峰值显存 23458 MB（baseline 同阶段 23065 MB）；3.87~4.02 s/iter（共卡） | 通过 |
| 载入 baseline 100 epoch `model_final.pth`，`POINT_SOURCE coarse`：segm AP 78.8017、`matched_F1@8` 0.6312、`count_MAE_spike` 2.5986、15665 点 | **与 baseline 末值逐值相同** |
| 同上，默认源（refine，精修头未训练，Δ ≡ 0）：`matched_F1@8` 0.5881、21575 点 | 走通（存在性头未训练，几乎全部槽都过阈值） |
| 同上，诊断 `--limit 20`：配对 84/90 穗，每穗 K = 25，AUC 0.497（未训练） | 走通 |
| 同上，训练集第一个 batch：`loss_point_ref_coord` 0.4452 vs `loss_point_coord` 0.3802（Δ ≡ 0 时之比 1.17 = 框系误差）；精修头梯度 0.42 / 全模型 240.4 | 框准时精修主损失生效 |

## 用法

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_mini_ifi_refine.yaml \
  OUTPUT_DIR /public/home/zncui/pycharm_project/maskdino-wheat50/output_v2/exp-<MMDD>-mini-ifi-refine
python myscripts/backup/vis_process/point_token_diag.py --run-dir output_v2/exp-<MMDD>-mini-ifi-refine --out <json>
```

A 与 B 建议并发起跑（同一张 A100，本仓库训练受 dataloader 限制，并发几乎不损速；显存各约 21 GB）。

## 结论（2026-10-07）

**预注册判定：主判据判否（峰值 0.6316 ≤ 0.6331）；M2 通过（沿轴 4.00 px）；计数护栏不过（默认 τ 0.35 后半程最低 5.19 @ iter 8999 > 2.71；baseline 2.46）。**
计数按默认读数判（2026-10-07 起不用阈值扫描的结果对比；启动前改成「校准后最优档」的那次修订不再使用），两种读法都不过。
详见 [exp-2026-10-06-mini-ifi-refine](../experiments/exp-2026-10-06-mini-ifi-refine.md)。

- AUC 0.659（baseline 0.667，A 0.630）：如假设所料，同样的监督下局部特征没有让存在性变成局部判别。
- 归因：A − B 末 4 轮 +0.0066 < 0.015 → APG 增量未决；APG 对存在性是负作用（A 的 AUC 低 0.03）。
- 同一 checkpoint 的 25 槽头 0.6413 > 精修头 0.6288。
