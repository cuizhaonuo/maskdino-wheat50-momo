# 想法：外层匈牙利匹配加小穗点代价

状态：**诊断后不跑**（零训练诊断，2026-10-07）：选择余量 +0.0026 < 0.015，不过；最终层改判率 1.1% 刚过 1% 的线，但与外层匹配自身的采样抖动（1.1%）一样大 ｜ 提出 2026-10-07 ｜ 分支 `mini/outer-point-cost`（从 `mini/baseline`@`0f71161` 开出，只放诊断脚本与本文档，没有训练）｜ 脚本 `myscripts/backup/vis_process/outer_point_cost_diag.py` ｜ 输出 `output_v2/diag-1007-outer-point-cost/`

## 动机

外层匈牙利（`maskdino/modeling/matcher.py`）的代价只有类别、框、mask：
`C_outer = 4·C_cls + 5·L1_box + 2·(−GIoU) + 5·BCE_mask + 5·Dice_mask`。
小穗点损失（`loss_point_coord` ×5、`loss_point_cls` ×2）是唯一在外层代价里没有对应项的实例级损失，
所以「哪个 query 当这个穗的正样本」完全不看点预测得好不好。DETR 系的惯例是代价与损失基本一一对应；
照此把点损失也加进外层代价，分类头就会学着给「点也好」的 query 打高分，推理时留下的就是它。

## 提出时就写明的疑虑

1. 点坐标是相对框的，这一项不知道穗在哪，只能在框和 mask 代价几乎打平的近似重复 query 之间改选择。
2. 近似重复的 query 特征相近、点头共享，点多半也差不多；而且只有配上的 query 有点监督，它的点本来就容易是几个里最好的。
3. KMO 这类「只在打平时起作用」的项，100 epoch 下的领先刚好落在 0.015 的噪声地板上，单种子分辨不了。

所以先做零训练诊断，再决定要不要占 A100 跑 100 epoch（参照 KMO，约 11.5 小时）。

## 诊断（零训练）

对象：`mini/baseline` 100 epoch 的 `output_v2/exp-0923-mini-ep100`。选择余量只测最终 checkpoint；
改判率另测 `model_0000999` / `model_0002999` / `model_0006499`（第 8 / 24 / 52 epoch），看训练途中的情形。

**点代价照点损失原样写（代价 = 损失）**：对每对 (query q, GT 穗 g) 先做与 `_inner_point_match` 相同的内层匈牙利，

    C_pt(q, g) = (5·Σ_配上 ‖p̂ − p‖₁ + 2·w_g·Σ_25槽 FL(s, t)) · N / N_pts

`N/N_pts` 把点损失的 ÷N_pts 换成框/mask 损失的 ÷N，与 `C_outer` 其余各项同一量纲。
Ni=0 的 wheat 穗没有内层匹配：t 全为 0，w = `POINT_EMPTY_CLS_WEIGHT` = 0.1。
加点项后的外层代价为 `C_outer + λ·C_pt`，λ ∈ {0.5, 1, 2}，主看 λ=1。

**A. 改判率**：train 集 125 个 batch（500 图），训练态前向 + LSJ，与训练时的外层匹配同口径。
在同一份预测、同一组 12544 个 mask 采样点上，看加点项前后每个带小穗 GT 分到的 query 有没有变；
11 次外层匹配（最后一层、aux `_0`..`_8`、interm）分开统计。
参照量「采样抖动」：只把 mask 采样点重抽一次、什么都不加，分配就会变的比例。

**B. 选择余量**：val 200 图，评估态前向，与内置 `SpikeletPointEvaluator` 同口径。
used 是评估器实际采用的配对（top-100 → wheat_2 且 score ≥ 0.4 → mask IoU 匈牙利，IoU ≥ 0.5）；
oracle 是对每个 used 配上的 GT 穗，在全部 300 个 query 里取 mask IoU ≥ 0.5、本穗 F1@8 最高的那个。
oracle 在评价指标上直接挑最大值，带挑选偏差，只当上界。

**正确性核对**
- 自算的 `C_outer` 与 `HungarianMatcher` 在同一随机种子下的分配逐个一致（每个 checkpoint 前 3 个 batch：570/570，最终 checkpoint 591/591）。
- B 的 used 复现了内置评估：`matched_F1@8` 0.6312、`count_MAE_spike` 2.5986、`spike_recall` 0.9429、`spike_precision` 0.6698，
  与 `metrics.json` 最后一次评估一致；同一批预测交给真的 `SpikeletPointEvaluator.evaluate()` 复算也是这组数。

## 判据（看结果前定的，2026-10-07）

两条都过才开训练：
- 最终 checkpoint 最后一层的改判率（λ=1）≥ 1%，否则加了等于没加；
- 选择余量（oracle − used 的 `matched_F1@8`）≥ 0.015（噪声地板），否则即使完全选对也测不出来。

## 结果

**B. 选择余量（最终 checkpoint）**

| | `matched_F1@3` | `@5` | `@8` |
|:--|--:|--:|--:|
| used（= 内置评估） | 0.1580 | 0.3502 | 0.6312 |
| oracle | 0.1584 | 0.3519 | 0.6337 |

- 余量 **+0.0026**，判据 0.015。
- 578 个配上的 GT 穗里，553 个（96%）只有一个 query 的 mask 与它 IoU ≥ 0.5，24 个有 2 个，1 个有 3 个。
  只有 11 个穗（1.9%）存在点更好的另一个 query；其他候选的 F1@8 均值 0.587，低于 used。

**A. 改判率（带小穗 GT，λ=1）**

| checkpoint | 最后一层改判率 | 最后一层采样抖动 | 11 次匹配合计改判率 | 合计采样抖动 |
|:--|--:|--:|--:|--:|
| `model_0000999`（第 8 epoch） | 15.6% | 8.4% | 20.8% | 10.6% |
| `model_0002999`（第 24 epoch） | 6.2% | 3.4% | 9.6% | 5.0% |
| `model_0006499`（第 52 epoch） | 2.2% | 2.1% | 4.0% | 2.6% |
| 最终（第 100 epoch） | **1.1%** | 1.1% | 3.1% | 2.0% |

λ=0.5 / 2 时，最终 checkpoint 最后一层为 0.7% / 2.1%。wheat（Ni=0）GT 的改判率在各 checkpoint、各层都低于它自己的采样抖动。

被改判的是什么样的 query（最后一层，λ=1）：

| checkpoint | 改判数 | 新旧 query 的 mask IoU 中位数 | 每点坐标 L1 中位数（框相对）旧 → 新 | 外层代价增加中位数 |
|:--|--:|--:|--:|--:|
| 第 8 epoch | 301 / 1926 | 0.98 | 0.0979 → 0.0978 | 0.03 |
| 第 24 epoch | 120 / 1926 | 0.94 | 0.0870 → 0.0811 | 0.17 |
| 第 52 epoch | 43 / 1926 | 0.44 | 0.0951 → 0.0729 | 0.46 |
| 第 100 epoch | 22 / 1951 | 0.21 | 0.0799 → 0.0736 | 1.11 |

- 训练早期改判多，但多是几乎重合的重复框之间的平局，点的质量没变。
- 训练后期改判少，而且多数换到 mask 明显不同的 query：框相对的点代价不知道穗在哪，正是提出时担心的失效方式。
- 只给每个 GT 按 `C_outer` 排前 10 的 query 算点代价（其余取这 10 个里的最大值），与精确版分配的不一致率在各层、各 checkpoint 都 ≤ 1.1%，省算力的版本可行。

## 结论（2026-10-07）

**诊断后不跑。** 选择余量 +0.0026 远低于 0.015；最终层改判率 1.1% 刚过 1% 的线，但与匹配自身的采样抖动一样大。
外层选择不是点丢分的地方：训练到后期，每个穗基本只剩一个候选 query，没有可选的。

顺带发现（不在本想法的判据内）：最终评估里有 285 个预测的 wheat_2 穗没配上 GT（占预测穗的 33%），
它们的点占全部预测点的 32%（`fp_point_ratio` 0.3234），把端到端 P@8 从约 0.59 拉到 0.40。
这 285 个里最多 26 个可能是已配上穗的重复框（即上面 IoU ≥ 0.5 的多余候选），其余都不是重复。

## 用法

从 worktree 根目录、在 qiuhao A100（GPU1）上（`output_v2` 在主仓库，worktree 里没有，要写绝对路径）：

    R=/public/home/zncui/pycharm_project/maskdino-wheat50/output_v2/exp-0923-mini-ep100
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python myscripts/backup/vis_process/outer_point_cost_diag.py \
        --config $R/config.yaml --weights $R/model_final.pth --batches 125 --out <json>

`--skip-b` 只测改判率（约 3 分钟），`--skip-a` 只测选择余量（约 0.5 分钟）。
