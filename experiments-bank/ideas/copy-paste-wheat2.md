# 想法：wheat_2 专项 copy-paste 数据增强

状态：**待验证** ｜ 提出 2026-09-03 ｜ 分支 `exp/copy-paste-wheat2`

## 动机：wheat_2 的缺口在「检出」，而它的样本数是全项目最稀缺的

`wheat_2`（完整可计数穗）是本项目的核心类别——点损失只监督它，计数指标只在它上面算。
但它在两个训练集里都是极少数：

| | 图 | 标注 | wheat_2 | 占比 | 每图 |
|:--|--:|--:|--:|--:|--:|
| Stage1 `coco_2cat` train | 2092 | 90851 | 5428 | **6.0%** | 2.6 |
| Stage2 `coco_2cat_with_points` train | 320 | 12885 | 988 | **7.7%** | 3.1 |

[exp-2026-09-03-decoder-layer-baseline](../experiments/exp-2026-09-03-decoder-layer-baseline.md)
给出了此前看不到的两条事实：

1. **Stage1 上 wheat_2 的 AP 并不比 wheat 差**（segm AP 73.43 vs 73.17），
   但 **AP50 低 9.9 点**（82.89 vs 92.83）。AP50 量的是「能不能被找到并分对类」，
   AP75/AP 量的是定位精度——**缺口全在检出，不在分割**。
2. 该缺口在编码器提案处就有 12.4 点，9 层 decoder 只收窄到 9.9 点，
   **decoder 的逐层精修基本不解决它**。

而 Stage2 解冻训练再把 wheat_2 打掉 8.6 点 AP50（→ 74.30），wheat 只掉 1.6 点。

copy-paste 直接冲检出缺口去：**增加 wheat_2 的实例数与位置多样性**。

## 改动

### 实例库（离线，`myscripts/backup/build_wheat2_bank.py`）

把每个 wheat_2 抠成「RGB 补丁 + 二值掩码 + 局部多边形 + 局部 points」存成 npz。
**三条质量闸门**，实测都真的刷掉了东西：

| 库 | 扫描 | 入库 | 刷掉 |
|:--|--:|--:|:--|
| pts（Stage2） | 988 | **934（94.5%）** | 47 贴边源、7 零点源 |
| nop（Stage1） | 5428 | **5265（97.0%）** | 162 贴边源、1 面积过小 |

- **贴边源必须刷掉**：贴边的 wheat_2 是被裁过的「半截穗」，标注却仍是 wheat_2。
  把它当完整穗贴出去 = 重新注入 2026-09-02 刚修掉的那类标签噪声
  （`INPUT.DEMOTE_BORDER_TRUNCATED`），只是换了个来源。
- **零点源必须刷掉**：训练集里确实存在 `len(points)==0` 的 wheat_2，
  贴出去会教模型「有穗但无小穗」。

### 贴图（`maskdino/data/copy_paste.py`）

在 mapper 的**原图阶段**调用（读图之后、LSJ 增广之前）。所有图都是同 GSD 的
1024×1024 航拍裁片，原图尺度一致，贴进去无尺度失配；贴完把新标注追加进
`dataset_dict["annotations"]`，**后续的多边形变换、points 变换、贴边降级、
空实例过滤全部复用 mapper 已有链路**。

两条几何约束（用户 2026-09-03 指定）：

- wheat_2 之间**零重叠**（与已有 + 已贴的 wheat_2 交集必须为 0）
- wheat_2 **可以**贴在 wheat 上面；被覆盖的 wheat 掩码减掉重叠区，
  残余面积比 < `MIN_KEEP=0.2` 则整个丢弃

复用 `myscripts/backup/coco_rle_to_polygon.py` 的
`mask_from_polygons` / `binary_mask_to_polygons` / `recompute_from_mask`，未重写。

### 配置

```yaml
# Stage1: TARGET_PER_IMG 5（现状 2.6，约 2 倍，占比 6.0% -> ~11%）
# Stage2: TARGET_PER_IMG 6（现状 3.1，约 2 倍，占比 7.7% -> ~14%）
COPY_PASTE_ENABLED / _BANK / _TARGET_PER_IMG / _MARGIN / _MIN_KEEP / _FLIP_PROB
```

`MARGIN=8 > DEMOTE_BORDER_MARGIN=2`，保证贴出的实例在原图上不贴边、不被降级逻辑误伤；
被 LSJ 裁到边的仍会正常降级，语义一致。只作用于训练，**val 完全不受影响**。

## 【⚠️ 已知风险，用户明确知情后选了激进档】

本项目有前科：[exp-2026-08-27-unfreeze-val-generalization](../experiments/exp-2026-08-27-unfreeze-val-generalization.md)
证实 overfit10 把 wheat_2 占比抬到 **14.0%**（真实 7.7%）后，val 上
**预测穗数 259 → 315~357（+39%）、`spike_precision` 0.7588 → 0.5462**。

本方案的 Stage2 档把占比推到约 14%，**与那个失败点基本重合**。
我已把该风险明确提出，用户在「保守/中等/激进/做成可扫旋钮」四档中选择了激进档。
**因此判否线必须写死并优先于主判据。**

## 跑之前写死的预期与判否线

> 上一个 run（[exp-2026-09-02-stage2-combo](../experiments/exp-2026-09-02-stage2-combo.md)）
> **没有预注册**，导致六变量同动、无法归因、且中途按错误口径读出过相反结论。
> 本文档在启动训练前提交。

对照：**Stage1** = `wheat_stage1_detector_v2_cosine`；**Stage2** = `output_v2/exp-0902-combo`。
所有评估统一 `INPUT.MIN/MAX_SIZE_TEST=1024`（Stage1 配置原本不覆盖、吃默认 800/1333，
09-03 因此错了一轮，本分支已在配置里写死）。

### 硬判否线（先验偏移，任一触发即否决或退到保守档）

| 条件 | 判定 |
|:--|:--|
| Stage2 `spike_precision` **< 0.70** | **重演 overfit10 的过检，方案否决** |
| 预测 wheat_2 穗数 / GT 257 **> 1.2**（即 > 308） | 同上 |

参考值：冻结 0.7529 / arm B 0.7200 / **combo 0.8257** / overfit10 失败 0.5462。

### 主判据（wheat_2 检出）

| 条件 | 判定 |
|:--|:--|
| Stage1 在 pts val 81 图上 wheat_2 **segm AP50 > 83.9**（基线 82.89 + 1.0） | 成立 |
| Stage2 wheat_2 **segm AP50 > 80.7**（combo 79.72 + 1.0） | 成立 |
| 提升 < 1.0 | 持平，说明先验从 6~8% 推到 11~14% 不足以改善检出 |

### 控制量（不得伤 wheat / 不得伤计数）

- wheat segm AP50 跌幅 **< 1.0**（Stage1 基线 92.83 / combo 91.75）
- Stage2 `count_MAE_spike`（各自扫十档取最优）不得差于 combo 的 **2.10** 超过 10%（即 ≤ 2.31）

### 冒烟必查（20 iter）

- 七项损失非零非 NaN；首次评估 `spike_recall` 不为 0
- 可训练参数与 combo 相同（**28,845,521**）——copy-paste 不改模型结构

## 实施前已完成的验证

**成分级**（40 张原图直接调 `Wheat2CopyPaste`，target=6）：

| | |
|:--|:--|
| wheat_2 每图 | **3.10 → 6.25** |
| 实际贴图 | 126 次，**无一张未补满** |
| wheat | 1517 → 1515：**掩码被减掉一块 122 个、完全遮挡删除 2 个** |
| 图像改动像素 | 每图 27,018（理论 3.15 × 掩码中位 8194 = 25,811，吻合） |
| 贴出实例缺点 / 违反 margin / wheat_2 互相重叠 | **0 / 0 / 0** |

**mapper 级**（60 张训练图，经完整 LSJ 链路）：

| | CP 关 | CP 开 |
|:--|--:|--:|
| wheat_2 数 | 84（每图 1.40） | **221（每图 3.68）** |
| wheat_2 上的点数 | 1098 | **2997** |
| wheat_2 两两最大 IoU | — | **0.000000** |
| 「类别=wheat 却带点」 | 0 | **0** |

> 注：LSJ 本身就会吃掉大半 wheat_2（原图 3.1 → 增广后 1.40），
> 所以增广后的 3.68 相当于**有效样本 2.6 倍**，点监督 2.7 倍。
>
> 注 2：copy-paste 消耗 `np.random`，会移动后续 LSJ 的随机流，
> 所以 CP 开/关两次跑的增广序列并不配对（上表 wheat 数的微小差异来源于此，
> 不是 bug——成分级测试证明 copy-paste 从不新增 wheat）。多进程 dataloader 下
> 各 worker 由 detectron2 自行播种，用全局 `np.random` 是正确选择。

## 归因限制（必须提前说明）

本路径一次动了两处：Stage1 的数据分布 + Stage2 的数据分布。**若结果变好，
分不清是新检测器的功劳还是 Stage2 贴图的功劳。** 要拆开需再跑一轮
「新 Stage1 + Stage2 不贴图」（+5h）。建议先跑完整路径看有无效果，
有效果再补消融；无效果则直接从判否线反推。

## 未处理的已知偏差

- **贴出来的 wheat_2 永远完整无遮挡。** 用户指定「wheat_2 可以在 wheat 上面」，
  所以贴图总在最上层，而真实田间的 wheat_2 常被别的穗叶部分遮挡。
  这会强化训练分布里「wheat_2 = 无遮挡」的相关性，可能反噬 val 上被遮挡的真穗。
- **未做边缘融合**（原论文的高斯模糊）与**颜色/光照对齐**。
  两者都是可选项，本次为保持改动最小、变量最少而未加。
- 只做水平翻转、不做旋转：wheat_2 框中位 75×177，竖直取向是强先验，旋转会破坏它。

## 成本

建库 0.5h（已完成）→ Stage1 26000 iter **~16h** → Stage1 评估 0.3h →
Stage2 8000 iter ~5h → 十档扫描 + 逐类别 AP 0.5h。**合计约 22h。**
