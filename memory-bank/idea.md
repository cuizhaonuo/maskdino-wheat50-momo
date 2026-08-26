# 待验证的想法

## Stage2 点存在性损失的正负样本失衡

`maskdino/modeling/criterion.py` 的 `loss_points` 对**每个**外层匹配上的实例都计算
P=50 个点的 focal 存在性损失。其中 `wheat` 实例（GT 小穗数 Ni=0）贡献 50 个纯负样本，
而两项损失共用的分母 `denom = num_pos` 只统计匹配上的小穗数。

训练集 wheat : wheat_2 ≈ 11897 : 988（每图约 40 个实例、仅约 3 个是 wheat_2），
负样本点数约为正样本的 20 倍。点头可能因此收敛到"所有点都输出低置信度"，
使得 `TEST.POINT_THRESHOLD=0.35` 下几乎筛不出点。

两个可选缓解（择一或组合）：

- **(a) 只对 GT 类别为 wheat_2 的实例计算点损失。** wheat 实例内不出点改由分类头保证
  （`instance_inference` 已经按 `pred_classes == 1` 门控）。顺带自动排除标注噪声。
- **(b) `cls_loss` 与 `coord_loss` 的分母解耦**：`coord_loss / num_pos` 保持不变，
  `cls_loss` 改为除以 `匹配实例数 × P`（真实的负样本规模），使两项量级不再互相牵制。

先按现状跑一版 Stage2 做基线，观察 `loss_point_cls` 是否塌到常数、推理出的每穗点数
是否系统性偏低，再决定是否改。

## Stage2 的 total_loss 只保留点损失 —— 已实现，待验证（2026-08-21）

Stage2 里 trunk 全部 `requires_grad=False`，检测各项损失（`loss_ce` / `loss_mask` /
`loss_dice` / `loss_bbox` / `loss_giou`）对唯一可训练的点头梯度恒为 0，却仍然计入
`total_loss`——baseline 实测末次迭代 total_loss 51.66 中，检测项占 30.25（58.6%），
点项只占 21.62。于是 `total_loss` 这条曲线里有一多半是与训练无关的量，
且随数据增广在 batch 间波动，没法当收敛曲线读。

改法：在 `wheat_stage2_pointhead.yaml` 里把检测各项的**损失**权重置 0。

```yaml
CLASS_WEIGHT: 0.0
MASK_WEIGHT: 0.0
DICE_WEIGHT: 0.0
BOX_WEIGHT: 0.0
GIOU_WEIGHT: 0.0
```

已核对的安全性：
- `weight_dict` 全仓库只在 `maskdino.py:286-289` 被使用，就是「乘一下、不在字典里就丢弃」，
  **没有任何 `> 0` 的分支**，置 0 不会触发别的代码路径。
- `matcher` 只吃 `COST_*`（`maskdino.py:144-151`），与这五个损失权重完全独立。
  **`COST_CLASS_WEIGHT` / `COST_MASK_WEIGHT` / `COST_DICE_WEIGHT` / `COST_BOX_WEIGHT` /
  `COST_GIOU_WEIGHT` 绝对不能置 0**——外层匈牙利靠它们把 query 分配给 GT 实例，
  cost 全零会退化成恒等分配（与 `COST_POINT_*=0` 踩的是同一个坑，但发生在外层、后果严重得多）。

需要知道的两点：
- **不省算力**：`criterion.forward` 无条件遍历 `self.losses`，检测损失照算再乘 0，
  反向也照样穿过那条图。想省时间得砍 `MAX_ITER`。
- **会丢掉一个健全性信号**：置 0 后 `loss_ce` 等全显示 0.0，看不出冻结的 trunk 前向是否正常。
  但更强的那个还在——`segm AP` 来自评估器、不受损失权重影响（baseline 中 8000 iter 恒为 73.30）。

预期：`total_loss` 降到约 21.6 且等于点损失之和；`loss_point_*` 与所有 `spikelet/*` 指标
**逐项不变**（口径没动）。若这些指标有任何变化，说明有未预料到的耦合，需要回头查。

已于 2026-08-21 写入 `wheat_stage2_pointhead.yaml`（五项检测损失权重置 0）。

## NUM_POINTS 从 50 降到 25 —— 暂存，之后再做

`exp-2026-08-21-innercost` 暴露出：**坐标损失的有效动态范围只有 27%。**
模拟「P 个点在框内均匀铺开」后与 GT 做同口径匈牙利匹配：

| 铺点数 | 平均 L1 | 等价 `loss_point_coord` | TP@8px |
|:--|--:|--:|--:|
| P=14 | .2728 | 1.364 | .142 |
| P=25 | .1658 | 0.829 | .231 |
| **P=50** | .0967 | **0.484** | .438 |
| 模型实测收敛值 | .0703 | **0.351** | — |

纯随机铺 50 个点就能拿到 0.484，模型训到底才 0.351。原因是 P=50 而每穗只有约 14 个
GT 小穗，匈牙利总能为每个 GT 找到碰巧靠近的 slot——损失可以靠「把框铺满」压低，
不必真的知道小穗在哪。这解释了为什么坐标损失降了 79%，F1@8 却只到 .29 就停住。

改法：`cfg.MODEL.MaskDINO.NUM_POINTS: 50 → 25`（GT 单穗最多 24 个点，25 刚好够用）。
随机基线随之从 0.484 抬到 0.829，模型必须真的定位才能把损失压下去。

副作用：会改变 `point_coord_embed` / `point_class_embed` 的输出维度
（50×2→25×2、50→25），从 Stage1 权重热启动时这两层 shape mismatch、checkpointer
会打 warning。因为它们本就是随机初始化、Stage2 才训练，没有实际损失。

预期（跑之前写死）：`loss_point_coord` 的收敛值应**显著高于** 0.351（不能再靠铺满取巧），
但 `F1@8` 与 `matched_F1@8` 应上升；`matched_F1@8 ≥ 0.45` 视为成立，
到不了 0.45 则说明 P 不是主因。顺带观察每穗计数 MAE 是否随 slot 数减少而改善
（slot 少了，正负分配可能更稳定，与上面那条「正负样本失衡」相关）。

排在 total_loss 那条之后做，保持一次 run 一个变量。

**2026-08-25 补充证据**（[exp-2026-08-25-slot-specialization](../experiments-bank/experiments/exp-2026-08-25-slot-specialization.md)）：
- 直接统计全库确认了「最多 24 点」：train `min=1 / max=24 / mean=13.9 / p95=18`
  （981 个有点实例 / 11904 无点），val `min=7 / max=22 / mean=14.0`。**P=25 刚好够，无需留余量。**
- 新增一条独立的收益论据：`P=50` 时 slot 的边际匹配频率被摊平到 `E[Ni]/P = 0.28`，
  而存在性头恰好停在这个边际频率对应的 focal 平衡点上（相关系数 0.964）。
  P 降到 25 会把边际频率抬到 0.56，**平衡点从 0.318 抬到约 0.46**，
  直接缓解「匹配点置信度峰值 0.30、卡在推理阈值 0.35 左边」这个计数漏检的根源。
- 但要清楚：**P 只是把平衡点往上挪，不解决判别力问题本身**（见下面第一、二条）。


---

# 2026-08-25 · 存在性头专项（来自 slot 分化诊断）

**四条已于 2026-08-25 全部跑完**（4 分支 / 6 次 run，overfit10 2000 iter，四卡并行）。
**2026-08-26 追加泛化检验后，横向结论已反转**（见
[exp-2026-08-26-val-generalization](../experiments-bank/experiments/exp-2026-08-26-val-generalization.md)）：
把六个 checkpoint 拿到从未见过的 val 上评，**五个改动的 mF1 优势全部消失**
（落差齐为 −59%~−65%，clsmlp 最大），但**计数收益真实存活**。

按泛化排序：**规范排序（canonical）是唯一实质收益**（val 计数 MAE −72%，
只用 10 张图就优于 320 张图训练的 innercost）；换 MLP 的 mF1 优势属记忆、
**继续加参数已被证否**；α 与 P=25 属标定类改善。逐条状态见下方各节开头。

以下四条同出一个诊断实验：
[exp-2026-08-25-slot-specialization](../experiments-bank/experiments/exp-2026-08-25-slot-specialization.md)。
**共同的事实基础**（84 个实例、P=50、α=0.25，模型已在这 10 张图上过拟合 2000 iter）：

- 内层匹配**高度稳定**：匹配 slot 集合 Jaccard 均值 0.998，98.8% 的观测完全没变。
  此前 `exp-2026-08-21-innercost` 归因的「跨迭代不稳定」**已被证伪**。
- slot **只有部分分化**：频率 std 0.1814（完美 rank 分化应为 0.4113），
  0/50 的 slot 频率超过 0.9，均值恰为 `E[Ni]/P = 0.280`。
- 存在性头**只学到了边际频率**：slot 平均置信度与其边际频率对应的 focal 平衡点
  相关系数 **0.964**；置信度方差中 **47.5%** 由 slot 边际差异解释；
  固定 slot 后跨实例平均 AUC 仅 **0.721**（0.5 = 完全不认识实例）。

即：目标是稳定且可学的（trunk 冻结、query embedding 是固定特征、同一实例标签每次一样），
但优化只找到了最容易的那个解——每个 slot 一个偏置项。

## 1. `focal_alpha` 独立配置并调高 —— ✅ 已验证 2026-08-25：**部分成立**

> 结论：标定有效、判别力无效。置信度按预测单调上升（0.324/0.461/0.565），
> 但 AUC 在 0.72 饱和、最佳 mF1 只动 +0.003~0.004（低于噪声地板）。
> **计数在各自最优阈值下有真实改善**（MAE 7.61→6.02/6.57），定位没有。
> 同样效果调阈值就能拿到且无需重训。
> 见 [ideas](../experiments-bank/ideas/point-focal-alpha.md) / [实验](../experiments-bank/experiments/exp-2026-08-25-point-focal-alpha.md)。

`criterion.py:170` 的 `self.focal_alpha = 0.25` **硬编码**，且
`loss_labels`（300 query、约 39 个正样本）与 `loss_points`（50 slot、每个 `wheat_2` 约 14 个正样本）
**共用同一个值**。全仓库没有任何 `FOCAL` / `ALPHA` 配置项。

0.25 是从 RetinaNet 稠密检测继承的，那里负正比约 1000:1；点存在性这里
每个 `wheat_2` 实例只有 36:14 ≈ 2.6:1，温和得多，α=0.25 把正样本相对负样本压了 3 倍。

某槽位若以频率 `f` 为正样本，focal 会把置信度推到的平衡点（数值解）：

| f | α=0.25 | α=0.5 | α=0.75 |
|--:|--:|--:|--:|
| 0.28 | **0.318** | 0.413 | 0.514 |
| 0.62 | 0.443 | 0.546 | 0.644 |
| 1.00 | 1.000 | 1.000 | 1.000 |

`f = E[Ni]/P = 0.28` 对应 0.318，实测匹配点置信度中位数 **0.33**——对得上。

改法：新增 `cfg.MODEL.MaskDINO.POINT_FOCAL_ALPHA`（默认 0.25 保持现状），
在 `loss_points` 与 `_inner_point_match` 里用它替换 `self.focal_alpha`，
**不要动 `loss_labels` 的那个**。跑 α ∈ {0.25, 0.5, 0.75} 三组对照。

预期（跑之前写死）：匹配点置信度分布整体右移，超过推理阈值 0.35 的点数从 86/102 上升；
`count_MAE_spike` 改善。判否线——若**固定 slot 后跨实例 AUC 仍停在 0.72 附近**，
说明 α 只是平移了标度、没有带来判别力，那就必须走第 2 条。

注意表格最后一行：**`f=1.0` 时无论 α 取多少平衡点都是 1.0**。所以 α 与「判别力不足」
是同一个问题的两面——α 是治标，第 2 条才是治本。

## 2. GT 小穗点规范排序 —— ⭐ 2026-08-26 订正：**成立，本批唯一实质收益**

> 结论：F1 前缀 0→0.762、AUC 0.671→0.842、slot→位置单调性 0.247→**1.000**，
> 机制完全按设计工作，预设的否决条件未触发。但最佳 mF1 0.365→0.278、
> 坐标损失 0.10→0.54；**计数却是最好的（MAE 7.61→1.62）**。
> 坐标损失的平台由 LR 衰减造成，按预设规则只能判「未收敛」。
> **val 检验反转了评价**：val 计数 MAE **2.38（baseline 8.60，−72%）**，
> 只用 10 张图就优于 320 张图训练的 innercost（7.6~8.2）。定位在 val 上无区分度
> （六个 run 落差齐为 −59%~−65%）。
> **优先级升为最高：上真实配置（train 320 / val 81）+ 延长 schedule。**
> 见 [ideas](../experiments-bank/ideas/point-canonical-order.md) / [实验](../experiments-bank/experiments/exp-2026-08-25-point-canonical-order.md)。

现在存在性头要学的目标是「84 个实例各自任意的一组 50 位模式」。
若把 GT 小穗点按穗轴排序、`slot k` 固定监督第 k 个点，目标就退化成
**`slot k 亮 ⟺ k < Ni`**——头只需从 embedding 读出 `Ni` 这一个标量，单层 Linear 完全够。

**注意这条的理由已经变了**：不是为了消除标签抖动（Jaccard 0.998 证明本来就没有抖动），
而是**把学习目标的结构简化**，从任意模式变成单调阈值。

穗轴的取法：GT mask 的 PCA 主轴，或 box 长边。需要处理朝向一致性
（约定主轴正方向，例如从穗基部指向穗尖）——**这是本条的主要风险，朝向定错会让标签系统性反向**。
建议先离线在标注上验证：按主轴投影排序后，相邻小穗点的间距是否单调、稳定。

与 `exp-2026-08-20-stage2-baseline` 的关系：那一版的恒等分配意外造成了
「slot k 为正当且仅当 k < Ni」，计数漂亮但坐标塌陷到中轴线。
**本条等于把当时那个好的部分（稳定的存在性目标）留下、坏的部分（坐标不看距离）去掉。**

## 3. 存在性头容量对齐 —— ⚠️ 2026-08-26 订正：**训练集成立，泛化否决**

> 训练集上曾是三方向同时改善（AUC 0.671→0.858、mF1 +0.061）。
> **但 val 检验证明 mF1 优势全属记忆**：val 0.1483 对 baseline 0.1495，
> 训练→val 落差 −65% 为六个 run 之最。存活的只有计数（val MAE 5.42 对 8.60）。
> **「容量是瓶颈」应更正为「是训练集拟合的瓶颈、不是泛化的瓶颈」；
> 继续加大参数量已被证否。**
> 见 [ideas](../experiments-bank/ideas/point-class-mlp.md) / [实验](../experiments-bank/experiments/exp-2026-08-25-point-class-mlp.md)。

`maskdino_decoder.py:147-148`：

```python
self.point_coord_embed = MLP(hidden_dim, hidden_dim, num_points * 2, 3)   # 3 层
self.point_class_embed = nn.Linear(hidden_dim, num_points)                # 单层
```

坐标头 3 层 MLP、存在性头单层线性。实测跨实例 AUC 0.721（有判别力但很弱），
与「容量不足」这个解释相容。

改法：`point_class_embed` 改成 `MLP(hidden_dim, hidden_dim, num_points, 3)`。
**若走第 2 条（规范排序），这条就不必做**——那时目标是单调阈值，单层足够。
所以这条应排在第 1 条之后、第 2 条之前作为廉价尝试，而不是和第 2 条并行。

副作用：改变该层参数形状，从旧权重热启动会 shape mismatch 并打 warning，属正常。

## 4. `NUM_POINTS` 下调 —— ✅ 已验证 2026-08-25：**部分成立**

> 结论：最佳 mF1 +0.025、计数 MAE −24%，但**判别力不升反降**（AUC 0.671→0.640）。
> 与第 1 条同属标定类改善，不解决根因。原判否线（matched_F1@8<0.45）
> 在 overfit10 上没有分辨力（baseline 自己就超过），不能据此判成立。
> 见 [ideas](../experiments-bank/ideas/point-num-points-25.md) / [实验](../experiments-bank/experiments/exp-2026-08-25-point-num-points-25.md)。

本次诊断给它补了新证据和新的收益论据（把边际频率从 0.28 抬到 0.56、
平衡点从 0.318 抬到约 0.46），详见该条目的「2026-08-25 补充证据」。
**它与第 1 条作用机制相同（都是抬高平衡点），不要在同一次 run 里一起改**，否则归因不清。
