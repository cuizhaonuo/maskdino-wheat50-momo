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
