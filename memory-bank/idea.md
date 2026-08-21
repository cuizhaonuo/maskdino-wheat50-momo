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

## Stage2 的 total_loss 只保留点损失

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

等 `COST_POINT_*` 那版实验跑完、单变量结论确认后再改，避免一次 run 混两个变量。
