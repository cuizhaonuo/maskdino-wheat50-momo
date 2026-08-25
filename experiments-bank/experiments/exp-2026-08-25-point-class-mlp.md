# exp-2026-08-25 · 存在性头换 3 层 MLP

| | |
|:--|:--|
| 验证想法 | [ideas/point-class-mlp.md](../ideas/point-class-mlp.md) |
| 对照 | 同期 baseline（同一天、同代码、同硬件，见下） |
| 分支 | `exp/point-class-mlp` |
| 输出 | `output_v2/exp-0825-clsmlp` |
| 状态 | 完成 2000/2000 iter · **成立** |

## 单变量改动

```python
# maskdino_decoder.py:148   POINT_CLASS_MLP=True 时
- self.point_class_embed = nn.Linear(hidden_dim, num_points)
+ self.point_class_embed = MLP(hidden_dim, hidden_dim, num_points, 3)
```

可训练参数 170,134 → 301,718（点头从 12,850 增至 144,434）。
`_POINT_HEAD_KEYS` 是子串匹配，`point_class_embed.layers.*` 仍被正确解冻，无需改 `train_net.py`。

## 结果

| | baseline | **clsmlp** | 差值 |
|:--|--:|--:|--:|
| F2 跨实例 AUC | 0.6714 | **0.8576** | **+0.186** |
| 最佳 mF1（阈值扫描） | 0.3648 @0.28 | **0.4260** @0.28 | **+0.061** |
| mF1 @0.35 | 0.3112 | 0.4061 | +0.095 |
| F1@3 @0.35 | 0.2158 | 0.2803 | +0.065 |
| matched_F1@8 @0.35 | 0.4935 | 0.6047 | +0.111 |
| count_MAE_spike @0.35 | 7.61 | **3.87** | −49% |
| 预测点数（GT 1178） | 842 | 1084 | 更接近 |
| `loss_point_coord` | 0.1016 | 0.0891 | −12% |
| `loss_point_cls` | 0.4477 | 0.3035 | −32% |

**这是六个 run 里唯一在判别力、定位、计数三个方向同时改善的。**
+0.061 的 mF1 是 `exp-2026-08-21-pointloss-only` 建立的噪声地板（0.008）的 7.6 倍。

## 预期对账

ideas 文档写死的判否线是「AUC 相对 baseline 提升不足 0.03 则容量不足被否决」。
实测 **+0.186**，远超该线，**判否线未触发**。

更重要的是它回答了 ideas 里刻意设计的互斥判据：

> 若容量是瓶颈 → 换 MLP 应当提升 AUC；
> 若可表达性是瓶颈（目标本身是任意模式）→ 换 MLP 帮助有限，必须简化目标。

**容量才是瓶颈。** 而且 clsmlp 的 AUC（0.8576）还略高于把目标彻底简化的
canonical-order（0.8421），却完全没有付出后者的定位代价。

## 这推翻了 exp-2026-08-25-slot-specialization 的建议排序

那份诊断把「存在性头只学到边际频率」解释为目标不可表达，
因而把规范排序列为**治本**、把换 MLP 列为**廉价尝试**。本实验证明反了：
目标一直是可学的，只是单层 `nn.Linear` 表达不了。
诊断的**观测**（AUC 0.721、方差 47.5% 由 slot 身份解释）没有错，
错的是从观测到病因的那一步归因。

结构上也说得通：Stage2 里 trunk 全冻结，点头是在固定特征上做探针。
`logit_k = w_k·q + b_k` 要预测「内层匈牙利分配的结果」——那是坐标头输出的复杂函数，
单层线性只能退化到每个 slot 一个偏置项，正是实测的边际频率解。

## 注意事项

- **overfit10 的 `DATASETS.TRAIN == DATASETS.TEST`，上表全部是训练集拟合质量，不是泛化指标。**
  ideas 文档已预先记录该风险：84 个实例配 144k 参数的 MLP，**提升可能相当部分来自记忆**。
  必须在 `wheat_stage2_pointhead.yaml`（train 320 / val 81）上复跑才能定论，这是本条的第一优先后续。
- 噪声地板 0.008 来自 val 集上的 `pointloss-only`，在 overfit10 上未重新标定，仅作量级参考。
- `spike_recall` 六个 run 全为 0.7262，证实冻结的 trunk 未被扰动，差异全部来自点头。
