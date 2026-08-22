# 想法：Stage2 的 total_loss 只保留点损失

状态：**已验证成立** ｜ 提出 2026-08-21 ｜ 验证于
[exp-2026-08-21-pointloss-only](../experiments/exp-2026-08-21-pointloss-only.md)

## 动机

Stage2 里 trunk 全部 `requires_grad=False`，检测各项损失（`loss_ce` / `loss_mask` /
`loss_dice` / `loss_bbox` / `loss_giou`）对唯一可训练的点头梯度恒为 0，却仍然计入
`total_loss` —— innercost 实测末次迭代 total 41.63 中检测项占 31.16（75%）。
于是 `total_loss` 这条曲线里绝大部分是与训练无关的量，且随数据增广在 batch 间波动，
没法当收敛曲线读。

## 改法

`wheat_stage2_pointhead.yaml` 中把检测各项的**损失**权重置 0：

```yaml
CLASS_WEIGHT: 0.0
MASK_WEIGHT: 0.0
DICE_WEIGHT: 0.0
BOX_WEIGHT: 0.0
GIOU_WEIGHT: 0.0
```

## 跑之前核对的安全性

- `weight_dict` 全仓库只在 `maskdino.py:286-289` 使用，就是「乘一下、不在字典里就丢弃」，
  **没有任何 `> 0` 的分支**，置 0 不会触发别的代码路径。
- `matcher` 只吃 `COST_*`（`maskdino.py:144-151`），与这五个损失权重完全独立。
  **`COST_CLASS/MASK/DICE/BOX/GIOU_WEIGHT` 绝对不能置 0** —— 外层匈牙利靠它们把 query
  分配给 GT 实例，cost 全零会退化成恒等分配（与 `COST_POINT_*=0` 同一个坑，
  但发生在外层、后果严重得多）。
- **不省算力**：`criterion.forward` 无条件遍历 `self.losses`，检测损失照算再乘 0，
  反向也照样穿过那条图。
- **代价**：置 0 后 `loss_ce` 等全显示 0.0，看不出冻结 trunk 的前向是否正常。
  但更强的信号还在 —— `segm AP` 来自评估器，不受损失权重影响。

## 预期（跑之前写死）

- 应该变：`total_loss` 降到约 21.6 且等于各份点损失之和；五项检测损失全为 0。
- 不该变：`loss_point_*` 与所有 `spikelet/*` 逐项不变，`segm AP` 恒定。
- 判否：任何一项「不该变」的出现变化，说明存在未预料的耦合，需停下来查。

## 结论

**全部达成。** `total_loss` 41.63 → 9.66，检测损失合计精确 0.00，点损失合计两次均为 9.75；
`loss_point_coord` 0.3513 → 0.3549（+1.0%）、`loss_point_cls` 0.5174 → 0.5163（−0.2%）；
`segm AP` 73.3021 → 73.3130。

`spikelet/*` 在默认阈值下有 +7~14% 的上移，一度看似真实效果，但换阈值复测后**排序会翻转**
（0.35 本次胜、0.28 innercost 胜、0.20 本次胜），确认是 run-to-run 噪声被阈值悬崖放大。
详见实验记录。

**副产物：拿到了噪声基线** —— 三个阈值下 `|Δ mF1|` 均在 0.005~0.008。
**以后任何改动要在 mF1 上超过约 5% 相对幅度才谈得上真实效果。**
