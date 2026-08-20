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
