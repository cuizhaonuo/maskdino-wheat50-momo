# 想法：把冻结线画在 pixel_decoder 与 predictor 之间（arm C）

状态：**待验证** ｜ 提出 2026-08-27 ｜ 承接
[unfreeze-decoder](unfreeze-decoder.md) / [exp-2026-08-26-unfreeze-decoder](../experiments/exp-2026-08-26-unfreeze-decoder.md)

## 动机：上一轮的两个 arm 各有缺口

[exp-2026-08-26-unfreeze-decoder](../experiments/exp-2026-08-26-unfreeze-decoder.md)
已经判定「Stage2 冻结 trunk 才是可行性闸门的真正瓶颈」，但那两个 arm 都不干净：

**① arm A 不是「只解冻 decoder」，框头被一起解冻了。**
`maskdino_decoder.py:168` 有一句别名赋值：

```python
self.bbox_embed = nn.ModuleList(box_embed_layerlist)   # L167
self.decoder.bbox_embed = self.bbox_embed              # L168
```

`named_parameters()` 默认 `remove_duplicate=True`，共享张量只按**最先注册的路径**报一次。
`self.decoder` 在 L155 注册、`self.bbox_embed` 在 L167，深度优先遍历先走到 `decoder`，
所以幸存的名字是 `predictor.decoder.bbox_embed.*` —— 含 `"predictor.decoder"` 子串。
算术精确对上 arm A 的日志值：

| arm A 可训练 | 参数量 |
|:--|--:|
| `predictor.decoder`（9 层 TransformerDecoder） | 14,109,824 |
| `predictor.decoder.bbox_embed`（9 层共享一个 MLP） | **132,612** |
| `predictor.decoder_norm` | 512 |
| 点头 `point_*_embed` | 170,134 |
| **合计** | **14,413,082** ← 日志实测值 |

**② A 与 B 之间空着一级。** B 比 A 多解冻的是 `pixel_decoder`(14,055,424) +
`mask_embed`(197,376) + `class_embed`(514) + `enc_output`/`enc_output_norm`/`label_enc`(66,816)
—— 一次动了编码器和头两侧，分不清 `segm AP` 86.6 → 99.3 是谁的功劳。

## 修法：arm C

冻结线精确画在 `pixel_decoder` 与 `predictor` 之间：**predictor 侧凡是产生预测的模块
全部解冻，编码器侧全部冻结**。

```yaml
UNFREEZE_KEYS: ["predictor.decoder", "predictor.class_embed", "predictor.mask_embed"]
```

| | `UNFREEZE_KEYS` | 可训练参数 |
|:--|:--|--:|
| arm A | `["predictor.decoder"]` | 14,413,082 |
| **arm C** | 上面三项 | **14,610,972** |
| arm B | `["sem_seg_head"]` | 28,733,212 |

`enc_output` / `enc_output_norm`（两阶段查询选择的投影）和 `label_enc`（DN 标签嵌入）
不算预测头，划在编码器侧。arm B 解冻了它们，相对 A 没有收益。

这样 **A ⊂ C ⊂ B** 成为干净的嵌套阶梯。同时 arm C 本身就是「全部头 + decoder」
这套配方的第一次完整落地，配上七项全非零的损失权重
（新配置 `configs/coco/instance-segmentation/wheat_stage2_unfreeze_allheads_overfit10.yaml`）。

## 【口径限制，必须写进实验文档】点指标预期无信息量

arm A 已在 overfit10 上打到 F1@3 0.9826 / F1@8 0.9987，**arm C 是 A 的超集，
点指标几乎必然同样饱和**。本 run 的信息量集中在 `segm AP` 与配方正确性，
不在点指标。真正的考题在真实数据集上，本次只做 overfit10。

## 跑之前写死的预期与判否线

**`segm AP`（本 run 唯一有信息量的指标）**
arm A（`mask_embed` 冻结）86.6，arm B（`mask_embed` + `pixel_decoder`）99.3。

- **arm C ≥ 97** → 掩码质量的瓶颈在 `mask_embed` 这个头，**`pixel_decoder` 不必解冻**，
  arm B 的 99.3 主要由头贡献。
- **arm C 仍在 86~90** → 瓶颈在 `pixel_decoder` 的特征，arm B 的 99.3 来自编码器，
  「全部头 + decoder」这套配方在掩码上有天花板。
- **落在 90~97** → 两者都有贡献，按差值分摊，**不得单方面下结论**。

**点指标（一致性检查，非判据）**
预期与 arm A 同级饱和。若 `F1@8 < 0.98`（明显低于 arm A 的 0.9987），说明解冻
mask/class 头**干扰**了点任务，需要单独解释，不能当噪声放过。

**`bbox AP`（控制量）**
arm A 已到 98.34（框头在 A 里就解冻了）。arm C 应**持平**；若明显下降，
说明新加的 class/mask 梯度干扰了框头。

**可训练参数数（硬控制量）**
日志必须精确打印 **14,610,972**（= 14,413,082 + `mask_embed` 197,376 + `class_embed` 514）。
对不上说明子串匹配越界或漏匹配，立即停。

**`spike_recall`（哨兵）**
应为 1.0，与 arm A/B 相同。若归零 → 检测被毁，重演 08-26 那个坑。
**崩溃期间 `loss_point_coord` 会照常下降，绝不能只看点损失判断实验是否正常。**

## 注意事项

- 与 arm A/B 的比较有一处已知的口径不平：arm A/B 当时只用了默认 `POINT_THRESHOLD=0.35`，
  而冻结组九个方案是八档扫描取各自最优。**后处理时必须把 A/B/C 三个都扫一遍**，
  否则「冻结组最优阈值 vs 解冻组固定 0.35」是偏袒冻结组的。
- 检测损失已恢复，`total_loss` 不再等于点损失之和，读收敛曲线要分项看。
- 本实验**不是**为了拿一个更好的模型，而是为了把 A/B 之间那一级补上、
  并验证这套配方在真实数据集上跑之前是正确的。
