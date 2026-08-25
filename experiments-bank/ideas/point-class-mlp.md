# 想法：存在性头只有单层线性，与坐标头容量不对称

状态：**待验证** ｜ 提出 2026-08-25 ｜ 验证于
[exp-2026-08-25-point-class-mlp](../experiments/exp-2026-08-25-point-class-mlp.md)

## 现象

`maskdino/modeling/transformer_decoder/maskdino_decoder.py:147-148`：

```python
self.point_coord_embed = MLP(hidden_dim, hidden_dim, num_points * 2, 3)   # 3 层
self.point_class_embed = nn.Linear(hidden_dim, num_points)                # 单层
```

坐标头 3 层 MLP、存在性头单层线性。而实测两者表现正好对应这个差距：
坐标头收敛到 `d=1.8px`（[exp-2026-08-25-slot-specialization](../experiments/exp-2026-08-25-slot-specialization.md)），
存在性头跨实例 AUC 只有 0.721、置信度方差 47.5% 由 slot 身份解释。

## 假设

在 Stage2 里 trunk 全冻结，`decoder_output` 是**固定特征**，点头是在做线性/浅层探针。
存在性头要预测的是「内层匈牙利分配的结果」——它是坐标头输出的复杂函数，
单层线性 `logit_k = w_k·q + b_k` 恐怕表达不了，只能退化到每个 slot 一个偏置项
（即边际频率），这与实测的「47.5% 方差由 slot 身份解释」一致。

注意这条与 [point-canonical-order](point-canonical-order.md) 是**互斥的解释**：
- 若容量是瓶颈 → 换 MLP 应当提升 AUC；
- 若可表达性是瓶颈（目标本身是任意模式）→ 换 MLP 帮助有限，必须简化目标。

两条一起跑正是为了把这两种解释分开。

## 修法

新增 `cfg.MODEL.MaskDINO.POINT_CLASS_MLP`（默认 False）。为 True 时
`point_class_embed` 换成 `MLP(hidden_dim, hidden_dim, num_points, 3)`，与坐标头对齐。

`train_net.py:76` 的 `_POINT_HEAD_KEYS` 是**子串匹配**，
`point_class_embed.layers.*` 仍会被 `FREEZE_EXCEPT_POINT_HEAD` 正确解冻，无需改动。
从 Stage1 权重热启动会有该层的 shape mismatch warning，属正常（本就随机初始化）。

## 预期与判否线（跑之前写死）

- 参数量从 `256×50 + 50 = 12,850` 增至约 `256×256×2 + 256×50 = 144,082`。
- **判否线：若跨实例 AUC 相对 baseline 的 0.721 提升不足 0.03，
  则「容量不足」被否决**，瓶颈是目标的可表达性而非头的容量，
  资源应全部投向 [point-canonical-order](point-canonical-order.md)。
- 若 AUC 明显上升但点级指标不动，说明判别力提升没能转化为定位质量，
  需要回头看推理阈值标定。
- 反例风险：Stage2 只有 84（overfit10）/981（全库）个有点实例，
  144k 参数的 MLP 在 overfit10 上**可能纯粹是记住了这 84 个实例**。
  所以本实验在 overfit10 上的 AUC 提升**不能**直接外推到泛化能力，
  必须在实验文档里明确标注这一点，后续要在真实配置上复跑才能定论。
