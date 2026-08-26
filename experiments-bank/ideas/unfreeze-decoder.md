# 想法：Stage2 冻结的 query embedding 可能就是定位的天花板

状态：**成立**（判据以两倍幅度触发） ｜ 提出 2026-08-26 ｜ 验证于
[exp-2026-08-26-unfreeze-decoder](../experiments/exp-2026-08-26-unfreeze-decoder.md)

## 现象：九个方案在过拟合闸门上全部没过

overfit10 是**可行性闸门**——连 10 张图都拟合不了的方案，没必要上全量数据。
九个方案（baseline / α×2 / canonical / clsmlp / P=25 / w×3）在**自己训练了 2000 迭代、
见过几百遍的那 10 张图上**，八档扫描取各自最优：

| | 最好的一个 | 值 |
|:--|:--|--:|
| F1@3 | clsmlp | **0.2989** |
| F1@5 | clsmlp | 0.4395 |
| F1@8 | clsmlp | 0.5396 |
| 计数 MAE | canonical | 1.62（均值 14 个小穗，≈12% 误差） |

**真正过拟合成功应当 F1@3 逼近 0.9，没有任何一个方案接近。**
3px 内只对上 30%，放宽到 8px 也只有 54%。

计数部分通过（canonical 1.62），**定位这一项九个方案全部没过闸门**。

## 假设

前九个实验调的旋钮（α、P、头容量、损失降权、分配方式）**全部在点头内部**。
闸门显示点头在最有利条件下也拟合不了坐标——这不像是超参问题，像是形式本身的问题。

Stage2 的设计是 `FREEZE_EXCEPT_POINT_HEAD: True`，
**`decoder_output` 那个 256 维向量是死的**。点头是在固定特征上做探针。
如果这个向量压根没编码「这根穗子上 14 个小穗分别在哪」，
那么无论点头多大、损失怎么加权、分配怎么改，都变不出这个信息。

支持这个判断的旁证：
[exp-2026-08-26-val-generalization](../experiments/exp-2026-08-26-val-generalization.md)
已经证明把点头从 12.8k 参数加到 144k（clsmlp）只带来记忆、不带来泛化——
**加容量解决不了信息不足**。

数量级上也说得通：GT 实测同穗内小穗最近邻间距中位 **16px**，
3px 约是它的五分之一。要在冻结特征上做到这个精度，很可能超出该向量携带的信息量。

## 修法

新增 `cfg.MODEL.MaskDINO.UNFREEZE_KEYS`（默认 `[]` = 现状），
`freeze_all_but_point_head` 接受额外解冻的参数名子串。两个 arm：

| arm | `UNFREEZE_KEYS` | 解冻内容 | 可训练张量数 |
|:--|:--|:--|--:|
| A | `["predictor.decoder"]` | transformer decoder + decoder_norm | 260（+点头 8） |
| B | `["sem_seg_head"]` | 整个分割头（decoder + pixel decoder + 各预测头） | 462 |

backbone 两个 arm 都保持冻结。其余一切不变（overfit10、2000 iter、同一 LR schedule）。

## 预期与判否线（跑之前写死）

- **arm A 的 F1@3 若明显跳升（> 0.45，即相对最好的 0.2989 提升 50%）
  → 冻结的 query embedding 就是天花板**，Stage2 两阶段设计需要改，
  点头内部的旋钮都是次要的。
- **两个 arm 的 F1@3 都上不去（< 0.35）→ 信息不在特征里，而在坐标头的表达形式**，
  下一步应换形式（例如 heatmap 而非回归 P 个点），而不是继续解冻。
- arm B 应当 ≥ arm A；若 B 明显优于 A，说明 pixel decoder 的特征分辨率也是瓶颈之一。

**必须同时报告的控制量**（三者钉死指标/数据集/口径，避免重蹈前三次判据缺陷）：

- `spike_recall`（训练集口径，前九个 run 恒为 **0.7262**）。
  解冻后检测器会变，这个数一定会动；**点级指标必须同时看
  `matched_F1@σ`（只在配对上的穗子上算，对检测变化不敏感）**，
  不能只看端到端的 `F1@σ`。
- `loss_point_coord` 的收敛值与轨迹（判断是否只是训练更久而非表达能力提升）。
- 八档阈值扫描后的 best-vs-best，不得固定在 0.35 比较。

## 【首次启动即踩坑，2026-08-26】解冻 trunk 必须同时恢复检测损失权重

第一次启动直接沿用了 stage2 配置，结果 **`segm/AP` 从 73.3 掉到 0.25 / 0.00，
`spike_recall` 归零、一个 wheat_2 都检不出来**，两个 arm 都是。

原因：`wheat_stage2_pointhead.yaml` 里 `CLASS/MASK/DICE/BOX/GIOU_WEIGHT` 全是 0，
这是 [exp-2026-08-21-pointloss-only](../experiments/exp-2026-08-21-pointloss-only.md)
置的，**其正当性依据恰恰是「trunk 全部 requires_grad=False，这些损失对唯一可训练的
点头梯度恒为 0」**。一旦解冻，该前提失效——decoder 变成只为点损失优化，
**没有任何东西惩罚它破坏检测**，于是它就把检测毁了换取点损失下降。

修法：解冻的 run 必须恢复 `CLASS 4.0 / MASK 5.0 / DICE 5.0 / BOX 5.0 / GIOU 2.0`。
代价是 `total_loss` 不再等于点损失之和，读收敛曲线时要分开看。

**这条要回填到 `pointloss-only` 的文档里**：那个改动是**有条件安全**的，
条件就是 trunk 冻结；任何解冻实验都必须先撤销它。

## 注意事项

- 解冻后检测损失权重仍为 0，`class_embed` / `mask_embed` / `bbox_embed` 会拿到零梯度
  但仍被 AdamW 施加 weight decay。按 `lr=1e-4 × wd=0.05` 估算 2000 步累计收缩约 1%，
  可忽略；但仍需用 `spike_recall` 作为哨兵确认。
- 本实验**不是**为了拿一个更好的模型，而是为了判定「继续在点头里调旋钮」是否还有意义。
  即便 arm A/B 效果好，也不等于该直接采用——那会推翻 Stage2 冻结 trunk 的前提，
  需要单独论证。

## 结论（2026-08-26）

见 [exp-2026-08-26-unfreeze-decoder](../experiments/exp-2026-08-26-unfreeze-decoder.md)。**假设成立。**

| | F1@3 | F1@8 | 计数 MAE |
|:--|--:|--:|--:|
| 冻结组最好（九方案 + 最优阈值） | 0.2989 | 0.5396 | 1.62 |
| **只解冻 decoder（arm A）** | **0.9826** | **0.9987** | **0.01** |

判否线是「arm A 的 F1@3 > 0.45」，实测 0.9826，**两倍于该线**。
另一条「两个 arm 都 < 0.35 则问题在表达形式」完全没有出现。

**arm A ≈ arm B**（F1@8 0.9987 vs 0.9966），说明瓶颈在 transformer decoder，
不在 pixel decoder——最小干预就够。

**九个点头方案是在天花板之下互相比较**，它们之间 ±0.01~0.03 的差异大多是噪声。
