# 想法：arm D —— 全部解冻（含 backbone）

状态：**否决（第二条判否线触发）** ｜ 提出 2026-08-28 ｜ 验证于 [exp-2026-08-28-unfreeze-full-model](../experiments/exp-2026-08-28-unfreeze-full-model.md) ｜ 承接
[unfreeze-real-dataset](unfreeze-real-dataset.md) /
[exp-2026-08-27-unfreeze-real-dataset](../experiments/exp-2026-08-27-unfreeze-real-dataset.md)

## 动机：真实数据上「解冻越多越好」这条趋势还没到头

真实数据集（320 图 / 8000 iter / val 八档最优）上，点指标随解冻范围**单调上升**：

| arm | 可训练参数 | `matched_F1@8` | 计数 MAE | segm AP |
|:--|--:|--:|--:|--:|
| A · decoder(+框头) | 14,413,082 | 0.5428 | 2.75 | **72.69** |
| C · +mask/class 头 | 14,610,972 | 0.5534 | 2.69 | 72.54 |
| **B · +pixel_decoder** | 28,733,212 | **0.5806** | **2.48** | 67.85 |

这条趋势与 overfit10 上的结论**正好相反**（那里 A>B，且 B 过拟合最重），
已经证实是样本量伪造的排序。既然 320 图上"解冻越多越好"，
**唯一还没解冻的 backbone 就是下一个该试的**。

同时掩码是反向的：A→C 几乎不变，C→B 掉 4.7 点。
两条趋势的分界正好在 `pixel_decoder`，backbone 会不会继续这个模式，未知。

## 改动

```yaml
UNFREEZE_KEYS: ["backbone", "sem_seg_head"]     # = 全模型
```

预期可训练参数 **52,188,124**（= arm A 日志里的 14,413,082 可训练 + 37,775,042 冻结）。
配置、数据、schedule、七项损失权重与 A/C/B **逐项相同**，唯一变量仍是 `UNFREEZE_KEYS`。

### 【必须记录】backbone 实际以 0.1× 学习率训练

`train_net.py:330` 的 `build_optimizer` 对 `"backbone" in module_name` 的参数
施加 `SOLVER.BACKBONE_MULTIPLIER`，而配置链里该值是 **0.1**。
所以 arm D 是「backbone @ 1e-5 + 其余 @ 1e-4」，不是严格意义的均匀全解冻。

**这是本 run 里 `BACKBONE_MULTIPLIER` 第一次真正生效**——A/C/B 的 backbone 全程冻结，
该旋钮一直是死的（`wheat_stage2_pointhead.yaml:103` 的注释已写明这一点）。

保留 0.1 而不是改成 1.0，是为了不引入第二个变量，且这是本仓库与 Stage1 一贯的约定。
若 D 的结论依赖学习率，需要单独再跑一档 `BACKBONE_MULTIPLIER: 1.0`。

## 跑之前写死的预期与判否线

对照是 **arm B**（当前最好）：`matched_F1@8` 0.5806 / 计数 MAE 2.48 / segm AP 67.85。
跨 arm 比较**必须用 `matched_F1@σ`**，`mF1` 因 `spike_recall` 摆动大不可作单点判据
（已在 [unfreeze-real-dataset](unfreeze-real-dataset.md) 记为方法论教训）。

**主判据（八档阈值各取最优）**

- **`matched_F1@8` > 0.5806** → 单调趋势延续到 backbone，**解冻范围应当拉满**。
- **`matched_F1@8` < 0.5806** → **B 就是最优点**，解冻范围存在最优值而非越多越好；
  这会同时说明「320 图足以支撑 28.7M 参数、但不足以支撑 52.2M」。
- 两者相差 < 0.01 → 记为**持平**，按参数量与训练成本取 B。

**掩码判据**

segm AP 在 A(72.69) → C(72.54) → B(67.85) 上的跌落集中发生在解冻 `pixel_decoder` 那一步。

- **D 的 segm AP 落在 66~68** → 掩码退化在 `pixel_decoder` 处已饱和，backbone 不再加剧。
- **D 明显 < 66** → backbone 解冻进一步伤掩码，代价随解冻范围继续累积。
- **D ≥ 70** → 掩码退化不是解冻范围的单调函数，B 的 67.85 另有原因，需重新解释。

**过拟合判据**

52.2M 可训练参数 / 320 张图。A/C/B 的 val 峰值都在 iter 7199、末次仅略低、无明显回落。

- 若 D 的 val 曲线在 **iter < 4000 见顶且随后回落 > 5%** → 判为过拟合，
  说明 320 图撑不住全模型，需要更短 schedule 或更强增广。

**控制量**

- 可训练参数须精确 **52,188,124**；对不上说明子串匹配越界。
- `spike_recall` 首次评估不得为 0。
- 七项损失全部非零非 NaN。
- 显存与耗时会明显上升（backbone 反传），预计 ~2.4 s/iter、8000 iter 约 5.5h。

## 注意事项

- backbone 是 ResNet-50 + **FrozenBN**，解冻只放开卷积权重，BN 统计量本就不训练，
  所以不存在"小 batch 下 BN 统计崩坏"的风险。
- 本实验若成立，Stage1/Stage2 的两阶段划分基本失去意义（等于端到端联合训练），
  需要重新论证两阶段设计还剩什么价值。

## 结论（2026-08-28）

见 [exp-2026-08-28-unfreeze-full-model](../experiments/exp-2026-08-28-unfreeze-full-model.md)。
**单调趋势在 B 处拐头，D 被否决。**

| arm | 可训练参数 | matched_F1@8 | 计数 MAE | segm AP |
|:--|--:|--:|--:|--:|
| A · decoder(+框头) | 14,413,082 | 0.5428 | 2.75 | **72.69** |
| C · +mask/class 头 | 14,610,972 | 0.5534 | 2.69 | 72.54 |
| **B · +pixel_decoder** | 28,733,212 | **0.5806** | 2.48 | 67.85 |
| D · 全部解冻 | 52,188,124 | 0.5520 | **2.46** | **65.10** |

- **主判据第二条触发**：`matched_F1@8` 0.5520 < B 的 0.5806，差 0.0286 ≈ 持平带的三倍。
  **B 就是最优点，解冻范围存在最优值而非越多越好。**
- **掩码判据第二档触发**：segm AP 65.10 < 66，backbone 解冻**继续加剧**掩码退化、未饱和。
  且 D 的 segm AP 是一路下滑的（68.10 → 65.10），而 A/C 是上升的。
- **过拟合判据未触发**：D 的峰值仍在 iter 7199（与 A/C/B 相同）。
  **它输给 B 不是过拟合，是学到的东西对点任务没帮助、反而持续损害掩码。**

**拐点位置有实际意义**：`sem_seg_head` 是任务相关部分，backbone 是通用视觉特征。
**320 图足以重训前者、不足以重训后者。**

一条限定：本次否决的是「backbone @ 1e-5 的全解冻」（`BACKBONE_MULTIPLIER=0.1`
在本 run 里第一次生效）。要彻底定论需再跑一档 1.0，但按掩码单调下滑的趋势大概率更差。
