# exp-2026-08-21 · total_loss 只保留点损失

| | |
|:--|:--|
| 验证想法 | [ideas/point-loss-only-total.md](../ideas/point-loss-only-total.md) |
| 对照 | [exp-2026-08-21-innercost](exp-2026-08-21-innercost.md) |
| 基线 commit | `5920709`（本次改动尚未提交时运行） |
| 输出 | `output_v2/wheat_stage2_pointhead_pointloss_only` |
| 耗时 | 08-21 17:27 → 18:16（约 49 分钟） |
| 状态 | 完成 2000/2000 iter |

## 单变量改动

相对 innercost **只改这一处**（`wheat_stage2_pointhead.yaml` +10 行）：五项检测**损失**权重
`CLASS/MASK/DICE/BOX/GIOU_WEIGHT` 由原值置 0。所有 `COST_*` 保持不变
（`COST_CLASS 4.0 / COST_MASK 5.0 / COST_DICE 5.0 / COST_BOX 5.0 / COST_GIOU 2.0`、
`COST_POINT 5.0/2.0`），`NUM_POINTS` 仍为 50。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_stage2_pointhead.yaml \
  SOLVER.MAX_ITER 2000 SOLVER.STEPS "(1600,1900)" \
  OUTPUT_DIR output_v2/wheat_stage2_pointhead_pointloss_only
```

## 结果：记账改动成功

| | innercost | 本次 |
|:--|--:|--:|
| `total_loss` | 41.63 | **9.66** |
| 检测损失合计 | 31.16 | **0.00** |
| 点损失合计 | 9.75 | 9.75 |
| `loss_point_coord` 收敛 | 0.3513 | 0.3549 |
| `loss_point_cls` 收敛 | 0.5174 | 0.5163 |
| `segm AP` | 73.3021 | 73.3130 |

`total_loss` 现已等于点损失之和（差 1%，是 detectron2 对不同 key 分别做中位数平滑所致），
可直接当收敛曲线读。

评估轨迹：

| iter | F1@5 | F1@8 | mF1 | mF1@8 | count_MAE_spike | 预测点 | segmAP |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 799 | .0855 | .1854 | .1015 | .2237 | 9.20 | 1391 | 73.31 |
| 1599 | .1072 | .2229 | .1247 | .2699 | 8.06 | 1872 | 73.31 |
| 2000 | .1072 | .2246 | .1257 | .2720 | 7.96 | 1928 | 73.31 |

## 那个 +14% 是噪声，不是效果

默认阈值下 `F1@5` .0938 → .1072（+14%）、`mF1` +7.2%，一度看似真实提升。
但零权重损失的梯度恒为 0，数学上不可能有效果。换阈值复测两个 checkpoint：

| POINT_THRESHOLD | innercost mF1 | 本次 mF1 | 谁更好 |
|:--|--:|--:|:--|
| 0.35（默认） | .1177 | **.1257** | 本次 +6.8% |
| 0.28 | **.1591** | .1544 | innercost +3.0% |
| 0.20 | .1670 | **.1719** | 本次 +2.9% |

**排序随阈值翻转** —— 真实效果不会这样。放大机制：T=0.35 卡在存在性分数分布的陡峭段，
预测点数只差 3.4%（1865 vs 1928），F1 就被放大成 14%。

两次 run 不逐位相同的原因：`SOLVER.AMP.ENABLED: true`，而 `total_loss` 量级掉了 4.3 倍，
AMP 的动态 GradScaler 稳定到不同 scale，fp16 反向舍入随之不同，2000 iter 累积后轨迹分开。

**噪声基线（重要）**：三个阈值下 `|Δ mF1|` 均在 0.005~0.008。
**以后任何改动要在 mF1 上超过约 5% 相对幅度才谈得上真实效果。**
（对比：innercost 相对 baseline 是 +67%，远在噪声之上，那个结论稳。）

## 主要发现：`count_MAE_spike` 上两个模型都不会数小穗

在 196 个配对成功的穗上做逐穗分析（`scratchpad/count_analysis.py`）：

| | 预测均值 | 预测标准差 | MAE | 相关 r | **最优重标定后 MAE** |
|:--|--:|--:|--:|--:|--:|
| GT | 14.57 | 2.67 | — | — | — |
| **闭眼恒猜 14 个** | 14.00 | 0 | **2.17** | 0 | 2.17 |
| baseline（COST=0） | 13.72 | 2.00 | **2.24** | +0.329 | **2.03** |
| pointloss_only @0.31 | 13.16 | **8.36** | **7.10** | +0.278 | **2.05** |

（pointloss 其它阈值：T=0.28 均值 16.99 / std 8.84 / MAE 7.26 / r=+.313；
T=0.35 均值 9.27 / std 7.11 / MAE 7.64 / r=+.261。全 val 257 穗上最优恒定值 14，MAE 2.342。）

三条结论：

**① baseline 的 `count_MAE_spike` = 2.30 是假象。** 它在配对穗上的 MAE 2.24 **比闭眼恒猜的
2.17 还差**，预测标准差只有 2.00（GT 2.67）—— 就是个被压缩的近似常数预测器。
[exp-2026-08-20](exp-2026-08-20-stage2-baseline.md) 里「计数学会了」的判断**应予撤回**，
当时缺了恒定基线这个对照。

**② 两个模型的计数信息量完全相同。** 最优线性重标定后 baseline 2.03、pointloss 2.05，
差异在噪声内。所以 2.24 vs 7.10 这个 3 倍差距 **100% 是离散度校准问题，与模型能力无关**。

**③ 天花板很低。** r ≈ 0.3 即 r² ≈ 10%，只能解释 10% 的方差；即使完美校准也只能到 2.03，
相比闭眼猜的 2.17 仅好 6%。

pointloss 离散度是 GT 的 3 倍的原因：**计数是 50 个 slot 各自独立过阈值再求和**，
本质是 50 个近似伯努利变量之和，方差天然大。baseline 因恒等匹配学到了
「slot k 为正当且仅当 k<Ni」这个平凡规则，slot 间高度相关，方差才小。
匹配修复后相关性消失，方差就炸开了。

## 下一步

1. **加一个标量计数头**（`decoder_output → MLP(256,256,1,3) → 每 query 的小穗数`，
   L1 回归 Ni，只在外层匹配上的实例算）。把计数从「50 个独立阈值判断求和」变成一次直接回归，
   离散度问题当场消失，且与定位完全解耦。约 30 行，散落在 decoder / criterion / maskdino 三处。
   这是直接优化主指标 `count_MAE_spike` 的一刀。
2. **评估器加「恒定预测基线」常驻参照列** —— 否则下次再看到 2.30 还会误判。
3. **`POINT_THRESHOLD` 默认值该动**（0.35 落在悬崖上，会污染往后每次对比）。
4. 暂存中的 `NUM_POINTS: 50 → 25`（见 `memory-bank/idea.md`）—— 提高 r 的唯一途径仍是改善定位。

## 注意事项

- `count_MAE_spike` **只在配对成功的穗上算**（196/257 = 76%），漏检的 24% 未计入。
  若最终目标是产量估计，`count_MAE_img`（整图总数，现为 16~22，相对整图 GT 均值 44 点
  是 36~50% 误差）更贴近实际需求。

---

## 【2026-08-26 补充前提条件】本改动只在 trunk 冻结时安全

把五项检测损失权重置 0 的正当性依据是「trunk 全部 `requires_grad=False`，
这些损失对唯一可训练的点头梯度恒为 0」。**该前提一旦不成立，本改动会摧毁检测器。**

[exp-2026-08-26-unfreeze-decoder](exp-2026-08-26-unfreeze-decoder.md) 首次启动时
沿用了本配置去解冻 decoder，结果 `segm/AP` 从 73.3 掉到 0.25 / 0.00、
`spike_recall` 归零——decoder 变成只为点损失优化，没有任何东西惩罚它破坏检测。

**任何解冻 trunk 的实验，都必须先把这五项权重恢复为
`CLASS 4.0 / MASK 5.0 / DICE 5.0 / BOX 5.0 / GIOU 2.0`。**
