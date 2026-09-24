# 想法：GroupPose 式分组 decoder —— 麦穗实例 token + 小穗 token

状态：**待验证** ｜ 提出 2026-09-23 ｜ 分支 `mini/group-decoder`（从 `mini/baseline` 拉出）｜ 配置
`wheat_joint_single_stage_mini_group.yaml` ｜ 输出 `output_v2/exp-0923-mini-group-decoder`

来源：Liu et al., *Group Pose: A Simple Baseline for End-to-End Multi-person Pose Estimation*（ICCV 2023），
本地源码 `/public/home/zncui/pycharm_project/GroupPose`。结合方案讨论见本次会话的计划文件（E3）。

## 现象

现在的小穗点由「实例 query 的 256 维向量 → MLP 一次吐 P 个框相对坐标 + P 个存在性 logit」得到。
**点本身从不看图像特征**：decoder 里只有实例的 4-D 参考框在做可变形 cross-attn，每个小穗的定位全靠实例向量隐式推出。
08-25 的 slot 诊断里存在性头只学到每个 slot 的边际频率（固定 slot 后跨实例 AUC 0.721），与此一致。
mini/baseline：matched_F1@8 0.578、每穗计数 MAE 3.81（GT 均 15.9 点，约 24%）。

## 假设

给每个小穗一个自己的 query（带 2-D 参考点、自己做可变形采样、逐层细化），并让同一穗的 token 互相看见：
1. 定位：每个点在自己的位置取特征、逐层修正，精度不再受「一个向量编码 30 个坐标」的容量限制；
2. 存在性：变成局部判别「此处有没有小穗」（P2PNet 式），不再是实例级的组合模式；
3. 去重：同穗 token 互相看见，配合内层一对一匈牙利实现无 NMS 去重。

## 修法

`cfg.MODEL.MaskDINO.POINT_GROUP_DECODER: True`（默认 False，关着时一个参数都不建、模型逐位不变）。
每个匹配 query（Ni=100）挂 `NUM_POINTS=30` 个小穗 token（DN query 不挂）。decoder 每层：

```
实例内 SA（新，1+30 个 token）
  → 实例 token 走原 DeformableTransformerDecoderLayer（跨实例 SA + DN mask / 4-D 框 cross-attn / FFN，一行不改）
  → 小穗 token：同 slot 跨实例 SA → 2-D 点可变形 cross-attn → FFN(1024)（新，独立权重）
  → 小穗参考点 ← sigmoid(inv_sig(ref) + Δ_l)，detach 后进下一层（同框的细化）
```

- 小穗 token 内容 = slot embedding + 所属实例 token 内容（detach）；初始参考点 = layer-0 的 MLP 点预测
  经初始框（mask2box）换成绝对坐标。原 MLP 点头保留为 interm 与 layer-0 的点预测，照旧按框相对坐标监督。
- 实例 token 接收实例内 SA 的更新时乘一个零初始化的标量门：开训时实例路径与原模型完全一致。
- 头：类别 / mask / 框 ← 实例 token；小穗点 = 细化后的参考点（图像归一化绝对坐标）；存在性 ← 每 token 1 个 logit。
- 损失骨架不变（外层实例匈牙利 → 内层点匈牙利 → 坐标 + 存在性 focal）。小穗 token 的输出先换到 GT 框相对系，
  误差用各向同性归一化 + ρ(e)=c·log(1+|e|/c)（c=1）；推理时点已是绝对坐标，不再经预测框换算。

与 GroupPose 源码的差异：pos 用标准写法（GroupPose 的 `with_pos_embed` 是原地 `+=`，pos 混进 v 与残差流）；
其 `MSDeformAttn` 写死 `(K+1)·2 ∈ {6,30,34,36}`，K=30 时直接报错，这里不用它；实例 token 保留 4-D 框、
与小穗 token 分两次 cross-attn；小穗 token 不与实例 token 共享权重；GroupPose 没有 DN，这里 DN query 不挂小穗组。

## 与对照臂 E2（mini/q100）的差异

四处绑在一起，作为「分组 decoder 方案」整体检验：

| 项 | E2 | E3 |
|:--|:--|:--|
| `POINT_GROUP_DECODER` | False | **True** |
| `POINT_COORD_NORM` | box_axis | **iso**（单独作用由 E1 `mini/iso-norm` 量） |
| `NUM_POINTS` | 25 | **30**（每穗 max 31、p99 22） |
| `POINT_COORD_ROBUST_C` | — | **1.0**（只作用于小穗 token 的输出） |

两臂同为 100 个实例 query；数据、schedule、其余旋钮一律相同（`POINT_EMPTY_CLS_WEIGHT` 仍为 0.1）。

## 实施前已完成的验证

- **回归**（qiuhao GPU0，同种子、同 batch、1 次训练态前向）：`POINT_GROUP_DECODER=False` 时与主仓库
  `exp/single-stage-joint-mini` 的 **127 项 loss 逐位相等**，参数量同为 52,300,433。
- **开启**（2 图前向 + 反向）：参数 65,322,300（+13.0M）；所有参数都有梯度；前向峰值 11.5 GB（基线 10.1 GB）。
- **初值**：layer-0 / interm 的 `loss_point_coord` 2.30（与 E1 的 iso 初值相同，MLP 头口径不变）；
  小穗 token 各层 17.8（ρ 后；纯 L1 时为 57）——开训时实例框由随机 mask 转来、近乎整图，
  匹配上的 token 离 GT 几十个框宽。
- **梯度范数分解**（开训时，点损失 vs 其余损失）：点损失对共享主干的梯度约 14（backbone）/ 16（pixel decoder），
  其余损失为 2390 / 3890，不存在点损失压过检测的问题；纯 L1 时小穗分支自身的梯度范数 290，ρ 后 48。
- **冒烟**（qiuhao A100，与 ep100 并发，bs 4，40 iter，iter 19 / 39 各做一次完整 val 评估）：训练与两次评估均跑通；
  3.8~3.9 s/iter（mini/baseline 3.84），峰值显存 25.9 GB（mini/baseline 23.1 GB）；未训练模型各指标为 0，与基线早期相同。
  4090（24 GB）放不下：iter 0 在 mask 损失的点采样处 OOM（已分配 22.8 GB）。
  另：`wheat_cat2_pts_coco_overfit10` 当前不可用作评估集——其标注里的图像文件名是 09-22 重新生成数据集之前的，文件已不存在。

## 预期与判否线（跑之前写死）

主对照是 **E2**（同 query 预算）。E2 的数值出来前，下面的阈值以 E2 的峰值 / 末值为变量写死：

### 机制（先判）

对 val 上配对的 wheat_2（mask IoU ≥ 0.5），用评估器同一套点级匈牙利配对，在最终 checkpoint 上，
脚本 `myscripts/backup/vis_process/point_token_diag.py`（自检：在 mini/baseline 最终 checkpoint 上重算的
matched_F1@8 为 0.57183，与训练日志 0.5718 一致）：

- **M1（存在性变成局部判别）**：全部 K 个点与 GT 做点级匈牙利，以「配上且距离 ≤ 8 px」为正，
  报存在性分数的 AUC。E3 须比 E2（同口径，MLP 头 25 个 slot）高 **≥ 0.05**。
  参照：mini/baseline（300 query）为 **0.680**（13950 个 slot，正样本率 0.447）。
  这与 08-25 的「固定 slot 跨实例 AUC 0.721」不是同一口径，不可互比。
- **M2（不引入重复点，护栏）**：阈值化后同一穗内距离 < 3 px 的预测点对数 / 预测点数 ≤ **1%**。

M1 不通过 → 判「机制未启动」，端到端差异不归因于局部存在性。

> 【2026-09-23 预注册修订，读任何 E3 结果之前】M2 原写作「E3 须低于 E2」。实测 mini/baseline 的重复点只有
> 2 / 10951（0.02%），MLP 头本来就几乎不出重复点，「更低」没有信息量；改为护栏：小穗 token 各自细化、
> 可能在同一小穗上扎堆，实例内 SA 应当防止这一点。

### 主判据：`matched_F1@8`

| 条件 | 判定 |
|:--|:--|
| 峰值 ≥ E2 峰值 **+0.02** 且末值 ≥ E2 末值 **+0.02**，护栏满足 | **成立** |
| 峰值 ≤ E2 峰值 | **判否** |
| 介于两者之间 | **歧义**（单种子，种子方差未测） |

### 次判据：计数

`count_MAE_spike` 峰值 ≤ E2 峰值 × **0.9** → 计数成立（与定位分开判）。

### 护栏

- `segm/AP` 峰值 ≥ E2 峰值 − **0.5**（实例路径靠零初始化门保护，但点损失经实例内 SA 会回传到实例 token）；
- `spike_recall` 峰值 ≥ E2 峰值 − **0.02**。

### 阈值口径

训练期评估都在 `POINT_THRESHOLD=0.35`。E3 的存在性头与 E2 完全不同，0.35 对两者对应的分位数不同；
主判据落在歧义区或判否时，必须对 E2、E3 的最终 checkpoint 做阈值扫描后比 best-vs-best，再下结论。

## 归因限制

- 四处改动绑在一起。iso 的作用用 E1（相对 mini/baseline）近似扣除（假设可加）；K=25→30 的作用未单独量；
  ρ 只在远离 GT 的区间与 L1 不同，收敛区差 < 3%。
- 若成立，后续消融：E4 点 token 不做跨实例 SA（`POINT_ACROSS_ATTN none`）；E5 `POINT_EMPTY_CLS_WEIGHT 0`、
  OKS 型损失、计数头；E6 点 DN。

## 成本

6250 iter；正式 run 开头实测 3.56 s/iter（qiuhao A100，与 ep100 并发），约 6.5 h + 25 次评估。峰值显存约 26 GB，需 A100。

## 结论（2026-09-24，对照 E2，按上面写死的规则判）

**机制未启动；主判据歧义（阈值扫描后仍未过 +0.02）；计数次判据成立；护栏全过。**
详见 [exp-2026-09-23-mini-group-decoder](../experiments/exp-2026-09-23-mini-group-decoder.md)。

| 判据 | 要求 | 实测 | |
|:--|:--|:--|:--:|
| M1 存在性 AUC | ≥ E2 + 0.05 = 0.7445 | **0.6825**（E2 0.6945，E0 0.6804） | ❌ |
| M2 重复点（护栏） | ≤ 1% | 0.29%（30 / 10499；E2 为 0） | ✅ |
| 主判据 `matched_F1@8` | 峰值、末值都 ≥ E2 + 0.02 | 末值 0.5951 vs 0.5887（+0.0064）；末 5 轮均值 +0.0061 | 歧义 |
| 阈值扫描 best-vs-best | （歧义时的预定补充） | 0.6048 @0.25 vs 0.5899 @0.30（+0.0149） | 未过 +0.02 |
| 次判据 计数 | 峰值 ≤ E2 峰值 × 0.9 = 3.12 | 2.19；末值 3.15 vs 3.59（−12%），末 5 轮 −17%，各自最优阈值下 1.55 vs 1.86（−17%） | ✅ |
| 护栏 `segm/AP` | 峰值 ≥ 73.19 | 73.90 | ✅ |
| 护栏 `spike_recall` | 峰值 ≥ 0.9033 | 0.9152 | ✅ |

峰值一律只在 spike_recall ≥ 0.8 的评估轮里取：E3 官方峰值 0.6500 出现在 iter 1249，那一轮 spike_recall 0.002
（约 1 个配对穗），是小样本假象，不采用；即便采用，末值条件也不满足，判定不变。

读法：

- **假设的机制没有出现。** 小穗 token 的存在性分数并没有比 MLP 头更会分辨「哪些点落在真小穗上」（AUC 反而略低）。
  所以即使端到端有改善，也不能归因于「存在性变成局部判别」。
- **定位**：`@8` 相对 E2 只 +0.006，在 E1/E2 暴露出的单种子波动（约 0.01~0.015）之内；
  `@5` 与 `@3` 更明显（末 5 轮 +0.016 / +0.008），横轴误差中位降 10%，沿轴不变。
- **计数是实打实的收益**：同阈值与各自最优阈值下都降 12%~17%。但它随训练往回漂（iter 3249 时 2.19 → 末值 3.15），
  且最优计数阈值（0.50）与最优 F1 阈值（0.25）相距很远，固定 0.35 两头都不是最优——「几个点」与「点在哪」需要分开决定。
- 归因限制仍在：四处改动绑在一起，iso 与 K=30 的作用未单独量。

建议：(1) 先测种子方差（E0 或 E2 再跑 2 个种子），否则 0.01 量级的差异都无法判；(2) 若继续这条线，
主攻计数：计数头（方案 ④）或把计数阈值与定位阈值解耦，而不是继续堆定位结构。
