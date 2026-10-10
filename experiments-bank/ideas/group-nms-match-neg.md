# 想法：小穗 token 逐层点 NMS 后再做内层匹配，被压掉的 token 判负 · arm A-neg（mini/group-nms-match-neg）

状态：**歧义（100 epoch，2026-10-10；主判据比 E0 高 0.0089，机制未启动，护栏 G 未过，计数次判据不成立，归因为推理 NMS，对 A 无差别）**，见 [exp-2026-10-09-mini-group-nms-match-neg](../experiments/exp-2026-10-09-mini-group-nms-match-neg.md) ｜ 提出 2026-10-09 ｜ 分支 `mini/group-nms-match-neg`（从 `mini/group-nms-match`@`bfe1d7e` 开出，100 epoch）｜ 配置
`wheat_joint_single_stage_mini_group_nms_match_neg.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_group_nms_match.yaml`，即 arm A 的配置）｜
输出 `output_v2/exp-1009-mini-group-nms-match-neg` ｜ 代码提交 `1dffc3b` ｜ 启动提交 `5673db5` ｜
启动时间 2026-10-09 15:58:03，qiuhao A100（GPU1）。开跑时卡上没有本组的其他 run，之后有别的用户的小任务同卡（约 5.6 GB）

本 arm 下文记作 **A-neg**。

**分叉点是有意的（用户指定）。** mini 实验的约定是新 arm 一律从 `mini/baseline` 开。用户 2026-10-09 要求在 arm A（`mini/group-nms-match`）上续做，
把被 NMS 压掉的 token 改回判负，所以从 A 的结果提交 `bfe1d7e` 开出。按约定（mini arm 的主对照一律是 E0），**主对照是 E0**
（`mini/baseline` 100 epoch，`output_v2/exp-0923-mini-ep100`）。A（`output_v2/exp-1008-mini-group-nms-match`）是只差一个开关的对照；
E3（`output_v2/exp-0925-mini-group-decoder`）与「E3 + 推理 NMS κ=0.05」只作归因参照。

## 来源与动机

- 做法的来源同 A：DDQ（Zhang et al., *Dense Distinct Query for End-to-End Object Detection*，CVPR 2023）的「匹配前 NMS，只让留下的 query 去匹配」。
  DDQ 里被压的 query 不算损失；本 arm 在这一点上**偏离 DDQ**，被压的 token 照常判负。
- A 的结论（[group-nms-match](group-nms-match.md)，[exp-2026-10-08-mini-group-nms-match](../experiments/exp-2026-10-08-mini-group-nms-match.md)）：
  主判据歧义（读数那次 F1@8 0.6228，与 E0 持平），机制未启动（M1 0.0325、M2 0.6663），护栏 G 未过（2.256），
  计数次判据成立（2.120），但归因为推理 NMS。
- A 的诊断（最终 checkpoint，全 val）指向「被压的 token 不判负」这个副作用：
  - 阈值后、不带推理 NMS 的 <3 px 重复点：A 0.69%（75 / 10809），E3 0.39%（42 / 10734）；
  - 匹配前 NMS 每层压掉的 token：A 14.2~18.2%，E3（同规则）12.0~14.8%；
  - 不带推理 NMS 的读数：A 计数 3.024、偏差 +2.85；E3 2.764、+2.53；
  - AUC_all：A 0.6896，E3 0.6989；AUC_surv：A 0.6663，E3 + κ0.05 推理 NMS 0.6905。
- 用户 2026-10-09 要求：把被压掉的 token 改成仍然判负，开新分支试。

## 假设

匹配仍只在 NMS 留下的 token 之间做（同 A），被压掉的 token 存在性目标为 0、focal 照常计入：

1. 近重复重新受到负监督：不带推理 NMS 的重复点回到 E3 的水平，AUC_all 回升（报告项，不作判据）；
2. 同时保留 A 的匹配规则：每个近重复簇只有分数最高的 token 能当正样本，同簇内谁判正不再因坐标的细微差别换人，
   而被压的 token 被往下压，簇内排序更稳；内层匹配跨层更稳（机制 M1），存在性分数更会挑点（机制 M2）；
3. 默认阈值下 matched_F1@8 与计数改善（主判据、次判据）。

## 修法

代码提交 `1dffc3b`。开关默认 `ignore`，等于 A 的行为（与 `bfe1d7e` 逐位等价，见「实施前已完成的验证」1）。

**配置**（`maskdino/config.py`）：新增 `MODEL.MaskDINO.POINT_NMS_SUPPRESSED`

- `"ignore"`（默认）：被压的 token 不算存在性损失，focal 只对留下的 token 求和。这是 A 的做法。
- `"negative"`（本 arm）：被压的 token 存在性目标为 0，focal 照常计入，即对全部 30 个 token 求和。这与 `POINT_NMS_MODE none`（E3）是同一个式子。
- 两种取值的匹配与坐标损失完全相同：只有留下的 token 进内层匹配，坐标损失只算配上的。
- 开关只在有存活掩码时起作用（`POINT_NMS_MODE="match"`，或输出里带 `pred_point_keep`）。存活掩码来自 `pred_point_keep`（arm B 的 prune）时语义同样成立，
  但本分支只验证 match 模式。

**训练侧**（`maskdino/modeling/criterion.py` 的 `loss_points`）：`keep` 非 None 且开关为 `"negative"` 时，走 `focal.sum() * w`（与 `keep=None` 同一行），
否则（`"ignore"`）走 A 的 `focal[kidx].sum() * w`。其余全部与 A 相同：

- κ 0.05（GT 框，iso 空间）、只在第 1~9 层的小穗 token 上做；layer-0 与 interm 的 MLP 点头不受影响；
- 不补回（留下的少于 Ni 时多出的 GT 小穗本层没有配对）；
- 损失分母（总 GT 小穗数）不变；Ni=0 的 wheat 实例同样做，被压 token 的负样本也乘 `POINT_EMPTY_CLS_WEIGHT` 0.1；
- `point_nms/suppressed_frac`、`point_nms/starved_frac` 照常写进 `metrics.json`。

**推理侧**：与 A 相同，最后一层对每个 wheat_2 实例的 30 个点做 NMS（κ=0.05，半径用预测框），再按 `POINT_THRESHOLD` 0.35 过滤。内置评估的点指标就是带 NMS 的。

**诊断**（`myscripts/backup/vis_process/point_token_diag.py`）：从配置读 `POINT_NMS_SUPPRESSED`，记进输出的 `nms` 段。

- 旧键的口径不变。
- 检查过要不要随开关变：
  - 旧 `auc`（AUC_all）与 `auc_surv` 的标签都是纯距离匈牙利，不依赖训练侧开关；
  - `flip` 用的是 criterion 的匹配规则，两种取值的匹配相同，也不用变。
- 被压的 token 在 `"negative"` 下应算负样本。旧指标里没有这样的口径：
  - AUC_all 包含它们，但按距离给标签，被压的 token 若离 GT 最近也会记正；
  - AUC_surv 直接不算它们。
- 所以新增 `auc_supneg`：
  - 最终层全部 30 个 token，被推理 NMS（∧ 存活掩码）压掉的一律记负，留下的标签同 `auc_surv`；
  - 另附 `value_labels_from_all_k`。
- 新增 `auc_rule`：按开关指向与训练规则一致的那个，ignore → `auc_surv`，negative → `auc_supneg`。
- `auc_supneg` 天然偏高。NMS 按分数降序压，被压的 token 总比附近某个留下的分低。所以它只能与别的 run 的 `auc_supneg` 互比。

## 与对照的差异

| 项 | E3 | A | 本 arm（A-neg） |
|:--|:--|:--|:--|
| `POINT_NMS_MODE` | none（默认） | match | match |
| `POINT_NMS_KAPPA` | —（none 下不生效） | 0.05 | 0.05 |
| `TEST.POINT_NMS_KAPPA` | 0（默认，关） | 0.05 | 0.05 |
| `POINT_NMS_SUPPRESSED` | —（none 下不生效） | 当时没有此键，行为等于 ignore | **negative** |

- **相对 A 的唯一差别**：存在性 focal 从「只对留下的求和」改成「对全部 30 个求和」。
- **相对 E3 的训练侧唯一差别**：正样本只能从 NMS 留下的 token 里选。推理侧另有 NMS（同 A）。

其余全部继承 A / E3：分组 decoder（100 个实例 query、K=30、iso、ρ c=1、`POINT_EMPTY_CLS_WEIGHT` 0.1、`POINT_FOCAL_ALPHA` 0.5）、
数据、schedule（12500 iter = 100 epoch，cosine，每 500 iter 评估一次，共 25 次）、`POINT_THRESHOLD` 0.35、`SEED` 42、AMP。
合并后的配置与 A 的 yaml 只差 `POINT_NMS_SUPPRESSED` 一个键（见验证 5）。正式 run 除 `OUTPUT_DIR` 外不带任何 CLI 覆盖。

## 实施前已完成的验证

脚本与原始输出：`output_v2/_smoke-1009-group-nms-match-neg/verify/`。

- 脚本是 `verify_neg.py`，一次性脚本，不进 git；GPU 部分是 `run_verify_gpu.sh` 与 `run_verify_gpu.log`。
- GPU 部分都在 qiuhao GPU1（A100）上跑。
- 验证在代码提交之前跑。当时的代码与 `1dffc3b` 相同，提交前只给诊断脚本补了一句 docstring。

1. **开关默认（ignore）与 A 逐位等价**。
   - 做法：固定 random / numpy / torch 种子，mini train 前 2 张图经同一个 LSJ mapper，训练态前向 1 次。每个进程只比第一次前向。
     本 worktree 与 A 的 worktree（`bfe1d7e`）各跑一次。
   - A 的 yaml（match、κ 0.05），随机初始化与 E3 最终 checkpoint：
     - **127 项 loss（加权前、加权后）逐位相等**；
     - 参数量同为 65,322,300，参数 sha256 相同（随机初始化 `655a2dce…`，E3 final `79ea05b7…`）；
     - point_nms 统计相同（第 1~9 层合计）：被压的 token 分别为 1690 / 22950 与 1057 / 22950，饿死 0 / 54 个「有点实例 × 层」。
   - E3 的 yaml（none）：两种起点下同样 127 项逐位相等。
2. **negative + κ=1e-9 与 E3 的监督逐位等价**。
   - 做法：新 yaml + `POINT_NMS_KAPPA 1e-9`，什么都压不掉（第 1~9 层合计 0 / 22950 个 token 被压）。
   - 对照两个 none：本 worktree 的 E3 yaml，以及 E3 代码（`mini/group-decoder`@`00a7aaf`）的 E3 yaml。
   - 随机初始化与 E3 final 下，127 项全部逐位相等。
   - 反过来的对照（negative + κ0.05）：
     - 相对 none 恰好差 18 项，即第 1~9 层的 `loss_point_cls` 与 `loss_point_coord`（匹配变了）；
     - 相对 ignore（A）恰好差 9 项，即第 1~9 层的 `loss_point_cls`，坐标项逐位相同；
     - 非点损失项、layer-0 与 interm 的 MLP 点头损失，在三者之间都逐位相同。
3. **negative + κ0.05 打开**。mini train 前 16 张图经 LSJ，训练态；R-50 起点与 E3 final 各测一次（`neg_{r50,e3ckpt}.json`）。
   - 每个 batch 先预热前向一次，再按 ignore / negative / ignore 各前向一次：
     - 两次 ignore 逐位相同，两种取值的存活掩码逐位相同；
     - 除第 1~9 层 `loss_point_cls` 外，118 项逐位相同。
   - **被压 token 的目标为 0、focal 被计入**：每层 negative − ignore 的 `loss_point_cls` 之差，等于用 float64 独立重算的
     「被压 token 以目标 0 算的 focal 之和 × w / 分母」，最大相对误差 1.6e-5（E3 final）/ 5.7e-6（R-50）。
   - 被压比例（逐层计数与 A 的验证 `_smoke-1008-group-nms/verify/k005/` 完全相同）：
     - E3 final：每层 3.8~5.4%，合计 4.84%；`loss_point_cls` 因此每层增大 3.2~6.4%；
     - R-50 起点：每层 9.9~10.5%，合计 10.2%；每层增大 18~25%。
     - A 的文档写的「全部 4.3%」（R-50 为 9.0%）在合计里多算了 κ=0 对照那次前向的 22950 个 token（8812 / (181980 + 22950) = 4.30%），
       逐层数字不受影响。
   - 前向 + 反向（batch 0）：fp32 与 AMP（autocast fp16，不乘 loss scale）下，**735 个可训练参数张量都有梯度，全部有限**；loss 也全部有限。
   - 被压 token 的 logit 梯度：
     - negative、fp32：全部非零且为正，即往低压（E3 final 1056 / 1056，R-50 1686 / 1686）；
     - negative、AMP：R-50 1697 / 1697；E3 final 830 / 1059。其余为 0 应是 fp16 下溢：同一次反向里留下的 token 也只有
       16942 / 21891 非零（fp32 下全部非零），两种取值相同；正式训练的 GradScaler 会先乘 2^16；
     - ignore：全部恰为 0。
   - CPU 合成数据单测（`lossunit_cpu.json`），20 个种子全过：
     - κ=0 与 κ=1e-9 的 negative 都与 none 逐位相同；
     - negative 与 ignore 的坐标损失、统计逐位相同；
     - negative 的 `loss_point_cls` 与独立参照一致。参照的做法：NMS → 只让留下的 token 匹配 → 配上的目标 1、其余（含被压的）目标 0 → 自写 focal；
     - 被压 token 的 logit 梯度在 negative 下全部为正，在 ignore 下全为 0；坐标梯度都为 0。
4. **诊断脚本**（`output_v2/_diag-1009-group-nms-match-neg/`，全 val）：
   - 用本分支的 `point_token_diag.py` 重跑 E3 plain、E3 κ0.05、A，旧键与已有输出逐位相同（分别 71 / 98 / 98 个键）；
   - `auc_supneg` 在什么都没压时等于 AUC_all（E3 plain 0.6989）；
   - 在冒烟 checkpoint 上（前 5 张图），从 run 的 `config.yaml` 读到 `negative`，`auc_rule` 指向 `auc_supneg`。
5. **冒烟**。最终 yaml + `SOLVER.MAX_ITER 40 TEST.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 1000`，
   输出 `output_v2/_smoke-1009-group-nms-match-neg/smoke_neg`。代码 `1dffc3b` + 未提交的 yaml，训练段独占 GPU1。
   - 40 iter 与 iter 19 / 39 两次完整 val 评估都跑通，0 报错（Traceback、CUDA error、OOM 都是 0）。
   - `metrics.json` 两条记录的 127 项 loss 全部有限：total_loss 521.4 → 508.5，`loss_point_cls`（加权后）0.694 → 0.669。
     A 的冒烟同两项为 520.2 → 507.6、0.622 → 0.597。
   - `point_nms/suppressed_frac` 0.081 → 0.071，`point_nms/starved_frac` 0 → 0（A 的冒烟 0.081 → 0.072、0 → 0）。
   - **3.33 s/iter**（38 iter / 2:06）。峰值显存 max_mem 25,888 MB，与 A 的冒烟相同；nvidia-smi 采样的 GPU1 占用峰值 34,024 MiB（含缓存与 CUDA 上下文）。
   - 未训练模型的两次评估指标全为 0（AP 0、matched_F1@8 0、预测点 0），`count_MAE_spike` 为 nan（没有预测点），与 A 的冒烟相同。
   - 落盘 `config.yaml` 与 A run 的 `config.yaml` 逐行 diff：除 `OUTPUT_DIR` 与冒烟覆盖的三项外，只多一行 `POINT_NMS_SUPPRESSED: negative`。

## 预期与判否线（跑之前写死，2026-10-09）

判据线照抄 A 的预注册（[group-nms-match](group-nms-match.md)「预期与判否线」），抄之前逐项核对过 A 的文档，并用 E0 / E3 的 `metrics.json`、
`output_v2/_diag-1008-group-nms/E3_{plain,k005}.json`、`output_v2/_eval-1008-group-decoder-testnms-k005/summary.json` 重算过来源数字。

**读数规则**（用户的硬性规定，同 A）：

- 一律用默认读数：训练内置评估，`POINT_THRESHOLD` = 0.35（配置值），推理侧 NMS κ=0.05（配置值）。
  不做阈值扫描，不用扫描最优档或在 val 上拟合的校准值做对比。
- 判据在同一次评估上读：取后半程（iter ≥ 6250）里 `count_MAE_spike` 最低的那次内置评估，该 iter 的全部指标都从这次读。
- 护栏（`segm/AP`、`spike_recall`）用全程峰值。
- 单种子。

**E0 的读数**（`output_v2/exp-0923-mini-ep100/metrics.json`，按上面的规则重算）：

- 读数那次在 iter 8999：`count_MAE_spike` 2.4648，`matched_F1@8` 0.6227659467，`spike_recall` 0.9511，`segm/AP` 77.40。
- 峰值：`segm/AP` 78.80（final），`spike_recall` 0.9511（iter 8999）。

### 机制（先判）

用 `point_token_diag.py`，本 arm 最终 checkpoint，按 run 自己的配置（匹配前与推理侧都是 κ=0.05），σ = 8。
必须传不带值的 `--sweep` 关掉扫描，且放在最前面（它是 `nargs="*"`，放在后面会吞掉后续参数）：

```
python myscripts/backup/vis_process/point_token_diag.py --sweep --run-dir output_v2/<本 arm 的 run> --out <json>
```

- **M1（匹配稳定）**：第 1→9 层相邻层的内层改判率均值 ≤ **0.0190**（E3 plain 0.02710 × 0.7 = 0.01897）。
  不过 → 判「**机制未启动**」，端到端差异不归因于匹配变稳。报告时并列 E3 plain 0.0271、E3 + κ0.05 NMS 0.0319、A 0.0325。
- **M2（挑点）**：`auc_surv` ≥ **0.7405**（E3 + κ0.05 推理 NMS 的 0.69053 + 0.05）。M2 仍按 A 的口径用 `auc_surv`；
  与本 arm 训练规则一致的 `auc_supneg` 只作报告项（见下）。
- **护栏 G（覆盖）**：NMS 造成的 8 px 覆盖损失 ≤ **2.1486 个百分点**（E3 + 同 κ 推理 NMS 的覆盖损失，全精度 2.14856）。
  A 的预注册与结果表都写作「≤ 2.15」，括号里注明来源是 E3 + κ0.05 的 2.1486。本 arm 按全精度 2.1486 判；落在 2.1486~2.15 之间时另行注明。

### 主判据：读数那次的 `matched_F1@8`（对 E0）

| 条件 | 判定 |
|:--|:--|
| ≥ **0.6428**（E0 + 0.02） | **成立** |
| ≤ **0.6227659**（E0 本身，全精度 0.6227659467） | **判否** |
| 介于两者之间 | **歧义** |

+0.02 高于 mini 实验族的噪声地板 0.015（见 [group-decoder](group-decoder.md)）。

### 次判据：计数（单独判）

读数那次的 `count_MAE_spike` ≤ **2.21** → 计数成立。

照抄 A 的线。A 写的来源是「E0 2.465 × 0.9」，但这个乘积是 2.2184。2.21 比乘积略严，为与 A 可比，不改。
落在 2.21~2.2184 之间时按 2.21 判（不成立），另行注明。

### 护栏

- `segm/AP` 峰值 ≥ **77.80**（E0 峰值 78.80 − 1.0）；
- `spike_recall` 峰值 ≥ **0.9311**（E0 峰值 0.9511 − 0.02）。

### 归因（预注册，不作判据）

- **训练侧有没有贡献**：对照「E3 + 推理 NMS κ=0.05」的读数（iter 9499：计数 2.3118，matched_F1@8 0.63485，同一读数规则）。
  本 arm 读数那次 matched_F1@8 ≥ **0.6548**（+0.02）或 `count_MAE_spike` ≤ **2.081**（× 0.9）→ 记「**训练侧有贡献**」；
  否则记「**收益来自推理 NMS**」。
- **对 A（新增）**：两者只差 `POINT_NMS_SUPPRESSED` 一个开关。
  - 读数那次 matched_F1@8 之差（A-neg − A，A 为 0.6228289 @ iter 9999）的绝对值 ≤ 0.015 → 记「**判负与否无差别**」；
  - 超过 0.015 → 按方向记「判负更好」或「判负更差」。
  - 这个差里含有 run-to-run 噪声，相当于换种子。同代码、同种子的 run 在 GPU 上到 iter 19 就会因反向的非确定性分叉
    （arm B 的记录：A、B 在 iter 2500 前代码路径相同、同种子，`metrics.json` 从 iter 19 起就不同，`loss_bbox` 11.2349 对 11.2392；
    见 `mini/group-nms-prune` 分支的 `experiments/exp-2026-10-08-mini-group-nms-prune.md`）。
  - 本 arm 与 A 从 iter 0 起损失就不同（R-50 起点约 10% 的 token 被压，见验证 3）。

### 新增报告项（不作判据）

用来看判负是否恢复了对近重复的压制。都取自上面那条诊断命令的输出（最终 checkpoint，全 val）。

| 报告项（诊断输出的键） | E3 | E3 + κ0.05 推理 NMS | A | B |
|:--|--:|--:|--:|--:|
| <3 px 重复点比例：阈值 0.35 后、不做 NMS（`dup.ratio`） | 0.39%（42 / 10734） | 同 E3 | 0.69%（75 / 10809） | 1.59%（181 / 11375） |
| AUC_all：全部 30 个 token（`auc.value`） | 0.6989 | 同 E3 | 0.6896 | 0.6661 |
| `auc_supneg`（被推理 NMS 压掉的记负） | 0.6989（什么都没压） | 0.7569 | 0.7526 | — |
| 不带推理 NMS 的计数 / 偏差（`readout_no_nms`） | 2.764 / +2.53 | 同 E3 | 3.024 / +2.85 | 3.784 / +3.66 |

- E3 来自 `output_v2/_diag-1008-group-nms/E3_plain.json`，A 来自 `output_v2/_diag-1009-group-nms/A_match.json`，
  B 来自 `output_v2/_diag-1009-group-nms/B_prune.json`。
- B 没有 `auc_supneg`：B 的 run 配置含本分支代码没有的 prune 键，不能在本分支重跑诊断；而且 B 的「被压」来自剔除掩码，口径也不同。
- `auc_supneg` 的参照由本分支的诊断脚本重跑得到：`output_v2/_diag-1009-group-nms-match-neg/E3_k005.json`、`A_match.json`、`E3_plain.json`。
  同次重跑的旧键与上面三个文件逐位相同。

### 参照（不作判据）

- E3 读数（iter 9499）：`count_MAE_spike` 2.422 / `matched_F1@8` 0.6326 / `spike_recall` 0.9364 / `segm/AP` 77.40；
  峰值 `segm/AP` 78.35、`spike_recall` 0.9657。
- A 读数（iter 9999）：`count_MAE_spike` 2.120 / `matched_F1@8` 0.6228 / `spike_recall` 0.9086 / `segm/AP` 78.01；
  峰值 `segm/AP` 78.15、`spike_recall` 0.9494。
- 穗长直线（不训练，每穗点数按 GT mask 长边一条直线估）：val MAE 1.43、r 0.635。
- 计数一律附偏差 / Pearson r / 去偏移 MAE。用读数那次的逐穗明细（`inference/spikelet_pair_counts.csv`）；训练期只留下末次评估的明细，
  读数那次的明细要对该 iter 的 checkpoint 重跑一次 `--eval_only`（B 的做法），说明差异是偏移还是相关。

### 预期（如实写）

- **机制大概率不过。**
  - M1：构造本身让改判率上升，E3 只加匹配前 NMS 就从 0.0271 到 0.0319，A 实测 0.0325。判负能让簇内排序更稳，
    但要过 0.0190 得比 A 低 40% 以上，不太可能。
  - M2：要比 E3 + 推理 NMS 高 0.05，A 反而低 0.024。判负大概只能把 AUC_surv 拉回 0.69 附近。
  - 护栏 G：不确定。A 的 2.256 只超线 0.11；判负改的是分数，不改位置。
- **主判据大概率歧义。** A 0.6228、B 0.6255、E3 + 推理 NMS 0.6348（各自读数那次）都落在歧义区。判负即使把 A 拉回 E3 + 推理 NMS 的水平，也还在歧义区。
  判否也可能：读数规则按计数挑 iter，A 那次的 F1@8 恰好与 E0 持平。
- **计数次判据不确定。** A 的 2.120 有一部分来自读数规则挑中了一次预测点偏少的评估（`spike_recall` 0.909）；E3 + 推理 NMS 的 2.312 不过线。
  判负减少近重复，主要改善的是「不带推理 NMS」的计数；读数本来就带推理 NMS，近重复大多会被它去掉。
- **对 A 大概率「判负与否无差别」。** A、B 读数那次只差 0.0026，远小于 0.015。
- **最可能看到的变化在报告项上**：重复点比例回到 E3 附近（≤ 0.5%），AUC_all 回到 0.698 附近，不带 NMS 的计数接近 E3 的 2.76。
  这些不进判据。

## 归因限制

- **推理 NMS 混在内置评估里**（同 A）。本 arm 的读数带推理 NMS，与 E0 / E3 的读数不同规则。「E3 + 推理 NMS」一行把只靠推理侧能拿到的部分单独量出来，
  归因只看本 arm 比它多出的部分。
- **对 A 的比较含有 run-to-run 噪声。** 见上文「对 A」。差在 0.015 内不能说明两种做法真的相同，只能说这个设计分辨不出。
- **起步阶段的 token 塌缩。** A 与 E3（同种子）在起步阶段都有 token 塌缩：iter 39~179 每个实例的 30 个 token 落在同一个半径内
  （被压比例 29/30、饿死 100%），到 iter 319 前后才散开。这一段三者的存在性监督不同：
  - E3：Ni 个正样本、30 − Ni 个负样本；
  - A：每个实例只有留下的 1 个 token 拿损失；
  - 本 arm：最多 1 个正样本，被压的 29 个全部判负，监督明显偏负。
  - 本 arm 的早期轨迹可能因此与 A、E3 都不同。跑完一并报告 `point_nms/*` 的轨迹与前几次评估的预测点数。
- **饿死不补回**（同 A）。留下的 token 少于 Ni 时，多出的 GT 小穗本层没有配对；本 arm 里被压的 token 还要判负，所以饿死时的监督比 A 更偏负。
- **`auc_supneg` 天然偏高**（见「修法」的诊断一段），只与别的 run 的 `auc_supneg` 互比。
- **结论只对分组 decoder 成立。** 单种子。

## 成本

- 12500 iter（100 epoch）。
  - 冒烟独占 A100 时 3.33 s/iter，约 11.6 h；A 的正式 run 独占时约 3.30 s/iter。
  - 与别的任务同卡时按 A 的实测（同卡约 4.55 s/iter）估，约 15.8 h。
  - 另加 25 次评估，每次约 2.5~3 min，共约 1 h。
- 峰值显存约 26~27 GB（冒烟 max_mem 25.9 GB，A 的正式 run 26.9 GB），只能用 qiuhao 的 A100（GPU1）。
- 每次存 checkpoint 785 MB × 25 个，约 20 GB 磁盘。

## 启动信息

- **启动提交**：`5673db5`（启动时 worktree 干净，`launch.log` 首行记有完整哈希）。
- **启动时间**：2026-10-09 15:58:03。
- **输出**：`output_v2/exp-1009-mini-group-nms-match-neg`，日志 `output_v2/exp-1009-mini-group-nms-match-neg.launch.log`。
- **GPU 与排队**：
  - 只用 qiuhao 的 A100（GPU1），`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`。
  - 15:41 起排队：当时别的用户的 vLLM 占了约 60 GB，只剩 20 GB。15:58 空出到 75 GB 后自动启动。
  - 启动后 nvidia-smi 上本 run 约 27 GB。
- **启动方式**：排队脚本 `output_v2/_queue/launch-group-nms-match-neg.sh`，日志为同目录下的 `launch-group-nms-match-neg.queue.log`。
  - 脚本先轮询 GPU1 空闲显存 ≥ 34000 MiB，再核对 HEAD = `5673db5` 且 worktree 干净，然后执行：

    ```
    PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      /public/home/zncui/anaconda3/envs/MaskDINO-310/bin/python train_net.py --num-gpus 1 \
      --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_mini_group_nms_match_neg.yaml \
      OUTPUT_DIR /public/home/zncui/pycharm_project/maskdino-wheat50/output_v2/exp-1009-mini-group-nms-match-neg
    ```
  - `expandable_segments` 只减少显存碎片，不改数值。arm B 也用了它，arm A 没用。
- **首个日志点**：iter 19，eta 12:37:50，0 报错。

## 结论（2026-10-10）

**歧义：主判据比 E0 高 0.0089；机制未启动（M1、M2 未过），护栏 G 未过；计数次判据不成立（2.232，差 0.022）；
segm / recall 护栏过；归因按预注册记「收益来自推理 NMS」；对 A 记「判负与否无差别」。**
按上面「预期与判否线」判，读数一律是默认读数（阈值 0.35、推理 NMS κ=0.05，未做阈值扫描）。
run：`output_v2/exp-1009-mini-group-nms-match-neg`（12500/12500 iter · 25/25 评估 · 0 报错）。
详见 [exp-2026-10-09-mini-group-nms-match-neg](../experiments/exp-2026-10-09-mini-group-nms-match-neg.md)。

| 判据 | 要求 | 实测 | |
|:--|:--|:--|:--:|
| M1 跨层改判率均值 | ≤ 0.0190 | 0.0333（A 0.0325，E3 + κ0.05 NMS 0.0319，E3 plain 0.0271） | ❌ |
| M2 AUC_surv | ≥ 0.7405 | 0.6569（A 0.6663，E3 + κ0.05 NMS 0.6905） | ❌ |
| 护栏 G 覆盖损失 | ≤ 2.1486 个百分点 | 2.243（A 2.256，E3 + κ0.05 NMS 2.149） | ❌ |
| 主判据：读数那次 `matched_F1@8`（对 E0） | ≥ 0.6428 成立；≤ 0.6227659 判否 | 0.6316923（iter 9499；比 E0 高 0.0089） | 歧义 |
| 次判据：读数那次 `count_MAE_spike` | ≤ 2.21 | 2.232 | ❌ |
| 护栏 `segm/AP` 全程峰值 | ≥ 77.80 | 78.47（@9999） | ✅ |
| 护栏 `spike_recall` 全程峰值 | ≥ 0.9311 | 0.9560（@7499） | ✅ |
| 归因：对「E3 + 推理 NMS κ0.05」（iter 9499：0.6348 / 2.312） | F1@8 ≥ 0.6548 或计数 ≤ 2.081 → 训练侧有贡献 | 0.6317 / 2.232 | 收益来自推理 NMS |
| 对 A（读数那次 F1@8 之差 ≤ 0.015 → 无差别） | — | 0.6317 − 0.6228 = +0.0089 | 判负与否无差别 |

诊断在 qiuhao GPU1 上跑（`output_v2/_diag-1009-group-nms-match-neg/Aneg_match_neg.json`），`readout_model` 的 matched_F1@8 与计数与训练日志末值逐位相同。
读数那次的逐穗明细来自 `model_0009499.pth` 的 `--eval_only` 重跑（`output_v2/_eval-1010-group-nms-match-neg-readout/9499/`），40 项指标与内置评估逐位相同。

报告项（不作判据）：

| | A-neg | E3 | E3 + κ0.05 推理 NMS | A | B |
|:--|--:|--:|--:|--:|--:|
| <3 px 重复点（阈值后、不带 NMS） | 0.41% | 0.39% | 同 E3 | 0.69% | 1.59% |
| AUC_all | 0.6921 | 0.6989 | 同 E3 | 0.6896 | 0.6661 |
| `auc_supneg` | 0.7591 | 0.6989（什么都没压） | 0.7569 | 0.7526 | — |
| 不带推理 NMS 的计数 / 偏差 | 2.715 / +2.54 | 2.764 / +2.53 | 同 E3 | 3.024 / +2.85 | 3.784 / +3.66 |

读法：

- **判负把近重复的分数压了回去，但只在分数上。** 重复点与不带 NMS 的计数回到 E3 的水平，`auc_supneg` 最高。
  但 token 在位置上的近重复比 A 还多（训练中被压约 7.8% 对 5.9%），AUC_all 只比 A 高 0.0025，留下的 token 之间的排序（AUC_surv）没有变好。
- **端到端回到「E3 + 推理 NMS」的水平，没有超过。** 后半程 F1@8 均值 0.6353 对 0.6357，末值 0.6415 对 0.6416。
  A 相对它的退步（后半程均值 −0.008）消失了。
- **对 A 方向一致、幅度在噪声内。** 后半程 13 次里 12 次 F1@8 高于 A，均值 +0.008；但这是同一个 run 内彼此相关的评估、单种子，按预注册记无差别。
- **计数的相关性变好**（r 0.593~0.599，A 0.563，E3 0.552），与 B 相同；计数 MAE 仍由 +1.9~+2.4 的偏差主导。
- **起步更慢，没有后果。** 塌缩窗口（iter 39~179）与 A、E3 相同，但到 iter 439 才散开（A 319），iter 999 的评估只出 28 个点，iter 1499 起追平。
