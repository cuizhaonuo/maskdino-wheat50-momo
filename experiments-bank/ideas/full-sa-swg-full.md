# full/sa-swg（完整集）：统一 CA 链路，SA 顺序 ss → ww → group

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-30 ｜ 分支 `full/sa-swg` ｜ 配置 `wheat_joint_single_stage_full_sa_swg.yaml`（_BASE_ = `wheat_joint_single_stage_full_combo.yaml`）｜ 输出 `output_v2/exp-1030-full-sa-swg`

## 动机

mini 6 臂顺序筛选（2026-10-28~30，500 图 / 12500 iter）结果：6 臂全部通过护栏，F1@8 落在 0.6026~0.6159（带宽 0.013），帕累托前沿为 gsw（0.6159 / MAE 1.423）与 swg（0.6115 / MAE 1.379）。本臂是前 2 名之一，用于在完整集上验证「统一 CA + 该 SA 顺序」相对 combo（原两路 CA 架构）的真实收益。

## 改动（相对 full/combo，仅两处）

1. `POINT_UNIFIED_CA: True`：每层改为 三个 SA（按 `POINT_SA_ORDER` 依次更新同一对缓冲）→ N×(1+K) 一次可变形 CA → 全部 token 共享 FFN；wheat 用 4D 框参考点、spike 用 4D 参考点（x,y = 点坐标，w,h = 所属实例框）。
2. `POINT_SA_ORDER: [ss, ww, group]`。

其余（A CoralHead / B KMO / C Group Decoder 三开关、NMS κ=0.05、top-n_k、lr、bs、种子）与 full/combo 完全一致。

## 训练口径

完整集 2292 图，bs4，50 epoch = 26000 iter，EVAL_PERIOD=1000。

**偏差记录**：mini 筛选期因父配置继承链实际跑了 12500 iter（100 ep），与 mini 预注册的 6250（50 ep）不一致；6 臂一致，臂间比较有效。本 full run 按 26000 iter 执行。

## 判定线（跑之前写死，原文）

参照：full/combo 的 guardrails 全过代表点 = iter 23999，`count_MAE_spike`=1.360、`matched_F1@8`=0.6144、segm AP=82.05、bbox AP=80.89、prec=0.7592。

护栏（同 combo 口径，任一不过则该点不参与判定）：segm/AP ≥ 81.47、bbox/AP ≥ 80.38、spike_precision ≥ 0.753。

主判据（相对 combo，须同一点位同时满足）：
- (i) **更优**：`matched_F1@8` ≥ 0.629 且 `count_MAE_spike` ≤ 1.36；
- (ii) **无显著差异**：|F1 − 0.614| ≤ 0.015 且 `count_MAE_spike` ≤ 1.41；
- (iii) 其余一律判 **劣于 combo**（含 F1 < 0.599 或 MAE > 1.41）。

次判据（两臂之间，gsw vs swg）：取 `matched_F1@8` 更高者为最终模型；|ΔF1| < 0.005 时取 `count_MAE_spike` 更低者。

报告口径：峰值与末次值同时给；同时报 8/16/32px 尺度与机制量（`point_nms/suppressed_frac`、`starved_frac`）。
