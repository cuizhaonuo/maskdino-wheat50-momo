# full/sa-wsg：统一 CA 链路下的 SA 执行顺序排列（ww → ss → group）

状态：**预注册**（判定线在跑之前写死）｜ 提出 2026-10-26 ｜ 分支 `full/sa-wsg`（从 `full/sa-order`@`b36f46b` 切出）｜ 配置 `wheat_joint_single_stage_mini_sa_wsg.yaml`（`_BASE_` 为 `wheat_joint_single_stage_mini_sa_order.yaml`）｜ 输出 `output_v2/exp-1026-mini-sa-wsg`

## 动机

分组解码器原本把 token 更新拆成"实例 token 一支 / 小穗 token 一支"两套 CA+FFN（① 实例内 SA → ② wheat 的 SA+CA+FFN → ③ spike 的 SA+CA+FFN）。改成四个组件——SA_group / SA_ww / SA_ss / 统一 CA——之后，三个 SA 的相对顺序成为一个待定变量：顺序决定"跨实例信息"与"实例内信息"的注入先后，可能影响收敛与最终点位质量。本臂是 6 种排列之一，mini 集筛选用。

## 改动（唯一变量 = SA 顺序）

相对 `full/sa-order`：只改 `POINT_SA_ORDER`。其余（A CoralHead / B KMO / C Group Decoder 三开关、NMS κ=0.05、top-n_k 语义、lr、bs、种子）完全一致。

统一 CA 的语义：N×(1+K) 个 token 一次 `MSDeformAttn`，wheat 用 4D 框参考点、spike 用 4D 参考点（x,y 取点坐标，w,h 取所属实例框），共享权重；FFN 对全部 token 共享。

## 训练口径

mini 500 图（`pts_mini`），bs4，50 epoch = 6250 iter，EVAL/CHECKPOINT_PERIOD=500，与既往 mini 臂一致。

## 判定线（跑之前写死，原文）

主判据（帕累托筛选）：在 val 曲线峰值点上，按 `matched_F1@8`（高优）与 `count_MAE_spike`（低优）构建帕累托前沿；前沿内按 `matched_F1@8` 降序取前 2 名进完整集。

护栏（任一不过则该臂淘汰，不参与排名）：
- (a) `segm/AP` ≥ 6 臂中位数 − 1.0；
- (b) `spike_precision` ≥ 6 臂中位数 − 0.02；
- (c) 训练无 NaN/发散。

报告口径：同时给峰值与末次值；同时报 8/16/32px 尺度。

注意：帕累托前沿与中位数均在 6 臂全部完成后统一计算（批量判定，不允许单臂先看后调）。
