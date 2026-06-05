# 实施计划：逐实例小穗点回归头（分步可验证）

> 本文件是对 `memory-bank/小穗点回归方案.md`（总体设计，**唯一真理来源**）的分步落地说明。
> 使用方式：每次只取**一个步骤**，进入 plan mode 与 Opus 讨论后再实现。
> 约定：
> - 本文件**不含任何代码**，只给"做什么 / 改哪里 / 怎么验证"的指令。具体代码由实现阶段产出。
> - 每一步都必须先通过"验证测试"才能进入下一步；验证脚本统一放在 `myscripts/_devtests/`（临时，可随时删除）。
> - 所有改动遵循 `小穗点回归方案.md` 的坐标系与命名约定（`gt_points`、`pred_point_coords`、
>   `pred_point_logits`、`pred_spikelet_points`、`pred_spikelet_counts`、`loss_point_coord`、`loss_point_cls`）。
> - 坐标系铁律：GT 框与 GT 点都用同一 `(w,h)=pre-pad image_size` 做图像归一化；点回归用"实例框相对归一化"。
> - 每完成一个里程碑（见末尾），更新 `memory-bank/architecture.md` 与 `progress.md`。

---

## 阶段 0：数据核查（不改动任何源码）

### Step 0.1 — 统计单穗最大小穗数，确认 NUM_POINTS=50 足够
- **目标**：核对 `NUM_POINTS=50` 是否覆盖数据集中单个 `wheat_2` 实例的最大小穗数。
- **改哪里**：仅新增临时脚本 `myscripts/_devtests/check_points_stats.py`，不动任何训练代码。
- **做什么**：读取训练/验证两个 `instances_*2017.json`，对所有 `category_id==2` 的标注取 `len(points)`，
  输出最大值、均值、95/99 分位数、以及"超过 50 的实例数量与占比"。
- **验证测试**：运行脚本，确认输出中**最大值 ≤ 50**。
  - 若最大值 > 50：在本文件此处记录实际最大值，并决定（a）调大 `NUM_POINTS`，或（b）接受裁剪并记录被裁剪占比。
    该决定将影响 Step 1.1 的默认值，必须先定下来再继续。
- **完成判据**：统计结果写入本步骤下方"结果记录"，并明确最终采用的 `NUM_POINTS` 值。
- **结果记录**：（实现时填写）最大小穗数 = ___，最终 NUM_POINTS = ___。

---

## 阶段 1：配置项（先让配置可用，后续步骤才能读取）

### Step 1.1 — 在 config.py 注册所有新增配置项
- **目标**：新增点头相关的全部 `cfg.MODEL.MaskDINO.*` 配置，给定默认值（与设计文档一致）。
- **改哪里**：`maskdino/config.py` 的 `add_maskdino_config`，MaskDINO 段内。
- **做什么**：新增以下键（默认值见设计文档"改动清单 5"）：
  `NUM_POINTS`、`POINT_COORD_WEIGHT`、`POINT_CLASS_WEIGHT`、`COST_POINT_COORD_WEIGHT`、
  `COST_POINT_CLASS_WEIGHT`、`TEST.POINT_THRESHOLD`。`NUM_POINTS` 用 Step 0.1 确定的值。
- **验证测试**：临时脚本 `myscripts/_devtests/check_config.py` 调用 `get_cfg()` + `add_maskdino_config(cfg)`，
  断言这 6 个键存在且类型/默认值正确；再用一个真实训练 config 文件 merge 一遍，确认不报"non-existent key"。
- **完成判据**：脚本无断言失败、无 merge 报错。
- **注意**：本步**只加配置、不读配置**；任何消费这些配置的代码在后续步骤完成。

---

## 阶段 2：数据加载（让一个实例带上 gt_points）

### Step 2.1 — 在 dataset mapper 中处理 points 并挂为 instances.gt_points
- **目标**：让 LSJ mapper 在增广后对每个标注的 `points` 应用同样的几何变换、过滤越界点，
  最终把对齐后的点挂到 `instances.gt_points`。
- **改哪里**：`maskdino/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py`。
- **做什么**（参照设计文档"改动清单 1"，**严禁引入任何密度图逻辑**）：
  - 训练与测试分支都在 `transform_instance_annotations` 之后，对每个 anno 的 `points` 调
    `transforms.apply_coords` 并过滤落到图像外的点。
  - 在 `filter_empty_instances` 之后，按保留下来的实例顺序，把点列表挂为
    `instances.gt_points = list[Tensor[Ni,2]]`，空实例用 `torch.zeros((0,2))` 占位。
  - 不要设置任何 `gt_density_*` 字段。
- **验证测试**：临时脚本 `myscripts/_devtests/check_mapper.py`，对 `wheat_cat2_coco_train` 取若干样本走 mapper，断言：
  1. `len(instances.gt_points) == len(instances)`（与其它 gt_* 字段一一对应）。
  2. 每个 `wheat`(类0) 实例的点张量形状为 `(0,2)`。
  3. 每个 `wheat_2`(类1) 实例的点数与原始 COCO 标注的 `len(points)` 一致（允许因越界过滤而 ≤，需打印被过滤数量并人工确认合理）。
  4. 所有点坐标都落在增广后图像尺寸范围内（`0 <= x <= W`，`0 <= y <= H`）。
- **完成判据**：4 项断言全部通过，被过滤点数在合理范围（通常为 0）。

---

## 阶段 3：构造训练 targets（让 GT 点进入归一化坐标系）

### Step 3.1 — 在 prepare_targets / prepare_targets_detr 中加入归一化 points
- **目标**：把 `gt_points` 从像素坐标转成"图像归一化 [0,1]"，与 `boxes` 用**同一** `(w,h)` 归一化，写入 target dict。
- **改哪里**：`maskdino/maskdino.py` 的 `prepare_targets` 与 `prepare_targets_detr`。
- **做什么**：每个 target dict 增加键 `"points"`，值为 `list[Tensor[Ni,2]]`，由 `gt_points` 除以归一化用的
  `(w,h)` 得到（与该函数内 `boxes` 归一化所用的 `(w,h)` 必须是同一个变量，禁止用 padded size）。
- **验证测试**：临时脚本 `myscripts/_devtests/check_targets.py`，构造一个真实 batch，调用模型的
  `prepare_targets`（可直接实例化模型或抽取该方法），断言：
  1. 每个 target dict 含 `"points"` 键，且 `len(points) == len(labels) == boxes.shape[0]`。
  2. 所有 `wheat_2` 实例的归一化点坐标都在 `[0,1]` 内（允许极小越界，打印越界比例确认 ≈0）。
  3. 抽查一个实例：用其归一化框 `cxcywh` 反推像素框，再把归一化点反推像素，与 Step 2.1 中
     `gt_points`（像素）数值一致（误差 < 1px）——确认点与框坐标系一致。
- **完成判据**：3 项断言通过；坐标系一致性抽查数值吻合。

---

## 阶段 4：解码器输出点预测（先有 shape，不接损失）

### Step 4.1 — 在 decoder 中新增两个点头模块并接入 configurable
- **目标**：实例化 `point_coord_embed`（MLP→`num_points*2`）与 `point_class_embed`（Linear→`num_points`），
  并让 `num_points` 从 config 流入。
- **改哪里**：`maskdino/modeling/transformer_decoder/maskdino_decoder.py` 的 `__init__`、`from_config`、
  `@configurable` 签名（参照设计文档"改动清单 2"第一条）。
- **做什么**：新增 `self.num_points`、`self.point_coord_embed`、`self.point_class_embed`；
  `from_config` 里 `ret["num_points"] = cfg.MODEL.MaskDINO.NUM_POINTS`；签名补 `num_points`。
- **验证测试**：临时脚本断言：用真实 config 构建 decoder（或整模型）后，
  `hasattr(decoder, 'point_coord_embed')` 与 `point_class_embed` 为真，且 `decoder.num_points == 配置值`；
  给一个 `(B, Q, hidden_dim)` 的随机张量过两个头，输出 reshape 后形状分别为 `(B,Q,P,2)`、`(B,Q,P,1)`（暂不 sigmoid）。
- **完成判据**：模块存在、`num_points` 正确、手工过头 shape 正确。本步**不修改 forward**，整模型前向行为不变。

### Step 4.2 — forward_prediction_heads 输出点预测并贯通所有调用点
- **目标**：让 `forward_prediction_heads` 多返回 `(pt_coord[sigmoid], pt_logit)`，并在三处调用点收集到
  `predictions_point_coord / predictions_point_logit`，与 `predictions_mask` 完全并行。
- **改哪里**：同文件 `forward_prediction_heads` 及其全部调用点（设计文档标注约 419/467/518 行）。
- **做什么**：返回值新增两项（coord 经 `sigmoid`，logit 不过激活）；每个调用点接收并 append 到新列表，
  处理方式逐行对照 `predictions_mask`。
- **验证测试**：临时脚本跑一次**训练前向**（`model.train()` + 真实 batch），断言：
  1. 不抛异常（说明所有调用点签名匹配）。
  2. 在 decoder 内部（可临时打印或返回）`predictions_point_coord` 列表长度 == `predictions_mask` 列表长度（层数一致）。
  3. 每层点坐标 `(B,Q,P,2)` 数值在 `[0,1]`、点 logit `(B,Q,P,1)` 为实数。
- **完成判据**：3 项通过；mask 相关原有输出 shape 不变（回归对照）。

### Step 4.3 — 把点预测纳入 out dict / aux loss / dn 切分
- **目标**：让最终输出字典、深监督 aux、DN 后处理都正确携带/切分点预测。
- **改哪里**：同文件：最终 out dict（约 328/566 区域）、`_set_aux_loss`、`dn_post_process`（约 315-332 行）、
  以及 two-stage 的 `interm_outputs` 组装处。
- **做什么**（参照设计文档"改动清单 2"第三条 + "需要注意的坑"）：
  - 主 out dict 加 `pred_point_coords`、`pred_point_logits`（取 `[-1]` 层）。
  - `_set_aux_loss` 的每个 aux dict 并列加入这两项。
  - `interm_outputs` 同样补这两项。
  - `dn_post_process` 对点预测做与 mask **完全相同**的 `[:, :, :pad_size]`/`[:, :, pad_size:]` 切分，
    **只把非 dn 部分写回**，dn 部分丢弃（不进损失）。
- **验证测试**：临时脚本跑训练前向后检查返回的 `outputs` 字典：
  1. 顶层含 `pred_point_coords (B,300,50,2)`、`pred_point_logits (B,300,50,1)`。
  2. `outputs['aux_outputs']` 每个元素都含这两键且 shape 同上。
  3. 若启用 two-stage，`interm_outputs` 也含这两键。
  4. 开启 DN 时（用启用 DN 的 config），写回主输出的 query 维度 == `num_object_queries`（已剔除 dn pad）；
     可通过对比关闭 DN 时的 query 维度一致来确认切分正确。
- **完成判据**：4 项通过。至此模型能产出点预测但尚未计算点损失（损失里还没注册 `points`）。

---

## 阶段 5：损失（外层沿用现有匹配，内层新写二次匈牙利）

### Step 5.1 — 在 criterion 内实现内层逐实例匈牙利匹配（独立可测）
- **目标**：先把"单实例内 P 个预测点 ↔ Ni 个 GT 小穗"的匹配函数写出来并单测，**不接入主损失**。
- **改哪里**：`maskdino/modeling/criterion.py` 内新增一个 `@torch.no_grad()` 私有方法
  （cost = `cost_coord * L1_cdist(pred_rel, gt_rel) + cost_class * focal_cost(pred_logit)`，
  参照 `CLTR/Networks/CDETR/matcher.py:66-80`）。
- **做什么**：实现"给定单实例 `pred_rel[P,2]`、`pred_logit[P,1]`、`gt_rel[Ni,2]`，返回匹配索引对"的逻辑；
  正确处理 `Ni==0`（返回空匹配）与 `Ni>0` 两个分支，避免空张量报错。
- **验证测试**：纯单元测试 `myscripts/_devtests/check_inner_matcher.py`：
  1. 构造 `gt_rel` = 某 P 中 Ni 个点的精确副本，断言匹配把这些预测点一一对到对应 GT（cost≈0 的对角匹配）。
  2. `Ni==0` 时返回的两个索引张量都为空、不报错。
  3. `Ni > P` 的极端输入下不崩溃（即便设计上不应出现，也要安全返回 `min(P,Ni)` 个匹配）。
- **完成判据**：3 项通过。该函数此时尚未被 `loss_points` 调用。

### Step 5.2 — 实现 loss_points 并注册进 criterion（含 weight_dict 与 DN 排除）
- **目标**：实现 `loss_points`，注册到 `loss_map`、`losses` 列表、`weight_dict`，且**确保 DN 不含点损失**。
- **改哪里**：`maskdino/modeling/criterion.py`（`__init__` 增参、`loss_points`、`loss_map`）；
  `maskdino/maskdino.py` 的 `from_config`（`weight_dict`、`losses`、把 `num_points`/cost 权重传入 criterion）。
- **做什么**（参照设计文档"改动清单 3/4"）：
  - `loss_points`：用外层匹配索引取出每个匹配 query 的点预测；用匹配到的 **GT 框** 把该实例 GT 小穗
    （图像归一化）转成框相对坐标并 clip 到 [0,1]；调 Step 5.1 内层匹配；累积匹配点 L1 坐标损失 +
    全部 P 点存在性 focal/BCE 损失；坐标损失按总匹配小穗数归一化、分类损失按总小穗数归一化。
  - `__init__` 保存 `num_points / point_cost_coord / point_cost_class`。
  - `loss_map` 注册 `"points": self.loss_points`；`losses` 列表追加 `"points"`。
  - `weight_dict`：在 two_stage/dn/aux 复制**之前**加入 `loss_point_coord/loss_point_cls`，使 `_i`/`_interm` 自动带上；
    但**确保 `_dn` 复制不含点项**（在 dn 复制后再加点项，或 dn 复制时显式排除点 key）。
- **验证测试**：临时脚本跑训练前向 + `loss = sum(...)` + `loss.backward()`：
  1. 返回 loss 字典含 `loss_point_coord`、`loss_point_cls`，以及对应的 `_0../_interm` aux 项。
  2. **不含**任何带 `_dn` 后缀的点损失键（断言遍历键名无 `point` 且含 `dn` 的组合）。
  3. `loss_point_*` 为有限正数（非 NaN/Inf）。
  4. `loss.backward()` 不报错，且 `point_coord_embed`/`point_class_embed` 参数的 `.grad` 非全零。
  5. 用一个 batch 内只含 `wheat`（无 wheat_2）的样本验证：坐标损失为 0、分类损失正常（全负样本），不报空张量错。
- **完成判据**：5 项通过。

---

## 阶段 6：训练前向冒烟测试（集成验证）

### Step 6.1 — 完整 model(batched_inputs) 训练前向 + 反向冒烟
- **目标**：在不依赖临时拆解的前提下，用真实训练入口路径验证端到端可训练。
- **改哪里**：无源码改动，仅 `myscripts/_devtests/check_train_forward.py`。
- **做什么**：用真实 config + 真实 `DatasetMapper` 取 2~4 张图组 batch，`model.train()` 跑一次前向得到 loss 字典，
  求和反向。
- **验证测试**：
  1. `outputs/loss` 字典同时含分割原有损失与 `loss_point_coord/loss_point_cls`（及 aux）。
  2. 所有 loss 值有限。
  3. 反向无 in-place / shape / device 报错。
  4. 连续跑 2 个 batch 不出现显存异常增长（粗看，排除明显泄漏）。
- **完成判据**：4 项通过。这是进入推理实现前的关键门槛。

---

## 阶段 7：推理（产出每实例小穗点与计数）

### Step 7.1 — forward 推理分支取出点预测并传入 instance_inference
- **目标**：推理时把 `pred_point_coords/pred_point_logits` 随 `mask_cls_results` 等按图遍历，传进
  `instance_inference`。
- **改哪里**：`maskdino/maskdino.py` 推理分支（设计文档约 279-341 行）与 `instance_inference` 签名。
- **做什么**：给 `instance_inference` 增加入参 `point_coords/point_logits`（形状 `[num_queries,P,2/1]`），
  本步只负责**正确传参**，内部使用逻辑在 Step 7.2。
- **验证测试**：`myscripts/_devtests/check_infer_wiring.py` 跑 `model.eval()` 前向（`INSTANCE_ON=True`），
  断言不抛异常，且 `instance_inference` 内部收到的两参 shape 为 `[num_queries,P,2]`/`[num_queries,P,1]`
  （可临时打印/断言）。
- **完成判据**：前向通、参数 shape 正确。

### Step 7.2 — instance_inference 内：框相对点还原为像素并按类输出
- **目标**：对选中的 topk query，把框相对点用预测框还原到图像像素，按阈值过滤，仅 `wheat_2` 实例输出点，
  挂为 `pred_spikelet_points` 与 `pred_spikelet_counts`。
- **改哪里**：`maskdino/maskdino.py` 的 `instance_inference`（参照设计文档"改动清单 3"推理部分）；
  从 config 读取 `TEST.POINT_THRESHOLD` 存为 `self.point_threshold`（在 `MaskDINO.from_config` 传入）。
- **做什么**：
  - 用已 `// num_classes` 还原的 `topk_indices` 取 `point_coords/point_logits`。
  - `pt_score = sigmoid(logit)`，`keep_pt = pt_score > self.point_threshold`。
  - 用该实例 `pred_boxes`（xyxy 像素）把框相对 `rel` 还原：`xy = [x0,y0] + rel*[w,h]`。
  - 仅当 `pred_classes == 1`(wheat_2) 输出点，否则置空；挂 `result.pred_spikelet_points`(list[Tensor[Ki,2]])
    与 `result.pred_spikelet_counts`(Tensor[K])。
- **验证测试**：`myscripts/_devtests/check_infer_points.py` 对含 wheat_2 的真实图推理，断言：
  1. 返回的 `Instances` 含 `pred_spikelet_points`、`pred_spikelet_counts`，长度 == 实例数。
  2. 所有 `wheat`(类0) 实例的 count == 0、点列表为空。
  3. 每个 wheat_2 实例的所有还原点都落在其 `pred_boxes` 内部（允许极小越界，打印越界比例）。
  4. `pred_spikelet_counts` 与各实例点列表长度逐一相等。
- **完成判据**：4 项通过。（此时模型未训练，点位置可以是乱的，只验证管线与约束正确性。）

---

## 阶段 8：学习能力验证（过拟合小集）

### Step 8.1 — 5~10 张含 wheat_2 图过拟合
- **目标**：证明损失能驱动点头真正学习。
- **改哪里**：无源码改动；用现有训练入口 + 一个小子集 config / 数据子集 + 少量 iter。
- **做什么**：取 5~10 张含 wheat_2 的图，关掉/调弱增广，跑数百~数千 iter 过拟合。
- **验证测试**：
  1. TensorBoard/日志中 `loss_point_coord` 与 `loss_point_cls` 明显下降并趋于很小。
  2. 用过拟合后的权重对训练图自身推理：每个 wheat_2 实例预测点数 ≈ GT 小穗数（计数 MAE 很小）。
  3. 所有 `wheat` 实例预测点数严格 == 0。
  4. 可视化叠加（见 Step 10.1 工具）点位与 GT 大致重合。
- **完成判据**：损失下降 + 计数接近 + wheat 零点 + 可视化合理。**这是方案可行性的核心判据。**

---

## 阶段 9：端到端训练配置与启动

### Step 9.1 — 更新 Wheat 训练 config 并启动正式训练
- **目标**：把 `NUM_OBJECT_QUERIES` 提到 300、确认 `INSTANCE_ON`、按需覆盖点损失权重，跑通完整训练。
- **改哪里**：`configs/coco/instance-segmentation/Wheat-COCO-InstanceSegmentation.yaml`（参照设计文档"改动清单 6"）。
- **做什么**：`MODEL.MaskDINO.NUM_OBJECT_QUERIES: 300`；确认 `MODEL.MaskDINO.TEST.INSTANCE_ON: True`；
  可选覆盖 `POINT_*_WEIGHT`。
- **验证测试**：用现有训练命令启动训练，确认：
  1. 训练正常启动、前若干 iter 无报错。
  2. 日志含 `loss_point_coord/loss_point_cls` 且随训练下降。
  3. 周期性 eval 不因点头报错；分割指标（AP）未明显退化（与改造前 baseline 对比）。
- **完成判据**：训练稳定、点损失收敛、分割指标不崩。

---

## 阶段 10：评估与可视化

### Step 10.1 — 小穗点可视化与计数指标
- **目标**：人工核对小穗定位/计数，并给出量化产量评估指标。
- **改哪里**：`myscripts/` 下新增/复用可视化脚本（非训练管线）。
- **做什么**：推理后把 `pred_spikelet_points` 叠加到原图；统计每穗 count 的 MAE/MSE（vs GT 小穗数）。
- **验证测试**：
  1. 抽查若干图，肉眼确认点落在穗内、数量合理、wheat 上无点。
  2. 输出验证集整体计数 MAE/MSE 数值，记录为基线指标。
- **完成判据**：可视化合理 + 指标数值产出并记录。（后续可把计数指标封装进 evaluator，属增强项。）

---

## 里程碑与文档更新

- **里程碑 A**（完成阶段 2~3）：数据侧点流水线打通 → 更新 `architecture.md` 数据部分、`progress.md`。
- **里程碑 B**（完成阶段 4~6）：模型可带点损失训练 → 更新 `architecture.md` 模型/损失部分。
- **里程碑 C**（完成阶段 7~8）：推理产出点 + 过拟合验证通过 → 更新 `progress.md` 记录可行性结论。
- **里程碑 D**（完成阶段 9~10）：端到端训练 + 指标 → 更新两份文档并归档基线指标。

## 全局回归红线（每个阶段都要顺手确认）
- 不得引入任何高斯密度图相关字段/逻辑（该方案已废弃）。
- 分割原有功能（cls/box/mask 的 shape、loss、AP）在每个阶段后不得退化。
- DN 路径永远不参与点损失；`_dn` weight_dict 永远不含点 key。
- 点与框始终用同一 `(w,h)=pre-pad image_size` 归一化。
