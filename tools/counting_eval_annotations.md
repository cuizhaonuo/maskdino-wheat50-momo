# 整图输入、固定预测 mask 的小穗计数评估

所有模型输入原始 `val2017` 图像。JSON 只规定评估区域、实例匹配和真实数量，
不生成裁剪图，也不对输入背景置黑。它是扩展的计数评估协议，不能直接当作
COCOEvaluator 的真实实例标注，不能用于训练。

**mask 归属规则仅用于没有实例归属输出的其他计数模型。full/combo 保留原有的
实例点归属、计数和评估结果，不需要重新按 mask 过滤或重新计算基线。**

当前导出文件：

`/public/home/zncui/datasets/wheat50/coco_2cat_with_points/annotations/counting_eval_val2017_full_combo_final.json`

来源是 `output_v2/exp-1009-full-combo/inference/coco_instances_results.json`
中保存的最后一次评估预测。日志与 `last_checkpoint` 指向最后一轮；本次没有重新加载
`model_final.pth` 推理，也没有选择最佳 checkpoint。JSON 保存了预测文件、原始真值、
配置及参考 CSV 的 SHA-256，`checkpoint_label_unverified` 只是来源说明。

## 固定协议

- COCO 原始类别 `category_id=2`（`wheat_2`），注意模型连续类别 ID 是 1。
- 保留实例分数 `>=0.4` 的预测。
- 在每张图内，使用原始预测 mask 与 GT mask 的 IoU 做匈牙利分配，再接受 `IoU>=0.5`
  的配对，与现有 `SpikeletPointEvaluator` 的顺序一致。
- 所有通过分数过滤的预测参与像素归属，包括未匹配预测。重叠像素属于分数最高的实例，
  同分时属于源 JSON 中先出现的实例。此步骤不使用真实标注决定归属。
- 最终互不重叠的计数区域保存在 `evaluation_segmentation`。原始预测区域仍保存在
  `segmentation`，用于追溯及复核匹配。
- 点使用原图绝对坐标 `(x,y)`，通过 `floor` 转成像素索引；图外点不吸附到图像边缘。
- 对比模型的逐穗计数只在 `matched=true` 的固定清单上计算 MAE、RMSE，
  与 full/combo 原评估使用同一批匹配实例及真实数量。
- 逐穗真实数量为对应 GT 实例的**全部真实点数**。真实点落在预测区域外也不能删除。
  因此这项指标包含分割区域不完整造成的计数损失，并非仅测计数网络自身。

## JSON 字段

| 字段 | 内容 |
| --- | --- |
| `protocol` | 来源、哈希、阈值、匹配和区域归属规则 |
| `images` | 全部原图记录，保留原始 ID、文件名、尺寸；增加每图真实点数等 |
| `annotations` | 739 个通过分数过滤的预测实例及计数区域 |
| `annotations[].id` | 源预测 JSON 中的零基索引加 1；可能不连续 |
| `annotations[].segmentation` | 原始预测 mask，COCO RLE |
| `annotations[].evaluation_segmentation` | 解决重叠后的计数区域，COCO RLE |
| `annotations[].matched` | 是否属于固定的逐穗评估清单 |
| `annotations[].gt_annotation_id` | 匹配的原始 GT 标注 ID；未匹配为 null |
| `annotations[].gt_count` | 匹配 GT 的完整真实点数；未匹配为 null，不能解释成真值零 |
| `annotations[].gt_points` | 匹配 GT 的全部真实点，原图坐标；仅匹配实例有此字段 |
| `annotations[].gt_points_outside_pred_mask` | 真实点落在原始预测 mask 外的数量 |
| `annotations[].gt_points_outside_owned_mask` | 真实点落在最终计数区域外的数量 |
| `ground_truth_annotations` | 全部 613 个目标类别 GT 实例，含未匹配实例、真实 mask 和点 |
| `images[].gt_count` | 每图全部目标类别 GT 点数，含漏检实例 |
| `images[].unmatched_gt_annotation_ids` | 漏配 GT 的 ID |
| `summary` | 数据和区域审计汇总 |

所有 200 张原图均保留；即使某图无预测，也必须参加整图指标。

## 对接计数模型

点预测模型先把坐标还原到原图，再按 `evaluation_segmentation` 归属。可复用
`export_counting_eval_annotations.py` 的 `points_inside(points, mask)`。
模型自己的点分数阈值在验证阶段选择并记录，不要求不同模型使用同一个数值。

密度模型先将输出与原图区域在空间上对齐，确保密度总和代表数量且分辨率转换守恒，
然后在相同区域内求和，保留浮点计数，不能先四舍五入。
若密度已经是原图尺寸、每像素数值表示该像素的计数质量：

```python
mask = mask_util.decode(annotation["evaluation_segmentation"]).astype(bool)
predicted_count = float(density[mask].sum())
```

低分辨率密度图不能直接使用原图像素索引，也不能只 resize 后不校正总量。
需要对密度做保持总量的空间映射，或在密度网格上计算对应区域的覆盖权重。

必须先采用全体预测确定好的区域，再筛选 `matched=true` 的实例计算逐穗指标。
不能先删除未匹配实例后重新分配重叠区域，这会利用 GT 改变评估输入。

## full/combo 的基线与整图指标

现有 `full/combo` 的 `count_MAE_spike=1.3698384201077198` 统计的是每个 query
输出的点数，直接沿用该原生评估结果作为这次对比的基线。

full/combo 自身已经给出了每根麦穗所属的小穗点，无需额外按 mask 重新归属，也无需
为本协议重新导出它的点坐标。其他点模型使用 mask 归属，密度模型在 mask 内积分。
共享的是原图、匹配实例清单和真实数量；不同方法获取每穗预测数量的方式按各自输出
形式确定。论文中应说明这一差异，以及其他模型借用了 full/combo 的预测分割区域。

对比模型可以另外报告以下整图指标，并明确是否使用了区域过滤：

1. `count_MAE_img_mask` / `count_RMSE_img_mask`：累计所有预测区域（包含未匹配区域）
   的预测计数，与 `images[].gt_count` 比较。它是共享固定分割区域的流程指标。
2. `count_MAE_img_raw` / `count_RMSE_img_raw`：直接统计各模型完整输出，与全部 GT
   比较，不使用区域过滤。它是该计数模型自身的整图计数指标。

full/combo 的整图指标继续按原评估器累计通过阈值的实例点数。上述两种对比模型
指标都以每图全部 GT 点数为目标，但预测汇总方式不同，应分别注明。
只累计匹配实例后计算的误差不能称为完整图像计数误差。

## 当前检查结果

- 200 张图；613 个 GT 穗；9747 个真实点。
- 739 个预测穗；557 个匹配；182 个未匹配预测；56 个未匹配 GT。
- 匹配穗合计 8889 个真实点。
- 6 张图的预测 mask 有重叠，共 31087 个重叠像素。
- 18 个匹配穗真实点在原始预测 mask 外；解决重叠后，40 个在其所属计数区域外。
- 每图实例数、真实点数、匹配数，与原 `spikelet_pred_vs_gt.csv` 一致。
- 匹配记录的文件名、真实点数、IoU 多重集合，与原 `spikelet_pair_counts.csv` 一致。
  原 CSV 没有实例 ID，所以此项不能独立验证同图同数量同 IoU 的歧义配对；导出本身使用
  原始标注和预测顺序复现相同匹配算法。
- 已检查重叠优先级、同分处理、坐标边界、图外点、区域外 GT 保留、零预测图及 RLE 往返。

## 重新导出

在仓库根目录运行；脚本拒绝覆盖已有协议文件，换模型时使用新文件名。

```bash
/public/home/zncui/anaconda3/envs/MaskDINO-310/bin/python tools/export_counting_eval_annotations.py \
  --gt-json /public/home/zncui/datasets/wheat50/coco_2cat_with_points/annotations/instances_val2017.json \
  --pred-json output_v2/exp-1009-full-combo/inference/coco_instances_results.json \
  --source-config output_v2/exp-1009-full-combo/config.yaml \
  --image-reference-csv output_v2/exp-1009-full-combo/inference/spikelet_pred_vs_gt.csv \
  --pair-reference-csv output_v2/exp-1009-full-combo/inference/spikelet_pair_counts.csv \
  --output /path/to/new_counting_eval_protocol.json
```

该文件仍属于验证集的派生评估协议，不构成独立测试集。
