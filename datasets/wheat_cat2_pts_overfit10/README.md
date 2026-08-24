# wheat_cat2_pts_overfit10

Stage2 点头调试用的 10 图过拟合集。它只保存 COCO 标注子集，图片复用：

`/public/home/zncui/datasets/wheat50/coco_2cat_with_points/train2017`

## 选择规则

1. 从 `coco_2cat_with_points` 的 train split 选取。
2. 图片至少含一个 `wheat_2`。
3. 排除任一 `wheat_2` 的 `points` 为空的图片。
4. 按 `wheat_2` 实例数降序、总点数降序、`image_id` 升序取前 10 张。

## 规模

| images | wheat | wheat_2 | spikelet points |
|---:|---:|---:|---:|
| 10 | 517 | 84 | 1178 |

选中的 `image_id` 为：`101, 195, 210, 10, 174, 313, 137, 226, 193, 253`。

训练与评估均注册为 `wheat_cat2_pts_coco_overfit10`，配套配置为
`configs/coco/instance-segmentation/wheat_stage2_pointhead_overfit10.yaml`。
