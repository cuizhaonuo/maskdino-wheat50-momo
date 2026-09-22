# 配置自己的数据集路径
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

# ====== 全局参数：是否使用带点的数据集 ======
WITH_POINTS = True

# ====== 注册 wheat 2类别 数据集 ======
_DATASET = "coco_2cat_with_points" if WITH_POINTS else "coco_2cat"

# 使用 load_coco_json 并传入 extra_annotation_keys=["points"] 以确保 points 字段被加载
DatasetCatalog.register(
    "wheat_cat2_coco_train",
    lambda: load_coco_json(
        f"/public/home/zncui/datasets/wheat50/{_DATASET}/annotations/instances_train2017.json",
        f"/public/home/zncui/datasets/wheat50/{_DATASET}/train2017",
        "wheat_cat2_coco_train",
        extra_annotation_keys=["points"],
    ),
)
DatasetCatalog.register(
    "wheat_cat2_coco_val",
    lambda: load_coco_json(
        f"/public/home/zncui/datasets/wheat50/{_DATASET}/annotations/instances_val2017.json",
        f"/public/home/zncui/datasets/wheat50/{_DATASET}/val2017",
        "wheat_cat2_coco_val",
        extra_annotation_keys=["points"],
    ),
)

MetadataCatalog.get("wheat_cat2_coco_train").set(
    thing_classes=["wheat", "wheat_2"],
    thing_colors=[(255, 0, 0), (0, 0, 255)],
    evaluator_type="coco",
    thing_dataset_id_to_contiguous_id={1: 0, 2: 1}
)
MetadataCatalog.get("wheat_cat2_coco_val").set(
    thing_classes=["wheat", "wheat_2"],
    thing_colors=[(255, 0, 0), (0, 0, 255)],
    evaluator_type="coco",
    thing_dataset_id_to_contiguous_id={1: 0, 2: 1}
)

# ====== 固定命名数据集：与全局 WITH_POINTS 解耦，各自 config 直接引用 ======
# nop      ：无点数据集 coco_2cat（Stage1 训检测/分类头用）。
# pts      ：带点数据集 coco_2cat_with_points。2026-09-22 扩容后 train=2092 图 /
#            90850 实例 / 5394 个带点 wheat_2，与无点集同规模。
# pts_mini ：带点小集 coco_2cat_with_points_mini。train 从 pts 的 2092 张里按
#            (wheat_2 实例数降序, 小穗点总数降序, file_name 升序) 取前 500 张，
#            val 就是 pts 的全部 200 张（逐条相同）。用 24% 的图拿到 51% 的 wheat_2
#            和 50% 的小穗点，单轮 50 epoch 从 26000 iter 降到 6250 iter。
#            ⚠️ 选图偏向密集图，train 的 wheat_2 占比 9.9%，而 pts 是 5.9%、val 是 7.1%。
#            类别先验偏移会压 spike_precision（见 experiments-bank/ideas/unfreeze-real-dataset.md），
#            所以 mini 上的 spike_precision / 预测穗数不可与 pts 直接互比。
_STAGE_DATASETS = {
    "nop": "coco_2cat",                        # 无点（no points），2 类
    "pts": "coco_2cat_with_points",            # 带点（points），2 类
    "pts_mini": "coco_2cat_with_points_mini",  # 带点小集：train 500 张，val 全留 200 张
}
for _tag, _ds in _STAGE_DATASETS.items():
    # 只有带点数据集传 extra_annotation_keys=["points"]；无点数据集的 ann 没有该字段，
    # 传入会触发 load_coco_json 的 KeyError。
    _extra = [] if _tag == "nop" else ["points"]
    for _split in ("train", "val"):
        _name = f"wheat_cat2_{_tag}_coco_{_split}"
        DatasetCatalog.register(
            _name,
            lambda ds=_ds, split=_split, name=_name, extra=_extra: load_coco_json(
                f"/public/home/zncui/datasets/wheat50/{ds}/annotations/instances_{split}2017.json",
                f"/public/home/zncui/datasets/wheat50/{ds}/{split}2017",
                name,
                extra_annotation_keys=extra,
            ),
        )
        MetadataCatalog.get(_name).set(
            thing_classes=["wheat", "wheat_2"],
            thing_colors=[(255, 0, 0), (0, 0, 255)],
            evaluator_type="coco",
            thing_dataset_id_to_contiguous_id={1: 0, 2: 1},
        )

# ====== Stage2 debug：10 图过拟合子集 ======
# 标注 JSON 放在仓库内便于复现；图片直接复用带点训练集，不重复拷贝 1024x1024 原图。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OVERFIT10_JSON = _PROJECT_ROOT / "datasets/wheat_cat2_pts_overfit10/annotations/instances_overfit10.json"
_OVERFIT10_IMAGE_ROOT = Path("/public/home/zncui/datasets/wheat50/coco_2cat_with_points/train2017")
_OVERFIT10_NAME = "wheat_cat2_pts_coco_overfit10"

DatasetCatalog.register(
    _OVERFIT10_NAME,
    lambda: load_coco_json(
        str(_OVERFIT10_JSON),
        str(_OVERFIT10_IMAGE_ROOT),
        _OVERFIT10_NAME,
        extra_annotation_keys=["points"],
    ),
)
MetadataCatalog.get(_OVERFIT10_NAME).set(
    thing_classes=["wheat", "wheat_2"],
    thing_colors=[(255, 0, 0), (0, 0, 255)],
    evaluator_type="coco",
    thing_dataset_id_to_contiguous_id={1: 0, 2: 1},
)
