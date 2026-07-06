# 配置自己的数据集路径
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

# ====== 两阶段迁移专用：与全局 WITH_POINTS 解耦的固定命名数据集 ======
# Stage1（训练检测/分类头）：无点大数据集 coco_2cat。
# Stage2（只训练小穗点头）：带点小数据集 coco_2cat_with_points。
# 两组名字固定，各自 config 直接引用，不依赖上面的 WITH_POINTS 开关。
_STAGE_DATASETS = {
    "nop": "coco_2cat",              # 无点（no points），2 类
    "pts": "coco_2cat_with_points",  # 带点（points），2 类
}
for _tag, _ds in _STAGE_DATASETS.items():
    # 只有带点数据集传 extra_annotation_keys=["points"]；无点数据集的 ann 没有该字段，
    # 传入会触发 load_coco_json 的 KeyError。
    _extra = ["points"] if _tag == "pts" else []
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
