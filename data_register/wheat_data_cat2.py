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
