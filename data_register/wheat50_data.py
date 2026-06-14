# 配置 wheat50 两类（cat2）数据集路径
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

DATASET_ROOT = "/public/home/zncui/datasets/wheat50/coco_2cat"

# ====== 注册 wheat50 coco_2cat 数据集 ======
register_coco_instances(
    "wheat_cat2_coco_train", {},
    f"{DATASET_ROOT}/annotations/instances_train2017.json",
    f"{DATASET_ROOT}",
)
register_coco_instances(
    "wheat_cat2_coco_val", {},
    f"{DATASET_ROOT}/annotations/instances_val2017.json",
    f"{DATASET_ROOT}",
)

for _name in ("wheat_cat2_coco_train", "wheat_cat2_coco_val"):
    MetadataCatalog.get(_name).set(
        thing_classes=["wheat", "wheat_2"],
        thing_colors=[(255, 0, 0), (0, 255, 0)],
        evaluator_type="coco",
        thing_dataset_id_to_contiguous_id={1: 0, 2: 1},
    )
