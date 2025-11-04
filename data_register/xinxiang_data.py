# 配置自己的数据集路径
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
# ====== 注册 wheat 数据集 ======
register_coco_instances(
    "xinxiang_coco_train", {},
    "/public/home/zncui/datasets/xinxiang/coco/annotations/instances_train2017.json",
    "/public/home/zncui/datasets/xinxiang/coco/train2017"
)
register_coco_instances(
    "xinxiang_coco_val", {},
    "/public/home/zncui/datasets/xinxiang/coco/annotations/instances_val2017.json",
    "/public/home/zncui/datasets/xinxiang/coco/val2017"
)

MetadataCatalog.get("xinxiang_coco_train").set(
    thing_classes=["wheat"],
    thing_colors=[(255, 0, 0)],
    evaluator_type="coco",
    thing_dataset_id_to_contiguous_id={1: 0}
)
MetadataCatalog.get("xinxiang_coco_val").set(
    thing_classes=["wheat"],
    thing_colors=[(255, 0, 0)],
    evaluator_type="coco",
    thing_dataset_id_to_contiguous_id={1: 0}
)