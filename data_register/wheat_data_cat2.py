# 配置自己的数据集路径
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
# ====== 注册 wheat 2类别 数据集 ======
register_coco_instances(
    "wheat_cat2_coco_train", {},
    "/public/home/zncui/datasets/wheat50/coco_2cat/annotations/instances_train2017.json",
    "/public/home/zncui/datasets/wheat50/coco_2cat/train2017"
)
register_coco_instances(
    "wheat_cat2_coco_val", {},
    "/public/home/zncui/datasets/wheat50/coco_2cat/annotations/instances_val2017.json",
    "/public/home/zncui/datasets/wheat50/coco_2cat/val2017"
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
