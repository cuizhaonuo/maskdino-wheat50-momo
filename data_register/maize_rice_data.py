# 配置自己的数据集路径
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# ====== 注册 maize_rice 数据集 ======
register_coco_instances(
    "maize_rice_coco_train", {},
    "/public/home/zncui/datasets/maize_rice_dataset/coco/annotations/instances_train2017.json",
    "/public/home/zncui/datasets/maize_rice_dataset/coco/train2017"
)
register_coco_instances(
    "maize_rice_coco_val", {},
    "/public/home/zncui/datasets/maize_rice_dataset/coco/annotations/instances_val2017.json",
    "/public/home/zncui/datasets/maize_rice_dataset/coco/val2017"
)

# 设置 metadata
MetadataCatalog.get("maize_rice_coco_train").set(
    thing_classes=["maize", "rice"],
    thing_colors=[(255, 255, 0), (0, 255, 0)],  # 黄色 maize，绿色 rice，可以按需改颜色
    evaluator_type="coco",
    thing_dataset_id_to_contiguous_id={1: 0, 2: 1}
)

MetadataCatalog.get("maize_rice_coco_val").set(
    thing_classes=["maize", "rice"],
    thing_colors=[(255, 255, 0), (0, 255, 0)],
    evaluator_type="coco",
    thing_dataset_id_to_contiguous_id={1: 0, 2: 1}
)
