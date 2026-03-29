from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


DATASET_ROOT = "/public/home/zncui/datasets/GWHD/gwhd_coco"


SPLITS = {
    "gwhd_train": (
        f"{DATASET_ROOT}/annotations/instances_train2017.json",
        f"{DATASET_ROOT}/train2017",
    ),
    "gwhd_val": (
        f"{DATASET_ROOT}/annotations/instances_val2017.json",
        f"{DATASET_ROOT}/val2017",
    ),
    "gwhd_test": (
        f"{DATASET_ROOT}/annotations/instances_test2017.json",
        f"{DATASET_ROOT}/test2017",
    ),
}


for dataset_name, (json_file, image_root) in SPLITS.items():
    register_coco_instances(dataset_name, {}, json_file, image_root)
    MetadataCatalog.get(dataset_name).set(
        thing_classes=["wheat"],
        thing_colors=[(255, 0, 0)],
        evaluator_type="coco",
        thing_dataset_id_to_contiguous_id={1: 0},
    )
