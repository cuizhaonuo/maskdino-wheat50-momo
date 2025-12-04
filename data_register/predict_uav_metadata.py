# uav_register_datasets.py
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.data.datasets import register_coco_panoptic
#from detectron2.data import MetadataCatalog

# 数据路径（改成你自己的路径）

# === COCO Panoptic Standard ===
# thing_dataset_id_to_contiguous_id 数量: 80
# stuff_dataset_id_to_contiguous_id 数量: 53
# thing_classes 数量: 80
# stuff_classes 数量: 133
# thing_colors 数量: 80
# stuff_colors 数量: 133

# 示例 (前 5 个 thing):
#   0: person, 颜色=[220, 20, 60]
#   1: bicycle, 颜色=[119, 11, 32]
#   2: car, 颜色=[0, 0, 142]
#   3: motorcycle, 颜色=[0, 0, 230]
#   4: airplane, 颜色=[106, 0, 228]

# 示例 (前 5 个 stuff):
#   0: person, 颜色=[220, 20, 60]
#   1: bicycle, 颜色=[119, 11, 32]
#   2: car, 颜色=[0, 0, 142]
#   3: motorcycle, 颜色=[0, 0, 230]
#   4: airplane, 颜色=[106, 0, 228]


ROOT ="/public/home/qwang/DJI/datasets/uav_coco"  #根目录

# 只注册 panoptic（没有 instances_json），必须提供 metadata
thing_classes = ["spike"]                  # 只有 spike 是 thing

stuff_classes = ["awn", "leaf", "stem"]    # stuff 包含所有类别
# 你的原始类别 id 约定：spike=1, awn=2, leaf=3, stem=4


thing_id_map = {1: 0}                                        
stuff_id_map = {2:1, 3: 2, 4: 3}    

meta = {
    "thing_classes": thing_classes,
    "stuff_classes": stuff_classes,
    "thing_dataset_id_to_contiguous_id": thing_id_map,
    "stuff_dataset_id_to_contiguous_id": stuff_id_map,
    # 可选：颜色 & 忽略标号（常用）

    "thing_colors":[(232, 206, 122)],   # spike = 红色基色
    # 按照 stuff_classes 顺序：spike,awn, leaf, stem
    "stuff_colors":[
                    (148,171,205),
                    (100,180,100),
                    (185,150,112)]
}
#  {"id": 1, "trainId": 0, "name": "spike", "isthing": 1, "color": [220, 20, 60]},
#     {"id": 2, "trainId": 1, "name": "awn",   "isthing": 0, "color": [119, 11, 32]},
#     {"id": 3, "trainId": 2, "name": "leaf",  "isthing": 0, "color": [0, 128, 0]},
#     {"id": 4, "trainId": 3, "name": "stem",  "isthing": 0, "color": [0, 0, 142]},
# # # ===== 注册 train =====
# register_coco_panoptic_separated(
#     name="uav_panoptic_train",
#     metadata={},
#     image_root=f"{ROOT}/train",
#     panoptic_root=f"{ROOT}/panoptic_train",
#     panoptic_json=f"{ROOT}/annotations/panoptic_train.json",
#     sem_seg_root=f"{ROOT}/panoptic_semseg_train",  # 如果你有语义分割标注可以写路径
#     instances_json=f"{ROOT}/annotations/instances_train.json",   # 如果你有 instance 标注可以写路径
# )

# # # ===== 注册 val =====
# register_coco_panoptic_separated(
#     name="uav_panoptic_val",
#     metadata={},
#     image_root=f"{ROOT}/val",
#     panoptic_root=f"{ROOT}/panoptic_val",
#     panoptic_json=f"{ROOT}/annotations/panoptic_val.json",
#     sem_seg_root=f"{ROOT}/panoptic_semseg_val",  # 如果你有语义分割标注可以写路径
#     instances_json=f"{ROOT}/annotations/instances_val.json",
# )

# # # ===== 注册 test =====
# register_coco_panoptic_separated(
#     name="uav_panoptic_test",
#     metadata={},
#     image_root=f"{ROOT}/test",
#     panoptic_root=f"{ROOT}/panoptic_test",
#     panoptic_json=f"{ROOT}/annotations/panoptic_test.json",
#     sem_seg_root=f"{ROOT}/panoptic_semseg_test",  # 如果你有语义分割标注可以写路径
#     instances_json=f"{ROOT}/annotations/instances_test.json",
# )


 #===== 注册 train =====
register_coco_panoptic(
    name="uav_panoptic_train",
    metadata=meta,
    image_root=f"{ROOT}/train",
    panoptic_root=f"{ROOT}/panoptic_train",
    panoptic_json=f"{ROOT}/annotations/panoptic_train.json",
    #sem_seg_root=f"{ROOT}/panoptic_semseg_train",
    instances_json=None
)

# ===== 注册 val =====
register_coco_panoptic(
    name="uav_panoptic_val",
    metadata=meta,
    image_root=f"{ROOT}/val",
    panoptic_root=f"{ROOT}/panoptic_val",
    panoptic_json=f"{ROOT}/annotations/panoptic_val.json",
    #sem_seg_root=f"{ROOT}/panoptic_semseg_val",
    instances_json=None
)