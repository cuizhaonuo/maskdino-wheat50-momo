from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

ann_file = "/public/home/zncui/datasets/xinxiang/coco_subset/annotations/instances_train2017.json"
coco = COCO(ann_file)

# 2️⃣ 获取所有图像 ID
img_ids = coco.getImgIds()
print(f"共有 {len(img_ids)} 张图片")


