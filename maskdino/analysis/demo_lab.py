import os
import json

def compute_mask_bbox_iou(predictions):
    coco_path = "/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/"
    test_ann_path = os.path.join(coco_path, "annotations/instances_test2017.json")
    img_path = "/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/test2017/D6-122-20230529.jpg"
    filename = os.path.basename(img_path)
    with open(test_ann_path) as f:
        gt_data = json.load(f)
    images = gt_data["images"]
    annotations = gt_data["annotations"]
    img_id = [img["id"] for img in images if img["file_name"] == filename][0]
    ann = [a for a in annotations if a["image_id"] == img_id]
    gt_masks = [a["segmentation"] for a in ann]
    pred_masks = predictions["instances"].pred_masks.numpy()
    # Compute IoU between gt_masks and pred_masks
    from pycocotools import mask as maskUtils
    import numpy as np
    def rle_to_mask(rle, height, width):
        return maskUtils.decode(rle)
    gt_mask_arrays = [rle_to_mask(maskUtils.frPyObjects(gt_mask, 1024, 1024), 1024, 1024) for gt_mask in gt_masks]
    ious = np.zeros((len(gt_mask_arrays), len(pred_masks)), dtype=np.float32)
    for i, gt_mask in enumerate(gt_mask_arrays):
        for j, pred_mask in enumerate(pred_masks):
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union == 0:
                ious[i, j] = 0.0
            else:
                ious[i, j] = intersection / union
    print("IoU matrix between GT masks and Pred masks:")
    print(ious)