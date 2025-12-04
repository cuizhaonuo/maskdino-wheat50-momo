import os
import random

coco_dir = "/public/home/zncui/datasets/xinxiang/coco"
split = "train2017"
split_coco_dir = os.path.join(coco_dir, split)
split_ann_file = os.path.join(coco_dir, "annotations", f"instances_{split}.json")


def extract_subset_coco(split_coco_dir, split_ann_file, out_coco_dir, num_images):
    import json
    from shutil import copy2

    with open(split_ann_file, 'r') as f:
        coco_data = json.load(f)

    image_ids = [img['id'] for img in coco_data['images']]
    selected_image_ids = set(random.sample(image_ids, num_images))

    subset_images = [img for img in coco_data['images'] if img['id'] in selected_image_ids]
    subset_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in selected_image_ids]

    subset_coco_data = {
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': coco_data['categories']
    }

    os.makedirs(out_coco_dir, exist_ok=True)
    out_ann_file = os.path.join(out_coco_dir, f"instances_{split}_subset.json")
    with open(out_ann_file, 'w') as f:
        json.dump(subset_coco_data, f)

    for img in subset_images:
        src_img_path = os.path.join(split_coco_dir, img['file_name'])
        dst_img_path = os.path.join(out_coco_dir, img['file_name'])
        copy2(src_img_path, dst_img_path)

    print(f"Extracted {num_images} images and annotations to {out_coco_dir}")

out_coco_dir = "/public/home/zncui/datasets/xinxiang/coco_subset/train2017"
num_images = 100  # Specify the number of images you want in the subset
extract_subset_coco(split_coco_dir, split_ann_file, out_coco_dir, num_images)