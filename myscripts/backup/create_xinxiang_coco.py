import glob
import os.path
import random
import json

xinxiang_dataset_dir = '/public/home/zncui/datasets/xinxiang/cropped_51images'
xinxiang_pred_dir = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/swinl_xinxiang_infer_iou_min_0.4_0.372_no_grid_coco'
xinxiang_coco_dir = '/public/home/zncui/datasets/xinxiang/coco'

image_files = []
for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG'):
    image_files.extend(glob.glob(os.path.join(xinxiang_dataset_dir, '*', ext)))

# Shuffle the list to ensure randomness
random.seed(2025)
random.shuffle(image_files)

ratio = (6, 2, 2)  # train, val, test
total = len(image_files)
num_train = total * ratio[0] // sum(ratio)
num_val = total * ratio[1] // sum(ratio)
num_test = total - num_train - num_val

train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

# Function to copy files to target directory
def copy_files_and_create_ann(file_list, subset):
    target_dir = os.path.join(xinxiang_coco_dir, subset)
    os.makedirs(target_dir, exist_ok=True)
    for file_path in file_list:
        base_name = os.path.basename(file_path)
        target_path = os.path.join(target_dir, base_name)
        os.system(f'cp "{file_path}" "{target_path}"')
    print(f'Copied {len(file_list)} files to {target_dir}')
    # 每一个图片对应一个 JSON 标注文件名，现在合并成一个 JSON
    ann_list = [os.path.join(xinxiang_pred_dir, os.path.basename(f)) for f in file_list]
    ann_list = [f.replace('.jpg', '.json') for f in ann_list]
    ann_file = os.path.join(xinxiang_coco_dir, 'annotations', f'instances_{subset}.json')
    os.makedirs(os.path.dirname(ann_file), exist_ok=True)
    annotations = {
        'info': {},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'wheat'}],
    }
    for ann_path in ann_list:
        with open(ann_path, 'r') as f:
            data = json.load(f)
        annotations['images'].extend(data['images'])
        annotations['annotations'].extend(data['annotations'])
    with open(ann_file, 'w') as f:
        json.dump(annotations, f)
    print(f'Created annotation file {ann_file} with {len(annotations["images"])} images and {len(annotations["annotations"])} annotations.')

copy_files_and_create_ann(train_files, 'train2017')
copy_files_and_create_ann(val_files, 'val2017')
copy_files_and_create_ann(test_files, 'test2017')

# 检查 image_id 和 annotation id 是否有重复
def check_image_ann_id():
    for subset in ['train2017', 'val2017', 'test2017']:
        ann_file = os.path.join(xinxiang_coco_dir, 'annotations', f'instances_{subset}.json')
        with open(ann_file, 'r') as f:
            data = json.load(f)
        image_ids = set()
        ann_ids = set()
        for img in data['images']:
            if img['id'] in image_ids:
                print(f'Duplicate image id {img["id"]} in {subset}')
            image_ids.add(img['id'])
        for ann in data['annotations']:
            if ann['id'] in ann_ids:
                print(f'Duplicate annotation id {ann["id"]} in {subset}')
            ann_ids.add(ann['id'])
        print(f'Checked {subset}: {len(image_ids)} unique images, {len(ann_ids)} unique annotations.')

check_image_ann_id()