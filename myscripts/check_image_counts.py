import json
ann_dir = '/public/home/zncui/datasets/maize_rice_dataset/coco/annotations'


for split in ['train', 'val', 'test']:
    ann_file = f'{ann_dir}/instances_{split}2017.json'
    with open(ann_file, 'r') as f:
        data = json.load(f)
        num_images = len(data['images'])
        print(f'Number of images in {split} set: {num_images}')