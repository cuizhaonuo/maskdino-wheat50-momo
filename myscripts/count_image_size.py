import json
import os

coco_ann_dir = '/public/home/zncui/datasets/maize_rice_dataset/coco/annotations/'
ann_files = ['instances_train2017.json', 'instances_val2017.json', 'instances_test2017.json']

for ann_file in ann_files:
    image_sizes = []
    ann_path = os.path.join(coco_ann_dir, ann_file)
    with open(ann_path, 'r') as f:
        data = json.load(f)
        for img_info in data['images']:
            width = img_info['width']
            height = img_info['height']
            image_sizes.append((width, height))
    # 绘制图像尺寸的分布直方图
    print(f'File: {ann_file}, Number of images: {len(image_sizes)}')
    size_count = {}
    for size in image_sizes:
        if size not in size_count:
            size_count[size] = 0
        size_count[size] += 1
    sorted_sizes = sorted(size_count.items(), key=lambda x: x[1], reverse=True)
    print('Top 10 most common image sizes:')
    for size, count in sorted_sizes[:10]:
        print(f'Size: {size}, Count: {count}')

'''
Top 10 most common image sizes:
Size: (1600, 1600), Count: 159
Size: (3500, 3500), Count: 142
Size: (600, 600), Count: 90
File: instances_val2017.json, Number of images: 48
Top 10 most common image sizes:
Size: (3500, 3500), Count: 23
Size: (1600, 1600), Count: 19
Size: (600, 600), Count: 6
File: instances_test2017.json, Number of images: 51
Top 10 most common image sizes:
Size: (1600, 1600), Count: 21
Size: (3500, 3500), Count: 18
Size: (600, 600), Count: 12
'''