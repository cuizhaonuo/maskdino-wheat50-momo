import json
import os

poly_annotations = '/public/home/zncui/datasets/maize_rice_dataset/coco/poly_annotations'
split = ['train2017', 'val2017', 'test2017']

for sp in split:
    ann_file = f'{poly_annotations}/instances_{sp}.json'
    with open(ann_file, 'r') as f:
        data = json.load(f)
    image_names = [img['file_name'] for img in data['images']]
    source_image_names = set(image_names)

    images_dir = f'/public/home/zncui/datasets/maize_rice_dataset/coco/{sp}'
    existing_image_names = set(os.listdir(images_dir))

    diff_image_names = existing_image_names - source_image_names
    print(f'{len(existing_image_names)} images')
    print(f'{len(source_image_names)} images in annotations')
    print(f'For split {sp}, found {len(diff_image_names)} images not in annotations.')

    # 将不在标注文件中的图片删除
    for img_name in diff_image_names:
        img_path = os.path.join(images_dir, img_name)
        os.remove(img_path)
    print(f'Removed {len(diff_image_names)} images from {images_dir}.')



