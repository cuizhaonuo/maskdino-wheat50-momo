import json

source_ann_file = "/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/annotations_polygon/instances_test2017.json"
target_ann_file = "/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/annotations_polygon/instances_test2017_one_image.json"

# 将 D6-108-20230519.jpg 相关的信息提取出来
target_image_filename = "D6-108-20230519.jpg"
with open(source_ann_file, 'r') as f:
    data = json.load(f)
target_images = [img for img in data['images'] if img['file_name'] == target_image_filename]
target_image_ids = {img['id'] for img in target_images}
target_annotations = [ann for ann in data['annotations'] if ann['image_id'] in target_image_ids]
target_data = {
    'images': target_images,
    'annotations': target_annotations,
    'categories': data['categories']
}
with open(target_ann_file, 'w') as f:
    json.dump(target_data, f)