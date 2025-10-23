import shutil
import os
import glob
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm
import random
from collections import defaultdict

dataset_dir = '/public/lab_share/nfs_data/annotations/lab/maize_rice'
target_dataset_dir = '/public/home/zncui/datasets/maize_rice_dataset/total'

maize_annotations_dir = '/public/lab_share/nfs_data/annotations/lab/maize/maize_spike'
rice_annotations_dir  = '/public/lab_share/nfs_data/annotations/lab/rice/rice_spike_ealy'

total_images_dir = os.path.join(target_dataset_dir, 'images')
maize_images_dir = os.path.join(total_images_dir, 'maize')
rice_images_dir  = os.path.join(total_images_dir, 'rice')

total_ann_dir = os.path.join(target_dataset_dir, 'annotations')
maize_ann_dir = os.path.join(total_ann_dir, 'maize')
rice_ann_dir  = os.path.join(total_ann_dir, 'rice')

os.makedirs(maize_images_dir, exist_ok=True)
os.makedirs(rice_images_dir,  exist_ok=True)
os.makedirs(maize_ann_dir,   exist_ok=True)
os.makedirs(rice_ann_dir,    exist_ok=True)

maize_dir_list = [d for d in os.listdir(dataset_dir)
                  if 'maize' in d and os.path.isdir(os.path.join(dataset_dir, d))]
rice_dir_list  = [d for d in os.listdir(dataset_dir)
                  if 'rice'  in d and os.path.isdir(os.path.join(dataset_dir, d))]

print('maize_dir_list:', maize_dir_list)
print('rice_dir_list :', rice_dir_list)

def copy_images_from(dir_list, out_dir):
    for d in dir_list:
        src_root = os.path.join(dataset_dir, d)
        for root, _, files in os.walk(src_root):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(root, f)
                    dst = os.path.join(out_dir, f)
                    shutil.copy2(src, dst)

def clean_and_copy_jsons(src_dir, dst_dir):
    for old in glob.glob(os.path.join(dst_dir, '**', '*.json'), recursive=True):
        os.remove(old)
    for src in glob.glob(os.path.join(src_dir, '**', '*.json'), recursive=True):
        basename = os.path.basename(src)
        dst = os.path.join(dst_dir, basename)
        shutil.copy2(src, dst)

# copy_images_from(maize_dir_list, maize_images_dir)
# copy_images_from(rice_dir_list,  rice_images_dir)

# clean_and_copy_jsons(maize_annotations_dir, maize_ann_dir)
# clean_and_copy_jsons(rice_annotations_dir,  rice_ann_dir)

filtered_dataset_dir = os.path.join(target_dataset_dir, 'filtered_images')
filtered_maize_dir = os.path.join(filtered_dataset_dir, 'maize')
filtered_rice_dir = os.path.join(filtered_dataset_dir, 'rice')
os.makedirs(filtered_dataset_dir, exist_ok=True)
os.makedirs(filtered_maize_dir, exist_ok=True)
os.makedirs(filtered_rice_dir, exist_ok=True)

def filter_images_with_json(images_dir, ann_dir, out_dir):
    json_basenames = set()
    for json_path in glob.glob(os.path.join(ann_dir, '*.json')):
        basename = os.path.basename(json_path)
        json_basenames.add(basename.replace('.json', ''))
    for img_path in glob.glob(os.path.join(images_dir, '*')):
        img_basename = os.path.basename(img_path)
        # img_name, ext = os.path.splitext(img_basename)
        if img_basename in json_basenames:
            dst = os.path.join(out_dir, img_basename)
            shutil.copy2(img_path, dst)

# filter_images_with_json(maize_images_dir, maize_ann_dir, filtered_maize_dir)
# filter_images_with_json(rice_images_dir,  rice_ann_dir,  filtered_rice_dir)
# ================== 工具函数 ==================
def parse_mask_rle_field(mask_rle_field):
    """
    将形如 [['93807 6 94400 14 ...']] 或 ['93807 6 94400 14 ...'] 或 '93807 6 ...'
    统一解析成整数列表 [start0, len0, start1, len1, ...]
    """
    if isinstance(mask_rle_field, list):
        # 可能是 [['...']] 或 ['...','...']
        parts = []
        for item in mask_rle_field:
            if isinstance(item, list):
                parts.extend(item)
            else:
                parts.append(item)
        s = ' '.join(parts)
    else:
        s = str(mask_rle_field)
    s = s.strip()
    # 兼容多重空白
    tokens = [t for t in s.replace('\n', ' ').split(' ') if t]
    nums = list(map(int, tokens))
    if len(nums) % 2 != 0:
        raise ValueError(f'mask_rle 的数字个数不是偶数：len={len(nums)}，内容示例：{nums[:10]}')
    return nums

def startlen_to_mask(startlen_nums, height, width, order='C'):
    """
    把 [start0, len0, start1, len1, ...] 转换成 HxW 二值掩码。
    order: 'C' 表示 start 基于 C 顺序展平（行优先），'F' 表示基于 Fortran 顺序（列优先）。
    """
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)

    for i in range(0, len(startlen_nums), 2):
        start = startlen_nums[i] - 1 # -1 避免越界，标注是从 1 开始的，而不是 0
        run   = startlen_nums[i+1]
        if start + run > total:
            run = total - start
        # if start < 0 or run < 0 or start + run > total:
        #     raise ValueError(f'RLE 片段越界: start={start}, run={run}, total={total}')
        flat[start:start+run] = 1

    mask2d = flat.reshape((height, width), order=order)
    return mask2d

def mask_to_coco_rle(mask2d):
    """
    将 HxW 的二值掩码编码为 COCO RLE（Fortran 顺序）。
    注意：pycocotools 期望的是 Fortran 连续内存。
    """
    rle = maskUtils.encode(np.asfortranarray(mask2d))
    # pycocotools 返回的 counts 是 bytes，需要转成 str 方便序列化
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def decode_coco_rle_to_mask(coco_rle):
    """
    将 COCO RLE 解码回二值掩码 (HxW)
    """
    rle = coco_rle.copy()
    # 确保 counts 是 bytes
    if isinstance(rle.get('counts'), str):
        rle['counts'] = rle['counts'].encode('ascii')
    return maskUtils.decode(rle).astype(np.uint8)

def overlay_mask_on_image(img_bgr, mask, color_bgr=None, alpha=0.5):
    """
    在 BGR 图上叠加单个二值掩码（0/1），返回新图。
    """
    if color_bgr is None:
        color_bgr = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    overlay = img_bgr.copy()
    overlay[mask.astype(bool)] = (
        overlay[mask.astype(bool)] * (1 - alpha) + np.array(color_bgr, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay

def draw_bbox(img_bgr, bbox, color=(0, 255, 255), thickness=2):
    """
    画矩形框。支持 [x, y, w, h] 或 [x1, y1, x2, y2]
    """
    if len(bbox) == 4:
        x, y, w, h = bbox
        if w >= 0 and h >= 0:  # [x,y,w,h]
            pt1, pt2 = (int(x), int(y)), (int(x + w), int(y + h))
        else:  # 兜底：当作 [x1,y1,x2,y2]
            pt1, pt2 = (int(x), int(y)), (int(w), int(h))
    else:
        raise ValueError(f'不支持的 bbox 格式：{bbox}')
    cv2.rectangle(img_bgr, pt1, pt2, color, thickness)

# 简单的类别着色（可替换为你自己的调色板）
def category_color():
    # random.seed(12345)
    return (random.randint(30, 255), random.randint(30, 255), random.randint(30, 255))


RLE_ORDER = 'C'   # 'C' 或 'F'
ALPHA = 0.8      # 掩码叠加透明度

# coco_ann = '/public/home/zncui/datasets/coco/annotations/instances_val2017.json'
# with open(coco_ann, 'r') as f:
#     coco_data = json.load(f)

def collect_all_categories(ann_list):
    category_list = []
    for ann in tqdm(ann_list):
        with open(ann, 'r') as f:
            data = json.load(f)
            for d in data:
                category_id = d['category_id']
                if category_id not in category_list:
                    category_list.append(category_id)
    print('category_list:', category_list)  # ['maize', 'other']

def draw_whole_masks(category, image_name, img, data, h, w):
    whole = img.copy()
    for d in data:
        mask_rle = d['mask_rle']  # 'mask_rle': [['93807 6 94400 14 ...']]
        # 可视化全部掩码
        startlen_nums = parse_mask_rle_field(mask_rle)
        bin_mask = startlen_to_mask(startlen_nums, h, w, order=RLE_ORDER)
        coco_rle = mask_to_coco_rle(bin_mask)
        decoded_mask = decode_coco_rle_to_mask(coco_rle)
        whole = overlay_mask_on_image(whole, decoded_mask, color_bgr=category_color(), alpha=ALPHA)
    save_dir = f'/public/home/zncui/datasets/maize_rice_dataset/total/vis/{category}/whole'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'whole_{image_name}')
    cv2.imwrite(save_path, whole)

def get_group_id_to_data(data):
    group_data_list = []
    for d in data:
        group_id = d['group_id']
        group_data_list.append((group_id, d))

    grouped_data = defaultdict(list)
    for group_id, data in group_data_list:
        grouped_data[group_id].append(data)
    return grouped_data

def build_coco_annotation(category, ann_dir, filtered_dir):
    ann_list = sorted(glob.glob(os.path.join(ann_dir, '*.json')))

    # 收集所有类别
    collect_all_categories(ann_list)

    coco_annotation = {
        'info': {},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': category}]
    }

    ann_start = 10000 if category == 'maize' else 20000
    image_start = 30000 if category == 'maize' else 40000

    for idx, ann in enumerate(tqdm(ann_list, desc='Visualizing grouped masks')):
        with open(ann, 'r') as f:
            data = json.load(f)
            image_name = data[0]['image_name']
            img_path = os.path.join(filtered_dir, image_name)
            img = cv2.imread(img_path)
            # 纯净白底
            # if img is not None:
            #     img = np.full_like(img, 255)
            h, w = img.shape[:2]
            grouped = img.copy()

            # 可视化全部掩码
            draw_whole_masks(category, image_name, img, data, h, w)

            # 分组
            grouped_data = get_group_id_to_data(data)


            image_idx = image_start + idx
            image_info = {
                'file_name': image_name,
                'height': h,
                'width': w,
                'id': image_idx
            }
            coco_annotation['images'].append(image_info)

            # 可视化每个组
            coco_rle_list = []

            for group_id, data in grouped_data.items():
                if group_id == '':
                    mask_idx_list = []
                    board = np.zeros((h, w), dtype=np.uint8)
                    for i, d in enumerate(data):
                        category_id = d['category_id']
                        mask_rle = d['mask_rle']
                        startlen_nums = parse_mask_rle_field(mask_rle)
                        bin_mask = startlen_to_mask(startlen_nums, h, w, order=RLE_ORDER)
                        if category_id == 'other' or category_id == '其他':
                            board[bin_mask.astype(bool)] = 0
                        else:
                            board[bin_mask.astype(bool)] = i + 1  # 不同实例用不同整数表示
                            mask_idx_list.append(i + 1)
                    for idx in mask_idx_list:
                        instance_mask = (board == idx).astype(np.uint8)
                        coco_rle = mask_to_coco_rle(instance_mask)
                        decoded_mask = decode_coco_rle_to_mask(coco_rle)
                        grouped = overlay_mask_on_image(grouped, decoded_mask, color_bgr=category_color(), alpha=ALPHA)
                        coco_rle_list.append(coco_rle)
                else:
                    board = np.zeros((h, w))
                    for d in data:
                        category_id = d['category_id']
                        mask_rle = d['mask_rle']
                        startlen_nums = parse_mask_rle_field(mask_rle)
                        bin_mask = startlen_to_mask(startlen_nums, h, w, order=RLE_ORDER)
                        if category_id == 'other' or category_id == '其他':
                            board[bin_mask.astype(bool)] = 0
                        else:
                            board = np.logical_or(board, bin_mask)
                    coco_rle = mask_to_coco_rle(board)
                    decoded_mask = decode_coco_rle_to_mask(coco_rle)
                    grouped = overlay_mask_on_image(grouped, decoded_mask, color_bgr=category_color(), alpha=ALPHA)
                    coco_rle_list.append(coco_rle)

            # save_dir = f'/public/home/zncui/datasets/maize_rice_dataset/total/vis/{category}/grouped'
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f'grouped_{image_name}')
            # cv2.imwrite(save_path, grouped)


            for rle_idx, coco_rle in enumerate(coco_rle_list):
                ann_info = {
                    'image_id': image_idx,
                    'category_id': 1 if category == 'maize' else 2,
                    'iscrowd': 0,
                }
                ann_info['segmentation'] = coco_rle
                ann_info['area'] = int(maskUtils.area(coco_rle))
                bbox = maskUtils.toBbox(coco_rle)
                ann_info['bbox'] = [float(b) for b in bbox]
                ann_start = ann_start + rle_idx
                ann_info['id'] = ann_start + rle_idx
                coco_annotation['annotations'].append(ann_info)

    coco_ann_path = os.path.join(target_dataset_dir, 'annotations', f'instances_total_{category}.json')
    with open(coco_ann_path, 'w') as f:
        json.dump(coco_annotation, f)
    print(f'Saved COCO annotation to {coco_ann_path}')
    return coco_ann_path

coco_maize_ann_path = build_coco_annotation('maize', maize_ann_dir, filtered_maize_dir)
coco_rice_ann_path = build_coco_annotation('rice',  rice_ann_dir,  filtered_rice_dir)

# 将标注绘制到图像上进行可视化
def visualize_coco_annotations(coco_ann_path, images_dir, out_dir):
    with open(coco_ann_path, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    # 将 images 随机打乱顺序
    random.shuffle(coco_data['images'])

    for img_info in tqdm(coco_data['images'][:50], desc='Visualizing COCO annotations'):
        img_name = img_info['file_name']
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f'Warning: failed to read image {img_path}')
            continue
        h, w = img.shape[:2]

        anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_info['id']]
        vis_img = img.copy()

        for ann in anns:
            coco_rle = ann['segmentation']
            decoded_mask = decode_coco_rle_to_mask(coco_rle)
            vis_img = overlay_mask_on_image(vis_img, decoded_mask, color_bgr=category_color(), alpha=ALPHA)
            draw_bbox(vis_img, ann['bbox'], color=(0, 255, 255), thickness=2)

        save_path = os.path.join(out_dir, img_name)
        cv2.imwrite(save_path, vis_img)
        # print(f'Saved visualization to {save_path}')

coco_maize_ann_path = os.path.join(target_dataset_dir, 'annotations', 'instances_total_maize.json')
coco_rice_ann_path  = os.path.join(target_dataset_dir, 'annotations', 'instances_total_rice.json')
vis_maize_outdir = os.path.join(target_dataset_dir, 'vis_coco', 'maize')
vis_rice_outdir  = os.path.join(target_dataset_dir, 'vis_coco', 'rice')
# visualize_coco_annotations(coco_maize_ann_path, filtered_maize_dir, vis_maize_outdir)
# visualize_coco_annotations(coco_rice_ann_path,  filtered_rice_dir,  vis_rice_outdir)

# 根据标注文件，按照 8:1:1 划分训练集、验证集和测试集
def split_dataset(coco_ann_path, train_ratio=0.8, val_ratio=0.1):
    with open(coco_ann_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    random.shuffle(images)

    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)

    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    def filter_annotations(image_list):
        image_ids = {img['id'] for img in image_list}
        anns = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
        return anns

    train_anns = filter_annotations(train_images)
    val_anns = filter_annotations(val_images)
    test_anns = filter_annotations(test_images)

    def save_split(split_name, image_list, ann_list):
        split_data = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'images': image_list,
            'annotations': ann_list,
            'categories': coco_data['categories']
        }
        split_ann_path = coco_ann_path.replace('.json', f'_{split_name}.json')
        with open(split_ann_path, 'w') as f:
            json.dump(split_data, f)
        print(f'Saved {split_name} split to {split_ann_path}')

    save_split('train', train_images, train_anns)
    save_split('val', val_images, val_anns)
    save_split('test', test_images, test_anns)

coco_maize_ann_path = os.path.join(target_dataset_dir, 'annotations', 'instances_total_maize.json')
coco_rice_ann_path  = os.path.join(target_dataset_dir, 'annotations', 'instances_total_rice.json')
split_dataset(coco_maize_ann_path)
split_dataset(coco_rice_ann_path)

# for split_name in ['train', 'val', 'test']:
#     coco_maize_ann_path = os.path.join(target_dataset_dir, 'annotations', f'instances_total_maize_{split_name}.json')
#     coco_rice_ann_path  = os.path.join(target_dataset_dir, 'annotations', f'instances_total_rice_{split_name}.json')
#     vis_maize_outdir = os.path.join(target_dataset_dir, 'split_vis', f'vis_coco_{split_name}', 'maize')
#     vis_rice_outdir  = os.path.join(target_dataset_dir, 'split_vis', f'vis_coco_{split_name}', 'rice')
#     visualize_coco_annotations(coco_maize_ann_path, filtered_maize_dir, vis_maize_outdir)
#     visualize_coco_annotations(coco_rice_ann_path,  filtered_rice_dir,  vis_rice_outdir)
# print('Visualization of splits completed.')

# 将两个类别的标注合并为一个总的 COCO 标注文件
def merge_coco_annotations(maize_ann_path, rice_ann_path, output_path):
    with open(maize_ann_path, 'r') as f:
        maize_data = json.load(f)
    with open(rice_ann_path, 'r') as f:
        rice_data = json.load(f)

    merged_data = {
        'info': maize_data['info'],
        'licenses': maize_data['licenses'],
        'images': maize_data['images'] + rice_data['images'],
        'annotations': maize_data['annotations'] + rice_data['annotations'],
        'categories': [
            {'id': 1, 'name': 'maize'},
            {'id': 2, 'name': 'rice'}
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(merged_data, f)
    print(f'Merged COCO annotation saved to {output_path}')
merged_ann_path = os.path.join(target_dataset_dir, 'annotations', 'instances_total_maize_rice.json')
merge_coco_annotations(coco_maize_ann_path, coco_rice_ann_path, merged_ann_path)

# images_dir = '/public/home/zncui/datasets/maize_rice_dataset/total/images/maize_rice'
# visualize_coco_annotations(merged_ann_path, images_dir, os.path.join(target_dataset_dir, 'vis_coco', 'maize_rice'))
# print('Visualization of merged annotations completed.')

# 以上问题已经排除

# 将训练集、验证集和测试集的划分也合并
def merge_coco_splits(maize_ann_path, rice_ann_path, output_prefix):
    splits = ['train', 'val', 'test']
    for split in splits:
        with open(maize_ann_path.replace('.json', f'_{split}.json'), 'r') as f:
            maize_data = json.load(f)
        with open(rice_ann_path.replace('.json', f'_{split}.json'), 'r') as f:
            rice_data = json.load(f)

        merged_data = {
            'info': maize_data['info'],
            'licenses': maize_data['licenses'],
            'images': maize_data['images'] + rice_data['images'],
            'annotations': maize_data['annotations'] + rice_data['annotations'],
            'categories': [
                {'id': 1, 'name': 'maize'},
                {'id': 2, 'name': 'rice'}
            ]
        }
        output_path = output_prefix + f'_{split}.json'
        with open(output_path, 'w') as f:
            json.dump(merged_data, f)
        print(f'Merged {split} split saved to {output_path}')
merge_coco_splits(coco_maize_ann_path, coco_rice_ann_path,
                 os.path.join(target_dataset_dir, 'annotations', 'instances_total_maize_rice'))

# for split_name in ['train', 'val', 'test']:
#     coco_maize_ann_path = os.path.join(target_dataset_dir, 'annotations', f'instances_total_maize_rice_{split_name}.json')
#     vis_maize_rice_outdir = os.path.join(target_dataset_dir, 'split_vis', f'vis_coco_{split_name}', 'maize')
#     images_dir = '/public/home/zncui/datasets/maize_rice_dataset/total/images/maize_rice'
#     visualize_coco_annotations(coco_rice_ann_path,  images_dir,  vis_maize_rice_outdir)
# print('Visualization of splits completed.')


coco_dir =  '/public/home/zncui/datasets/maize_rice_dataset/coco'
for split in ['train', 'val', 'test']:
    prefix = os.path.join(target_dataset_dir, 'annotations', 'instances_total_maize_rice')
    split_path = prefix + f'_{split}.json'
    with open(split_path, 'r') as f:
        data = json.load(f)
        images = data['images']
        split_dir = os.path.join(coco_dir, f'{split}2017')
        os.makedirs(split_dir, exist_ok=True)
        for img_info in tqdm(images, desc=f'Copying {split} images'):
            img_name = img_info['file_name']
            src_path = os.path.join(total_images_dir, 'maize_rice', img_name)
            dst_path = os.path.join(split_dir, img_name)
            shutil.copy2(src_path, dst_path)
        ann_path = os.path.join(coco_dir, 'annotations', f'instances_{split}2017.json')
        os.makedirs(os.path.dirname(ann_path), exist_ok=True)
        shutil.copy2(split_path, ann_path)
        print(f'Copied {split} images to {split_dir} and annotation to {ann_path}')
    # visualize_coco_annotations(split_path, split_dir, os.path.join(coco_dir, 'vis_coco', split))

for split in ['train', 'val', 'test']:
    ann_path = os.path.join(coco_dir, 'annotations', f'instances_{split}2017.json')
    with open(ann_path) as f:
        coco_data = json.load(f)
        print(f'{split} split: {len(coco_data["images"])} images, {len(coco_data["annotations"])} annotations.')


