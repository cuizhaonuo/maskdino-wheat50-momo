import glob
import os
import json
import cv2
import numpy as np

xinxiang_blue_polygon_dir = '/public/home/zncui/datasets/xinxiang/blue_polygon_test'

image_files = []
for ext in ['*.JPG', '*.jpg']:
    image_files.extend(glob.glob(os.path.join(xinxiang_blue_polygon_dir, ext)))

def get_red_point_masks(image_path, out_dir=None):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色区间（分两段）
    lower_red1, upper_red1 = (0, 150, 80), (10, 255, 255)
    lower_red2, upper_red2 = (170, 150, 80), (180, 255, 255)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 进行一次开运算，去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        basename = os.path.basename(image_path)
        save_path = os.path.join(out_dir, f"{basename}_red_mask.png")
        cv2.imwrite(save_path, mask)

    # 查找红色连通域，每个连通域对应一个红点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    masks = []
    for i, contour in enumerate(contours):
        # 创建与原图大小相同的全黑 mask
        point_mask = np.zeros_like(mask)
        # 在 mask 上填充该连通域
        cv2.drawContours(point_mask, [contour], -1, 255, -1)
        masks.append(point_mask)

    print(f"Detected {len(masks)} red points.")
    return masks

def polygon_to_mask(polygon, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask

# for image_file in image_files:
#     print(f"Processing {image_file}...")
#     get_red_point_masks(image_file, out_dir=os.path.join(xinxiang_blue_polygon_dir, 'red_point_masks'))

for image_file in image_files:
    img = cv2.imread(image_file)
    red_point_masks = get_red_point_masks(image_file)
    base_name = os.path.basename(image_file)
    json_file = image_file.replace('.JPG', '.json').replace('.jpg', '.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    for ann in annotations:
        polygon = ann['segmentation'][0]
        # 将 polygon 转化为二值 mask
        mask = polygon_to_mask(polygon, img.shape)
        # 检查该 polygon mask 是否与任一红点 mask 有重叠
        keep = False
        for red_mask in red_point_masks:
            intersection = cv2.bitwise_and(mask, red_mask)
            if np.any(intersection):
                keep = True
                break
        ann['keep'] = keep
    # 过滤掉不需要保留的 annotations
    filtered_annotations = [ann for ann in annotations if ann.get('keep', False)]
    data['annotations'] = filtered_annotations
    # 保存新的 json 文件
    out_json_file = os.path.join(xinxiang_blue_polygon_dir, 'filtered_annotations', base_name.replace('.JPG', '.json').replace('.jpg', '.json'))
    os.makedirs(os.path.dirname(out_json_file), exist_ok=True)
    with open(out_json_file, 'w') as f:
        json.dump(data, f)
    print(f"Saved filtered annotations to {out_json_file}")

    # 将过滤后得到的 annotations 绘制在图像上进行可视化
    vis_img = img.copy()
    for ann in filtered_annotations:
        polygon = ann['segmentation'][0]
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    vis_out_path = os.path.join(xinxiang_blue_polygon_dir, 'filtered_visualizations', base_name)
    os.makedirs(os.path.dirname(vis_out_path), exist_ok=True)
    cv2.imwrite(vis_out_path, vis_img)
    print(f"Saved visualization to {vis_out_path}")
