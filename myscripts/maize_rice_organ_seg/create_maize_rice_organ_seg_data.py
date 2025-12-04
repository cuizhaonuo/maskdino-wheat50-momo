import os
import random
import shutil
from tqdm import tqdm
import glob

# 设置随机种子以确保可重复性
random.seed(42)

rice_source_dir = "/public/datasets/WGSR"
maize_source_dir = "/public/datasets/WGSM"

rice_target_dir = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/rice"
maize_target_dir = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/maize"
os.makedirs(rice_target_dir, exist_ok=True)
os.makedirs(maize_target_dir, exist_ok=True)

# 首先从 rice_source_dir 中筛选得到 300 张图片
rice_image_list = [img for img in os.listdir(rice_source_dir) if "top" not in img]
selected_rice_image_list = random.sample(rice_image_list, 300)

for img_name in tqdm(selected_rice_image_list):
    source_path = os.path.join(rice_source_dir, img_name)
    target_path = os.path.join(rice_target_dir, img_name)
    shutil.copy(source_path, target_path)

# 然后从 maize_source_dir 中筛选得到 300 张图片
maize_image_list = glob.glob(os.path.join(maize_source_dir, "**", "*.png"), recursive=True)
selected_maize_image_list = random.sample(maize_image_list, 300)
for img_path in tqdm(selected_maize_image_list):
    img_name = os.path.basename(img_path)
    target_path = os.path.join(maize_target_dir, img_name)
    shutil.copy(img_path, target_path)