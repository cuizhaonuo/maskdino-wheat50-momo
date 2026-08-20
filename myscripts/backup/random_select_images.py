import os
import shutil
import random
import pandas as pd
import random
from tqdm import tqdm

# target_dir = '/public/home/zncui/datasets/xinxiang/yolo/images/test'

xinxiang_0509_dir = '/public/lab_share/nfs_field/level_1/DJI/P1/xinxiang2025/20250509DJI'
real_xinxiang_file = '/public/home/zncui/datasets/xinxiang/MuSuiShu.xlsx'
xinxiang_dir = '/public/home/zncui/datasets/xinxiang'
xinxiang_797_dir = '/public/home/zncui/datasets/xinxiang/797Areas'
xinxiang_150_dir = '/public/home/zncui/datasets/xinxiang/150Areas'
os.makedirs(xinxiang_797_dir, exist_ok=True)
os.makedirs(xinxiang_150_dir, exist_ok=True)

# 读取 real_xinxiang_file 的 sheet1 表
df = pd.read_excel(real_xinxiang_file, sheet_name='Sheet1')

# 查看数据
# print(df)

PlotID = [pid.replace('P', 'C') for pid in list(df['PlotID'])]
print(len(PlotID))

created = False
if not created:
    for pid in tqdm(PlotID):
        source_dir = os.path.join(xinxiang_0509_dir, pid)
        one_image = random.sample(os.listdir(source_dir), 1)[0]
        source_image = os.path.join(source_dir, one_image)
        target_image = os.path.join(xinxiang_797_dir, one_image)
        shutil.copy(source_image, target_image)
        # print(f'Copied {source_image} to {target_image}')

# 确保 xinxiang_150_dir 中有 150 张图片
xinxiang_150_images = os.listdir(xinxiang_150_dir)
xinxiang_150_txt = os.path.join(xinxiang_dir, 'selected_xinxiang_150_images.txt')
if len(xinxiang_150_images) < 150:
    all_797_images = os.listdir(xinxiang_797_dir)

    if not os.path.exists(xinxiang_150_txt):
        xinxiang_150_images = random.sample(all_797_images, 150)
        with open(xinxiang_150_txt, 'w') as f:
            for image in xinxiang_150_images:
                f.write(f"{image}\n")
        for image in xinxiang_150_images:
            source_image = os.path.join(xinxiang_797_dir, image)
            target_image = os.path.join(xinxiang_150_dir, image)
            shutil.copy(source_image, target_image)

    # 已经选择的图片，不会再选择
    with open(xinxiang_150_txt, 'r') as f:
        xinxiang_150_images = [line.strip() for line in f.readlines()]
    remaining_images = list(set(all_797_images) - set(xinxiang_150_images))
    images_to_select = 150 - len(xinxiang_150_images)
    new_selected_images = random.sample(remaining_images, images_to_select)
    with open(xinxiang_150_txt, 'a') as f:
        for image in new_selected_images:
            f.write(f"{image}\n")
    for image in new_selected_images:
        source_image = os.path.join(xinxiang_797_dir, image)
        target_image = os.path.join(xinxiang_150_dir, image)
        shutil.copy(source_image, target_image)






