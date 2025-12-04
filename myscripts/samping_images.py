import os
import random
import shutil

areas797_dir = "/public/home/zncui/datasets/xinxiang/797Areas"
areas150_dir = "/public/home/zncui/datasets/xinxiang/150Areas"

images797_list = os.listdir(areas797_dir)
images150_list = os.listdir(areas150_dir)

# 797区域中有而150区域中没有的图片
only_in_797 = set(images797_list) - set(images150_list)
print(len(only_in_797))

# 随机选择 100 张图片，然后复制到一个新的文件夹中
sampled_images = random.sample(only_in_797, 100)
output_dir = "/public/home/zncui/datasets/xinxiang/100images_for_train"
os.makedirs(output_dir, exist_ok=True)

for img_name in sampled_images:
    src_path = os.path.join(areas797_dir, img_name)
    dst_path = os.path.join(output_dir, img_name)
    shutil.copyfile(src_path, dst_path)

