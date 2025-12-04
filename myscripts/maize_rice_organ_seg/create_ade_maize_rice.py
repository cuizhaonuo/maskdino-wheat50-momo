import os
import cv2
import shutil
import numpy as np
import random

random.seed(42)



# ------------------ 一些小工具函数 ------------------ #
def list_masks(mask_dir):
    """列出目录下所有非包含 'mask' 字样的文件的绝对路径。"""
    return [
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if "mask" not in f
    ]


def get_unique_gray_values(mask_paths):
    """统计一批灰度掩码的所有唯一像素值，并输出含有异常值的路径。"""
    values = set()
    for p in mask_paths:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        unique_vals = np.unique(mask)
        values.update(unique_vals.tolist())
    return sorted(values)


def train_val_split(mask_paths, train_num=40):
    """打乱后按数量划分 train / val。"""
    paths = mask_paths[:]  # 复制一份，避免原列表顺序被修改
    random.shuffle(paths)
    return paths[:train_num], paths[train_num:]


def masks_to_images(mask_paths, img_dir):
    """根据掩码文件名在对应的图像目录中构造图像路径。"""
    return [os.path.join(img_dir, os.path.basename(m)) for m in mask_paths]


def copy_pair_list(mask_list, img_list, ann_dir, img_dir):
    """成对拷贝 mask 和 image 到指定目录。"""
    for mask_path, img_path in zip(mask_list, img_list):
        shutil.copy(mask_path, ann_dir)
        shutil.copy(img_path, img_dir)


def convert_mask_values(mask_path):
    """
    将掩码值映射为：
    0 -> 0（背景）
    [1, 129, 130, 131] -> 1（穗）
    [254, 255] -> 2（叶）
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    converted_mask = np.zeros_like(mask, dtype=np.uint8)
    # converted_mask[mask == 0] = 0
    # converted_mask[mask == 130] = 1
    # converted_mask[mask == 255] = 2
    # print(f'before {mask_path}', np.unique(converted_mask))
    converted_mask[np.isin(mask, [1, 129, 130, 131])] = 1
    converted_mask[np.isin(mask, [254, 255])] = 1
    # new_path = mask_path.replace(".jpg", ".png")
    # cv2.imwrite(new_path, converted_mask)
    cv2.imwrite(mask_path, converted_mask)



def convert_split_masks(train_masks, val_masks, train_ann_dir, val_ann_dir):
    """对 train/val 两个集合中的掩码做数值转换。"""
    for m in train_masks:
        convert_mask_values(os.path.join(train_ann_dir, os.path.basename(m)))
    for m in val_masks:
        convert_mask_values(os.path.join(val_ann_dir, os.path.basename(m)))


def print_mask_values(ann_dir):
    """打印某个 annotations 目录下所有掩码的唯一值。"""
    values = set()
    for fname in os.listdir(ann_dir):
        mask_path = os.path.join(ann_dir, fname)
        # 删除该目录下所有的 png 格式以外的文件
        if not fname.lower().endswith(".png"):
            if os.path.isfile(mask_path):
                os.remove(mask_path)
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        values.update(np.unique(mask).tolist())
    print(f"Mask unique values in {ann_dir}:", sorted(values))


def clean_dir(dir_path):
    """删除目录下的所有文件（不删目录本身）。"""
    if not os.path.exists(dir_path):
        return
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)


# ------------------ 1. 统计 Krita 掩码的像素值 ------------------ #
krita_maize_mask_dir = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/krita_huazhong_maize_2019"
krita_rice_mask_dir  = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/krita_rice_mask"

maize_masks = list_masks(krita_maize_mask_dir)
rice_masks  = list_masks(krita_rice_mask_dir)

print("Maize mask unique values:", get_unique_gray_values(maize_masks))
print("Rice mask unique values:", get_unique_gray_values(rice_masks))
# 结论：两者的掩码值是一致的，主值为 0（背景）、130（穗子）、255（叶片），
# 其中 maize 掩码还包含 1, 129, 131, 254 等，需要一起映射。


# ------------------ 2. 准备 ADE 目录 ------------------ #
ade_dir = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/ade_v2/ADEChallengeData2016/"

# maize 和 rice 共用一套 ADE 目录（和原逻辑一致）
ade_maize_train_ann_dir = os.path.join(ade_dir, "annotations/training")
ade_maize_val_ann_dir   = os.path.join(ade_dir, "annotations/validation")
ade_maize_train_img_dir = os.path.join(ade_dir, "images/training")
ade_maize_val_img_dir   = os.path.join(ade_dir, "images/validation")

ade_rice_train_ann_dir = ade_maize_train_ann_dir
ade_rice_val_ann_dir   = ade_maize_val_ann_dir
ade_rice_train_img_dir = ade_maize_train_img_dir
ade_rice_val_img_dir   = ade_maize_val_img_dir

os.makedirs(ade_maize_train_ann_dir, exist_ok=True)
os.makedirs(ade_maize_val_ann_dir, exist_ok=True)
os.makedirs(ade_maize_train_img_dir, exist_ok=True)
os.makedirs(ade_maize_val_img_dir, exist_ok=True)

# 关键：清空 annotations 目录，避免旧文件中的 3、4 干扰统计
clean_dir(ade_maize_train_ann_dir)
clean_dir(ade_maize_val_ann_dir)

# 如果你希望每次也重置 images，可以打开下面两行
# clean_dir(ade_maize_train_img_dir)
# clean_dir(ade_maize_val_img_dir)


# ------------------ 3. train/val 划分，并生成图像路径 ------------------ #
maize_images_dir = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/huazhong_maize_2019"
rice_images_dir  = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/rice"

maize_train_masks, maize_val_masks = train_val_split(maize_masks, train_num=50)
rice_train_masks,  rice_val_masks  = train_val_split(rice_masks,  train_num=40)

maize_train_images = masks_to_images(maize_train_masks, maize_images_dir)
maize_val_images   = masks_to_images(maize_val_masks,   maize_images_dir)

rice_train_images  = masks_to_images(rice_train_masks,  rice_images_dir)
rice_val_images    = masks_to_images(rice_val_masks,    rice_images_dir)


# ------------------ 4. 拷贝 maize/rice 的掩码和图像 ------------------ #
# maize
copy_pair_list(maize_train_masks, maize_train_images,
               ade_maize_train_ann_dir, ade_maize_train_img_dir)
copy_pair_list(maize_val_masks,   maize_val_images,
               ade_maize_val_ann_dir,   ade_maize_val_img_dir)

# rice
copy_pair_list(rice_train_masks, rice_train_images,
               ade_rice_train_ann_dir, ade_rice_train_img_dir)
copy_pair_list(rice_val_masks,   rice_val_images,
               ade_rice_val_ann_dir,   ade_rice_val_img_dir)

print("Done copying maize and rice masks/images.")

# 将 ade_dir 目录下的文件格式都改为 png 格式
def convert_images_to_png(root_dir, valid_exts=(".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    for curr_root, _, files in os.walk(root_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".png" or ext not in valid_exts:
                continue
            src_path = os.path.join(curr_root, fname)
            image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            dst_path = os.path.splitext(src_path)[0] + ".png"
            cv2.imwrite(dst_path, image)
            os.remove(src_path)
convert_images_to_png(ade_dir)
print("Converted all images to PNG format.")

# ------------------ 5. 转换掩码数值 ------------------ #
# 将 maize_train_masks, maize_val_masks, rice_train_masks, rice_val_masks 的后缀都改为 .png
maize_train_masks = [os.path.splitext(m)[0] + ".png" for m in maize_train_masks]
maize_val_masks   = [os.path.splitext(m)[0] + ".png" for m in maize_val_masks]
rice_train_masks  = [os.path.splitext(m)[0] + ".png" for m in rice_train_masks]
rice_val_masks    = [os.path.splitext(m)[0] + ".png" for m in rice_val_masks]
convert_split_masks(maize_train_masks, maize_val_masks,
                    ade_maize_train_ann_dir, ade_maize_val_ann_dir)
convert_split_masks(rice_train_masks, rice_val_masks,
                    ade_rice_train_ann_dir, ade_rice_val_ann_dir)

print("Done converting mask values.")


# ------------------ 6. 检查转换后的掩码值 ------------------ #
print_mask_values(ade_maize_train_ann_dir)
print_mask_values(ade_maize_val_ann_dir)
print_mask_values(ade_rice_train_ann_dir)
print_mask_values(ade_rice_val_ann_dir)


# ------------------ 7. 可视化 ADE 标签 ------------------ #
# 0 为黑色，1 为红色，2 为绿色
def visualize_ade_masks(ann_dir, vis_dir):
    os.makedirs(vis_dir, exist_ok=True)
    for fname in os.listdir(ann_dir):
        mask_path = os.path.join(ann_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        vis_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        vis_mask[mask == 1] = [255, 0, 0]   # 红色：穗
        vis_mask[mask == 2] = [0, 255, 0]   # 绿色：叶
        cv2.imwrite(os.path.join(vis_dir, fname), vis_mask)


vis_train_dir = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/ade_v2/ADEChallengeData2016/vis/training"
vis_val_dir   = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/ade_v2/ADEChallengeData2016/vis/validation"

visualize_ade_masks(ade_maize_train_ann_dir, vis_train_dir)
visualize_ade_masks(ade_maize_val_ann_dir,   vis_val_dir)

print("Done visualizing ADE masks.")