import os
from PIL import Image

# yolo_val100_test500_test_root = '/public/home/zncui/datasets/maize_rice_dataset/yolo_val100_test500/images/test'
# swinl_maize_rice_val100_test500_search_root = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/swinl_maize_rice_val100_test500_search'

nannong_maize_yolo_root = '/public/home/zncui/datasets/nannong-maize-rice/maskdino_data/maize_yolo/images/test'
nannong_maize_retangle_root = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/nannong_maize_rect_infer_for_ppt'
nannong_maize_mask_root = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/nannong_maize_mask_infer_for_ppt'

nannong_rice_yolo_root = '/public/home/zncui/datasets/nannong-maize-rice/maskdino_data/rice_yolo/images/test'
nannong_rice_retangle_root = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/nannong_rice_rect_infer_for_ppt'
nannong_rice_mask_root = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/nannong_rice_mask_infer_for_ppt'


def get_oriHW_images(src_root, tgt_root):
    # 可选：输出目录（不覆盖原图更安全）
    output_root = tgt_root + '_restored'
    os.makedirs(output_root, exist_ok=True)

    # 遍历 swinl 预测结果图片
    for img_name in os.listdir(tgt_root):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        swin_img_path = os.path.join(
            tgt_root, img_name
        )
        yolo_img_path = os.path.join(
            src_root, img_name
        )

        # 如果原始 YOLO 图不存在，跳过
        if not os.path.exists(yolo_img_path):
            print(f'[Warning] 原始图不存在，跳过: {img_name}')
            continue

        # 读取原始 YOLO 图尺寸
        with Image.open(yolo_img_path) as yolo_img:
            orig_w, orig_h = yolo_img.size

        # 读取 swinl 输出图并 resize 回原始尺寸
        with Image.open(swin_img_path) as swin_img:
            restored_img = swin_img.resize(
                (orig_w, orig_h), resample=Image.BILINEAR
            )

        # 保存
        save_path = os.path.join(output_root, img_name)
        restored_img.save(save_path)

    print('✅ 所有可匹配图片已还原至原始宽高')

get_oriHW_images(nannong_maize_yolo_root, nannong_maize_retangle_root)
get_oriHW_images(nannong_maize_yolo_root, nannong_maize_mask_root)
get_oriHW_images(nannong_rice_yolo_root, nannong_rice_retangle_root)
get_oriHW_images(nannong_rice_yolo_root, nannong_rice_mask_root)