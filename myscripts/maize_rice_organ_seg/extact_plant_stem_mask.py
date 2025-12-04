import os
import cv2
import numpy as np

# ================== 配置 ==================
img_dir = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset/maize_stem"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def get_red_mask_hsv(img_bgr):
    """
    在 HSV 空间中提取红色区域，返回 0/255 二值 mask。
    这里假设茎为接近纯红 (255,0,0)，使用较严格阈值。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 红色通常分为两个区间，需要拼起来
    # 可根据实际情况略微调整阈值（尤其是 S/V）
    lower_red1 = np.array([0, 150, 150], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, 150, 150], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    # 确保是干净的 0/255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def process_image(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[跳过] 无法读取图像: {img_path}")
        return

    # ================== 提取所有红色区域 ==================
    red_mask = get_red_mask_hsv(img_bgr)

    # ================== 保存 mask ==================
    base, _ = os.path.splitext(os.path.basename(img_path))
    mask_path = os.path.join(os.path.dirname(img_path), f"{base}_stem_mask.png")
    cv2.imwrite(mask_path, red_mask)
    print(f"[保存] mask: {mask_path}")

    # ================== 叠加蓝色可视化 ==================
    # 题目要求：新 mask 区域绘制成 RGB(0,0,255) 蓝色
    # OpenCV 是 BGR，因此设为 (255, 0, 0)
    overlay_bgr = img_bgr.copy()
    overlay_bgr[red_mask == 255] = (255, 0, 0)

    overlay_path = os.path.join(os.path.dirname(img_path), f"{base}_stem_overlay.png")
    cv2.imwrite(overlay_path, overlay_bgr)
    print(f"[保存] overlay: {overlay_path}")


def main():
    for fname in sorted(os.listdir(img_dir)):
        if fname.lower().endswith(IMG_EXTS):
            img_path = os.path.join(img_dir, fname)
            process_image(img_path)


if __name__ == "__main__":
    main()
