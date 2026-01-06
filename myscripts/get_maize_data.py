import os
import json
import random
import shutil
import csv

import numpy as np
import cv2

ANN_PATH = "/public/home/zncui/datasets/maize_rice_dataset/coco/annotations/instances_train2017.json"
IMG_DIR  = "/public/home/zncui/datasets/maize_rice_dataset/coco/train2017"
OUT_DIR  = "/public/home/zncui/datasets/maize_rice_dataset/maize100"

SAMPLE_N = 100
CATEGORY_NAME = "maize"
SEED = 42  # 想每次随机不同就改成 None

VIS_DIRNAME = "vis"
VIS_ALPHA = 0.45  # mask 叠加透明度

# 尝试支持 RLE（iscrowd/segmentation 为 dict 的情况）
try:
    from pycocotools import mask as mask_utils  # type: ignore
    HAS_PYCOCOTOOLS = True
except Exception:
    HAS_PYCOCOTOOLS = False


def get_random_bright_bgr(rng: random.Random, min_val: int = 120):
    """显眼颜色（BGR），保证至少有一个通道很高，避免灰暗。"""
    # 先随机一个亮色，再强制一个通道接近 255
    b = rng.randint(min_val, 255)
    g = rng.randint(min_val, 255)
    r = rng.randint(min_val, 255)
    ch = rng.randint(0, 2)
    if ch == 0:
        b = 255
    elif ch == 1:
        g = 255
    else:
        r = 255
    return (b, g, r)


def decode_segmentation_to_mask(segmentation, h, w):
    """
    segmentation 可以是:
    - list: polygon 格式（可能是 list of lists）
    - dict: RLE 格式（需要 pycocotools）
    返回 uint8 mask: (h,w), 0/1
    """
    if segmentation is None:
        return None

    # polygon: list
    if isinstance(segmentation, list):
        mask = np.zeros((h, w), dtype=np.uint8)
        # COCO polygon: segmentation = [ [x1,y1,x2,y2,...], [...], ... ]
        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    # RLE: dict
    if isinstance(segmentation, dict):
        if not HAS_PYCOCOTOOLS:
            return None
        rle = segmentation
        m = mask_utils.decode(rle)  # (h,w,1) or (h,w)
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m > 0).astype(np.uint8)

    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vis_dir = os.path.join(OUT_DIR, VIS_DIRNAME)
    os.makedirs(vis_dir, exist_ok=True)

    with open(ANN_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 1) 找到 maize 的 category_id
    cat_id = None
    for c in coco.get("categories", []):
        if c.get("name") == CATEGORY_NAME:
            cat_id = c.get("id")
            break
    if cat_id is None:
        raise ValueError(f"Category name '{CATEGORY_NAME}' not found in categories.")

    # 2) 找到包含 maize 标注的 image_id 集合
    maize_image_ids = set()
    for ann in coco.get("annotations", []):
        if ann.get("category_id") == cat_id:
            maize_image_ids.add(ann.get("image_id"))

    if not maize_image_ids:
        raise ValueError("No images found containing maize annotations.")

    maize_image_ids = sorted(maize_image_ids)

    # 3) 随机采样 100 张（不足则全取）
    if SEED is not None:
        random.seed(SEED)

    if len(maize_image_ids) <= SAMPLE_N:
        chosen_image_ids = set(maize_image_ids)
        print(f"[Info] Only {len(maize_image_ids)} maize images found; taking all.")
    else:
        chosen_image_ids = set(random.sample(maize_image_ids, SAMPLE_N))
        print(f"[Info] Sampled {SAMPLE_N} images from {len(maize_image_ids)} maize images.")

    # 4) 构建 image_id -> image 结构
    images_by_id = {img.get("id"): img for img in coco.get("images", [])}

    # 5) 过滤 images + 复制图片
    out_images = []
    missing_files = []
    for img_id in chosen_image_ids:
        img_info = images_by_id.get(img_id)
        if img_info is None:
            continue
        out_images.append(img_info)

        file_name = img_info.get("file_name")
        if not file_name:
            continue

        src_path = os.path.join(IMG_DIR, file_name)
        dst_path = os.path.join(OUT_DIR, file_name)

        if not os.path.isfile(src_path):
            missing_files.append(src_path)
            continue

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)  # 防止 file_name 含子目录
        shutil.copy2(src_path, dst_path)

    # 6) 过滤 annotations（只保留选中图片的 maize 类标注）
    out_annotations = []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") in chosen_image_ids and ann.get("category_id") == cat_id:
            out_annotations.append(ann)

    # 7) categories：只保留 maize
    out_categories = [c for c in coco.get("categories", []) if c.get("id") == cat_id]

    # 8) 生成 csv：两列 [image, instance_count]
    inst_count_by_image_id = {img_id: 0 for img_id in chosen_image_ids}
    for ann in out_annotations:
        img_id = ann.get("image_id")
        if img_id in inst_count_by_image_id:
            inst_count_by_image_id[img_id] += 1

    csv_path = os.path.join(OUT_DIR, "maize100_image_instance_count.csv")
    rows = []
    for img_info in out_images:
        img_id = img_info.get("id")
        file_name = img_info.get("file_name", "")
        rows.append((file_name, inst_count_by_image_id.get(img_id, 0)))
    rows.sort(key=lambda x: x[0])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "instance_count"])
        writer.writerows(rows)

    # 9) 保存新 json 标注
    out_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": out_images,
        "annotations": out_annotations,
        "categories": out_categories,
    }
    out_ann_path = os.path.join(OUT_DIR, "instances_train2017_maize100.json")
    with open(out_ann_path, "w", encoding="utf-8") as f:
        json.dump(out_coco, f, ensure_ascii=False, indent=2)

    # 10) 可视化：每个实例一种显眼颜色，结果保存到 OUT_DIR/vis
    # 先把 annotations 按 image_id 聚合，便于画图
    anns_by_img = {}
    for ann in out_annotations:
        img_id = ann.get("image_id")
        anns_by_img.setdefault(img_id, []).append(ann)

    # 提醒：没有 pycocotools 时，RLE mask 可能无法画出来（仍会画 bbox）
    if not HAS_PYCOCOTOOLS:
        print("[Warning] pycocotools not available: RLE masks (segmentation as dict) may NOT be visualized. Only polygons/bboxes will show.")

    vis_ok = 0
    vis_fail = 0
    for img_info in out_images:
        img_id = img_info.get("id")
        file_name = img_info.get("file_name")
        if not file_name:
            continue

        src_img_path = os.path.join(IMG_DIR, file_name)
        if not os.path.isfile(src_img_path):
            continue

        img_bgr = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            vis_fail += 1
            continue

        h, w = img_bgr.shape[:2]
        overlay = img_bgr.copy()

        anns = anns_by_img.get(img_id, [])
        # 为每个实例生成稳定的随机颜色：SEED + ann_id
        for ann in anns:
            ann_id = ann.get("id", 0)
            rng = random.Random((SEED if SEED is not None else 0) * 1000003 + int(ann_id))

            color = get_random_bright_bgr(rng)

            # 画 mask（优先）
            seg = ann.get("segmentation", None)
            mask01 = decode_segmentation_to_mask(seg, h, w)
            if mask01 is not None:
                # 用 overlay 着色
                overlay[mask01.astype(bool)] = color

                # 画轮廓让边界更清晰
                contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_bgr, contours, -1, color, 2)

            # 画 bbox + id
            bbox = ann.get("bbox", None)
            if bbox is not None and len(bbox) == 4:
                x, y, bw, bh = bbox
                x1, y1 = int(round(x)), int(round(y))
                x2, y2 = int(round(x + bw)), int(round(y + bh))
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img_bgr,
                    f"id:{ann_id}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )

        # alpha blend
        img_vis = cv2.addWeighted(overlay, VIS_ALPHA, img_bgr, 1 - VIS_ALPHA, 0)

        # 保存到 vis：用 basename，避免子目录嵌套太深
        base = os.path.basename(file_name)
        name, ext = os.path.splitext(base)
        if ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            ext = ".jpg"
        out_vis_path = os.path.join(vis_dir, f"{name}_vis{ext}")
        cv2.imwrite(out_vis_path, img_vis)
        vis_ok += 1

    print(f"[Done] Images copied to: {OUT_DIR}")
    print(f"[Done] Annotation saved to: {out_ann_path}")
    print(f"[Done] CSV saved to: {csv_path}")
    print(f"[Done] Visualizations saved to: {vis_dir} (ok={vis_ok}, fail={vis_fail})")
    print(f"[Stats] images={len(out_images)}, annotations={len(out_annotations)}, categories={len(out_categories)}")

    if missing_files:
        print("\n[Warning] Some files were missing (first 10 shown):")
        for p in missing_files[:10]:
            print("  -", p)
        print(f"[Warning] Missing files count: {len(missing_files)}")


if __name__ == "__main__":
    main()
