import os
import json
import random
import shutil
import csv

ANN_PATH = "/public/home/zncui/datasets/maize_rice_dataset/coco/annotations/instances_train2017.json"
IMG_DIR  = "/public/home/zncui/datasets/maize_rice_dataset/coco/train2017"
OUT_DIR  = "/public/home/zncui/datasets/maize_rice_dataset/maize100"

SAMPLE_N = 100
CATEGORY_NAME = "maize"
SEED = 42  # 想每次随机不同就改成 None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

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
    images_by_id = {}
    for img in coco.get("images", []):
        images_by_id[img.get("id")] = img

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

        # 防止 file_name 含子目录
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

    # 6) 过滤 annotations（只保留选中图片的 maize 类标注）
    out_annotations = []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") in chosen_image_ids and ann.get("category_id") == cat_id:
            out_annotations.append(ann)

    # 7) categories：只保留 maize
    out_categories = [
        c for c in coco.get("categories", [])
        if c.get("id") == cat_id
    ]

    # 8) 生成 csv：两列 [image, instance_count]
    # 注意：这里的 instance_count 统计的是 maize 类实例数量（因为 out_annotations 已经过滤为 maize）
    inst_count_by_image_id = {}
    for img_id in chosen_image_ids:
        inst_count_by_image_id[img_id] = 0
    for ann in out_annotations:
        img_id = ann.get("image_id")
        if img_id in inst_count_by_image_id:
            inst_count_by_image_id[img_id] += 1

    csv_path = os.path.join(OUT_DIR, "maize100_image_instance_count.csv")
    # 为了输出稳定：按 file_name 排序
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

    print(f"[Done] Images copied to: {OUT_DIR}")
    print(f"[Done] Annotation saved to: {out_ann_path}")
    print(f"[Done] CSV saved to: {csv_path}")
    print(f"[Stats] images={len(out_images)}, annotations={len(out_annotations)}, categories={len(out_categories)}")

    if missing_files:
        print("\n[Warning] Some files were missing (first 10 shown):")
        for p in missing_files[:10]:
            print("  -", p)
        print(f"[Warning] Missing files count: {len(missing_files)}")

if __name__ == "__main__":
    main()
