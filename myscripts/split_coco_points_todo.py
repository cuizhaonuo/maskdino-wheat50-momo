#!/usr/bin/env python3
"""
把 coco_2cat_points_todo 按每 100 张切成多个独立 COCO 子集, 与 labelme 版本一一对应,
一并放到 labelme_2cat_points_todo 目录下。

每个 chunk 生成一个目录 coco_{start}_{end}_{train|val}/:
    images/                 该 chunk 全部图片
    annotations/instances.json   该 chunk 的 COCO 标注 (images/annotations/categories)

分组顺序与 coco_to_labelme_points_todo.py 完全一致 (json 中图片顺序)。
标注保持原样 (id/bbox/segmentation/category, file_name 改为 basename)。
"""
import json
import os
import shutil
import collections

SRC = "/public/home/zncui/datasets/wheat50/coco_2cat_points_todo"
DST = "/public/home/zncui/datasets/wheat50/labelme_2cat_points_todo"
CHUNK = 100


def build_split(split, suffix):
    with open(os.path.join(SRC, "annotations", f"instances_{split}.json")) as f:
        d = json.load(f)
    categories = d["categories"]
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)

    images = d["images"]
    total = len(images)
    for start in range(0, total, CHUNK):
        chunk = images[start:start + CHUNK]
        end = start + len(chunk) - 1
        root = os.path.join(DST, f"coco_{start}_{end}_{suffix}")
        img_dir = os.path.join(root, "images")
        ann_dir = os.path.join(root, "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        out_images, out_anns = [], []
        for im in chunk:
            base = os.path.basename(im["file_name"])
            shutil.copy2(os.path.join(SRC, split, base), os.path.join(img_dir, base))
            im2 = dict(im)
            im2["file_name"] = base  # 图片就在 images/ 下
            out_images.append(im2)
            out_anns.extend(by_img.get(im["id"], []))

        coco = {"images": out_images, "annotations": out_anns, "categories": categories}
        with open(os.path.join(ann_dir, "instances.json"), "w") as f:
            json.dump(coco, f)
        print(f"  {os.path.basename(root)}: {len(out_images)} 张, {len(out_anns)} 标注")

    print(f"[{split}] 共 {total} 张 -> {(total + CHUNK - 1) // CHUNK} 个 coco 目录")


def main():
    build_split("train2017", "train")
    build_split("val2017", "val")
    print("完成 ->", DST)


if __name__ == "__main__":
    main()
