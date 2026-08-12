#!/usr/bin/env python3
"""
把 coco_2cat_points_todo 转成 LabelMe 格式 (每图一个 json), 便于人工补点标注。

- 每个 split (train/val) 的图片按 json 中顺序, 每 100 张放一个子目录:
    labelme_0_99_train, labelme_100_199_train, ...   (train2017)
    labelme_0_99_val,   labelme_100_199_val,   ...   (val2017)
- 图片与其 LabelMe json 放在同一子目录, 每张图一个 json。
- COCO polygon segmentation -> LabelMe polygon shape, label = 类别名 (wheat / wheat_2)。
  多部件实例用同一 group_id 关联; 单部件 group_id=null。
"""
import json
import os
import shutil
import collections

SRC = "/public/home/zncui/datasets/wheat50/coco_2cat_points_todo"
DST = "/public/home/zncui/datasets/wheat50/labelme_2cat_points_todo"
CHUNK = 100
LABELME_VERSION = "5.5.0"


def to_pairs(flat):
    return [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]


def build_split(split, suffix):
    with open(os.path.join(SRC, "annotations", f"instances_{split}.json")) as f:
        d = json.load(f)
    catname = {c["id"]: c["name"] for c in d["categories"]}

    # image_id -> annotations
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)

    images = d["images"]  # 保持 json 顺序
    total = len(images)
    gid_counter = 0  # 全局递增的实例 group_id (仅多部件用)

    for start in range(0, total, CHUNK):
        chunk = images[start:start + CHUNK]
        end = start + len(chunk) - 1
        sub = os.path.join(DST, f"labelme_{start}_{end}_{suffix}")
        os.makedirs(sub, exist_ok=True)

        for im in chunk:
            base = os.path.basename(im["file_name"])
            # 拷贝图片
            shutil.copy2(os.path.join(SRC, split, base), os.path.join(sub, base))

            shapes = []
            for a in by_img.get(im["id"], []):
                seg = a["segmentation"]  # list of polygons (flat coord lists)
                multipart = len(seg) > 1
                gid = None
                if multipart:
                    gid_counter += 1
                    gid = gid_counter
                for poly in seg:
                    if len(poly) < 6:  # 少于3个点的退化多边形跳过
                        continue
                    shapes.append({
                        "label": catname[a["category_id"]],
                        "points": to_pairs(poly),
                        "group_id": gid,
                        "shape_type": "polygon",
                        "flags": {},
                    })

            lm = {
                "version": LABELME_VERSION,
                "flags": {},
                "shapes": shapes,
                "imagePath": base,
                "imageData": None,
                "imageHeight": im["height"],
                "imageWidth": im["width"],
            }
            stem = os.path.splitext(base)[0]
            with open(os.path.join(sub, stem + ".json"), "w") as f:
                json.dump(lm, f, ensure_ascii=False)

        print(f"  {os.path.basename(sub)}: {len(chunk)} 张")

    print(f"[{split}] 共 {total} 张 -> {(total + CHUNK - 1) // CHUNK} 个目录")


def main():
    os.makedirs(DST, exist_ok=True)
    build_split("train2017", "train")
    build_split("val2017", "val")
    print("完成 ->", DST)


if __name__ == "__main__":
    main()
