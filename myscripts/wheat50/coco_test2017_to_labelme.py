# 将 coco_2cat_with_points/test2017 的 COCO 伪标注转为 labelme 格式供人工修订。
# 与历史版（clean-wheat50-model-1.1）的区别：实例除 polygon 外还带 wheat_2 的小穗点，
# 小穗点导出为 shape_type=point 的 shape，与所属实例的 polygon 共用 group_id(=ann id)，
# 这样在 labelme 里增删点即可修订小穗数，不必动密集 polygon。
# 每张图片在输出目录下生成同名 .jpg + .json（json 内嵌 imageData，可单独搬走修订）。
import base64
import json
import os
import shutil
from collections import defaultdict

COCO_JSON = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points/annotations/instances_test2017.json"
IMG_DIR = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points/test2017"
OUT_DIR = "/public/home/zncui/datasets/wheat50/coco_2cat_with_points/labelme_test2017"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    coco = json.load(open(COCO_JSON))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    n_done = 0
    for img in coco["images"]:
        file_name = img["file_name"]
        src = os.path.join(IMG_DIR, file_name)
        dst = os.path.join(OUT_DIR, file_name)
        shutil.copy2(src, dst)
        with open(src, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("ascii")

        shapes = []
        for a in anns_by_img.get(img["id"], []):
            label = cat_name[a["category_id"]]
            # group_id 统一用 ann id：多段 polygon 之间、polygon 与小穗点之间都靠它关联
            group_id = a["id"]
            for poly in a["segmentation"]:
                points = [[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)]
                if len(points) < 3:
                    continue
                shapes.append({
                    "label": label,
                    "points": points,
                    "group_id": group_id,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                })
            for x, y in a.get("points", []):
                shapes.append({
                    "label": "spikelet",
                    "points": [[x, y]],
                    "group_id": group_id,
                    "description": "",
                    "shape_type": "point",
                    "flags": {},
                })

        labelme = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": file_name,
            "imageData": image_data,
            "imageHeight": img["height"],
            "imageWidth": img["width"],
        }
        with open(os.path.splitext(dst)[0] + ".json", "w") as f:
            json.dump(labelme, f)
        n_done += 1

    print(f"converted {n_done} images -> {OUT_DIR}")


if __name__ == "__main__":
    main()
