# 将 coco_2cat/test2017 的 COCO 实例标注转为 labelme 格式，
# 每张图片在输出目录下生成一个同名 .json，格式与 image_json_labelme 中已有文件保持一致
# （label 用类别名、shape_type=polygon、内嵌 imageData）。
import base64
import json
import os
from collections import defaultdict

COCO_JSON = "/public/home/zncui/datasets/wheat50/coco_2cat/annotations/instances_test2017.json"
OUT_DIR = "/public/home/zncui/datasets/wheat50/total_images/image_json_labelme_new/val"

def main():
    coco = json.load(open(COCO_JSON))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    n_done = 0
    for img in coco["images"]:
        file_name = img["file_name"]
        img_path = os.path.join(OUT_DIR, file_name)
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("ascii")

        shapes = []
        for a in anns_by_img.get(img["id"], []):
            polys = a["segmentation"]
            # 多段 polygon 的实例用 group_id 关联为同一目标
            group_id = a["id"] if len(polys) > 1 else None
            for poly in polys:
                points = [[poly[i], poly[i + 1]] for i in range(0, len(poly), 2)]
                if len(points) < 3:
                    continue
                shapes.append({
                    "label": cat_name[a["category_id"]],
                    "points": points,
                    "group_id": group_id,
                    "description": "",
                    "shape_type": "polygon",
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
        out_path = os.path.splitext(img_path)[0] + ".json"
        with open(out_path, "w") as f:
            json.dump(labelme, f)
        n_done += 1

    print(f"converted {n_done} images -> {OUT_DIR}")

if __name__ == "__main__":
    main()
