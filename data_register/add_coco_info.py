import json
import os

# 你的 JSON 文件路径列表
json_files = [
    "/public/home/zncui/datasets/GWHD/gwhd_coco/annotations/instances_train2017.json",
    "/public/home/zncui/datasets/GWHD/gwhd_coco/annotations/instances_val2017.json",
    "/public/home/zncui/datasets/GWHD/gwhd_coco/annotations/instances_test2017.json"
]

for file_path in json_files:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"Processing {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 检查并添加 info 字段
    if 'info' not in data:
        data['info'] = {}  # 赋予空字典即可满足 pycocotools 依赖
        # 如果 pycocotools 严格要求内部字段，可以换成:
        # data['info'] = {"description": "", "url": "", "version": "", "year": 2021, "contributor": "", "date_created": ""}
        print("  Added 'info' key.")

        # 覆写保存
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print("  Saved.")
    else:
        print("  'info' key already exists.")
