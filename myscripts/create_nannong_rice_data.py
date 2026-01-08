# create_ricespike_uav_data.py
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ============ 你只需要改这里 ============
DATA_ROOT = Path("/home/ubuntu/文档/xwechat_files/wxid_4lzw0utprwyt22_8011/msg/file/2026-01/ricespike_uav标注好的备份/ricespike_uav")

COCO_ROOT = DATA_ROOT / "coco"
YOLO_ROOT = DATA_ROOT / "yolo"
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # 你的数据是 JPG，但这里做宽松支持
JSON_EXTS = {".json"}
# ======================================


def parse_labelme(json_path: Path) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    解析 LabelMe JSON，返回:
      width, height, objects: [{'name': label, 'bbox': [xmin,ymin,xmax,ymax], 'segmentation': [[x1,y1,x2,y2,...]] or []}]
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    w = int(data.get("imageWidth", 0))
    h = int(data.get("imageHeight", 0))
    if w <= 0 or h <= 0:
        raise ValueError(f"imageWidth/imageHeight missing or invalid in {json_path}")

    objects = []
    for shp in data.get("shapes", []):
        label = shp.get("label")
        points = shp.get("points", [])
        shape_type = shp.get("shape_type")  # 可能是 'polygon'/'rectangle'/None

        if not label or not points:
            continue

        # points: [[x,y], [x,y], ...]
        xs = [float(p[0]) for p in points if len(p) >= 2]
        ys = [float(p[1]) for p in points if len(p) >= 2]
        if len(xs) == 0 or len(ys) == 0:
            continue

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # clamp to image range
        xmin = max(0.0, min(xmin, w - 1.0))
        ymin = max(0.0, min(ymin, h - 1.0))
        xmax = max(0.0, min(xmax, w - 1.0))
        ymax = max(0.0, min(ymax, h - 1.0))

        # COCO segmentation（仅当 polygon 且点数>=3 时写入）
        segmentation = []
        if (shape_type == "polygon" or shape_type is None) and len(points) >= 3:
            poly = []
            for x, y in points:
                x = max(0.0, min(float(x), w - 1.0))
                y = max(0.0, min(float(y), h - 1.0))
                poly.extend([x, y])
            # COCO 要求 polygon 至少 6 个数（3个点）
            if len(poly) >= 6:
                segmentation = [poly]

        objects.append({
            "name": str(label),
            "bbox": [xmin, ymin, xmax, ymax],
            "segmentation": segmentation,  # 可能为空
        })

    return w, h, objects


def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """
    依据 stem 进行 jpg/json 配对
    """
    imgs: Dict[str, Path] = {}
    anns: Dict[str, Path] = {}

    for p in root.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in IMG_EXTS:
            imgs[p.stem] = p
        elif suf in JSON_EXTS:
            anns[p.stem] = p

    common = sorted(set(imgs.keys()) & set(anns.keys()))
    missing_img = sorted(set(anns.keys()) - set(imgs.keys()))
    missing_ann = sorted(set(imgs.keys()) - set(anns.keys()))

    if missing_img or missing_ann:
        print(f"[WARN] pairs not perfect: missing_img={len(missing_img)} missing_ann={len(missing_ann)}")
        if missing_img:
            print("  examples missing image:", missing_img[:10])
        if missing_ann:
            print("  examples missing json :", missing_ann[:10])

    return [(imgs[k], anns[k]) for k in common]


def split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed=SEED):
    s = train_ratio + val_ratio + test_ratio
    if s <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")
    train_ratio, val_ratio, test_ratio = train_ratio / s, val_ratio / s, test_ratio / s

    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]
    return train, val, test


def create_coco(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    生成 COCO（bbox + 可选 polygon segmentation）
    目录结构：
      coco/
        train2017/ val2017/ test2017/
        annotations/
          instances_train2017.json
          instances_val2017.json
          instances_test2017.json
    """
    COCO_ROOT.mkdir(parents=True, exist_ok=True)
    (COCO_ROOT / "annotations").mkdir(parents=True, exist_ok=True)
    for split in ["train2017", "val2017", "test2017"]:
        (COCO_ROOT / split).mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(DATA_ROOT)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, train_ratio, val_ratio, test_ratio)

    # 统计类别
    cat_names = set()
    for _, json_path in pairs:
        w, h, objs = parse_labelme(json_path)
        for o in objs:
            cat_names.add(o["name"])
    cat_names = sorted(cat_names)
    name_to_catid = {n: i + 1 for i, n in enumerate(cat_names)}
    categories = [{"id": name_to_catid[n], "name": n, "supercategory": "none"} for n in cat_names]

    def build(split_name: str, split_pairs: List[Tuple[Path, Path]]):
        images = []
        annotations = []
        img_id = 1
        ann_id = 1

        split_dir = COCO_ROOT / split_name
        for img_path, json_path in split_pairs:
            w, h, objs = parse_labelme(json_path)

            dst_img = split_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            })

            for o in objs:
                xmin, ymin, xmax, ymax = o["bbox"]
                bw = max(0.0, xmax - xmin)
                bh = max(0.0, ymax - ymin)

                seg = o.get("segmentation", [])
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": name_to_catid[o["name"]],
                    "bbox": [xmin, ymin, bw, bh],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "segmentation": seg if seg else [],  # polygon 有就写，没有就空
                })
                ann_id += 1

            img_id += 1

        return {
            "info": {"description": "LabelMe->COCO converted dataset"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    split_map = {
        "train2017": train_pairs,
        "val2017": val_pairs,
        "test2017": test_pairs,
    }

    for split_name, split_pairs in split_map.items():
        coco_dict = build(split_name, split_pairs)
        out_json = COCO_ROOT / "annotations" / f"instances_{split_name}.json"
        out_json.write_text(json.dumps(coco_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[COCO] {split_name}: images={len(coco_dict['images'])}, anns={len(coco_dict['annotations'])} -> {out_json}")


def create_yolo(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    生成 YOLO detection（bbox）
    目录结构：
      yolo/
        images/train|val|test/
        labels/train|val|test/
        data.yaml
    每个 label txt：class cx cy w h（归一化）
    """
    YOLO_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(DATA_ROOT)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, train_ratio, val_ratio, test_ratio)

    # 类别（YOLO class id 从 0 开始）
    cat_names = set()
    for _, json_path in pairs:
        w, h, objs = parse_labelme(json_path)
        for o in objs:
            cat_names.add(o["name"])
    cat_names = sorted(cat_names)
    name_to_classid = {n: i for i, n in enumerate(cat_names)}

    def write_split(split_name: str, split_pairs: List[Tuple[Path, Path]]):
        img_out = YOLO_ROOT / "images" / split_name
        lab_out = YOLO_ROOT / "labels" / split_name

        n_img, n_lab = 0, 0
        for img_path, json_path in split_pairs:
            w, h, objs = parse_labelme(json_path)

            dst_img = img_out / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            n_img += 1

            txt_path = lab_out / f"{img_path.stem}.txt"
            lines = []
            for o in objs:
                xmin, ymin, xmax, ymax = o["bbox"]
                bw = max(0.0, xmax - xmin)
                bh = max(0.0, ymax - ymin)
                cx = xmin + bw / 2.0
                cy = ymin + bh / 2.0

                # normalize
                cx /= float(w)
                cy /= float(h)
                bw /= float(w)
                bh /= float(h)

                cls = name_to_classid[o["name"]]
                lines.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")

            txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_lab += 1

        print(f"[YOLO] {split_name}: images={n_img}, labels={n_lab}")

    write_split("train", train_pairs)
    write_split("val", val_pairs)
    write_split("test", test_pairs)

    # data.yaml（简单写，不依赖 pyyaml）
    yaml_path = YOLO_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {YOLO_ROOT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(cat_names)}\n")
        f.write("names:\n")
        for i, n in enumerate(cat_names):
            f.write(f"  {i}: {n}\n")
    print(f"[YOLO] data.yaml -> {yaml_path}")


if __name__ == "__main__":
    # 例子：你也可以改比例，比如 train=0.0, val=0.1, test=0.9
    create_coco(train_ratio=0.0, val_ratio=0.1, test_ratio=0.9)
    create_yolo(train_ratio=0.0, val_ratio=0.1, test_ratio=0.9)
