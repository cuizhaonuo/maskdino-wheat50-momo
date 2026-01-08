import os
import json
import random
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path


SRC_ROOT = Path("/public/home/zncui/datasets/nannong-maize-rice")
OUT_IMAGES = SRC_ROOT / "total_images"
OUT_LABELS = SRC_ROOT / "total_labels"

COCO_ROOT = SRC_ROOT / "coco"
YOLO_ROOT = SRC_ROOT / "yolo"

# 支持的扩展名（大小写都匹配）
IMG_EXTS = {".jpg", ".jpeg"}
XML_EXTS = {".xml"}


def copy_files(overwrite: bool = True):
    """
    将 SRC_ROOT 下所有 XML / JPG(JPEG) 递归复制到：
      - total_labels/  (仅保留原文件名)
      - total_images/  (仅保留原文件名)

    注意：不同子目录可能存在同名文件 -> 会发生覆盖（如果 overwrite=True）。
    """
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {SRC_ROOT}")

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    img_cnt = 0
    xml_cnt = 0
    skipped = 0
    overwritten = 0
    name_conflicts = 0

    for p in SRC_ROOT.rglob("*"):
        if not p.is_file():
            continue

        # 跳过输出目录自身，避免递归把复制后的文件再次扫描进去
        try:
            rel = p.relative_to(SRC_ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in {OUT_IMAGES.name, OUT_LABELS.name}:
            continue

        suffix = p.suffix.lower()

        # XML
        if suffix in XML_EXTS:
            dst = OUT_LABELS / p.name  # 仅保留原文件名
            if dst.exists():
                name_conflicts += 1
                if not overwrite:
                    continue
                overwritten += 1
            shutil.copy2(p, dst)
            xml_cnt += 1
            continue

        # JPG / JPEG（大小写不敏感）
        if suffix in IMG_EXTS:
            dst = OUT_IMAGES / p.name  # 仅保留原文件名
            if dst.exists():
                name_conflicts += 1
                if not overwrite:
                    continue
                overwritten += 1
            shutil.copy2(p, dst)
            img_cnt += 1
            continue

        skipped += 1

    print("Done.")
    print(f"Copied XML : {xml_cnt} -> {OUT_LABELS}")
    print(f"Copied JPG : {img_cnt} -> {OUT_IMAGES}")
    print(f"Skipped other files: {skipped}")
    print(f"Name conflicts encountered: {name_conflicts}")
    print(f"Overwritten files: {overwritten} (overwrite={overwrite})")


def clean_files():
    """
    基于“文件名（去掉最后一个后缀）”检查 total_images 与 total_labels 是否一一对应。
    例：0001.jpg <-> 0001.xml
    """
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    total_images = set(Path(f).stem for f in os.listdir(OUT_IMAGES) if Path(f).suffix.lower() in IMG_EXTS)
    total_labels = set(Path(f).stem for f in os.listdir(OUT_LABELS) if Path(f).suffix.lower() in XML_EXTS)

    images_without_labels = total_images - total_labels
    labels_without_images = total_labels - total_images

    print("Images without labels:", len(images_without_labels))
    print("Labels without images:", len(labels_without_images))

    # 将没有对应标签的图片删除
    for img_stem in images_without_labels:
        img_path = OUT_IMAGES / f"{img_stem}.JPG"
        if img_path.exists():
            img_path.unlink()
            print(f"Deleted image without label: {img_path}")
            break
    # 将没有对应图片的标签删除
    for label_stem in labels_without_images:
        label_path = OUT_LABELS / f"{label_stem}.xml"
        if label_path.exists():
            label_path.unlink()
            print(f"Deleted label without image: {label_path}")
            break




def _parse_voc_xml(xml_path: Path):
    """
    解析 VOC XML，返回:
      width, height, objects: [{'name': str, 'bbox': [xmin,ymin,xmax,ymax]}]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"No <size> in {xml_path}")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        if name is None or bnd is None:
            continue
        xmin = int(float(bnd.findtext("xmin")))
        ymin = int(float(bnd.findtext("ymin")))
        xmax = int(float(bnd.findtext("xmax")))
        ymax = int(float(bnd.findtext("ymax")))
        objects.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})

    return width, height, objects


def _collect_pairs(images_dir: Path, labels_dir: Path, img_exts={".jpg", ".jpeg"}, xml_exts={".xml"}):
    """
    依据 stem 建立一一对应 (image_path, xml_path)
    """
    img_map = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in img_exts:
            img_map[p.stem] = p

    xml_map = {}
    for p in labels_dir.iterdir():
        if p.is_file() and p.suffix.lower() in xml_exts:
            xml_map[p.stem] = p

    common = sorted(set(img_map.keys()) & set(xml_map.keys()))
    missing_img = sorted(set(xml_map.keys()) - set(img_map.keys()))
    missing_xml = sorted(set(img_map.keys()) - set(xml_map.keys()))

    if missing_img or missing_xml:
        print(f"[WARN] pairs not perfect: missing_img={len(missing_img)} missing_xml={len(missing_xml)}")
        # 只用 common，保证一一对应
    pairs = [(img_map[k], xml_map[k]) for k in common]
    return pairs


def _split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed=42):
    ratios = [train_ratio, val_ratio, test_ratio]
    s = sum(ratios)
    if s <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")
    # 允许你传 0.0/0.1/0.9 这类，做归一化更稳
    train_ratio, val_ratio, test_ratio = [r / s for r in ratios]

    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # 末尾都给 test，确保总数不丢
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]
    return train, val, test


def create_coco(train_ratio, val_ratio, test_ratio, seed=42):
    """
    生成 COCO 检测格式：
    coco/
      train2017/  val2017/  test2017/
      annotations/
        instances_train2017.json
        instances_val2017.json
        instances_test2017.json
    """
    COCO_ROOT.mkdir(parents=True, exist_ok=True)
    (COCO_ROOT / "annotations").mkdir(parents=True, exist_ok=True)
    for split in ["train2017", "val2017", "test2017"]:
        (COCO_ROOT / split).mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(OUT_IMAGES, OUT_LABELS, IMG_EXTS, XML_EXTS)
    train_pairs, val_pairs, test_pairs = _split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed=seed)

    # 1) 先扫一遍类别集合
    cat_names = set()
    for _, xml_path in pairs:
        _, _, objs = _parse_voc_xml(xml_path)
        for o in objs:
            cat_names.add(o["name"])
    cat_names = sorted(cat_names)
    name_to_catid = {n: i + 1 for i, n in enumerate(cat_names)}
    categories = [{"id": name_to_catid[n], "name": n, "supercategory": "none"} for n in cat_names]

    def build_coco_for_split(split_name: str, split_pairs):
        images = []
        annotations = []
        img_id = 1
        ann_id = 1

        split_dir = COCO_ROOT / split_name
        for img_path, xml_path in split_pairs:
            w, h, objs = _parse_voc_xml(xml_path)

            # 拷贝图片到 split 目录（保持原文件名）
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
                # clamp，避免越界
                xmin = max(0, min(xmin, w - 1))
                ymin = max(0, min(ymin, h - 1))
                xmax = max(0, min(xmax, w - 1))
                ymax = max(0, min(ymax, h - 1))
                bw = max(0, xmax - xmin)
                bh = max(0, ymax - ymin)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": name_to_catid[o["name"]],
                    "bbox": [xmin, ymin, bw, bh],  # COCO: [x,y,w,h]
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "segmentation": [],  # 检测任务可为空
                })
                ann_id += 1

            img_id += 1

        coco = {
            "info": {"description": "VOC->COCO converted dataset"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        return coco

    split_map = {
        "train2017": train_pairs,
        "val2017": val_pairs,
        "test2017": test_pairs,
    }

    for split_name, split_pairs in split_map.items():
        coco_dict = build_coco_for_split(split_name, split_pairs)
        out_json = COCO_ROOT / "annotations" / f"instances_{split_name}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=2)
        print(f"[COCO] {split_name}: images={len(coco_dict['images'])}, anns={len(coco_dict['annotations'])} -> {out_json}")


def create_yolo(train_ratio, val_ratio, test_ratio, seed=42, resize=None):
    """
    生成 YOLO 检测格式：
    - resize=None        -> yolo/
    - resize=(1024,1024)-> yolo_1024x1024/

    yolo_xxx/
      images/train|val|test/
      labels/train|val|test/
      data.yaml
    """
    from PIL import Image

    # ------------------------------------------------
    # 1. 根据 resize 动态决定 YOLO_ROOT
    # ------------------------------------------------
    if resize is None:
        yolo_root = SRC_ROOT / "yolo"
        resize_tag = "orig"
    else:
        assert isinstance(resize, tuple) and len(resize) == 2
        new_w, new_h = resize
        yolo_root = SRC_ROOT / f"yolo_{new_w}x{new_h}"
        resize_tag = f"{new_w}x{new_h}"

    # ------------------------------------------------
    # 2. 创建目录结构
    # ------------------------------------------------
    yolo_root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # 3. 收集 & 划分数据
    # ------------------------------------------------
    pairs = _collect_pairs(OUT_IMAGES, OUT_LABELS, IMG_EXTS, XML_EXTS)
    train_pairs, val_pairs, test_pairs = _split_pairs(
        pairs, train_ratio, val_ratio, test_ratio, seed=seed
    )

    # ------------------------------------------------
    # 4. 类别映射（YOLO class id 从 0 开始）
    # ------------------------------------------------
    cat_names = set()
    for _, xml_path in pairs:
        _, _, objs = _parse_voc_xml(xml_path)
        for o in objs:
            cat_names.add(o["name"])
    cat_names = sorted(cat_names)
    name_to_classid = {n: i for i, n in enumerate(cat_names)}

    # ------------------------------------------------
    # 5. 图片保存函数
    # ------------------------------------------------
    def save_image(src: Path, dst: Path):
        if resize is None:
            if not dst.exists():
                shutil.copy2(src, dst)
            return True

        new_w, new_h = resize
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im = im.resize((new_w, new_h), Image.BILINEAR)
                im.save(dst, quality=95, subsampling=0)
            return True
        except Exception as e:
            print(f"[BAD IMAGE] {src} -> {e}")
            return False

    # ------------------------------------------------
    # 6. 写每个 split
    # ------------------------------------------------
    def write_split(split_name: str, split_pairs):
        img_out = yolo_root / "images" / split_name
        lab_out = yolo_root / "labels" / split_name

        n_img, n_lab = 0, 0

        for img_path, xml_path in split_pairs:
            w, h, objs = _parse_voc_xml(xml_path)

            # image
            dst_img = img_out / img_path.name
            ok = save_image(img_path, dst_img)
            if not ok:
                # 失败就跳过这个样本，避免生成不对应的标签
                continue
            n_img += 1

            # label
            txt_path = lab_out / f"{img_path.stem}.txt"
            lines = []

            for o in objs:
                xmin, ymin, xmax, ymax = o["bbox"]

                xmin = max(0, min(xmin, w - 1))
                ymin = max(0, min(ymin, h - 1))
                xmax = max(0, min(xmax, w - 1))
                ymax = max(0, min(ymax, h - 1))

                bw = max(0.0, xmax - xmin)
                bh = max(0.0, ymax - ymin)
                cx = xmin + bw / 2.0
                cy = ymin + bh / 2.0

                # YOLO 归一化（纯 resize 不需要改）
                cx /= w
                cy /= h
                bw /= w
                bh /= h

                cls = name_to_classid[o["name"]]
                lines.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

            n_lab += 1

        print(f"[YOLO-{resize_tag}] {split_name}: images={n_img}, labels={n_lab}")

    write_split("train", train_pairs)
    write_split("val", val_pairs)
    write_split("test", test_pairs)

    # ------------------------------------------------
    # 7. data.yaml
    # ------------------------------------------------
    yaml_path = yolo_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {yolo_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(cat_names)}\n")
        f.write("names:\n")
        for i, n in enumerate(cat_names):
            f.write(f"  {i}: {n}\n")

    print(f"[YOLO-{resize_tag}] data.yaml -> {yaml_path}")




if __name__ == "__main__":
    # copy_files(overwrite=True)   # 若不想覆盖：overwrite=False
    # clean_files()
    # create_coco(train_ratio=0.0, val_ratio=0.1, test_ratio=0.9)
    create_yolo(train_ratio=0.0, val_ratio=0.1, test_ratio=0.9, resize=(800, 800))
