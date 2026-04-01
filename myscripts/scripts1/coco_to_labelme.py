#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import base64
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    raise ImportError("请先安装 pillow: pip install pillow")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height


def build_category_map(coco_data):
    cat_map = {}
    for cat in coco_data.get("categories", []):
        cid = cat.get("id")
        name = cat.get("name", str(cid))
        cat_map[cid] = name
    return cat_map


def collect_images_from_dir(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_dir = Path(image_dir)
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    by_name = {p.name: p for p in files}
    by_stem = {}
    for p in files:
        if p.stem not in by_stem:
            by_stem[p.stem] = p
    return by_name, by_stem


def parse_coco(coco_json_path):
    data = load_json(coco_json_path)

    # 两种情况：
    # 1) 标准 COCO dict: {images, annotations, categories}
    # 2) 纯结果 list: [{image_id, category_id, segmentation, score}, ...]
    if isinstance(data, list):
        return {
            "images": [],
            "annotations": data,
            "categories": []
        }
    return data


def group_annotations_by_image_id(annotations):
    grouped = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is not None:
            grouped[image_id].append(ann)
    return grouped


def build_image_id_to_file_map(coco_data, by_name, by_stem):
    image_id_to_file = {}

    # 标准 COCO 中有 images 字段
    for img in coco_data.get("images", []):
        image_id = img.get("id")
        file_name = img.get("file_name")
        if image_id is None or not file_name:
            continue

        p = by_name.get(Path(file_name).name)
        if p is None:
            p = by_stem.get(Path(file_name).stem)

        if p is not None:
            image_id_to_file[image_id] = p

    return image_id_to_file


def infer_image_id_to_file_from_stem(grouped_anns, by_stem):
    # 当 coco 是纯 results，没有 images 字段时：
    # 尝试 image_id=123 对应 123.jpg / 123.png
    image_id_to_file = {}
    for image_id in grouped_anns.keys():
        key = str(image_id)
        p = by_stem.get(key)
        if p is not None:
            image_id_to_file[image_id] = p
    return image_id_to_file


def bbox_to_rectangle_shape(bbox, label, score=None, group_id=None):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    x, y, w, h = bbox
    try:
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
    except (TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    return {
        "label": label,
        "points": [[x, y], [x + w, y + h]],
        "group_id": group_id,
        "description": "" if score is None else f"score={score:.6f}",
        "shape_type": "rectangle",
        "flags": {}
    }


def simplify_polygon_points(points, simplify_epsilon_ratio=0.003, max_points=80):
    """Simplify dense polygon points for easier manual editing in LabelMe."""
    if not isinstance(points, list) or len(points) < 3:
        return points

    simplified = points

    # Prefer Douglas-Peucker simplification when OpenCV/Numpy are available.
    if cv2 is not None and np is not None and simplify_epsilon_ratio > 0:
        try:
            contour = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
            perimeter = cv2.arcLength(contour, True)
            epsilon = max(0.5, perimeter * float(simplify_epsilon_ratio))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if approx is not None and len(approx) >= 3:
                simplified = [[float(p[0][0]), float(p[0][1])] for p in approx]
        except Exception:
            simplified = points

    # Hard cap point count to avoid over-dense polygons.
    if isinstance(max_points, int) and max_points >= 3 and len(simplified) > max_points:
        step = max(1, len(simplified) // max_points)
        sampled = simplified[::step]
        if len(sampled) > max_points:
            sampled = sampled[:max_points]
        if len(sampled) >= 3:
            simplified = sampled

    return simplified


def rle_segmentation_to_shapes(
    segmentation,
    label,
    score=None,
    group_id=None,
    simplify_epsilon_ratio=0.003,
    polygon_max_points=80,
):
    shapes = []

    if mask_utils is None or cv2 is None or np is None:
        return shapes

    if not isinstance(segmentation, dict):
        return shapes

    counts = segmentation.get("counts")
    size = segmentation.get("size")
    if counts is None or not isinstance(size, (list, tuple)) or len(size) != 2:
        return shapes

    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        return shapes

    try:
        if isinstance(counts, list):
            # Uncompressed RLE: convert counts list to encoded RLE first.
            rle_objs = mask_utils.frPyObjects(counts, height, width)
            if isinstance(rle_objs, list):
                if not rle_objs:
                    return shapes
                rle = rle_objs[0]
            else:
                rle = rle_objs
        else:
            rle = segmentation

        mask = mask_utils.decode(rle)
        if mask is None:
            return shapes

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        mask = (mask > 0).astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        return shapes

    for contour in contours:
        if contour is None or len(contour) < 3:
            continue

        points = []
        for p in contour:
            x, y = p[0]
            points.append([float(x), float(y)])

        points = simplify_polygon_points(
            points,
            simplify_epsilon_ratio=simplify_epsilon_ratio,
            max_points=polygon_max_points,
        )

        if len(points) < 3:
            continue

        shapes.append({
            "label": label,
            "points": points,
            "group_id": group_id,
            "description": "" if score is None else f"score={score:.6f}",
            "shape_type": "polygon",
            "flags": {}
        })

    return shapes


def segmentation_to_shapes(
    segmentation,
    label,
    score=None,
    group_id=None,
    simplify_epsilon_ratio=0.003,
    polygon_max_points=80,
):
    """
    支持两种格式：
    1) polygon: [[x1,y1,x2,y2,...], [...]]
    2) RLE: {"counts": ..., "size": [h,w]}
    返回 (shapes, mode)
    """
    shapes = []

    if isinstance(segmentation, list):
        for seg in segmentation:
            if not isinstance(seg, list):
                continue
            if len(seg) < 6 or len(seg) % 2 != 0:
                continue

            points = []
            for i in range(0, len(seg), 2):
                x = float(seg[i])
                y = float(seg[i + 1])
                points.append([x, y])

            points = simplify_polygon_points(
                points,
                simplify_epsilon_ratio=simplify_epsilon_ratio,
                max_points=polygon_max_points,
            )
            if len(points) < 3:
                continue

            shape = {
                "label": label,
                "points": points,
                "group_id": group_id,
                "description": "" if score is None else f"score={score:.6f}",
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)

        return shapes, "polygon"

    if isinstance(segmentation, dict):
        shapes = rle_segmentation_to_shapes(
            segmentation=segmentation,
            label=label,
            score=score,
            group_id=group_id,
            simplify_epsilon_ratio=simplify_epsilon_ratio,
            polygon_max_points=polygon_max_points,
        )
        return shapes, "rle"

    return shapes, "unknown"


def build_labelme_json(image_path, shapes):
    width, height = get_image_size(image_path)
    image_data = image_to_base64(image_path)

    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": image_data,
        "imageHeight": height,
        "imageWidth": width
    }


def copy_images_to_output(image_files, output_dir):
    copied = 0
    skipped_same_path = 0

    output_dir = Path(output_dir)
    for src in image_files:
        dst = output_dir / src.name

        # Avoid copying when source and destination are the same file.
        try:
            if src.resolve() == dst.resolve():
                skipped_same_path += 1
                continue
        except FileNotFoundError:
            pass

        shutil.copy2(src, dst)
        copied += 1

    return copied, skipped_same_path


def main():
    parser = argparse.ArgumentParser(description="将 COCO 实例分割结果转换为每张图一个 LabelMe JSON")
    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="COCO 结果 json 路径"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="原图目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出 LabelMe json 目录"
    )
    parser.add_argument(
        "--default_label",
        type=str,
        default="object",
        help="当 categories 中没有类别名时的默认标签名"
    )
    parser.add_argument(
        "--include_score",
        action="store_true",
        help="是否在 description 中写入 score"
    )
    parser.add_argument(
        "--polygon_simplify_epsilon_ratio",
        type=float,
        default=0.003,
        help="多边形简化强度(相对轮廓周长比例)，越大点越少，默认 0.003"
    )
    parser.add_argument(
        "--polygon_max_points",
        type=int,
        default=80,
        help="每个 polygon 最多保留点数，默认 80"
    )
    args = parser.parse_args()

    coco_json_path = Path(args.coco_json)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    coco_data = parse_coco(coco_json_path)
    annotations = coco_data.get("annotations", [])
    cat_map = build_category_map(coco_data)
    grouped_anns = group_annotations_by_image_id(annotations)

    by_name, by_stem = collect_images_from_dir(image_dir)

    # Copy all images to output_dir so LabelMe json and images are colocated.
    all_images = list(by_name.values())
    copied_count, skipped_same_path_count = copy_images_to_output(all_images, output_dir)

    image_id_to_file = build_image_id_to_file_map(coco_data, by_name, by_stem)

    # 若没有 images 字段映射成功，则尝试 image_id -> stem
    if not image_id_to_file:
        image_id_to_file = infer_image_id_to_file_from_stem(grouped_anns, by_stem)

    total_written = 0
    total_missing = 0
    total_rle_ok = 0
    total_bbox_fallback = 0
    total_empty_ann = 0

    for image_id, anns in grouped_anns.items():
        image_path = image_id_to_file.get(image_id)
        if image_path is None:
            print(f"[WARN] 找不到 image_id={image_id} 对应的原图，跳过")
            total_missing += 1
            continue

        shapes = []
        for idx, ann in enumerate(anns):
            category_id = ann.get("category_id")
            label = cat_map.get(category_id, args.default_label)
            score = ann.get("score")
            segmentation = ann.get("segmentation", [])

            ann_shapes, seg_mode = segmentation_to_shapes(
                segmentation=segmentation,
                label=label,
                score=score if args.include_score else None,
                group_id=idx,
                simplify_epsilon_ratio=args.polygon_simplify_epsilon_ratio,
                polygon_max_points=args.polygon_max_points,
            )

            if seg_mode == "rle" and ann_shapes:
                total_rle_ok += 1

            if not ann_shapes:
                rect_shape = bbox_to_rectangle_shape(
                    ann.get("bbox"),
                    label=label,
                    score=score if args.include_score else None,
                    group_id=idx,
                )
                if rect_shape is not None:
                    ann_shapes = [rect_shape]
                    total_bbox_fallback += 1
                else:
                    total_empty_ann += 1

            shapes.extend(ann_shapes)

        labelme_data = build_labelme_json(image_path, shapes)
        out_path = output_dir / f"{image_path.stem}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        total_written += 1
        print(f"[OK] 写入: {out_path}")

    print("=" * 60)
    print(f"已复制图片 {copied_count} 张到: {output_dir}")
    if skipped_same_path_count:
        print(f"跳过同路径图片 {skipped_same_path_count} 张")
    if mask_utils is None or cv2 is None or np is None:
        print("[WARN] 未安装 pycocotools/cv2/numpy，RLE 分割将回退为 bbox 或被跳过")
    print(f"RLE 转 polygon 成功 {total_rle_ok} 个标注")
    print(f"bbox 回退写入 {total_bbox_fallback} 个标注")
    print(f"无法转换并跳过 {total_empty_ann} 个标注")
    print(
        "polygon 简化参数: "
        f"epsilon_ratio={args.polygon_simplify_epsilon_ratio}, "
        f"max_points={args.polygon_max_points}"
    )
    print(f"完成，总共写入 {total_written} 个 LabelMe json")
    print(f"缺失原图映射 {total_missing} 个")


if __name__ == "__main__":
    main()

