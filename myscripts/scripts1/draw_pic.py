import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as MplPath

root = Path("/public/home/zncui/datasets/wheat50/total_images")
img_dir = root / "results"
coco_json_path = root / "annotations_results_infered_poly.json"
select_dir = root / "select_images"
out_dir = root / "vis_results_infered_poly"
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- utils ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def load_labelme_points(labelme_json: Path) -> np.ndarray:
    """从 labelme json 读取所有点，返回 shape=(N,2) 的 (x,y)"""
    data = json.loads(labelme_json.read_text(encoding="utf-8"))
    pts = []
    for sh in data.get("shapes", []):
        for p in sh.get("points", []):
            if len(p) >= 2:
                pts.append([float(p[0]), float(p[1])])
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)

def coco_seg_to_polys(seg, h=None, w=None):
    """
    支持：
    - polygon: list[list[float]]
    - RLE: dict(counts, size)
    返回 list of polygons (each Nx2)
    """
    polys = []

    # polygon
    if isinstance(seg, list):
        for one in seg:
            if isinstance(one, list) and len(one) >= 6:
                arr = np.asarray(one, dtype=np.float32).reshape(-1, 2)
                polys.append(arr)
        return polys

    # RLE
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        from pycocotools import mask as mask_utils
        m = mask_utils.decode(seg)  # (H,W) uint8 (0/1)
        m = (m * 255).astype(np.uint8)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) >= 3:
                c = c.reshape(-1, 2).astype(np.float32)
                polys.append(c)
        return polys

    return polys

def count_points_in_poly(points_xy: np.ndarray, poly_xy: np.ndarray, radius=0.0) -> int:
    """统计 polygon 内有多少个点（radius>0 会把边界附近点也算进去）"""
    if points_xy.shape[0] == 0:
        return 0
    path = MplPath(poly_xy)
    inside = path.contains_points(points_xy, radius=radius)
    return int(np.sum(inside))

def draw_poly_outline(img_bgr, poly_xy, color_bgr, thickness=2):
    """只画 polygon 轮廓（内部透明）"""
    pts = np.round(poly_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        img_bgr,
        [pts],
        isClosed=True,
        color=color_bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

# ---------- load coco ----------
coco = json.loads(coco_json_path.read_text(encoding="utf-8"))

# image_id -> file_name (可能带子目录)
id2name = {im["id"]: im["file_name"] for im in coco.get("images", [])}
# 关键：用 basename 做 key，避免目录不一致导致匹配不到
id2base = {img_id: Path(fname).name for img_id, fname in id2name.items()}

# basename -> list[annotation]
name2anns = {}
for ann in coco.get("annotations", []):
    base = id2base.get(ann["image_id"])
    if base is None:
        continue
    name2anns.setdefault(base, []).append(ann)

# ---------- process selected images ----------
selected_imgs = sorted([p for p in select_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

# 颜色规则：
# >=2 点：绿色；1 点：红色；0 点：蓝色（BGR）
BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# 边界附近点是否算进去（可按需要调大，比如 2.0）
POINT_RADIUS = 0.0

for sel_img in selected_imgs:
    img_name = sel_img.name
    src_img_path = img_dir / img_name
    if not src_img_path.exists():
        print(f"[skip] image not found in results: {src_img_path}")
        continue

    # labelme points
    labelme_json = sel_img.with_suffix(".json")
    if not labelme_json.exists():
        print(f"[skip] labelme json not found: {labelme_json}")
        continue
    points_xy = load_labelme_points(labelme_json)

    # read image
    img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[skip] cannot read image: {src_img_path}")
        continue

    anns = name2anns.get(img_name, [])
    if len(anns) == 0:
        print(f"[warn] no coco annotations for: {img_name}")

    n_green = 0
    n_red = 0
    n_blue = 0

    H, W = img.shape[:2]
    for ann in anns:
        seg = ann.get("segmentation", None)
        polys = coco_seg_to_polys(seg, H, W)
        if len(polys) == 0:
            continue

        for poly in polys:
            num_points = count_points_in_poly(points_xy, poly, radius=POINT_RADIUS)

            if num_points >= 2:
                draw_poly_outline(img, poly, GREEN)
                n_green += 1
            elif num_points == 1:
                draw_poly_outline(img, poly, RED)
                n_red += 1
            else:
                draw_poly_outline(img, poly, BLUE)
                n_blue += 1

    out_path = out_dir / img_name
    cv2.imwrite(str(out_path), img)
    print(f"[ok] {img_name} -> {out_path} (green={n_green}, red={n_red}, blue={n_blue})")

print("Done.")