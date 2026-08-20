"""合并小穗点标注并重新划分 train/val。

背景：`coco_2cat_with_points` 只剩 train2017（301 图，历史上 val 被 merge_val_into_train.py
并了进去），而 Stage2 需要一个验证集；同时新到了 100 张已标注小穗点的图
（labelme_2cat_points_todo/xianyu/...）。本脚本把两者合并成 401 图，再按 8:2 重新划分。

要点：
- 新 100 图的多边形不需要从 LabelMe 重转 —— labelme_2cat_points_todo/coco_0_99_val 里
  已有对应的 COCO（多边形与 LabelMe 逐图逐类别一致），只需把点挂上去。
- LabelMe 返回的点 shape 的 group_id 全为 null，点与所属穗没有显式关联，只能几何归属：
  点在哪个 wheat_2 的 segmentation 内就属于哪个穗；贴边落在外面的用最近多边形兜底。
- points 写成嵌套 [[x, y], ...] 的绝对像素坐标，且**每个** annotation 都必须有该键
  （data_register/wheat_data_cat2.py 用 extra_annotation_keys=["points"] 加载，缺键会 KeyError）。

用法：
    python myscripts/backup/merge_points_and_split.py --dry_run   # 只打印统计，不落盘
    python myscripts/backup/merge_points_and_split.py             # 正式执行
"""

import argparse
import collections
import json
import os
import random
import shutil

import numpy as np
from matplotlib.path import Path as MplPath

# ====== 路径配置 ======
ROOT = "/public/home/zncui/datasets/wheat50"
DST = os.path.join(ROOT, "coco_2cat_with_points")          # 就地更新的目标数据集
OLD_JSON = os.path.join(DST, "annotations", "instances_train2017.json")
OLD_IMG_DIR = os.path.join(DST, "train2017")
# 备份目录：不能叫 backup/，那个名字已被更早一次 merge 前的 267/34 划分占用。
BAK_DIR = os.path.join(DST, "backup_before_merge_xianyu100")

TODO = os.path.join(ROOT, "labelme_2cat_points_todo")
NEW_JSON = os.path.join(TODO, "coco_0_99_val", "annotations", "instances.json")
NEW_IMG_DIR = os.path.join(TODO, "coco_0_99_val", "images")
NEW_LABELME_DIR = os.path.join(
    TODO, "xianyu", "xianyu_labelme_0_99_val", "labelme_0_99_val")

# ====== 参数 ======
SEED = 0
VAL_RATIO = 0.2
# 点落在所有 wheat_2 多边形外时，吸附到最近多边形的容差（像素）。
# 实测 6 个越界点到最近边界的距离为 0.03~2.10 px，纯属贴边误差。
SNAP_TOL_PX = 5.0
# 分层划分的分桶边界：按每张 tile 的 wheat_2 实例数分桶，桶内再按 8:2 取。
STRATA = [(0, 0), (1, 2), (3, 4), (5, 10 ** 9)]
CAT_WHEAT, CAT_WHEAT2 = 1, 2


def seg_to_xy(seg):
    """COCO 扁平多边形 [x1,y1,x2,y2,...] -> (N,2) 数组。"""
    return np.asarray(seg, dtype=float).reshape(-1, 2)


def poly_boundary_dist(poly_xy, p):
    """点到多边形边界的最短距离（逐边线段距离，避免引入 shapely 依赖）。"""
    a = poly_xy
    b = np.roll(poly_xy, -1, axis=0)
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(1)
    t = np.where(denom > 0, (ap * ab).sum(1) / np.maximum(denom, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(proj - p, axis=1).min())


def attach_points_to_new_coco():
    """读新 100 图的 COCO，把 xianyu LabelMe 里的 spikelet 点按几何归属挂到 wheat_2 实例上。"""
    d = json.load(open(NEW_JSON))
    anns_by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        a["points"] = []          # 所有 ann 统一补键，无点为 []
        anns_by_img[a["image_id"]].append(a)

    stat = collections.Counter()
    for im in d["images"]:
        lm_path = os.path.join(
            NEW_LABELME_DIR, os.path.splitext(im["file_name"])[0] + ".json")
        if not os.path.exists(lm_path):
            print(f"[警告] 缺少 LabelMe 标注: {lm_path}")
            stat["缺少labelme的图"] += 1
            continue
        lm = json.load(open(lm_path))

        w2 = [a for a in anns_by_img[im["id"]] if a["category_id"] == CAT_WHEAT2]
        polys = [seg_to_xy(a["segmentation"][0]) for a in w2]
        paths = [MplPath(pg) for pg in polys]

        for s in lm["shapes"]:
            if s["shape_type"] != "point":
                continue
            p = np.asarray(s["points"][0], dtype=float)
            hit = [i for i, pa in enumerate(paths) if pa.contains_point(p)]
            if len(hit) == 1:
                idx = hit[0]
                stat["点·命中唯一穗"] += 1
            elif not polys:
                print(f"[警告] {im['file_name']} 有点但图中没有 wheat_2 实例，丢弃 {p.round(1)}")
                stat["点·丢弃"] += 1
                continue
            else:
                dists = [poly_boundary_dist(pg, p) for pg in polys]
                idx = int(np.argmin(dists))
                if len(hit) > 1:
                    print(f"[警告] {im['file_name']} 点 {p.round(1)} 落在 {len(hit)} 个 wheat_2 内，取最近的")
                    stat["点·落在多个穗内"] += 1
                elif dists[idx] <= SNAP_TOL_PX:
                    stat["点·贴边吸附"] += 1
                else:
                    print(f"[警告] {im['file_name']} 点 {p.round(1)} 距最近 wheat_2 {dists[idx]:.1f}px，超出容差，丢弃")
                    stat["点·丢弃"] += 1
                    continue
            w2[idx]["points"].append([float(p[0]), float(p[1])])

    print("[新 100 图挂点]", dict(stat))
    return d


def fix_stray_wheat_points(d):
    """修正标注噪声：category_id==1(wheat) 却带 points 的实例改判为 wheat_2。"""
    n = 0
    for a in d["annotations"]:
        if a.get("points") and a["category_id"] == CAT_WHEAT:
            a["category_id"] = CAT_WHEAT2
            n += 1
    if n:
        print(f"[修正] {n} 个带 points 的 wheat 实例已改判为 wheat_2")
    return n


def merge(old, new):
    """按项目既有的 offset 惯例（见 coco_2cat_with_points/merge_val_into_train.py）合并两份 COCO。"""
    img_off = max(im["id"] for im in old["images"])
    ann_off = max(a["id"] for a in old["annotations"])
    merged = {
        "info": old.get("info", {}),
        "licenses": old.get("licenses", []),
        "categories": old["categories"],     # 两份 id/name 一致，取旧的（supercategory 为 "none"）
        "images": [dict(im) for im in old["images"]],
        "annotations": [dict(a) for a in old["annotations"]],
    }
    for im in new["images"]:
        im2 = dict(im)
        im2["id"] = im["id"] + img_off
        merged["images"].append(im2)
    for a in new["annotations"]:
        a2 = dict(a)
        a2["id"] = a["id"] + ann_off
        a2["image_id"] = a["image_id"] + img_off
        merged["annotations"].append(a2)
    return merged


def stratified_split(d, seed=SEED, val_ratio=VAL_RATIO):
    """tile 级分层随机划分：按每张 tile 的 wheat_2 实例数分桶，桶内 8:2。

    注意：401 张 tile 仅来自 48 张 DJI 母图，tile 级划分会让同母图的相邻 tile
    同时出现在 train 和 val，验证指标因此偏乐观 —— 这是明确选择的取舍。
    """
    cnt = collections.Counter()
    for a in d["annotations"]:
        if a["category_id"] == CAT_WHEAT2:
            cnt[a["image_id"]] += 1

    buckets = collections.defaultdict(list)
    for im in d["images"]:
        c = cnt.get(im["id"], 0)
        for lo, hi in STRATA:
            if lo <= c <= hi:
                buckets[(lo, hi)].append(im["id"])
                break

    rng = random.Random(seed)
    val_ids = set()
    for key in STRATA:
        ids = sorted(buckets.get(key, []))
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_ratio))
        val_ids.update(ids[:n_val])
        print(f"  分层桶 wheat_2={key[0]}-{key[1] if key[1] < 10**9 else '+'}: "
              f"{len(ids)} 张 -> val {n_val}")
    train_ids = {im["id"] for im in d["images"]} - val_ids
    return train_ids, val_ids


def subset_and_reindex(d, keep_ids):
    """按 image id 集合取子集，并把 image_id / ann_id 重新编成从 1 开始的连续整数。"""
    images = [im for im in d["images"] if im["id"] in keep_ids]
    images.sort(key=lambda im: im["file_name"])
    remap = {im["id"]: i + 1 for i, im in enumerate(images)}

    out_images = []
    for im in images:
        im2 = dict(im)
        im2["id"] = remap[im["id"]]
        im2["file_name"] = os.path.basename(im["file_name"])
        out_images.append(im2)

    anns = [a for a in d["annotations"] if a["image_id"] in keep_ids]
    anns.sort(key=lambda a: (remap[a["image_id"]], a["id"]))
    out_anns = []
    for i, a in enumerate(anns):
        a2 = dict(a)
        a2["id"] = i + 1
        a2["image_id"] = remap[a["image_id"]]
        out_anns.append(a2)

    out = {k: v for k, v in d.items() if k not in ("images", "annotations")}
    out["images"] = out_images
    out["annotations"] = out_anns
    return out


def describe(name, d):
    per_cat = collections.Counter(a["category_id"] for a in d["annotations"])
    withpts = [a for a in d["annotations"] if a["points"]]
    npts = [len(a["points"]) for a in withpts]
    total = sum(npts)
    line = (f"[{name}] {len(d['images'])} 图 | {len(d['annotations'])} 实例 "
            f"(wheat {per_cat[CAT_WHEAT]}, wheat_2 {per_cat[CAT_WHEAT2]}) | "
            f"带点实例 {len(withpts)} | 小穗点 {total}")
    if npts:
        line += f" | 每穗点数 min/med/max = {min(npts)}/{int(np.median(npts))}/{max(npts)}"
    print(line)
    return {"images": len(d["images"]), "anns": len(d["annotations"]),
            "wheat_2": per_cat[CAT_WHEAT2], "points": total}


def validate(d, name):
    assert all("points" in a for a in d["annotations"]), f"{name}: 有 ann 缺 points 键"
    bad = [a["id"] for a in d["annotations"] if a["points"] and a["category_id"] != CAT_WHEAT2]
    assert not bad, f"{name}: 非 wheat_2 实例带 points: {bad[:5]}"


def copy_images(images, src_dirs, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    missing = []
    for im in images:
        base = os.path.basename(im["file_name"])
        for sd in src_dirs:
            src = os.path.join(sd, base)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, base))
                break
        else:
            missing.append(base)
    if missing:
        raise FileNotFoundError(f"{len(missing)} 张图片在源目录中找不到，例: {missing[:3]}")
    print(f"  复制 {len(images)} 张 -> {dst_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true", help="只打印统计，不备份/不写文件")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    args = ap.parse_args()

    old = json.load(open(OLD_JSON))
    describe("原 train2017", old)

    new = attach_points_to_new_coco()
    describe("新 100 图", new)

    merged = merge(old, new)
    fix_stray_wheat_points(merged)
    validate(merged, "合并后")
    describe("合并后", merged)

    print("[分层划分]")
    train_ids, val_ids = stratified_split(merged, args.seed, args.val_ratio)
    assert len(train_ids) + len(val_ids) == len(merged["images"])
    assert not (train_ids & val_ids)

    train = subset_and_reindex(merged, train_ids)
    val = subset_and_reindex(merged, val_ids)
    for name, d in (("train2017", train), ("val2017", val)):
        validate(d, name)
        describe(name, d)

    tn = {im["file_name"] for im in train["images"]}
    vn = {im["file_name"] for im in val["images"]}
    assert not (tn & vn), f"train/val 文件名重叠: {sorted(tn & vn)[:3]}"

    if args.dry_run:
        print("\n[dry_run] 未写入任何文件。")
        return

    # 1) 备份原 train2017（图片 + 标注）
    if os.path.exists(BAK_DIR):
        raise FileExistsError(f"备份目录已存在，请先确认: {BAK_DIR}")
    os.makedirs(os.path.join(BAK_DIR, "annotations"))
    print(f"\n[备份] {OLD_IMG_DIR} -> {BAK_DIR}/train2017")
    shutil.copytree(OLD_IMG_DIR, os.path.join(BAK_DIR, "train2017"))
    shutil.copy2(OLD_JSON, os.path.join(BAK_DIR, "annotations", "instances_train2017.json"))

    # 2) 重建图片目录（原 train2017 已备份，可安全清空重建）
    src_dirs = [os.path.join(BAK_DIR, "train2017"), NEW_IMG_DIR]
    shutil.rmtree(OLD_IMG_DIR)
    print("[写出图片]")
    copy_images(train["images"], src_dirs, os.path.join(DST, "train2017"))
    val_dir = os.path.join(DST, "val2017")
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    copy_images(val["images"], src_dirs, val_dir)

    # 3) 写标注
    ann_dir = os.path.join(DST, "annotations")
    for name, d in (("train", train), ("val", val)):
        p = os.path.join(ann_dir, f"instances_{name}2017.json")
        with open(p, "w") as f:
            json.dump(d, f)
        print(f"[写出] {p}")

    print("\n完成。")


if __name__ == "__main__":
    main()
