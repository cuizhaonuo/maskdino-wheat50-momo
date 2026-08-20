#!/usr/bin/env python3
"""
从 coco_2cat 抽取一个"待标注点"的 COCO 子集数据集。

- val2017: 全部 200 张
- train2017: 按每张图 wheat_2 (category_id==2) 实例数量排序, 取最多的 500 张
             (并列时用 image_id 升序做确定性 tie-break)
输出目录结构与 coco_2cat 一致 (train2017/ + val2017/ + train/val 软链接),
标注保持原样 (不加 points 字段)。
"""
import json
import os
import shutil
import collections

SRC = "/public/home/zncui/datasets/wheat50/coco_2cat"
DST = "/public/home/zncui/datasets/wheat50/coco_2cat_points_todo"
N_TRAIN = 500


def load(split):
    with open(os.path.join(SRC, "annotations", f"instances_{split}.json")) as f:
        return json.load(f)


def select_train(d):
    cnt = collections.Counter()
    for a in d["annotations"]:
        if a["category_id"] == 2:  # wheat_2
            cnt[a["image_id"]] += 1
    # 每张图的 wheat_2 数(没有 wheat_2 的图计为 0), 排序: 数量降序, image_id 升序
    ranked = sorted(
        (im["id"] for im in d["images"]),
        key=lambda iid: (-cnt.get(iid, 0), iid),
    )
    keep = set(ranked[:N_TRAIN])
    return keep, cnt


def subset(d, keep_ids):
    images = [im for im in d["images"] if im["id"] in keep_ids]
    anns = [a for a in d["annotations"] if a["image_id"] in keep_ids]
    out = {k: v for k, v in d.items() if k not in ("images", "annotations")}
    out["images"] = images
    out["annotations"] = anns
    return out


def copy_images(images, split):
    """file_name 可能带 'val/' 前缀; 图片实际落在 <split>/ 下, 用 basename。"""
    dst_dir = os.path.join(DST, split)
    os.makedirs(dst_dir, exist_ok=True)
    missing = []
    for im in images:
        base = os.path.basename(im["file_name"])
        src = os.path.join(SRC, split, base)
        if not os.path.exists(src):
            missing.append(src)
            continue
        shutil.copy2(src, os.path.join(dst_dir, base))
    return missing


def main():
    os.makedirs(os.path.join(DST, "annotations"), exist_ok=True)

    # ---- train ----
    dtr = load("train2017")
    keep, cnt = select_train(dtr)
    tr_out = subset(dtr, keep)
    kept_counts = [cnt.get(iid, 0) for iid in keep]
    print(f"[train] 选中 {len(tr_out['images'])} 张, "
          f"wheat_2 数量范围 {min(kept_counts)}~{max(kept_counts)}, "
          f"annotations {len(tr_out['annotations'])}")

    # ---- val ----
    dva = load("val2017")
    va_out = dva  # 全部保留
    print(f"[val] 保留 {len(va_out['images'])} 张, annotations {len(va_out['annotations'])}")

    # ---- 写 json ----
    for name, obj in [("train2017", tr_out), ("val2017", va_out)]:
        with open(os.path.join(DST, "annotations", f"instances_{name}.json"), "w") as f:
            json.dump(obj, f)

    # ---- 拷贝图片 ----
    m1 = copy_images(tr_out["images"], "train2017")
    m2 = copy_images(va_out["images"], "val2017")
    if m1 or m2:
        print(f"[警告] 缺失图片: train {len(m1)}, val {len(m2)}")
        for p in (m1 + m2)[:10]:
            print("   缺:", p)

    # ---- 软链接, 与 coco_2cat 结构一致 ----
    for link, target in [("train", "train2017"), ("val", "val2017")]:
        lp = os.path.join(DST, link)
        if not os.path.islink(lp) and not os.path.exists(lp):
            os.symlink(target, lp)

    print("完成 ->", DST)


if __name__ == "__main__":
    main()
