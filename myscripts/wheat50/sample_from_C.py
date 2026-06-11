#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从集合 C 随机取样，补足到目标张数（默认 50）。

集合定义：
  A = total_images/train_val_originals_50 下的原图（train2017 / val2017 的来源）
  B = raw_data .../DCIM 下所有 *-250cm-A50 子目录里的全部 .JPG
  C = B - A
输出：把抽到的图片复制到 total_images/<OUT_NAME>/。

“补足”逻辑（可反复运行）：
  - 脚本始终把输出目录补到 TARGET 张。
  - 已经抽过的文件名都会记到 history 文件里；即使你之后把不满意的删掉，
    下次运行也【不会】再抽到它们（避免重复推荐被你否决的图）。
  - 因此典型用法：
      1) 第一次运行 -> 得到 50 张，你人工挑选、删掉不满意的；
      2) 再次运行   -> 自动按缺口补足到 50 张（只补新的、没抽过的）。
每次抽样都会把记录追加到 manifest CSV，便于回溯。
"""
import argparse
import csv
import datetime as dt
import os
import random
import shutil

# ---------------- 路径配置 ----------------
RAW = "/public/lab_share/nfs_field/raw_data/DJI/P1/zhaoxian2023/wheat/20240516DJI/DCIM"
A_DIR = "/public/home/zncui/datasets/wheat50/total_images/train_val_originals_50"
TOTAL_IMAGES = "/public/home/zncui/datasets/wheat50/total_images"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(SCRIPT_DIR, "picked_history.txt")   # 所有抽过的文件名（含已删除的）
MANIFEST_CSV = os.path.join(SCRIPT_DIR, "picked_manifest.csv")  # 每次抽样的明细记录

IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")


def build_B():
    """返回 {basename: fullpath}，B = 所有 -250cm-A50 子目录下的图片。"""
    b = {}
    for sub in sorted(os.listdir(RAW)):
        if not sub.endswith("-250cm-A50"):
            continue
        d = os.path.join(RAW, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(IMG_EXTS):
                if fn in b:
                    print(f"[警告] 跨子目录重名: {fn}（保留先出现的）")
                    continue
                b[fn] = os.path.join(d, fn)
    return b


def build_A():
    """A = train_val_originals_50 下的图片文件名集合。"""
    return {fn for fn in os.listdir(A_DIR) if fn.endswith(IMG_EXTS)}


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE) as f:
        return {ln.strip() for ln in f if ln.strip()}


def append_history(names):
    with open(HISTORY_FILE, "a") as f:
        for n in names:
            f.write(n + "\n")


def append_manifest(rows, batch_tag):
    new = not os.path.exists(MANIFEST_CSV)
    with open(MANIFEST_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["batch", "filename", "subdir", "src_path", "dst_path"])
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="从 C=B-A 随机补足取样")
    ap.add_argument("--target", type=int, default=50, help="输出目录的目标张数（默认 50）")
    ap.add_argument("--out-name", default="random_50_from_C", help="total_images 下的输出目录名")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可选，便于复现）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要抽取的，不复制")
    ap.add_argument("--reset-history", action="store_true",
                    help="清空 history（慎用：之后可能重新抽到曾经被你删除的图）")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_dir = os.path.join(TOTAL_IMAGES, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    if args.reset_history and os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        print("[已清空 history]")

    B = build_B()
    A = build_A()
    history = load_history()
    current = {fn for fn in os.listdir(out_dir) if fn.endswith(IMG_EXTS)}

    # C = B - A；可抽样池还要排除 history（含已删的）和当前目录已有的
    pool = sorted(set(B.keys()) - A - history - current)

    need = args.target - len(current)
    print(f"B={len(B)}  A={len(A)}  C=B-A={len(set(B)-A)}")
    print(f"history(已抽过)={len(history)}  当前输出目录已有={len(current)}  可抽样池={len(pool)}")
    print(f"目标={args.target}  需补={max(need,0)}")

    if need <= 0:
        print("已达目标张数，无需补充。")
        return
    if not pool:
        print("[警告] 可抽样池已空，无法继续补充。")
        return

    k = min(need, len(pool))
    if k < need:
        print(f"[警告] 池中仅剩 {k} 张，少于需补的 {need} 张，按 {k} 张抽取。")
    picked = random.sample(pool, k)

    batch_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for fn in picked:
        src = B[fn]
        sub = os.path.basename(os.path.dirname(src))
        dst = os.path.join(out_dir, fn)
        if args.dry_run:
            print(f"[dry-run] {sub}/{fn}")
        else:
            shutil.copy2(src, dst)
        rows.append([batch_tag, fn, sub, src, dst])

    if not args.dry_run:
        append_history(picked)
        append_manifest(rows, batch_tag)

    print(f"\n本批抽取 {len(picked)} 张（batch={batch_tag}）:")
    for _, fn, sub, *_ in rows:
        print(f"  {sub}/{fn}")
    print(f"\n输出目录: {out_dir}  现共 {len(current)+ (0 if args.dry_run else len(picked))} 张")
    print(f"history : {HISTORY_FILE}")
    print(f"manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
