#!/usr/bin/env python
"""汇总阈值扫描结果：每个 run 一张 POINT_THRESHOLD 表 + 跨 run 的 best-vs-best。

用法: python collect_sweep.py [run 名...]   默认 last1 last3 last5 last7 last9
"""
import re, sys
from pathlib import Path

R = Path("/public/home/zncui/pycharm_project/maskdino-wheat50/output_v2")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
WANT = ["matched_F1@8", "mF1", "count_MAE_spike", "count_MAE_img",
        "spike_recall", "spike_precision", "n_pred_points"]


def read_eval(log: Path):
    """从一次 --eval_only 的 log.txt 里取 segm/bbox AP 与小穗点各项。"""
    txt = ANSI.sub("", log.read_text(errors="ignore"))
    out = {}
    # copypaste 块：Task: <name> / 表头 / 数值
    blocks = re.findall(r"copypaste: Task: (\w+)\s*\n.*?copypaste: ([^\n]+)\n.*?copypaste: ([^\n]+)", txt)
    for task, head, vals in blocks:
        keys = head.split(","); vs = vals.split(",")
        if len(keys) != len(vs):
            continue
        for k, v in zip(keys, vs):
            try:
                out[f"{task}/{k}"] = float(v)
            except ValueError:
                pass
    return out


def sweep(arm):
    rows = []
    for d in sorted((R / f"exp-0904-unfz-{arm}" / "sweep").glob("T*"),
                    key=lambda p: float(p.name[1:])):
        log = d / "log.txt"
        if not log.exists():
            continue
        m = read_eval(log)
        if "spikelet/matched_F1@8" not in m:
            continue
        rows.append((float(d.name[1:]), m))
    return rows


if __name__ == "__main__":
    arms = sys.argv[1:] or ["last1", "last3", "last5", "last7", "last9"]
    best = {}
    for a in arms:
        rows = sweep(a)
        if not rows:
            print(f"\n### {a}: 暂无结果"); continue
        ap = rows[0][1]
        print(f"\n### {a}   （segm AP {ap.get('segm/AP', float('nan')):.2f}"
              f" / bbox AP {ap.get('bbox/AP', float('nan')):.2f}，与阈值无关）")
        print(f"{'T':>5} " + " ".join(f"{k:>16s}" for k in WANT))
        for t, m in rows:
            print(f"{t:5.2f} " + " ".join(f"{m.get('spikelet/'+k, float('nan')):16.4f}" for k in WANT))
        bf = max(rows, key=lambda r: r[1]["spikelet/matched_F1@8"])
        bc = min(rows, key=lambda r: r[1]["spikelet/count_MAE_spike"])
        best[a] = dict(n=len(rows),
                       f1=(bf[0], bf[1]["spikelet/matched_F1@8"]),
                       mae=(bc[0], bc[1]["spikelet/count_MAE_spike"]),
                       segm=rows[0][1].get("segm/AP"), bbox=rows[0][1].get("bbox/AP"))
    print("\n### best-vs-best（各自扫描取最优）")
    print(f"{'run':8s} {'档数':>4s} {'matched_F1@8':>14s} {'@T':>5s} {'count_MAE_spike':>16s} {'@T':>5s} {'segm AP':>8s} {'bbox AP':>8s}")
    for a, b in best.items():
        print(f"{a:8s} {b['n']:4d} {b['f1'][1]:14.4f} {b['f1'][0]:5.2f} "
              f"{b['mae'][1]:16.2f} {b['mae'][0]:5.2f} {b['segm']:8.2f} {b['bbox']:8.2f}")
    print(f"{'combo':8s} {10:4d} {0.5638:14.4f} {0.28:5.2f} {2.10:16.2f} {0.50:5.2f} {70.32:8.2f} {69.73:8.2f}   <- 对照")
