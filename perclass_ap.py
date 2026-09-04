#!/usr/bin/env python
"""逐类别 AP / AP50 / AP75（segm + bbox）。

detectron2 的 COCOEvaluator 只在日志里打逐类别 AP，不打逐类别 AP50/AP75，
而 wheat_2 的缺口恰恰主要在 AP50（见 exp-2026-09-03-decoder-layer-baseline）。
本脚本直接从 eval 产物的 coco_instances_results.json 重算。

用法: python perclass_ap.py <eval_dir> [<eval_dir> ...]
      eval_dir 需含 inference/coco_instances_results.json 与 *_coco_format.json
"""
import sys, json, io, contextlib
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def perclass(d: Path):
    inf = d / "inference"
    gtp = next(inf.glob("*_coco_format.json"))
    dtp = inf / "coco_instances_results.json"
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gtp))
        dt = gt.loadRes(str(dtp)) if json.load(open(dtp)) else None
    if dt is None:
        return None
    names = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    out = {}
    for iou_type in ("segm", "bbox"):
        for cid, name in names.items():
            E = COCOeval(gt, dt, iou_type)
            E.params.catIds = [cid]
            with contextlib.redirect_stdout(io.StringIO()):
                E.evaluate(); E.accumulate(); E.summarize()
            out[(iou_type, name)] = (E.stats[0] * 100, E.stats[1] * 100, E.stats[2] * 100)
    return out

if __name__ == "__main__":
    print(f"{'run':22s} {'task':5s} {'类别':9s} {'AP':>7s} {'AP50':>7s} {'AP75':>7s}")
    for a in sys.argv[1:]:
        d = Path(a); r = perclass(d)
        if r is None:
            print(f"{d.name:22s} 无预测"); continue
        for (t, n), (ap, ap50, ap75) in r.items():
            print(f"{d.name:22s} {t:5s} {n:9s} {ap:7.2f} {ap50:7.2f} {ap75:7.2f}")
