import json, math
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

ann_file = "/public/home/zncui/datasets/maize_rice_dataset/coco/poly_annotations/instances_val2017.json"  # 改成你的 val/test json
coco = COCO(ann_file)

bad_anns = []
for ann in coco.dataset["annotations"]:
    img = coco.imgs.get(ann["image_id"])
    if not img or "height" not in img or "width" not in img:
        bad_anns.append((ann.get("id"), "missing image or size"))
        continue
    h, w = int(img["height"]), int(img["width"])
    if h <= 0 or w <= 0:
        bad_anns.append((ann.get("id"), f"non-positive size {h}x{w}"))
        continue

    segm = ann.get("segmentation", None)
    if segm is None:
        bad_anns.append((ann.get("id"), "segmentation is None"))
        continue

    try:
        if isinstance(segm, list):  # polygon(s)
            if len(segm) == 0:
                raise ValueError("empty polygon list")
            for poly in segm:
                if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2 != 0:
                    raise ValueError(f"invalid polygon len={len(poly)}")
                for v in poly:
                    if not isinstance(v, (int, float)) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        raise ValueError("non-numeric or NaN/Inf coord")
            _ = maskUtils.frPyObjects(segm, h, w)  # 让它实测一下
        elif isinstance(segm, dict):  # RLE
            if "counts" not in segm or "size" not in segm:
                raise ValueError("RLE missing counts/size")
            size = segm["size"]
            if not (isinstance(size, list) and len(size) == 2 and int(size[0]) == h and int(size[1]) == w):
                raise ValueError(f"RLE size mismatch: got {size}, expect {[h,w]}")
            _ = maskUtils.frPyObjects(segm, h, w)
        else:
            raise ValueError(f"unknown segm type: {type(segm)}")
    except Exception as e:
        bad_anns.append((ann.get("id"), str(e)))

print(f"Found {len(bad_anns)} bad annotations")
for x in bad_anns[:50]:
    print(x)
# 按需要把 bad_anns 存文件，方便定位
