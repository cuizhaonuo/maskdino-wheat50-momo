# -*- coding: utf-8 -*-
"""wheat_2 专项 copy-paste 增强。

动机见 experiments-bank/ideas/copy-paste-wheat2.md：wheat_2 在两个训练集里都只占
6.0% / 7.7%，而它是点损失与计数指标唯一作用的类别；Stage1 上它的 segm AP50 比
wheat 低 9.9 点（82.89 vs 92.83），缺口全在「能不能被找到并分对类」。

在 mapper 的**原图阶段**调用（读图之后、LSJ 增广之前），把实例库里的 wheat_2
贴到当前图上并追加标注。之后的多边形变换、points 变换、贴边降级、空实例过滤
全部复用 mapper 已有链路，本模块不重复实现。

两条几何约束（用户 2026-09-03 指定）：
  · wheat_2 之间**不得重叠**（与已有 + 已贴的 wheat_2 交集必须为 0）
  · wheat_2 **可以**贴在 wheat 上面；被覆盖的 wheat 掩码要减掉重叠区，
    残余面积比过小则整个丢弃

注意 category_id 用的是 **contiguous id**（0=wheat, 1=wheat_2），
因为 detectron2 的 load_coco_json 已按 thing_dataset_id_to_contiguous_id 映射过；
实例库 npz 里存的则是原始 COCO id 口径下抠出来的补丁，两者不冲突。
"""

import logging
import os

import numpy as np
from detectron2.structures import BoxMode

from myscripts.backup.coco_rle_to_polygon import (
    binary_mask_to_polygons,
    mask_from_polygons,
)

WHEAT_ID = 0        # contiguous
WHEAT2_ID = 1       # contiguous

_logger = logging.getLogger(__name__)


def _polys_of(ann):
    """取标注的多边形列表；非多边形（RLE）返回空，调用方跳过。"""
    seg = ann.get("segmentation", None)
    if not isinstance(seg, list):
        return []
    return [p for p in seg if isinstance(p, list) and len(p) >= 6]


def _rasterize(polys, x0, y0, w, h):
    """把绝对坐标多边形栅格化到以 (x0,y0) 为原点、尺寸 (h,w) 的局部画布。"""
    shifted = [
        (np.asarray(p, dtype=np.float32).reshape(-1, 2) - [x0, y0]).reshape(-1).tolist()
        for p in polys
    ]
    return mask_from_polygons(shifted, h, w)


def _poly_bbox(polys):
    pts = np.concatenate([np.asarray(p, np.float32).reshape(-1, 2) for p in polys], axis=0)
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


class Wheat2CopyPaste:
    """从实例库随机取 wheat_2 贴到当前图，直到该图 wheat_2 数达到 target_per_img。"""

    def __init__(self, bank_path, target_per_img=6, max_tries=30, margin=8,
                 min_keep=0.2, flip_prob=0.5, approx_eps=1.0, min_poly_area=16.0):
        if not bank_path or not os.path.exists(bank_path):
            raise FileNotFoundError(
                f"copy-paste 实例库不存在: {bank_path!r}；"
                "先用 myscripts/backup/build_wheat2_bank.py 建库")
        z = np.load(bank_path, allow_pickle=True)
        self.patches = z["patches"]
        self.masks = z["masks"]
        self.polys = z["polys"]
        self.points = z["points"]
        self.n = len(self.patches)
        self.target_per_img = int(target_per_img)
        self.max_tries = int(max_tries)
        self.margin = int(margin)
        self.min_keep = float(min_keep)
        self.flip_prob = float(flip_prob)
        self.approx_eps = float(approx_eps)
        self.min_poly_area = float(min_poly_area)
        _logger.info(
            f"[COPY_PASTE] 实例库 {bank_path} 载入 {self.n} 条 | "
            f"每图补到 {self.target_per_img} 个 wheat_2 | margin={self.margin} | "
            f"min_keep={self.min_keep}")

    # ---------------------------------------------------------------- 主入口
    def __call__(self, image, annotations):
        """就地返回 (新图, 新标注列表, 实际贴图数)。image 为 HxWx3 RGB。"""
        H, W = image.shape[:2]
        anns = list(annotations)

        # 占位掩码：所有 wheat_2（已有 + 已贴）。wheat_2 之间不许重叠。
        occ = np.zeros((H, W), dtype=np.uint8)
        n_exist = 0
        for a in anns:
            if int(a.get("category_id", -1)) != WHEAT2_ID:
                continue
            polys = _polys_of(a)
            if polys:
                occ |= mask_from_polygons(polys, H, W)
            n_exist += 1

        need = self.target_per_img - n_exist
        if need <= 0 or self.n == 0:
            return image, anns, 0

        # utils.read_image 返回的数组是只读的，且 ascontiguousarray 对已连续数组
        # 不会复制、也不改可写性 —— 必须显式 copy 才能就地合成。
        image = np.array(image, copy=True)
        pasted = 0
        for _ in range(need):
            if self._paste_one(image, anns, occ, H, W):
                pasted += 1
        return image, anns, pasted

    # ------------------------------------------------------------ 单次贴图
    def _paste_one(self, image, anns, occ, H, W):
        i = int(np.random.randint(self.n))
        patch = self.patches[i]
        pm = (self.masks[i] > 0).astype(np.uint8)
        polys = [list(p) for p in self.polys[i]]
        pts = np.asarray(self.points[i], dtype=np.float32).reshape(-1, 2)

        ph, pw = pm.shape[:2]
        if np.random.rand() < self.flip_prob:
            patch = patch[:, ::-1]
            pm = pm[:, ::-1]
            polys = [
                (np.asarray(p, np.float32).reshape(-1, 2) * [-1, 1] + [pw - 1, 0]).reshape(-1).tolist()
                for p in polys
            ]
            if len(pts):
                pts = pts * [-1, 1] + [pw - 1, 0]

        lo_x, hi_x = self.margin, W - self.margin - pw
        lo_y, hi_y = self.margin, H - self.margin - ph
        if hi_x < lo_x or hi_y < lo_y:
            return False  # 补丁比可用画布还大，换一条

        for _ in range(self.max_tries):
            x0 = int(np.random.randint(lo_x, hi_x + 1))
            y0 = int(np.random.randint(lo_y, hi_y + 1))
            # 约束一：wheat_2 之间零重叠
            if np.any(occ[y0:y0 + ph, x0:x0 + pw] & pm):
                continue

            m = pm.astype(bool)
            image[y0:y0 + ph, x0:x0 + pw][m] = patch[m]
            occ[y0:y0 + ph, x0:x0 + pw] |= pm

            abs_polys = [
                (np.asarray(p, np.float32).reshape(-1, 2) + [x0, y0]).reshape(-1).tolist()
                for p in polys
            ]
            area = float(pm.sum())
            anns.append({
                "bbox": [float(x0), float(y0), float(pw), float(ph)],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": WHEAT2_ID,
                "segmentation": abs_polys,
                "iscrowd": 0,
                "area": area,
                "points": (pts + [x0, y0]).tolist() if len(pts) else [],
            })
            # 约束二：被覆盖的 wheat 减掉重叠区
            self._subtract_from_wheat(anns, pm, x0, y0, pw, ph)
            return True
        return False

    # ------------------------------------------------- 遮挡：从 wheat 里减掉
    def _subtract_from_wheat(self, anns, pm, px0, py0, pw, ph):
        px1, py1 = px0 + pw, py0 + ph
        drop = []
        for k, a in enumerate(anns):
            if int(a.get("category_id", -1)) != WHEAT_ID:
                continue
            polys = _polys_of(a)
            if not polys:
                continue
            wx0, wy0, wx1, wy1 = _poly_bbox(polys)
            if wx1 <= px0 or wx0 >= px1 or wy1 <= py0 or wy0 >= py1:
                continue  # bbox 不相交，跳过（绝大多数走这条，省栅格化）

            lw, lh = max(wx1 - wx0, 1), max(wy1 - wy0, 1)
            wm = _rasterize(polys, wx0, wy0, lw, lh)
            before = float(wm.sum())
            if before <= 0:
                continue

            # 把贴图掩码搬到该 wheat 的局部画布上再相减
            ix0, iy0 = max(px0, wx0), max(py0, wy0)
            ix1, iy1 = min(px1, wx1), min(py1, wy1)
            wm[iy0 - wy0:iy1 - wy0, ix0 - wx0:ix1 - wx0] &= \
                1 - pm[iy0 - py0:iy1 - py0, ix0 - px0:ix1 - px0]

            after = float(wm.sum())
            if after == before:
                continue
            if after < self.min_keep * before:
                drop.append(k)
                continue

            new_polys = binary_mask_to_polygons(
                wm, approx_eps=self.approx_eps, min_area=self.min_poly_area,
                keep_holes=False, img_w=lw, img_h=lh)
            if not new_polys:
                drop.append(k)
                continue
            a["segmentation"] = [
                (np.asarray(p, np.float32).reshape(-1, 2) + [wx0, wy0]).reshape(-1).tolist()
                for p in new_polys
            ]
            ys, xs = np.where(wm > 0)
            a["bbox"] = [float(wx0 + xs.min()), float(wy0 + ys.min()),
                         float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
            a["bbox_mode"] = BoxMode.XYWH_ABS
            a["area"] = after

        for k in sorted(drop, reverse=True):
            anns.pop(k)


def build_copy_paste(cfg):
    """按 cfg 构造增强器；未开启返回 None。"""
    if not cfg.INPUT.COPY_PASTE_ENABLED:
        return None
    return Wheat2CopyPaste(
        bank_path=cfg.INPUT.COPY_PASTE_BANK,
        target_per_img=cfg.INPUT.COPY_PASTE_TARGET_PER_IMG,
        max_tries=cfg.INPUT.COPY_PASTE_MAX_TRIES,
        margin=cfg.INPUT.COPY_PASTE_MARGIN,
        min_keep=cfg.INPUT.COPY_PASTE_MIN_KEEP,
        flip_prob=cfg.INPUT.COPY_PASTE_FLIP_PROB,
    )
