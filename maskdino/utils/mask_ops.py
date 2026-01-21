from torch.nn import functional as F
import torch
import torch.nn.functional as F
from typing import Optional, Literal

ReturnType = Literal["iou", "dice", "ioa"]  # ioa = inter / min(area_a, area_b) 类似你 numpy 里的 else 分支


def pairwise_mask_overlap_roi(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    return_type: ReturnType = "iou",
    pred_is_logits: bool = True,
    pred_thresh: float = 0.5,
    gt_thresh: float = 0.5,
    resize_gt: bool = True,
    gt_resize_mode: str = "nearest",  # GT 是 binary 时建议 nearest
    eps: float = 1.0,
    candidate_iou_thr: float = 0.0,   # bbox IoU 阈值；0.0 表示只要 bbox 有 overlap 就算候选
    max_pairs_per_image: Optional[int] = None,  # 可选：限制每张图最多算多少对（进一步控时/控显存）
    pair_chunk: int = 512,            # 分块处理候选对，减少 Python 负担
    reduce: Optional[Literal["mean", "sum"]] = None,  # None -> 返回 (bs,nq,ngt); "mean"/"sum" -> (nq,ngt)
) -> torch.Tensor:
    """
    pred_mask: (nq, bs, Hp, Wp)  或 (bs, nq, Hp, Wp)
    gt_mask:   (ngt, bs, Hg, Wg) 或 (bs, ngt, Hg, Wg)
    pred_boxes:(bs, nq, 4)  xyxy in pixel coords
    gt_boxes:  (bs, ngt, 4) xyxy in pixel coords

    输出:
      - reduce is None: (bs, nq, ngt)
      - reduce in {"mean","sum"}: (nq, ngt)
    """

    # ---------- 0) 统一维度到 (bs, nq, H, W) / (bs, ngt, H, W) ----------
    if pred_mask.dim() != 4 or gt_mask.dim() != 4:
        raise ValueError("pred_mask and gt_mask must be 4D tensors.")

    # pred_mask: allow (nq,bs,H,W) -> (bs,nq,H,W)
    if pred_mask.shape[0] != pred_boxes.shape[0] and pred_mask.shape[1] == pred_boxes.shape[0]:
        # (nq, bs, H, W)
        pred_mask = pred_mask.permute(1, 0, 2, 3).contiguous()
    # gt_mask: allow (ngt,bs,H,W) -> (bs,ngt,H,W)
    if gt_mask.shape[0] != gt_boxes.shape[0] and gt_mask.shape[1] == gt_boxes.shape[0]:
        gt_mask = gt_mask.permute(1, 0, 2, 3).contiguous()

    bs, nq, Hp, Wp = pred_mask.shape
    bs2, ngt, Hg, Wg = gt_mask.shape
    if bs != bs2:
        raise ValueError(f"Batch mismatch: pred bs={bs}, gt bs={bs2}")
    if pred_boxes.shape[:2] != (bs, nq) or gt_boxes.shape[:2] != (bs, ngt):
        raise ValueError("pred_boxes must be (bs,nq,4), gt_boxes must be (bs,ngt,4)")

    device = pred_mask.device

    # ---------- 1) 需要的话，把 gt resize 到 pred 分辨率 ----------
    if resize_gt and (Hg != Hp or Wg != Wp):
        gt_mask = gt_mask.float()
        gt_mask = F.interpolate(gt_mask, size=(Hp, Wp), mode=gt_resize_mode, align_corners=False if gt_resize_mode in ("bilinear", "bicubic") else None)

    # ---------- 2) 二值化 / 预处理 ----------
    if pred_is_logits:
        pred_bin = (pred_mask.sigmoid() > pred_thresh)
    else:
        pred_bin = (pred_mask > pred_thresh)

    # GT 若是 soft（比如 bilinear 后），阈值化一下更符合 IoU 语义
    gt_bin = (gt_mask > gt_thresh)

    # ---------- 3) 预计算每个 mask 的面积（很省） ----------
    # (bs,nq)
    pred_area = pred_bin.flatten(2).sum(-1).to(torch.float32)
    # (bs,ngt)
    gt_area = gt_bin.flatten(2).sum(-1).to(torch.float32)

    # ---------- 4) 输出张量 ----------
    out = torch.zeros((bs, nq, ngt), device=device, dtype=torch.float32)

    # ---------- 5) 每张图：bbox 先筛候选 pairs，再 ROI 计算 ----------
    for b in range(bs):
        pb = pred_boxes[b]  # (nq,4)
        gb = gt_boxes[b]    # (ngt,4)

        # bbox overlap (向量化)：计算交叠矩形
        # pb: (nq,1,4), gb: (1,ngt,4) -> broadcast
        pb_e = pb[:, None, :]  # (nq,1,4)
        gb_e = gb[None, :, :]  # (1,ngt,4)

        xA = torch.maximum(pb_e[..., 0], gb_e[..., 0])
        yA = torch.maximum(pb_e[..., 1], gb_e[..., 1])
        xB = torch.minimum(pb_e[..., 2], gb_e[..., 2])
        yB = torch.minimum(pb_e[..., 3], gb_e[..., 3])

        inter_w = (xB - xA).clamp(min=0)
        inter_h = (yB - yA).clamp(min=0)
        inter_area = inter_w * inter_h  # (nq,ngt)

        # 先用 overlap_area > 0 做候选；也可加 bbox IoU 阈值
        cand = inter_area > 0

        if candidate_iou_thr > 0:
            pb_area = ((pb[:, 2] - pb[:, 0]).clamp(min=0) * (pb[:, 3] - pb[:, 1]).clamp(min=0))[:, None]
            gb_area = ((gb[:, 2] - gb[:, 0]).clamp(min=0) * (gb[:, 3] - gb[:, 1]).clamp(min=0))[None, :]
            bbox_union = pb_area + gb_area - inter_area
            bbox_iou = torch.where(bbox_union > 0, inter_area / bbox_union, torch.zeros_like(bbox_union))
            cand = cand & (bbox_iou >= candidate_iou_thr)

        pairs = cand.nonzero(as_tuple=False)  # (K,2) each row: [i_pred, i_gt]

        if pairs.numel() == 0:
            continue

        if max_pairs_per_image is not None and pairs.shape[0] > max_pairs_per_image:
            # 简单截断（你也可以按 bbox_iou 排序取 topk）
            pairs = pairs[:max_pairs_per_image]

        # 取出该图的二值 mask
        pm = pred_bin[b]  # (nq,Hp,Wp)
        gm = gt_bin[b]    # (ngt,Hp,Wp)

        # 分块处理候选 pairs，减少 Python 循环压力
        for s in range(0, pairs.shape[0], pair_chunk):
            blk = pairs[s:s + pair_chunk]  # (M,2)

            for i, j in blk.tolist():
                # overlap ROI 坐标（从 bbox overlap 得到），裁到图像范围内
                # 这里用 round/floor/ceil 的策略你可以按坐标定义调整；模板用 floor/ceil 更保守
                x1 = int(torch.floor(xA[i, j]).item())
                y1 = int(torch.floor(yA[i, j]).item())
                x2 = int(torch.ceil(xB[i, j]).item())
                y2 = int(torch.ceil(yB[i, j]).item())

                # clamp
                x1 = max(0, min(x1, Wp))
                x2 = max(0, min(x2, Wp))
                y1 = max(0, min(y1, Hp))
                y2 = max(0, min(y2, Hp))

                if x2 <= x1 or y2 <= y1:
                    continue

                a_roi = pm[i, y1:y2, x1:x2]
                b_roi = gm[j, y1:y2, x1:x2]

                inter = (a_roi & b_roi).sum().to(torch.float32)  # scalar

                if inter <= 0:
                    # 没交集，直接 0
                    continue

                pa = pred_area[b, i]
                ga = gt_area[b, j]

                if return_type == "iou":
                    union = pa + ga - inter
                    out[b, i, j] = (inter + eps) / (union + eps) if union > 0 else 0.0
                elif return_type == "dice":
                    denom = pa + ga
                    out[b, i, j] = (2 * inter + eps) / (denom + eps) if denom > 0 else 0.0
                elif return_type == "ioa":
                    denom = torch.minimum(pa, ga)
                    out[b, i, j] = (inter + eps) / (denom + eps) if denom > 0 else 0.0
                else:
                    raise ValueError("return_type must be 'iou' or 'dice' or 'ioa'")

    if reduce is None:
        return out  # (bs,nq,ngt)
    elif reduce == "mean":
        return out.mean(dim=0)  # (nq,ngt)
    elif reduce == "sum":
        return out.sum(dim=0)   # (nq,ngt)
    else:
        raise ValueError("reduce must be None/'mean'/'sum'")

def compute_mask_iou(pred_mask, gt_mask, return_type='iou'):
    # pred_mask: (nq, bs, H, W)
    # gt_mask:   (ngt, bs, H_gt, W_gt)
    nq, bs, H, W = pred_mask.shape
    ngt = gt_mask.shape[0]
    gt_mask = F.interpolate(
        gt_mask.float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )
    pred_mask = (pred_mask.sigmoid() > 0.5).float().reshape(nq, 1, bs * H * W)
    gt_mask = gt_mask.reshape(1, ngt, bs * H * W)
    result = None
    if return_type == 'iou':
        intersection = (pred_mask * gt_mask).sum(-1)
        union = pred_mask.sum(-1) + gt_mask.sum(-1) - intersection
        iou = (intersection + 1) / (union + 1)
        result = iou
    if return_type == 'dice':
        intersection = (pred_mask * gt_mask).sum(-1)
        union = pred_mask.sum(-1) + gt_mask.sum(-1)
        dice = (2 * intersection + 1) / (union + 1)
        result = dice
    return result  # (nq, ngt)

