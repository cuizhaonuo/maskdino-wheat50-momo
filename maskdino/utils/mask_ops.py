from torch.nn import functional as F

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