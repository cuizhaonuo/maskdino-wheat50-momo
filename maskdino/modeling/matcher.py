# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample

from maskdino.utils.box_ops import generalized_box_iou,box_cxcywh_to_xyxy

from torch.utils.tensorboard import SummaryWriter

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,
                 cost_box: float = 0, cost_giou: float = 0, panoptic_on: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou

        self.panoptic_on = panoptic_on

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

        # 增加 tensorboard 日志记录功能

        self.tf_writer = SummaryWriter(log_dir="/public/home/zncui/pycharm_project/MaskDINO/output/maskdino_swinl-time-10-19-20-00-v1/tf_cost_logs")
        self.tf_step = 0

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, cost=["cls", "box", "mask"]):
        """More memory-friendly matching. Change cost to compute only certain loss in matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        C_list = []
        # Iterate through batch size
        for b in range(bs):
            out_bbox = outputs["pred_boxes"][b]
            # 检查问题
            if not torch.isfinite(out_bbox).all():
                print("FOUND NaN/Inf in out_bbox", out_bbox[~torch.isfinite(out_bbox)])
                raise

            # 检查 w/h 是否为负（cxcywh格式下不应出现）
            wh = out_bbox[:, 2:]
            if (wh < 0).any():
                print("NEGATIVE w/h in out_bbox:", wh.min().item(), wh.max().item())
                # 看看是不是没sigmoid
                print("out_bbox range:", out_bbox.min().item(), out_bbox.max().item())
                raise

            xyxy = box_cxcywh_to_xyxy(out_bbox)
            bad = (xyxy[:, 2:] < xyxy[:, :2]).any(dim=1)
            if bad.any():
                idx = bad.nonzero(as_tuple=False).squeeze(1)[:10]
                print("BAD xyxy examples:", xyxy[idx])
                print("source cxcywh:", out_bbox[idx])
                raise

            if 'box' in cost:
                tgt_bbox=targets[b]["boxes"]
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            else:
                cost_bbox = torch.tensor(0).to(out_bbox)
                cost_giou = torch.tensor(0).to(out_bbox)

            out_prob = outputs["pred_logits"][b].sigmoid()  # [num_queries, num_classes]
            out_iou = outputs['pred_logits_iou'][b].sigmoid()
            out_prob = out_prob * out_iou
            tgt_ids = targets[b]["labels"]
            # focal loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            # debug: 获取 neg_cost_class 和 pos_cost_class 的柱状图
            # from maskdino.analysis.tool_lab import draw_neg_and_post
            # draw_neg_and_post(neg_cost_class, pos_cost_class)

            # debug
            # from maskdino.analysis.tool_lab import temporal_diff_bar_scatter
            # temporal_diff_bar_scatter(self, neg_cost_class, pos_cost_class)

            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]
            if 'mask' in cost:
                out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
                # gt masks are already padded when preparing target
                tgt_mask = targets[b]["masks"].to(out_mask)

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                # 旧版
                # with autocast(enabled=False):
                # 新版
                with torch.amp.autocast('cuda', enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # If there's no annotations
                    if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                        # Compute the focal loss between masks
                        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                        # Compute the dice loss betwen masks
                        cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    else:
                        cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                        cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            else:
                cost_mask = torch.tensor(0).to(out_bbox)
                cost_dice = torch.tensor(0).to(out_bbox)
            
            # Final cost matrix
            if self.panoptic_on:
                isthing = tgt_ids<80
                cost_bbox[:, ~isthing] = cost_bbox[:, isthing].mean()
                cost_giou[:, ~isthing] = cost_giou[:, isthing].mean()
                cost_bbox[cost_bbox.isnan()] = 0.0
                cost_giou[cost_giou.isnan()] = 0.0

            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_box*cost_bbox
                + self.cost_giou*cost_giou
            )
            C = C.reshape(num_queries, -1).cpu()

            # if 'interm_outputs' in outputs:
            #     interm_outputs = outputs['interm_outputs']
            #     pred_boxes = interm_outputs['pred_boxes'][b].cpu() # (bs, nq, 4) --> (nq, 4)
            #     tgt_boxes = targets[b]['boxes'].to(pred_boxes.device) # (nt, 4)
            #     from maskdino.utils.box_ops import box_iou
            #     iou_matrix = box_iou(box_cxcywh_to_xyxy(pred_boxes),
            #                          box_cxcywh_to_xyxy(tgt_boxes))[0]
            #     # ===== min matching results
            #     # iou_indices = C.min(dim=-1)[1] # (nq, nt) --> (nq,)
            #     # nq = pred_boxes.shape[0]
            #     # query_indices = torch.arange(nq, dtype=torch.int64, device=pred_boxes.device)
            #     # pred_iou = interm_outputs['pred_logits_iou'][b].sigmoid().to(pred_boxes.device) # (nq,)
            #     # ===== hungary matching results
            #     query_indices, iou_indices = linear_sum_assignment(C)
            #     tgt_iou = iou_matrix[query_indices, iou_indices]  # (nq,)
            #     pred_iou = interm_outputs['pred_logits_iou'][b].sigmoid()[query_indices].to(pred_boxes.device) # (nq,)
            #     from matplotlib import pyplot as plt
            #     plt.figure()
            #     nq = query_indices.shape[0]
            #     x_indices = torch.arange(nq, dtype=torch.int64, device=pred_boxes.device)
            #     plt.plot(x_indices, tgt_iou, marker='o', color='orange', label='tgt_iou')
            #     plt.plot(x_indices, pred_iou, marker='s', color='green', label='pred_iou')
            #     plt.xlabel('query id')
            #     plt.ylabel('iou value')
            #     plt.title('tgt_iou vs pred_iou')
            #     plt.legend()
            #     plt.savefig(f'hungary_matching_iou_plot_batch{b}.png')
            #     plt.close()

            # debug: 可视化 cost_xxx 在 C 中的占比
            # from maskdino.analysis.tool_lab import visualize_cost_components, visualize_cost_components_bar_sorted
            # no weighted
            # visualize_cost_components(C, cost_class, cost_mask, cost_dice,
            #                           cost_bbox, cost_giou)
            # add weighted
            # visualize_cost_components(C,
            #                           self.cost_class * cost_class,
            #                           self.cost_mask * cost_mask,
            #                           self.cost_dice * cost_dice,
            #                           self.cost_box * cost_bbox,
            #                           self.cost_giou * cost_giou,
            #                           outputs,
            #                           targets)
            # visualize_cost_components_bar_sorted(C,
            #                           self.cost_class * cost_class,
            #                           self.cost_mask * cost_mask,
            #                           self.cost_dice * cost_dice,
            #                           self.cost_box * cost_bbox,
            #                           self.cost_giou * cost_giou,
            #                           outputs,
            #                           targets)


            # debug: 绘制cost矩阵热力图
            # from maskdino.analysis.tool_lab import save_cost_matrix_heatmap
            # src_id, tgt_id = linear_sum_assignment(C)
            # save_cost_matrix_heatmap(C, src_id, tgt_id, 'cost_matrix_C.jpg')

            # debug: 检验是否 cost 最小的行就是匹配结果
            # print(f"batch idx: {b}")
            # import numpy as np
            # C = np.asarray(C)
            # src_id, tgt_id = linear_sum_assignment(C)
            # min_cost_rows = np.sum(C, axis=1)
            # idx = np.argsort(min_cost_rows)[:len(tgt_id)]
            # # min_vals = min_cost_rows[idx]
            # # for i, v in zip(idx, min_vals):
            # #     print(f"Row {i} min cost: {v}")
            # matched_count = 0
            # for i in src_id:
            #     if i in idx:
            #         matched_count += 1
            #         # print("index match found in min cost rows:", i)
            # print(f"Matched count in min cost rows: {matched_count} / {len(tgt_id)}")

            C_list.append(C)
            # C_list.append(self.cost_class * cost_class.cpu())
            indices.append(linear_sum_assignment(C))


        # debug: 统计每个query被匹配的次数
        # from maskdino.analysis.tool_lab import count_query_frequency
        # count_query_frequency(indices)

        # debug: 可视化 target 对应的每个 query
        # from maskdino.analysis.tool_lab import visualize_matched_queries
        # visualize_matched_queries(outputs, targets, indices, C_list)

        # debug: 绘制 class 柱形图
        # from maskdino.analysis.tool_lab import plot_class_distribution
        # plot_class_distribution(outputs)

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, cost=["cls", "box", "mask"]):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, cost)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
