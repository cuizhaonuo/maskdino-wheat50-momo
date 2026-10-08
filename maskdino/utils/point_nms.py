# -*- coding: utf-8 -*-
"""小穗点的贪心 NMS：criterion 的内层匹配前（POINT_NMS_MODE）、推理（TEST.POINT_NMS_KAPPA）
与诊断脚本（point_token_diag.py）共用 greedy_point_nms；decoder 的逐层剔除（POINT_NMS_MODE="prune"）
用规则完全相同的 GPU 向量化版 greedy_point_nms_batched。

做法来自 DDQ（Zhang et al., "Dense Distinct Query for End-to-End Object Detection", CVPR 2023）：
匹配前先去掉近重复，被压掉的不进匹配、也不算损失；推理时同样做。
"""
import numpy as np
import torch


@torch.no_grad()
def greedy_point_nms(xy, scores, radius):
    """按分数降序贪心做点 NMS，返回留下的掩码。支持前置批维度：每组点各自独立做，半径整批共用。

    规则：按分数降序逐个处理，分数并列时下标小的先处理（结果稳定）。还没被压掉的点留下，
    并压掉所有与它中心距**严格小于** radius 的未处理点；被压掉的点不再去压别人。
    等价表述：一个点留下，当且仅当比它先处理、且已经留下的点里没有一个离它 < radius。
    处理顺序只看分数，所以「先 NMS 再按分数阈值过滤」与「先过滤再 NMS」结果相同：
    高于阈值的点全排在不高于阈值的点之前，后者压不到前者。

    实现：中心距按平方比较（d² < r²，r > 0），只用减、乘、加，CPU 上向量化与标量路径逐位一致，
    所以批量做与逐组做的结果相同。距离在输入所在设备上一次算完；每行的近邻压成位掩码搬到 CPU，
    压制用 Python 整数位运算（P <= 62）。

    Args:
        xy:     [..., P, 2] 点坐标（任意设备；坐标单位与 radius 一致即可）
        scores: [..., P] 或 [..., P, 1] 分数（越大越优先）
        radius: 标量或 0-dim tensor；<= 0 时不压任何点
    Returns:
        BoolTensor [..., P]，与 xy 同设备；True = 留下
    """
    device = xy.device
    *batch, P, _ = xy.shape
    if P == 0 or xy.numel() == 0:
        return torch.zeros(*batch, P, dtype=torch.bool, device=device)
    assert P <= 62, f"位掩码实现要求每组点数 P <= 62，实际 {P}"
    xy = xy.detach().reshape(-1, P, 2)
    scores = scores.detach().reshape(-1, P)
    diff = xy[:, :, None, :] - xy[:, None, :, :]                              # [M, P, P, 2]
    d2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]            # [M, P, P]
    r = torch.as_tensor(radius, dtype=d2.dtype, device=device)
    close = (d2 < r * r) & (r > 0)                                            # 严格小于
    bits = (close.cpu().numpy().astype(np.int64) << np.arange(P, dtype=np.int64)).sum(-1).tolist()   # [M][P]
    orders = torch.sort(scores, dim=-1, descending=True, stable=True).indices.cpu().tolist()         # [M][P]
    keep = np.zeros((len(orders), P), dtype=bool)
    for m, (order, row) in enumerate(zip(orders, bits)):
        removed = 0
        for i in order:
            if (removed >> i) & 1:
                continue
            keep[m, i] = True
            removed |= row[i]
    return torch.from_numpy(keep).to(device).reshape(*batch, P)


@torch.no_grad()
def greedy_point_nms_batched(xy, scores, radius, valid=None):
    """greedy_point_nms 的 GPU 向量化批量版：N 组点各自独立做贪心 NMS，规则与 greedy_point_nms 完全一致，
    另加 valid 掩码。给 decoder 的逐层剔除用（每层 B×Q 个实例、训练与推理都做），全程不离开输入设备、
    不做 GPU→CPU 同步。

    规则（与 greedy_point_nms 相同）：按分数降序处理，并列时下标小的先处理；还没被压掉的点留下，
    并压掉之后所有与它中心距**严格小于** radius 的点（d² < r²）；被压掉的点不再压别人；r <= 0 时不压。
    valid=False 的点不留下、也不压别人，即等价于「只对 valid 子集调 greedy_point_nms」：
    稳定降序排序限制到子集上，就是子集自己的稳定降序排序，处理顺序与压制关系都不变。

    实现：每组按分数稳定降序排好，距离在排好序的点上一次算完（d² 的逐元素运算与 greedy_point_nms 相同，
    同设备、同 dtype 下逐位一致）；再按处理顺序循环 K 步，每步对 N 组同时做：第 t 个点若仍存活就留下，
    并把它之后与它相近的点标为失效。第 t 步之后不再改动前 t+1 个位置，所以循环结束时的存活标记就是结果。

    Args:
        xy:     [N, K, 2] 点坐标（任意设备；坐标单位与 radius 一致即可）
        scores: [N, K] 或 [N, K, 1] 分数（越大越优先）
        radius: 标量、0-dim tensor，或 [N]（逐组半径）；<= 0 的组不压任何点
        valid:  [N, K] bool 或 None（None = 全部有效）
    Returns:
        BoolTensor [N, K]，与 xy 同设备；True = 留下（必为 valid 的子集）
    """
    device = xy.device
    N, K, _ = xy.shape
    if valid is None:
        valid = torch.ones(N, K, dtype=torch.bool, device=device)
    else:
        valid = valid.to(device=device, dtype=torch.bool)
    if N == 0 or K == 0:
        return torch.zeros(N, K, dtype=torch.bool, device=device)
    xy = xy.detach()
    scores = scores.detach().reshape(N, K)
    # 处理顺序：分数降序、并列按下标（稳定排序，与 greedy_point_nms 同一调用）
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices          # [N, K]
    xy_s = xy.gather(1, order.unsqueeze(-1).expand(N, K, 2))                          # 按处理顺序排好的点
    diff = xy_s[:, :, None, :] - xy_s[:, None, :, :]                                  # [N, K, K, 2]
    d2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]                    # [N, K, K]
    r = torch.as_tensor(radius, dtype=d2.dtype, device=device)
    if r.dim() == 1:
        r = r[:, None, None]                                                          # 逐组半径
    close = (d2 < r * r) & (r > 0)                                                    # 严格小于
    alive = valid.gather(1, order)                                                    # [N, K]，按处理顺序
    for t in range(K - 1):
        # 第 t 个点仍存活 → 留下，压掉它之后与它相近的点；已失效的点不压别人
        alive[:, t + 1:] &= ~(close[:, t, t + 1:] & alive[:, t:t + 1])
    keep = torch.empty_like(alive)
    keep.scatter_(1, order, alive)                                                    # 还原到原下标
    return keep
