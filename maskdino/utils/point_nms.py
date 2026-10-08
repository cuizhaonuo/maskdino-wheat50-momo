# -*- coding: utf-8 -*-
"""小穗点的贪心 NMS：criterion 的内层匹配前（POINT_NMS_MODE）、推理（TEST.POINT_NMS_KAPPA）
与诊断脚本（point_token_diag.py）共用这一份实现。

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
