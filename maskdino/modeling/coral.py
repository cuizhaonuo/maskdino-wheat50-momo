"""CORAL：秩一致的序数回归（Cao, Mirjalili, Raschka, Pattern Recognition Letters 2020，arXiv 1901.07884）。

本地源码 /public/home/zncui/pycharm_project/coral-cnn。K 个序数类拆成 T=K-1 个二分类任务「y > k」，
所有任务【共享同一个 1 维打分 g(x)】，各自只有一个独立 bias：logit_k = g(x) + b_k。
论文定理 1：最优解上 b_1 >= b_2 >= ... >= b_T，于是 P(y>k) 随 k 单调不增，任务之间不会互相矛盾。

与论文代码（model-code/afad-coral.py）的对应与差异：
  - CoralHead         <-> resnet 末端的 fc(512 -> 1, bias=False) + linear_1_bias（:275-276、:316-317）。
                         打分从单层 fc 换成 MLP，与本仓库其余预测头同构；bias 按 coral_pytorch 的
                         preinit_bias 初始化为降序 T/T, (T-1)/T, ..., 1/T，论文仓库是全零（两者都满足定理的前提）。
  - coral_levels      <-> CustomDataset 里的 levels = [1]*label + [0]*(K-1-label)（:167）。
  - coral_loss        <-> cost_fn（:335-338）。论文对 T 个任务【求和】，这里【取均值】（差常数 1/T），
                         以便与本仓库其他损失、深监督的 11 份输出放在同一量级；样本维做加权均值。
                         论文的 task importance 权重（imp，默认全 1）不用。
  - coral_predict     <-> predicted_labels = sum(probas > 0.5)（:358-359）。按求和计数，不要求 bias 已排好序。
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.utils import MLP


class CoralHead(nn.Module):
    """共享 1 维打分 + T 个独立 bias：输出 [..., T] 的 CORAL logit。"""

    def __init__(self, input_dim, hidden_dim, num_tasks, num_layers=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.score = MLP(input_dim, hidden_dim, 1, num_layers)
        self.bias = nn.Parameter(torch.arange(num_tasks, 0, -1).float() / num_tasks)

    def forward(self, x):
        return self.score(x) + self.bias


def coral_levels(y, num_tasks):
    """序数标签 y（整数张量，任意形状）-> levels [..., T]，第 k 位 = 1[y > k]。y 先截到 [0, T]。"""
    y = y.long().clamp(0, num_tasks)
    k = torch.arange(num_tasks, device=y.device)
    return (y.unsqueeze(-1) > k).float()


def coral_loss(logits, levels, weights=None):
    """logits / levels: [N, T]；weights: [N] 或 None。返回标量。

    每个样本：-mean_k[ logsig(l_k)*y_k + (logsig(l_k) - l_k)*(1 - y_k) ]，即 T 个 BCE 的均值；
    样本之间按 weights 加权平均。N=0 时返回 0（保持计算图）。
    """
    logits = logits.float()
    per_task = F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels)
    per_sample = -per_task.mean(dim=-1)
    if per_sample.numel() == 0:
        return logits.sum() * 0.0
    if weights is None:
        return per_sample.mean()
    weights = weights.to(per_sample)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)


def coral_predict(logits, threshold=0.5):
    """[..., T] -> [...] 的整数计数：sum_k 1[sigmoid(l_k) > threshold]。"""
    return (logits.sigmoid() > threshold).sum(dim=-1)
