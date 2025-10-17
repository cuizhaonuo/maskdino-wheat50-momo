"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial

from detectron2.config import get_cfg, configurable
from detectron2.config import CfgNode as CN
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.backbone import build_backbone


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)    # 1/4
        c2 = self.conv2(c1)  # 1/8
        c3 = self.conv3(c2)  # 1/16
        c4 = self.conv4(c3)  # 1/32

        return c1, c2, c3, c4

@BACKBONE_REGISTRY.register()
class DINOv3STAs(Backbone):
    @configurable()
    def __init__(
            self,
            name=None,
            weights_path=None,
            interaction_indexes=[],
            finetune=True,
            embed_dim=192,
            num_heads=3,
            patch_size=16,
            use_sta=True,
            conv_inplane=16,
            hidden_dim=None,
    ):
        super(DINOv3STAs, self).__init__()
        if 'dinov3' in name:
            self.dinov3 = torch.hub.load('/public/home/zncui/pycharm_project/dinov3', name, source='local', weights=weights_path)
            while len(self.dinov3.blocks) != (interaction_indexes[-1] + 1):
                del self.dinov3.blocks[-1]
            del self.dinov3.head

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane * 1, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane * 2, 384, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane * 4, 768, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane * 4, 1536, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(192,),
            nn.SyncBatchNorm(384,),
            nn.SyncBatchNorm(768),
            nn.SyncBatchNorm(1536)
        ])

        self._out_features =  ["res2", "res3", "res4", "res5"]

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2":192,
            "res3":384,
            "res4":768,
            "res5":1536,
        }

    @classmethod
    def from_config(cls, cfg, input_shape=None):
        name = cfg.MODEL.DINOV3_STA.NAME
        weights_path = cfg.MODEL.DINOV3_STA.WEIGHTS_PATH
        interaction_indexes = cfg.MODEL.DINOV3_STA.INTERACTION_INDEXES
        finetune = cfg.MODEL.DINOV3_STA.FINETUNE
        embed_dim = cfg.MODEL.DINOV3_STA.EMBED_DIM
        num_heads = cfg.MODEL.DINOV3_STA.NUM_HEADS
        patch_size = cfg.MODEL.DINOV3_STA.PATCH_SIZE
        use_sta = cfg.MODEL.DINOV3_STA.USE_STA
        conv_inplane = cfg.MODEL.DINOV3_STA.CONV_INPLANE
        hidden_dim = cfg.MODEL.DINOV3_STA.HIDDEN_DIM

        return {
            "name": name,
            "weights_path": weights_path,
            "interaction_indexes": interaction_indexes,
            "finetune": finetune,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "patch_size": patch_size,
            "use_sta": use_sta,
            "conv_inplane": conv_inplane,
            "hidden_dim": hidden_dim,
        }

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        bs, C, h, w = x.shape

        all_layers = self.dinov3.get_intermediate_layers(
            x, n=self.interaction_indexes, return_class_token=True
        )

        if len(all_layers) == 1:  # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2 ** (num_scales - i)), int(W_c * 2 ** (num_scales - i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))
        c5 = self.norms[3](self.convs[3](fused_feats[3]))

        out = {'res2': c2, 'res3': c3, 'res4': c4, 'res5': c5}
        return out

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.MODEL.BACKBONE.NAME = "DINOv3STAs"
    # model config
    cfg.MODEL.DINOV3_STA = CN()
    # dinov3
    cfg.MODEL.DINOV3_STA.NAME = "dinov3_vit7b16"
    cfg.MODEL.DINOV3_STA.WEIGHTS_PATH = "/public/home/zncui/pycharm_project/dinov3/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    cfg.MODEL.DINOV3_STA.INTERACTION_INDEXES = [9, 19, 29, 39]
    cfg.MODEL.DINOV3_STA.FINETUNE = False
    cfg.MODEL.DINOV3_STA.EMBED_DIM = 4096
    cfg.MODEL.DINOV3_STA.HIDDEN_DIM = 256
    # optional
    cfg.MODEL.DINOV3_STA.NUM_HEADS = 3
    cfg.MODEL.DINOV3_STA.PATCH_SIZE = 16
    # STA
    cfg.MODEL.DINOV3_STA.USE_STA = True
    cfg.MODEL.DINOV3_STA.CONV_INPLANE = 32

    backbone = build_backbone(cfg)
    print(backbone)

    inputs = torch.randn(1, 3, 1024, 1024)

    import time
    start = time.time()
    outputs = backbone(inputs)
    end = time.time()
    print("backbone time: ", end - start)

    for k, out in outputs.items():
        print(out.shape)