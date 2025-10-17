from torch import nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.config import get_cfg, configurable
from detectron2.config import CfgNode as CN
from detectron2.modeling.backbone import build_backbone

class ToyBackbone(Backbone):
  @configurable()
  def __init__(self, cfg, a=1, b=2):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.a = a
    self.b = b

  @classmethod
  def from_config(self, cfg):
    a = cfg.MODEL.TOYBACKBONE.A
    b = cfg.MODEL.TOYBACKBONE.B
    return {
      "cfg": cfg,
      "a": a,
      "b": b
    }

  def forward(self, x):
    return {'conv1': self.conv1(x)}

@BACKBONE_REGISTRY.register()
def build_toy_backbone(cfg, input_shape=None):
  return ToyBackbone(cfg)

if __name__ == "__main__":
  cfg = get_cfg()
  cfg.MODEL.BACKBONE.NAME = "build_toy_backbone"
  cfg.MODEL.TOYBACKBONE = CN()
  cfg.MODEL.TOYBACKBONE.A = 10
  cfg.MODEL.TOYBACKBONE.B = 20
  backbone = build_backbone(cfg)
  print(backbone)
  print(backbone.a, backbone.b)