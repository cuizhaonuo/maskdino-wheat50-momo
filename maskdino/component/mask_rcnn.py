from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
import torch

def create_maskrcnn_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    # 你可以加载权重（建议），也可以随机初始化
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.INPUT.FORMAT = "BGR"  # 这里只影响 DefaultPredictor 的预处理，我们不使用它，无所谓

    model = build_model(cfg)
    # model.eval()
    model.to(device)
    print(model)
    return model

if __name__ == "__main__":
    model = create_maskrcnn_model()
    # 模拟前向传播
    inputs = [{"image": torch.randn(3, 224, 224).to(model.device), "height": 224, "width": 224}]
    with torch.no_grad():
        outputs = model(inputs)
    print(outputs)
