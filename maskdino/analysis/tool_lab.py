import torch

from maskdino.utils import box_ops

def check_recall(refpoint_embed, targets):
    pred_bbox = refpoint_embed
    for pred_bbox, tgt in zip(pred_bbox, targets):
        tgt_bbox = tgt['boxes']

        def compute_recall(pred, tgt):
            if len(tgt) == 0:
                return 1
            # pred, tgt 都是相对坐标，需要转成绝对坐标
            pred = box_ops.box_cxcywh_to_xyxy(pred)
            tgt = box_ops.box_cxcywh_to_xyxy(tgt)
            h, w = 1024, 1024
            pred = pred * torch.as_tensor([w, h, w, h], dtype=torch.float).to(pred.device)
            tgt = tgt * torch.as_tensor([w, h, w, h], dtype=torch.float).to(tgt.device)
            # 计算 iou
            ious, unious = box_ops.box_iou(pred, tgt)  # (num_queries, num_target)
            iou_max, _ = ious.max(dim=0)
            recall_50 = (iou_max >= 0.5).sum() / len(tgt)
            recall_75 = (iou_max >= 0.75).sum() / len(tgt)
            return recall_50.item(), recall_75.item()

        recall_50, recall_75 = compute_recall(pred_bbox, tgt_bbox)
        print('recall_50', recall_50)
        print('recall_75', recall_75)


# 显示 images 图像
def show_image(images):
    import cv2
    for idx, img in enumerate(images):
        img = img.permute(1, 2, 0).cpu().numpy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_path = f"/public/home/zncui/pycharm_project/MaskDINO/myscripts/output_{idx}.png"
        cv2.imwrite(save_path, img_bgr)

# 实现 backbone 热力图
def _show_heatmap(features, batched_inputs):
    # features 是一个字典，key 是层名字，value 是 Tensor, (B, C, H, W)
    # batched_inputs 是一个列表，里面是字典，字典里有 'image' 键，对应的值是 Tensor, (C, H, W)
    from mmengine.visualization import Visualizer
    import numpy as np
    # image 需要 rgb, 0-255, hwc, uint8, ndarray(numpy())
    # feature 需要 Tensor, chw
    image = batched_inputs[0]['image'].permute(1, 2, 0).cpu().numpy()
    image = image.astype(np.uint8)
    visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
                            save_dir='/public/home/zncui/pycharm_project/MaskDINO/myscripts/output_heatmap')
    for layer, feature in features.items():
        feature = feature[0]  # 取第一个样本
        drawn_image = visualizer.draw_featmap(feature, image, channel_reduction='select_max')
        visualizer.add_image(f"heatmap_{layer}", drawn_image)

def show_heatmap(features, batched_inputs, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from mmengine.visualization import Visualizer

    # 1) 原始图像转 HWC uint8
    image = batched_inputs[0]['image'].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # 2) 可视化器
    visualizer = Visualizer(image=image)

    # 3) 收集所有层的可视化结果
    drawn_images = []
    layer_names = []
    for layer, feat in features.items():
        feat = feat[0]  # B,C,H,W → C,H,W
        drawn = visualizer.draw_featmap(feat, image, channel_reduction='squeeze_mean')
        drawn_images.append(drawn)
        layer_names.append(layer)

    # 4) 用 matplotlib 拼接显示/保存
    n = len(drawn_images)
    cols = min(4, n)        # 一行最多放 4 张
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    for i, (img, name) in enumerate(zip(drawn_images, layer_names)):
        axes[i].imshow(img)
        axes[i].set_title(name)
        axes[i].axis("off")

    # 把多余的 subplot 去掉
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"拼接热力图已保存到: {save_path}")
    else:
        plt.show()

def check_topk_proposals(topk_proposals, level_start_index):
    import math
    level1_counts = 0
    level2_counts = 0
    level3_counts = 0
    level4_counts = 0
    level5_counts = 0

    topk_proposals = topk_proposals[0]
    for proposal in topk_proposals:
        if level_start_index[0] <= proposal < level_start_index[1]:
            level5_counts += 1
        elif level_start_index[1] <= proposal < level_start_index[2]:
            level4_counts += 1
        elif level_start_index[2] <= proposal < level_start_index[3]:
            level3_counts += 1
        elif level_start_index[3] <= proposal < level_start_index[4]:
            level2_counts += 1
        elif level_start_index[4] <= proposal < math.inf:
            level1_counts += 1
        else:
            raise ValueError("proposal 超出范围")
    print(f"level1: {level1_counts}, level2: {level2_counts}, level3: {level3_counts}, level4: {level4_counts}, level5: {level5_counts}")

def draw_bboxes_on_image(bboxes):
    import mmcv
    import numpy as np
    import cv2
    image = mmcv.imread("/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/test2017/D6-017-20230529.jpg", channel_order='bgr')
    image = image.astype(np.uint8)
    bboxes = box_ops.box_cxcywh_to_xyxy(bboxes).cpu().numpy()
    h, w = image.shape[:2]
    bboxes = bboxes * np.array([w, h, w, h], dtype=np.float32)
    bboxes = bboxes.astype(np.int32)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/bboxes.jpg"
    mmcv.imwrite(image, save_path)

def draw_encoder_heatmap(x):
    import mmcv
    import numpy as np
    from mmengine.visualization import Visualizer
    from matplotlib import pyplot as plt

    image = mmcv.imread("/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/test2017/D6-017-20230529.jpg", channel_order='rgb')
    image = image.astype(np.uint8)
    visualizer = Visualizer()
    drawn_images = []
    for idx, feat in enumerate(x):
        feat = feat[0]
        drawn_image = visualizer.draw_featmap(feat, image, channel_reduction='squeeze_mean')
        drawn_images.append(drawn_image)

    cols = 5
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    for i, img in enumerate(drawn_images):
        axes[i].imshow(img)
        axes[i].set_title(f"level_{i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/encoder_heatmap.jpg"
    plt.savefig(save_path, dpi=200)
    print(f"拼接热力图已保存到: {save_path}")

def _draw_bboxes_on_heatmap(x, bboxes, topk_proposals, level_start_index):
    import mmcv
    import numpy as np
    import cv2
    from mmengine.visualization import Visualizer
    from matplotlib import pyplot as plt
    import math

    image = mmcv.imread("/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/test2017/D6-017-20230529.jpg", channel_order='rgb')
    image = image.astype(np.uint8)
    visualizer = Visualizer()
    drawn_images = []
    for idx, feat in enumerate(x):
        feat = feat[0]
        drawn_image = visualizer.draw_featmap(feat, image, channel_reduction='squeeze_mean')
        drawn_images.append(drawn_image)

    bboxes = box_ops.box_cxcywh_to_xyxy(bboxes).cpu().numpy()
    h, w = image.shape[:2]
    bboxes = bboxes * np.array([w, h, w, h], dtype=np.float32)
    bboxes = bboxes.astype(np.int32)

    def _draw_box_cv(img, box_xyxy, color=(255, 0, 0), thickness=2):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, box_xyxy)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        # 绘制中心点
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(img, (cx, cy), radius=2, color=color, thickness=5, lineType=cv2.LINE_AA)

    # 用 cv2 在对应 level 的图上画框
    for idx, proposal in enumerate(topk_proposals.squeeze(0)):
        p = int(proposal)  # 防止是 tensor/np 标量
        if level_start_index[0] <= p < level_start_index[1]:
            _draw_box_cv(drawn_images[0], bboxes[idx], color=(255, 0, 0), thickness=2)
        elif level_start_index[1] <= p < level_start_index[2]:
            _draw_box_cv(drawn_images[1], bboxes[idx], color=(255, 0, 0), thickness=2)
        elif level_start_index[2] <= p < level_start_index[3]:
            _draw_box_cv(drawn_images[2], bboxes[idx], color=(255, 0, 0), thickness=2)
        elif level_start_index[3] <= p < level_start_index[4]:
            _draw_box_cv(drawn_images[3], bboxes[idx], color=(255, 0, 0), thickness=2)
        elif level_start_index[4] <= p < math.inf:
            _draw_box_cv(drawn_images[4], bboxes[idx], color=(255, 0, 0), thickness=2)
        else:
            raise ValueError("proposal 超出范围")

    n = len(drawn_images)
    cols = 5
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    for i, img in enumerate(drawn_images):
        axes[i].imshow(img)
        axes[i].set_title(f"level_{n - i}")
        axes[i].axis("off")
    fig.tight_layout()
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/encoder_heatmap_circles.jpg"
    plt.savefig(save_path, dpi=200)

def draw_bboxes_on_heatmap(x, bboxes, topk_proposals, level_start_index):
    import mmcv
    import numpy as np
    import cv2
    from mmengine.visualization import Visualizer
    from matplotlib import pyplot as plt
    import math

    image = mmcv.imread("/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/test2017/D6-017-20230529.jpg", channel_order='rgb')
    image = image.astype(np.uint8)
    visualizer = Visualizer()

    # 绘制 feature map 背景图
    drawn_images = []
    for idx, feat in enumerate(x):
        feat = feat[0]
        drawn_image = visualizer.draw_featmap(feat, image, channel_reduction='squeeze_mean')
        drawn_images.append(drawn_image)

    # 转换 bbox 坐标到 xyxy
    bboxes = box_ops.box_cxcywh_to_xyxy(bboxes).cpu().numpy()
    h, w = image.shape[:2]
    bboxes = bboxes * np.array([w, h, w, h], dtype=np.float32)
    bboxes = bboxes.astype(np.int32)

    def _draw_rect(img, box_xyxy, color=(255, 0, 0), thickness=2):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, box_xyxy)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    def _draw_circle(img, box_xyxy, color=(255, 0, 0), thickness=5):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, box_xyxy)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(img, (cx, cy), radius=2, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # 准备两份绘制结果：矩形图 & 圆点图
    drawn_images_rect = [img.copy() for img in drawn_images]
    drawn_images_circle = [img.copy() for img in drawn_images]

    topk_proposals = topk_proposals.squeeze(0)
    for idx, proposal in enumerate(topk_proposals):
        p = int(proposal)
        if level_start_index[0] <= p < level_start_index[1]:
            _draw_rect(drawn_images_rect[0], bboxes[idx])
            _draw_circle(drawn_images_circle[0], bboxes[idx])
        elif level_start_index[1] <= p < level_start_index[2]:
            _draw_rect(drawn_images_rect[1], bboxes[idx])
            _draw_circle(drawn_images_circle[1], bboxes[idx])
        elif level_start_index[2] <= p < level_start_index[3]:
            _draw_rect(drawn_images_rect[2], bboxes[idx])
            _draw_circle(drawn_images_circle[2], bboxes[idx])
        elif level_start_index[3] <= p < level_start_index[4]:
            _draw_rect(drawn_images_rect[3], bboxes[idx])
            _draw_circle(drawn_images_circle[3], bboxes[idx])
        elif level_start_index[4] <= p < math.inf:
            _draw_rect(drawn_images_rect[4], bboxes[idx])
            _draw_circle(drawn_images_circle[4], bboxes[idx])
        else:
            raise ValueError("proposal 超出范围")

    # 拼接显示：第一行矩形，第二行中心点
    n = len(drawn_images)
    cols = n
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    for i in range(n):
        axes[0, i].imshow(drawn_images[i])
        axes[0, i].set_title(f"level_{n - i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(drawn_images_rect[i])
        axes[1, i].set_title(f"level_{n - i} (rect)")
        axes[1, i].axis("off")

        axes[2, i].imshow(drawn_images_circle[i])
        axes[2, i].set_title(f"level_{n - i} (center)")
        axes[2, i].axis("off")

    fig.tight_layout()
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/encoder_heatmap_rect_and_circles_1000.jpg"
    plt.savefig(save_path, dpi=200)
