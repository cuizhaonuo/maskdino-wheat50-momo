import torch

from maskdino.utils import box_ops

IMG_PATH = "/public/home/zncui/datasets/xinxiang/cropped_100images_for_train/P177-19-pos_5435_5460_5163_151_2960_264_3223_5460_5435_5460-20250508075707_0142_colorchart_146.75/P177-19-pos_5435_5460_5163_151_2960_264_3223_5460_5435_5460-20250508075707_0142_colorchart_146.75_y0_x0.jpg"

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
    image = mmcv.imread(IMG_PATH, channel_order='bgr')
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

    image = mmcv.imread(IMG_PATH, channel_order='rgb')
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

    image = mmcv.imread(IMG_PATH, channel_order='rgb')
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



def draw_bboxes_on_heatmap(x, bboxes, proposal_boxes, topk_proposals, level_start_index):
    '''
    x: List[Tensor], 每个 Tensor 形状为 (B, C, H, W)
    bboxes: Tensor, 形状为 (num_boxes, 4)，相对坐标格式为 cxcywh
    topk_proposals: Tensor, 形状为 (num_boxes,)，每个值表示该 bbox 来自哪个特征层
    level_start_index: List[int], 每个特征层的起始索引，用于判断 bbox 属于哪个层级
    该函数会在每个特征层的热力图上绘制对应的 bbox 框和中心点，并保存拼接后的图像。
    结果图像会包含三行：第一行是原始热力图，第二行是绘制矩形框的热力图，第三行是绘制中心点的热力图。
    结果图像会保存到指定路径。
    '''
    import mmcv
    import numpy as np
    import cv2
    from mmengine.visualization import Visualizer
    from matplotlib import pyplot as plt
    import math

    image = mmcv.imread(IMG_PATH, channel_order='rgb')
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

    # 转换 proposal_bboxes 坐标到 xyxy
    proposal_boxes = box_ops.box_cxcywh_to_xyxy(proposal_boxes).cpu().numpy()
    proposal_boxes = proposal_boxes * np.array([w, h, w, h], dtype=np.float32)
    proposal_boxes = proposal_boxes.astype(np.int32)

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

    def _draw_line(img, box_xyxy, proposal_xyxy, color=(255, 0, 0), thickness=2):
        x1, y1, x2, y2 = box_xyxy
        px1, py1, px2, py2 = proposal_xyxy
        # 计算中心点
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        pcx = (px1 + px2) // 2
        pcy = (py1 + py2) // 2
        # 绘制中心点
        cv2.circle(img, (cx, cy), radius=1, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
        cv2.circle(img, (pcx, pcy), radius=1, color=(255, 0, 0), thickness=5, lineType=cv2.LINE_AA)
        # 绘制直线
        cv2.line(img, (pcx, pcy), (cx, cy), color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # 准备两份绘制结果：矩形图 & 圆点图
    drawn_images_rect = [img.copy() for img in drawn_images]
    drawn_images_circle = [img.copy() for img in drawn_images]
    drawn_images_lines = [img.copy() for img in drawn_images]

    for idx, proposal in enumerate(topk_proposals):
        p = int(proposal)
        if level_start_index[0] <= p < level_start_index[1]:
            _draw_rect(drawn_images_rect[0], bboxes[idx])
            _draw_circle(drawn_images_circle[0], bboxes[idx])
            _draw_line(drawn_images_lines[0], bboxes[idx], proposal_boxes[idx])
        elif level_start_index[1] <= p < level_start_index[2]:
            _draw_rect(drawn_images_rect[1], bboxes[idx])
            _draw_circle(drawn_images_circle[1], bboxes[idx])
            _draw_line(drawn_images_lines[1], bboxes[idx], proposal_boxes[idx])
        elif level_start_index[2] <= p < level_start_index[3]:
            _draw_rect(drawn_images_rect[2], bboxes[idx])
            _draw_circle(drawn_images_circle[2], bboxes[idx])
            _draw_line(drawn_images_lines[2], bboxes[idx], proposal_boxes[idx])
        elif level_start_index[3] <= p < level_start_index[4]:
            _draw_rect(drawn_images_rect[3], bboxes[idx])
            _draw_circle(drawn_images_circle[3], bboxes[idx])
            _draw_line(drawn_images_lines[3], bboxes[idx], proposal_boxes[idx])
        elif level_start_index[4] <= p < math.inf:
            _draw_rect(drawn_images_rect[4], bboxes[idx])
            _draw_circle(drawn_images_circle[4], bboxes[idx])
            _draw_line(drawn_images_lines[4], bboxes[idx], proposal_boxes[idx])
        else:
            raise ValueError("proposal 超出范围")

    # 拼接显示：第一行矩形，第二行中心点
    n = len(drawn_images)
    cols = n
    rows = 4
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

        axes[3, i].imshow(drawn_images_lines[i])
        axes[3, i].set_title(f"level_{n - i} (line)")
        axes[3, i].axis("off")

    fig.tight_layout()
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/encoder_heatmap_rect_and_circles_1000.jpg"
    plt.savefig(save_path, dpi=200)

def draw_center_point_on_image(pred_bboxes, proposals):
    '''
    pred_bboxes: Tensor, 形状为 (num_boxes, 4)，相对坐标格式为 cxcywh
    proposals: Tensor, 形状为 (num_boxes, 4)，相对坐标格式为 cxcywh
    '''
    import mmcv
    import numpy as np
    import cv2
    image = mmcv.imread(IMG_PATH, channel_order='bgr')
    image = image.astype(np.uint8)

    num_bboxes = pred_bboxes.shape[0]

    bboxes = box_ops.box_cxcywh_to_xyxy(pred_bboxes).cpu().numpy()
    h, w = image.shape[:2]
    bboxes = bboxes * np.array([w, h, w, h], dtype=np.float32)
    bboxes = bboxes.astype(np.int32)

    proposals = box_ops.box_cxcywh_to_xyxy(proposals).cpu().numpy()
    proposals = proposals * np.array([w, h, w, h], dtype=np.float32)
    proposals = proposals.astype(np.int32)

    # 将 proposals 的中心点为蓝色，pred_bboxes 的中心点为红色，另外每个索引取出两个点，绘制直线，直线为黑色
    for i in range(num_bboxes):
        x1, y1, x2, y2 = bboxes[i]
        px1, py1, px2, py2 = proposals[i]
        # 计算中心点
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        pcx = (px1 + px2) // 2
        pcy = (py1 + py2) // 2
        # 绘制中心点
        cv2.circle(image, (cx, cy), radius=1, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
        cv2.circle(image, (pcx, pcy), radius=1, color=(255, 0, 0), thickness=5, lineType=cv2.LINE_AA)
        # 绘制直线
        cv2.line(image, (pcx, pcy), (cx, cy), color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # 保存图像
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/center_points_and_lines.jpg"
    mmcv.imwrite(image, save_path)

def draw_encoder_class_heatmap(class_logits, level_start_indices, spatial_shapes):
    '''
    class_logits: Tensor, 形状为 (num_queries,)
    level_start_indices: List[int], 每个特征层的起始索引
    spatial_shapes, Tensor (
    '''
    # 根据 level_start_indices 将 class_logits 分成多个部分，分别绘制热力图
    num_feature_levels = 5
    split_logits = []
    for i in range(num_feature_levels):
        start_idx = level_start_indices[i]
        end_idx = level_start_indices[i + 1] if i + 1 < len(level_start_indices) else class_logits.shape[0]
        split_logits.append(class_logits[start_idx:end_idx])
    class_features = []
    spatial_shapes = list(spatial_shapes)
    for logit, shape in zip(split_logits, spatial_shapes):
        h, w = shape
        logit_reshaped = logit.reshape(h, w)
        class_features.append(logit_reshaped)

    import mmcv
    import numpy as np
    from mmengine.visualization import Visualizer
    image = mmcv.imread(IMG_PATH, channel_order='rgb')
    image = image.astype(np.uint8)
    visualizer = Visualizer()
    drawn_images = []
    for idx, feat in enumerate(class_features):
        feat = feat.unsqueeze(0)  # chw
        drawn_image = visualizer.draw_featmap(feat, image, channel_reduction='squeeze_mean')
        drawn_images.append(drawn_image)
    from matplotlib import pyplot as plt
    rows, cols = 1, num_feature_levels
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    for i, img in enumerate(drawn_images):
        idx = len(drawn_images) - i
        axes[i].imshow(img)
        axes[i].set_title(f"level_{idx}")
        axes[i].axis("off")
    plt.tight_layout()
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/encoder_class_heatmap.jpg"
    plt.savefig(save_path, dpi=200)

def visualize_matching(indices, outputs, targets):
    # image_path = '/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/vis/segmentation/train/D7-330-20230531.jpg'
    # image_path = '/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/train2017/D7-330-20230531.jpg'
    image_path = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/transformed_image.jpg'
    bidx = 0
    matching_indices = indices[bidx]
    bboxes = outputs['pred_boxes'][bidx]
    targets = targets[bidx]['boxes']
    coord_bboxes = bboxes[:, :2]
    coord_targets = targets[:, :2]

    import mmcv
    import numpy as np
    import cv2
    image = mmcv.imread(image_path, channel_order='bgr')
    image = image.astype(np.uint8)
    for coord_target in coord_targets:
        x, y = coord_target
        x = int(x * image.shape[0])
        y = int(y * image.shape[1])
        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    for coord_bbox in coord_bboxes:
        x, y = coord_bbox
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    for pred_idx, tgt_idx in zip(matching_indices[0], matching_indices[1]):
        coord_bbox = coord_bboxes[pred_idx]
        coord_target = coord_targets[tgt_idx]
        x1, y1 = coord_bbox
        x2, y2 = coord_target
        x1 = int(x1 * image.shape[1])
        y1 = int(y1 * image.shape[0])
        x2 = int(x2 * image.shape[1])
        y2 = int(y2 * image.shape[0])
        cv2.line(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/matching_vis.jpg"
    mmcv.imwrite(image, save_path)

# 绘制 cost 矩阵热力图
def save_cost_matrix_heatmap(
    C,
    src_id, tgt_id,
    save_name="cost_matrix_C.jpg",
    cmap="coolwarm"
):
    """
    根据矩阵 C 画热力图 + 每个格子的数值，并保存到指定路径。

    参数
    ----
    C : array-like
        任意形状为 (H, W) 的二维矩阵。
    save_path : str
        输出图片路径。
    cmap : str
        使用的 colormap，默认 'coolwarm'。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    C = np.asarray(C)
    if C.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {C.ndim}D with shape {C.shape}")

    H, W = C.shape

    # 对称颜色范围（适合有正有负的 cost）
    vmax = float(np.max(np.abs(C))) if np.size(C) > 0 else 1.0
    vmax = vmax if vmax > 0 else 1.0
    vmin = -vmax

    save_path = os.path.join('/public/home/zncui/pycharm_project/MaskDINO/myscripts/', save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 图像尺寸：随宽高略调节
    # 保证大矩阵不至于太糊，小矩阵不至于太空
    base_size = 0.4  # 每个格子大约的尺寸因子
    fig_w = max(6, W * base_size)
    fig_h = max(4, H * base_size * 0.5)
    plt.figure(figsize=(fig_w, fig_h))

    # 画热力图
    im = plt.imshow(
        C,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )

    # 关闭坐标轴刻度
    plt.xticks([])
    plt.yticks([])

    # 计算字体大小（矩阵大时自动缩小）
    # 大致让长边上字体数量合适
    max_dim = max(H, W)
    if max_dim <= 20:
        fontsize = 10
    elif max_dim <= 50:
        fontsize = 8
    elif max_dim <= 100:
        fontsize = 6
    else:
        fontsize = 4

    # 在每个格子上写数值
    # 使用 im.norm 来根据颜色深浅选择文字颜色（亮底用黑字，暗底用白字）
    norm = im.norm
    for i in range(H):
        for j in range(W):
            val = C[i, j]
            # 控制显示精度，防止太长
            text_str = f"{val:.5g}"
            # 根据颜色强度切换文字颜色
            color = "white" if norm(val) > 0.5 else "black"
            plt.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
            )

    # src_id 和 tgt_id 作为坐标轴标签,将对应的框用蓝色标出
    for s_id, t_id in zip(src_id, tgt_id):
        plt.gca().add_patch(plt.Rectangle((t_id-0.5, s_id-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2))

    # 加 colorbar（细一点）
    cbar = plt.colorbar(im, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def count_query_frequency(indices):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import csv

    num_queries = 300
    bs = 3 # 每个 GPU 的 batch size
    # query_match_count = np.zeros(num_queries)
    for i in range(bs):
        query_match_count = np.zeros(num_queries)
        src_id, tgt_id = indices[i]
        matched_queries = src_id
        for q in matched_queries:
            query_match_count[q] += 1
        # 将数据追加保存到 CSV 文件中
        import csv
        csv_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/query_match_count.csv"
        # 追加新数据
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(query_match_count.astype(int).tolist())

    # plt.bar(range(num_queries), query_match_count)
    # plt.title("Query Match Frequency")
    # # 自动递增文件名
    # base_name = "query_match_freq"
    # ext = ".png"
    # counter = 0
    # filename = f"{base_name}{ext}"
    # while os.path.exists(filename):
    #     counter += 1
    #     filename = f"{base_name}{counter}{ext}"
    #
    # plt.savefig(filename, dpi=150)
    # plt.close()

def visualize_matched_queries(outputs, targets, indices, C):
    import numpy as np
    bid = 0
    pred_boxes = outputs['pred_boxes'][bid] # (300, 4)
    tgt_bboxes = targets[bid]['boxes'] # (300, 4)
    tgt_masks = targets[bid]['masks'] # (300, 1024, 1024)
    src_idx, tgt_idx = indices[bid] # (num_matched,), (num_matched,)
    pred_cost = C[bid]

    img_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/transformed_image.jpg"
    import mmcv
    image = mmcv.imread(img_path, channel_order='rgb')
    image = image.astype(np.uint8)

    test_id = 22

    import cv2
    # 第一个图绘制 target boxes + masks
    img_with_targets = image.copy()
    for idx, (bbox, mask) in enumerate(zip(tgt_bboxes, tgt_masks)):
        if idx != test_id:
            continue
        bbox = box_ops.box_cxcywh_to_xyxy(bbox).cpu().numpy()
        h, w = image.shape[:2]
        bbox = bbox * np.array([w, h, w, h], dtype=np.float32)
        bbox = bbox.astype(np.int32)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_with_targets, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mask = mask.cpu().numpy().astype(np.uint8) * 255
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = mask  # 绿色通道
        img_with_targets = cv2.addWeighted(img_with_targets, 1.0, colored_mask, 0.5, 0)
        # 在框上写上编号,如果靠近边缘则调整位置
        text_x = x1
        text_y = y1 - 10 if y1 > 30 else y1 + 30  # 如果靠近顶部边缘，则放在框下方
        cv2.putText(img_with_targets, str(idx), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 第二个图绘制 matched pred boxes + masks
    img_with_preds = image.copy()
    selected_cost = np.asarray(pred_cost[:, test_id])

    # 建立颜色映射函数
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # 归一化到 [0, 1]
    norm = Normalize(vmin=selected_cost.min(), vmax=selected_cost.max())
    cmap = plt.cm.hot  # 热力图配色方案，也可以用 'jet', 'viridis', 'plasma' 等

    # 创建 ScalarMappable 用于 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    def value_to_rgb(val):
        """将数值映射到 RGB 颜色"""
        normalized_val = norm(val)
        rgba = cmap(normalized_val)
        # 转换为 0-255 的 RGB
        rgb = tuple(int(c * 255) for c in rgba[:3])
        return rgb

    # 从 selected_cost 中选择数值最小的 50 个元素的 idx
    topk = 300
    topk_indices = np.argsort(selected_cost)[:topk]
    print('visualize topk indices:', topk_indices)

    # pred_images = []
    # for idx in topk_indices:
    #     img_copy = image.copy()
    #     pred_box = pred_boxes[idx]
    #     C_val = selected_cost[idx]
    #
    #     # 转换坐标
    #     pred_box_xyxy = box_ops.box_cxcywh_to_xyxy(pred_box).cpu().numpy()
    #     h, w = image.shape[:2]
    #     pred_box_xyxy = pred_box_xyxy * np.array([w, h, w, h], dtype=np.float32)
    #     pred_box_xyxy = pred_box_xyxy.astype(np.int32)
    #     x1, y1, x2, y2 = pred_box_xyxy
    #
    #     def value_to_rgb(val):
    #         """将数值映射到 RGB 颜色"""
    #         normalized_val = norm(val)
    #         rgba = cmap(normalized_val)
    #         # 转换为 0-255 的 RGB
    #         rgb = tuple(int(c * 255) for c in rgba[:3])
    #         return rgb
    #
    #     # 绘制框
    #     color = value_to_rgb(C_val.item())
    #     cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 6)
    #
    #     # 添加编号
    #     text_x = x1
    #     text_y = y1 - 10 if y1 > 30 else y1 + 30
    #     cv2.putText(img_copy, str(idx), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
    #
    #     pred_images.append(img_copy)
    #
    # # 使用 matplotlib 拼接显示所有小图
    # import math
    # n_images = len(pred_images)
    # cols = 10  # 每行显示10张图
    # rows = math.ceil(n_images / cols)
    #
    # fig_grid, axes_grid = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    # axes_grid = axes_grid.flatten() if n_images > 1 else [axes_grid]
    #
    # for i, (img, idx) in enumerate(zip(pred_images, topk_indices)):
    #     axes_grid[i].imshow(img)
    #     axes_grid[i].set_title(f"Q{idx}\nC={selected_cost[idx]:.3f}", fontsize=8)
    #     axes_grid[i].axis("off")
    #
    # # 隐藏多余的子图
    # for j in range(i + 1, len(axes_grid)):
    #     axes_grid[j].axis("off")
    #
    # plt.tight_layout()
    # grid_save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/topk_queries_grid.jpg"
    # plt.savefig(grid_save_path, dpi=150)
    # plt.close(fig_grid)

    for idx, (pred_box, C_val) in enumerate(zip(pred_boxes, selected_cost)):
        if idx not in topk_indices:
            continue
        pred_box = box_ops.box_cxcywh_to_xyxy(pred_box).cpu().numpy()
        h, w = image.shape[:2]
        pred_box = pred_box * np.array([w, h, w, h], dtype=np.float32)
        pred_box = pred_box.astype(np.int32)
        x1, y1, x2, y2 = pred_box
        color = value_to_rgb(C_val.item())
        cv2.rectangle(img_with_preds, (x1, y1), (x2, y2), color, 2)
        # 在框上写上编号,如果靠近边缘则调整位置
        text_x = x1
        text_y = y1 - 10 if y1 > 30 else y1 + 30  # 如果靠近顶部边缘，则放在框下方
        cv2.putText(img_with_preds, str(idx), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 构建一个图片拼接显示
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(img_with_targets)
    axs[0].set_title("Target Boxes and Masks")
    axs[0].axis("off")
    axs[1].imshow(img_with_preds)
    axs[1].set_title("Matched Pred Boxes and Masks")
    axs[1].axis("off")
    # 添加 colorbar
    cbar = plt.colorbar(sm, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label('Cost Value', rotation=270, labelpad=15)

    # 保存结果
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/matched_queries_visualization.jpg"
    plt.savefig(save_path, dpi=200)
    plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_cost_components(C,
                              cost_class,
                              cost_mask,
                              cost_dice,
                              cost_bbox,
                              cost_giou,
                              outputs,
                              targets,
                              test_id=22,
                              topk=50,
                              save_dir="/public/home/zncui/pycharm_project/MaskDINO/myscripts"):
    """
    可视化指定 test_id 下的 cost 组成:
    1. 6 个子图的柱状图 (bar)
       - 根据 C_total 中最小的 topk 个 query，高亮对应柱为红色
    2. 1 张总体散点图 (scatter_all)
       - 所有 cost 叠加
       - 对 topk query 的 total cost 点标红，并在相应 query id 位置画竖直线
    3. 6 个单独散点图 (scatter_multi)
       - 每个 cost 一个子图
       - topk 对应点标红
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---- 工具函数：安全转 numpy ----
    def to_numpy(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        return np.asarray(t)

    assert 0 <= test_id < C.shape[1], f"test_id {test_id} 超出列范围 0~{C.shape[1]-1}"

    # outputs 与 targets 暂未使用，预留接口


    # ---- 取出该 gt 列对应的各项 cost ----
    C_total = to_numpy(C[:, test_id])
    C_class = to_numpy(cost_class[:, test_id])
    C_mask = to_numpy(cost_mask[:, test_id])
    C_dice = to_numpy(cost_dice[:, test_id])
    C_box = to_numpy(cost_bbox[:, test_id])
    C_giou = to_numpy(cost_giou[:, test_id])

    num_queries = C_total.shape[0]
    x = np.arange(num_queries)

    titles = [
        "Total Cost",
        "Classification Cost",
        "Mask Cost",
        "Dice Cost",
        "Box L1 Cost",
        "GIoU Cost",
    ]
    costs = [C_total, C_class, C_mask, C_dice, C_box, C_giou]

    # ==================================================
    # 选出 total cost 最小的 topk 个 query index
    # ==================================================
    topk = min(topk, num_queries)
    indices_cost = np.argsort(C_total)[:topk]
    print('visiualize_cost_components:', indices_cost)
    indices_cost = np.sort(indices_cost)  # 排序方便查看

    # 通用的高亮颜色配置
    base_color = "tab:blue"
    highlight_color = "red"

    # ==================================================
    # 1) 柱状图 (6 个子图)，topk 高亮为红色
    # ==================================================
    fig_bar, axes = plt.subplots(2, 3, figsize=(120, 12))
    axes = axes.flatten()

    for ax, title, cost in zip(axes, titles, costs):
        # 为每个柱单独设置颜色,topk 为红色
        colors = np.full(num_queries, base_color, dtype=object)
        colors[indices_cost] = highlight_color

        ax.bar(x,
               cost,
               width=0.6,
               antialiased=False,
               color=colors)

        ax.set_title(title)
        ax.set_xlabel("Query Index")
        ax.set_ylabel("Cost Value")
        ax.set_xlim(-0.5, num_queries - 0.5)

        # 只显示 indices_cost 的刻度
        ax.set_xticks(indices_cost)
        ax.set_xticklabels([str(i) for i in indices_cost], fontsize=6)

    plt.tight_layout()
    bar_path = os.path.join(save_dir, f"weighted_cost_components_bar_test_{test_id}.jpg")
    plt.savefig(bar_path, dpi=200)
    plt.close(fig_bar)

    # ==================================================
    # 2) 整体散点图 (所有 cost 叠加)
    #    - topk 的 total cost 点标红
    #    - 在这些 query index 位置画细长竖线
    # ==================================================
    fig_all, ax = plt.subplots(figsize=(18, 6))

    # 所有分量的普通散点
    ax.scatter(x, C_total, s=18, alpha=0.9, label="Total", c=base_color)
    ax.scatter(x, C_class, s=14, alpha=0.7, label="Cls")
    ax.scatter(x, C_mask,  s=14, alpha=0.7, label="Mask")
    ax.scatter(x, C_dice,  s=14, alpha=0.7, label="Dice")
    ax.scatter(x, C_box,   s=14, alpha=0.7, label="Box L1")
    ax.scatter(x, C_giou,  s=14, alpha=0.7, label="GIoU")

    # 高亮 topk 的 total cost 点
    ax.scatter(indices_cost,
               C_total[indices_cost],
               s=40,
               c=highlight_color,
               alpha=0.95,
               label=f"Top-{topk} (lowest total cost)",
               zorder=5)

    # 计算 y 轴范围，用于画竖线
    ymin = min(np.min(c) for c in costs)
    ymax = max(np.max(c) for c in costs)
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = 0.02 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    # 在 topk 的 query index 位置画细竖线
    for idx in indices_cost:
        ax.axvline(x=idx,
                   ymin=0,
                   ymax=1,
                   color=highlight_color,
                   linewidth=0.5,
                   alpha=0.6,
                   zorder=1)

    ax.set_title(f"Cost Components Scatter (test_id={test_id})")
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Cost Value")
    ax.set_xlim(-0.5, num_queries - 0.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=3, fontsize=9)

    plt.tight_layout()
    scatter_all_path = os.path.join(save_dir, f"weighted_cost_components_scatter_all_test_{test_id}.jpg")
    plt.savefig(scatter_all_path, dpi=200)
    plt.close(fig_all)

    # ==================================================
    # 3) 每个 cost 的单独散点子图，topk 点标红
    # ==================================================
    fig_scatters, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, title, cost in zip(axes, titles, costs):
        # 普通点
        ax.scatter(x, cost, s=16, alpha=0.8, c=base_color)
        # 高亮点
        ax.scatter(indices_cost,
                   cost[indices_cost],
                   s=30,
                   alpha=0.95,
                   c=highlight_color)

        ax.set_title(title)
        ax.set_xlabel("Query Index")
        ax.set_ylabel("Cost Value")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(-0.5, num_queries - 0.5)

    plt.tight_layout()
    scatter_multi_path = os.path.join(save_dir, f"weighted_cost_components_scatter_multi_test_{test_id}.jpg")
    plt.savefig(scatter_multi_path, dpi=200)
    plt.close(fig_scatters)

    # ==================================================
    # 输出路径
    # ==================================================
    print(f"Bar plot saved to: {bar_path}")
    print(f"All-in-one scatter plot saved to: {scatter_all_path}")
    print(f"Individual scatter subplots saved to: {scatter_multi_path}")
    print(f"Highlighted indices (Top-{topk} lowest total cost): {indices_cost.tolist()}")

import os
import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_cost_components_bar_sorted(
    C,
    cost_class,
    cost_mask,
    cost_dice,
    cost_bbox,
    cost_giou,
    outputs,
    targets,
    test_id=22,
    topk=50,
    save_dir="/public/home/zncui/pycharm_project/MaskDINO/myscripts",
):
    """
    只绘制柱状图的版本：
    1. 6 个子图 (bar)，每个 cost 自己从小到大排序之后再画柱子
    2. topk 仍然根据 Total Cost 选，排序后对应的柱子用红色高亮
       - 排序后的 x 轴为 [0, num_queries-1]，表示排序后的 rank
       - x 轴刻度只标出 topk 对应的位置，刻度文字是“原始 query index”
    """

    os.makedirs(save_dir, exist_ok=True)

    # ---- 工具函数：安全转 numpy ----
    def to_numpy(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        return np.asarray(t)

    assert 0 <= test_id < C.shape[1], f"test_id {test_id} 超出列范围 0~{C.shape[1]-1}"

    # ---- 取出该 gt 列对应的各项 cost ----
    C_total = to_numpy(C[:, test_id])
    C_class = to_numpy(cost_class[:, test_id])
    C_mask = to_numpy(cost_mask[:, test_id])
    C_dice = to_numpy(cost_dice[:, test_id])
    C_box = to_numpy(cost_bbox[:, test_id])
    C_giou = to_numpy(cost_giou[:, test_id])

    num_queries = C_total.shape[0]

    titles = [
        "Total Cost",
        "Classification Cost",
        "Mask Cost",
        "Dice Cost",
        "Box L1 Cost",
        "GIoU Cost",
    ]
    costs = [C_total, C_class, C_mask, C_dice, C_box, C_giou]

    # ==================================================
    # 1) 先根据 total cost 选出 topk 的 query（以原始 index 为准）
    # ==================================================
    topk = min(topk, num_queries)
    indices_cost = np.argsort(C_total)[:topk]       # 原始 query index（按 total cost 最小的 topk）
    indices_cost = np.sort(indices_cost)            # 排序方便查看
    print('visualize_cost_components_sorted_bar (topk by total):', indices_cost)

    base_color = "tab:blue"
    highlight_color = "red"

    # ==================================================
    # 2) 只绘制柱状图；每个 cost 自己排序
    # ==================================================
    fig_bar, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, title, cost in zip(axes, titles, costs):
        # ---- 按该 cost 自己进行从小到大的排序 ----
        sorted_idx = np.argsort(cost)         # 排序后的位置 -> 原始 query index
        sorted_cost = cost[sorted_idx]        # 排序后的 cost 值

        # x 轴为排序后的 rank
        x_rank = np.arange(num_queries)

        # ---- 确定哪些柱子需要高亮（红色）----
        # 条件：该位置对应的原始 query index 在 indices_cost 里
        highlight_mask = np.isin(sorted_idx, indices_cost)

        colors = np.full(num_queries, base_color, dtype=object)
        colors[highlight_mask] = highlight_color

        ax.bar(
            x_rank,
            sorted_cost,
            width=0.6,
            antialiased=False,
            color=colors
        )

        ax.set_title(title)
        ax.set_xlabel("Sorted Rank (ascending)")
        ax.set_ylabel("Cost Value")

        # 只对红色柱子做刻度，并显示原始 query index
        topk_positions = np.nonzero(highlight_mask)[0]  # 这些 rank 对应 topk
        ax.set_xticks(topk_positions)
        ax.set_xticklabels(
            [str(idx) for idx in sorted_idx[topk_positions]],  # 原始 query index
            fontsize=6
        )

    plt.tight_layout()
    bar_path = os.path.join(save_dir, f"weighted_cost_components_bar_sorted_test_{test_id}.jpg")
    plt.savefig(bar_path, dpi=200)
    plt.close(fig_bar)

    print(f"Sorted bar plot saved to: {bar_path}")
    print(f"Top-{topk} lowest total-cost indices (original query id): {indices_cost.tolist()}")



def plot_class_distribution(outputs):
    bid = 0
    pred_logits = outputs['pred_logits'][bid]
    import matplotlib.pyplot as plt
    import numpy as np

    # --- 数据准备 ---
    logits = pred_logits.squeeze().cpu().numpy()  # (300,1) → (300,)
    probs = pred_logits.sigmoid().squeeze().cpu().numpy()

    indices = np.arange(len(logits))  # [0..299]

    # 排序后数据（从大到小）
    sorted_logits = np.sort(logits)[::-1]
    sorted_probs = np.sort(probs)[::-1]

    plt.figure(figsize=(16, 12))

    # ===================== 1. 原始 logits 柱状图 =====================
    plt.subplot(2, 2, 1)
    plt.bar(indices, logits, color='blue', alpha=0.7)
    plt.title("Original Predicted Class Logits")
    plt.xlabel("Query Index")
    plt.ylabel("Logit Value")
    plt.grid(True, linestyle='--', alpha=0.3)

    # ===================== 2. 原始 probs 柱状图 =====================
    plt.subplot(2, 2, 2)
    plt.bar(indices, probs, color='green', alpha=0.7)
    plt.title("Original Predicted Class Probabilities")
    plt.xlabel("Query Index")
    plt.ylabel("Probability")
    plt.grid(True, linestyle='--', alpha=0.3)

    # ===================== 3. 排序后的 logits 柱状图 =====================
    plt.subplot(2, 2, 3)
    plt.bar(np.arange(len(sorted_logits)), sorted_logits, color='blue', alpha=0.7)
    plt.title("Sorted Logits (Descending)")
    plt.xlabel("Sorted Index")
    plt.ylabel("Logit Value")
    plt.grid(True, linestyle='--', alpha=0.3)

    # ===================== 4. 排序后的 probs 柱状图 =====================
    plt.subplot(2, 2, 4)
    plt.bar(np.arange(len(sorted_probs)), sorted_probs, color='green', alpha=0.7)
    plt.title("Sorted Probabilities (Descending)")
    plt.xlabel("Sorted Index")
    plt.ylabel("Probability")
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/logits_probs_original_sorted_model_0004999.pth.jpg"
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_neg_and_post(neg_cost_class, pos_cost_class):
    '''
    输出：
    1. 一个 2×2 柱状图文件（原图 + 排序）
    2. 一个 2×2 折线图文件（原图 + diff + 排序 + diff_sorted）
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 保存路径
    save_dir = "/public/home/zncui/pycharm_project/MaskDINO/myscripts"
    os.makedirs(save_dir, exist_ok=True)

    save_bar_fig = f"{save_dir}/neg_pos_class_cost_all.jpg"
    save_line_fig = f"{save_dir}/neg_pos_class_cost_line.jpg"

    # ============================
    # 数据处理
    # ============================
    neg_costs = neg_cost_class.squeeze().cpu().numpy()
    pos_costs = pos_cost_class.squeeze().cpu().numpy()
    num_queries = len(neg_costs)
    indices = np.arange(num_queries)

    diff = pos_costs - neg_costs

    # 排序
    neg_sorted_idx = np.argsort(neg_costs)
    pos_sorted_idx = np.argsort(pos_costs)
    diff_sorted_idx = np.argsort(diff)

    neg_costs_sorted = neg_costs[neg_sorted_idx]
    pos_costs_sorted = pos_costs[pos_sorted_idx]
    diff_sorted = diff[diff_sorted_idx]

    # ======================================================
    # PART 1: 2×2 柱状图文件
    # ======================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    # neg 原图
    ax = axes[0]
    ax.bar(indices, neg_costs, width=0.6, color='red', alpha=0.7)
    ax.set_title("Negative Class Cost per Query")
    ax.grid(True, linestyle='--', alpha=0.3)

    # pos 原图
    ax = axes[1]
    ax.bar(indices, pos_costs, width=0.6, color='blue', alpha=0.7)
    ax.set_title("Positive Class Cost per Query")
    ax.grid(True, linestyle='--', alpha=0.3)

    # neg 排序
    ax = axes[2]
    ax.bar(np.arange(num_queries), neg_costs_sorted, width=0.6, color='red', alpha=0.7)
    ax.set_title("Negative Class Cost (Sorted)")
    ax.grid(True, linestyle='--', alpha=0.3)

    # pos 排序
    ax = axes[3]
    ax.bar(np.arange(num_queries), pos_costs_sorted, width=0.6, color='blue', alpha=0.7)
    ax.set_title("Positive Class Cost (Sorted)")
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_bar_fig, dpi=200)
    plt.close()
    print("Saved bar figure:", save_bar_fig)

    # ======================================================
    # PART 2: 2×2 折线图文件（你的新增需求）
    # ======================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    # -------------------------
    # (1) 原始折线图 neg vs pos
    # -------------------------
    ax = axes[0]
    ax.plot(indices, neg_costs, color='red', linewidth=2, label='Negative Cost')
    ax.plot(indices, pos_costs, color='blue', linewidth=2, label='Positive Cost')
    ax.set_title("Negative vs Positive Class Cost (Line Plot)")
    ax.set_ylabel("Cost Value")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    # -------------------------
    # (2) diff 折线图 pos - neg
    # -------------------------
    ax = axes[1]
    ax.plot(indices, diff, color='black', linewidth=2, label='pos_cost - neg_cost')
    ax.set_title("Difference (Positive - Negative)")
    ax.set_ylabel("Diff Value")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    # -------------------------
    # (3) 排序后的 neg & pos 折线图
    # -------------------------
    ax = axes[2]
    ax.plot(np.arange(num_queries), neg_costs_sorted, color='red', linewidth=2, label='Negative (Sorted)')
    ax.plot(np.arange(num_queries), pos_costs_sorted, color='blue', linewidth=2, label='Positive (Sorted)')
    ax.set_title("Sorted Negative & Positive Class Cost")
    ax.set_ylabel("Sorted Cost")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    # -------------------------
    # (4) 排序后的 diff 折线图
    # -------------------------
    ax = axes[3]
    ax.plot(np.arange(num_queries), diff_sorted, color='black', linewidth=2, label='Sorted (pos - neg)')
    ax.set_title("Sorted Difference (pos_cost - neg_cost)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Sorted Diff Value")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_line_fig, dpi=200)
    plt.close()
    print("Saved line figure:", save_line_fig)


def temporal_diff_bar_scatter(self, neg_cost_class, pos_cost_class):
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_query_value(writer, tag, values, step, mode="line"):
        vals = values.detach().cpu().numpy().flatten()
        indices = np.arange(len(vals))

        fig = plt.figure(figsize=(8, 4))

        if mode == "line":
            plt.plot(indices, vals, linewidth=2)
            plt.title(f"{tag} (Line Plot)")
            plt.xlabel("Query Index")
            plt.ylabel("Value")
        elif mode == "bar":
            plt.bar(indices, vals, width=0.8)
            plt.title(f"{tag} (Bar Plot)")
            plt.xlabel("Query Index")
            plt.ylabel("Value")

        plt.tight_layout()

        # ---- 关键部分：兼容 Agg/InterAgg，RGB 图像提取 ----
        fig.canvas.draw()

        try:
            # 如果当前后端支持 tostring_rgb（大多数本地后端）
            buf = fig.canvas.tostring_rgb()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)

        except Exception:
            # Agg / InterAgg 环境，使用 ARGB → RGB
            buf = fig.canvas.tostring_argb()
            w, h = fig.canvas.get_width_height()
            argb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)

            # ARGB → RGB 转换：忽略 Alpha 通道
            rgb = argb[:, :, 1:]  # [A,R,G,B] 变成 [R,G,B]
            img = rgb.copy()

        plt.close(fig)

        # 写入 TensorBoard
        writer.add_image(tag, img, step, dataformats='HWC')

    step = self.tf_step  # or global_step
    # print('current: ', step)
    # 每 800 次 iteration 记录一次直方图
    if step % 800 == 0:
        print('logging class cost to tensorboard...', step)
        neg_mean = neg_cost_class.mean().item()
        pos_mean = pos_cost_class.mean().item()
        diff_mean = (pos_cost_class - neg_cost_class).mean().item()

        self.tf_writer.add_scalar("class_cost/neg_mean", neg_mean, step)
        self.tf_writer.add_scalar("class_cost/pos_mean", pos_mean, step)
        self.tf_writer.add_scalar("class_cost/diff_mean", diff_mean, step)

        self.tf_writer.add_histogram("class_cost/neg_hist", neg_cost_class.cpu(), step)
        self.tf_writer.add_histogram("class_cost/pos_hist", pos_cost_class.cpu(), step)
        self.tf_writer.add_histogram("class_cost/diff_hist", (pos_cost_class - neg_cost_class).cpu(), step)

        diff = (pos_cost_class - neg_cost_class)

        # line plot
        plot_query_value(self.tf_writer, "class_cost/diff_query_line", diff, step, mode="line")

        # bar plot
        plot_query_value(self.tf_writer, "class_cost/diff_query_bar", diff, step, mode="bar")

        # self.tf_writer.add_histogram("class_cost/diff_hist_indexed", diff, step, bins=300)

    self.tf_step += 1

# 可视化 attn_output_weights 矩阵
def visualize_attn_weights(attn_weights, average_attn_weights=True):
    """
    可视化 attention weights:
        average_attn_weights = True:
            attn_weights 形状: (Q, Q)
            画一张 heatmap
        average_attn_weights = False:
            attn_weights 形状: (n_heads, Q, Q)
            每个 head 一个 subplot，保存为一张大图
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import math

    save_dir = "/public/home/zncui/pycharm_project/MaskDINO/myscripts"
    os.makedirs(save_dir, exist_ok=True)

    attn_weights_np = attn_weights.detach().cpu().numpy()

    # ---------- 情况 1: 已经是 average 后的 (Q, Q) ----------
    if average_attn_weights:
        assert attn_weights_np.ndim == 2, \
            f"expected shape (Q,Q), got {attn_weights_np.shape}"

        save_path = os.path.join(save_dir, "attn_weights_avg.jpg")

        plt.figure(figsize=(10, 8))
        im = plt.imshow(attn_weights_np, cmap='viridis')
        plt.title("Average Attention Weights (All Heads)")
        plt.xlabel("Key Query Index")
        plt.ylabel("Query Index")
        cbar = plt.colorbar(im)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"✅ Average attention visualization saved to: {save_path}")

    # ---------- 情况 2: 每个 head 单独可视化 (H, Q, Q) ----------
    else:
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        from matplotlib.gridspec import GridSpec

        n_heads, Q, _ = attn_weights_np.shape

        cols = int(math.ceil(math.sqrt(n_heads)))
        rows = int(math.ceil(n_heads / cols))

        save_path = os.path.join(save_dir, "attn_weights_per_head.jpg")

        fig = plt.figure(figsize=(4 * cols + 1.2, 4 * rows))
        gs = GridSpec(rows, cols + 1, width_ratios=[1] * cols + [0.04], wspace=0.25, hspace=0.3)

        axes = []
        for i in range(rows * cols):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            axes.append(ax)

        # 统一色值范围
        vmin = attn_weights_np.min()
        vmax = attn_weights_np.max()

        for i in range(n_heads):
            ax = axes[i]
            im = ax.imshow(attn_weights_np[i], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"Head {i}", fontsize=10)

            # 只在左边和底部画坐标轴，避免拥挤
            if i % cols == 0:
                ax.set_ylabel("Query Index", fontsize=8)
            else:
                ax.set_yticks([])

            if i // cols == rows - 1:
                ax.set_xlabel("Key Index", fontsize=8)
            else:
                ax.set_xticks([])

        # 隐藏多余 subplot
        for i in range(n_heads, len(axes)):
            axes[i].axis("off")

        # colorbar 单独一列
        cax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=15)

        plt.suptitle("Attention Weights Per Head", fontsize=16)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"✅ Layout-optimized visualization saved to: {save_path}")

def visualize_attn_indices(attn_output_weights,
                           save_dir="/public/home/zncui/pycharm_project/MaskDINO/myscripts",
                           fname="attn_argmax_indices.jpg"):
    """
    可视化每个 head、每个 query 的最大 attention 所指向的 key 索引。

    attn_output_weights: Tensor
        形状可以是 (n_heads, nq, nk) 或 (bs, n_heads, nq, nk)，
        如果有 batch 维，只取第 0 个 batch。
    """
    import os
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fname)

    attn = attn_output_weights.detach().cpu()
    if attn.dim() == 4:
        # (bs, n_heads, nq, nk) -> 取第 0 个 batch
        attn = attn[0]
    assert attn.dim() == 3, f"expected (n_heads, nq, nk), got {attn.shape}"

    n_heads, nq, nk = attn.shape

    # indices[h, q] = 对于 head h、query q，attention 最大的 key 的索引
    indices = attn.argmax(dim=-1).numpy()   # (n_heads, nq)

    # 子图布局：尽量接近正方形
    cols = int(math.ceil(math.sqrt(n_heads)))
    rows = int(math.ceil(n_heads / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 3 * rows),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    x = np.arange(nq)  # query index

    for h in range(n_heads):
        ax = axes[h]
        y = indices[h]   # (nq, )

        # 每个点代表：query q 的 argmax key index
        ax.scatter(x, y, s=3)
        ax.set_title(f"Head {h}", fontsize=10)

        # 只在左边一列画 y 轴
        if h % cols == 0:
            ax.set_ylabel("Key index", fontsize=8)

        # 只在最后一行画 x 轴
        if h >= (rows - 1) * cols:
            ax.set_xlabel("Query index", fontsize=8)

        ax.set_xlim(-1, nq)
        ax.set_ylim(-1, nk)

    # 隐藏多余子图
    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    plt.suptitle("Argmax Attention Key Index per Query", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ Argmax index visualization saved to: {save_path}")

def visualize_attn_indices_heatmap(attn_output_weights):
    import matplotlib.pyplot as plt
    import numpy as np
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/attn_argmax_indices_heatmap.jpg"
    attn = attn_output_weights.detach().cpu()
    if attn.dim() == 4:
        attn = attn[0]
    n_heads, nq, nk = attn.shape

    indices = attn.argmax(dim=-1).numpy()  # (n_heads, nq)
    # 归一化到 [0,1] 方便配色
    norm_indices = indices / (nk - 1 + 1e-6)

    plt.figure(figsize=(10, 4))
    im = plt.imshow(norm_indices, aspect="auto", cmap="viridis",
                    extent=[0, nq, n_heads - 0.5, -0.5])
    plt.colorbar(im, label="Normalized key index")
    plt.xlabel("Query index")
    plt.ylabel("Head index")
    plt.title("Argmax Attention Key Index (Heads × Queries)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_attn_indices_to_csv(attn_output_weights):
    import numpy as np
    import os
    import pandas as pd
    import csv

    save_dir = "/public/home/zncui/pycharm_project/MaskDINO/myscripts"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "attn_argmax_indices.csv")

    attn = attn_output_weights.detach().cpu()
    n_heads, nq, nk = attn.shape

    indices = attn.argmax(dim=-1).numpy()  # (n_heads, nq)

    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            writer.writerow(['Head Index'] + [f'Query {i}' for i in range(nq)])
        for head_idx, row in enumerate(indices):
            writer.writerow([head_idx] + row.tolist())






