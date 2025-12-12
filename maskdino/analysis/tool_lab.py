import pandas as pd
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

import torch

import torch

def draw_encoder_class_heatmap(class_logits, level_start_indices, spatial_shapes, *vars):
    """
    class_logits: Tensor, (num_tokens,)
    level_start_indices: list[int]
    spatial_shapes: Tensor or list, [(h1,w1), (h2,w2), ...]
    """

    import mmcv
    import numpy as np
    from mmengine.visualization import Visualizer
    import cv2
    from matplotlib import pyplot as plt

    # ====== 1. dtype & device 统一 ======
    class_logits = class_logits.detach().float().cpu()
    class_scores = class_logits.sigmoid()

    if isinstance(level_start_indices, torch.Tensor):
        level_start_indices = level_start_indices.tolist()
    if isinstance(spatial_shapes, torch.Tensor):
        spatial_shapes = spatial_shapes.tolist()

    num_feature_levels = len(spatial_shapes)

    # ====== 2. TopK proposals ======
    TOPK = 300
    topk_indices = torch.topk(class_logits, TOPK)[1].tolist()
    # topk_indices = topk_indices[-1:]
    query_id = vars[0] if len(vars) > 1 else None
    selected_attn_output_weights = vars[1] if len(vars) > 2 else None

    # ====== 3. token index -> (level, y, x) 映射 ======
    token_map = []   # 每个 token => (level, y, x)
    for lvl, (h, w) in enumerate(spatial_shapes):
        start = level_start_indices[lvl]
        for idx in range(h * w):
            y = idx // w
            x = idx % w
            token_map.append((lvl, y, x))

    # ====== 4. split + reshape ======
    class_features = []
    for lvl, (h, w) in enumerate(spatial_shapes):
        start = level_start_indices[lvl]
        end = start + h * w
        fmap = class_logits[start:end].reshape(h, w)
        class_features.append(fmap)

    # ====== 5. 读取原图 ======
    image_path = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/transformed_image_v1.jpg'
    image = mmcv.imread(image_path, channel_order="rgb")
    image = image.astype(np.uint8)

    visualizer = Visualizer()
    drawn_images = []

    # ====== 6. 每层画热力图 & overlay 蓝色 top-k ======
    topk_score_indices = None
    if selected_attn_output_weights is not None:
        score = selected_attn_output_weights
        # 获取 score 最大的 10 个数值的索引
        topk_score_indices = torch.topk(score, 10)[1]

    for lvl, fmap in enumerate(class_features):

        fmap = fmap.unsqueeze(0).float().cpu()  # (1,H,W)

        drawn = visualizer.draw_featmap(
            fmap,
            image,
            channel_reduction="squeeze_mean"
        )
        if hasattr(drawn, "get_image"):
            drawn = drawn.get_image()

        # ---- 在图上叠加蓝点 ----
        img = drawn.copy()
        H, W = fmap.shape[-2], fmap.shape[-1]
        H_img, W_img = img.shape[:2]

        sx = W_img / W
        sy = H_img / H

        for i, tidx in enumerate(topk_indices):
            lvl0, y, x = token_map[tidx]
            if lvl0 != lvl:
                continue

            # 映射到像素坐标
            px = int((x + 0.5) * sx)
            py = int((y + 0.5) * sy)

            cv2.circle(img, (px, py), radius=5, color=(255, 0, 0), thickness=-1)

            # score = class_scores[tidx]
            # cv2.putText(img, f"{score:.3f}", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # if score < 0.3:
            #     cv2.circle(img, (px, py), radius=5, color=(255, 0, 0), thickness=-1)
            # if 0.3 <= score < 0.6:
            #     cv2.circle(img, (px, py), radius=5, color=(0, 255, 0), thickness=-1)
            # if score >= 0.6:
            #     cv2.circle(img, (px, py), radius=5, color=(0, 0, 255), thickness=-1)

            if query_id is not None and i == query_id:
                cv2.circle(img, (px, py), radius=10, color=(0, 255, 0), thickness=-1)

            if topk_score_indices is not None:
                if i in topk_score_indices :
                    cv2.circle(img, (px, py), radius=8, color=(0, 0, 255), thickness=-1)


            # if score > 0.001:
            #     cv2.putText(img, f"{score:.3f}", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # 将 i 也画上去
            # cv2.putText(img, str(i), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        drawn_images.append(img)

    # ====== 7. plot 保存 ======
    rows, cols = 1, num_feature_levels
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

    if cols == 1:
        axes = [axes]

    for i, img in enumerate(drawn_images):
        lvl = num_feature_levels - i
        axes[i].imshow(img)
        axes[i].set_title(f"level_{lvl}")
        axes[i].axis("off")

    plt.tight_layout()
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/encoder_class_heatmap.jpg"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    # # ====== 画 TopK class_scores 的柱状图 ======
    # topk_scores = class_scores[topk_indices].cpu().numpy()  # (TOPK,)
    #
    # # 简单版本：x 轴就是 0~TOPK-1
    # plt.figure(figsize=(12, 4))
    # plt.bar(np.arange(len(topk_scores)), topk_scores)
    # plt.xlabel("TopK token rank")
    # plt.ylabel("class_scores (sigmoid)")
    # plt.title(f"Top-{TOPK} class scores")
    # plt.tight_layout()
    #
    # bar_save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/encoder_class_scores_bar.jpg"
    # plt.savefig(bar_save_path, dpi=200)
    # plt.close()




def visualize_matching(indices, outputs, targets):
    # image_path = '/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/vis/segmentation/train/D7-330-20230531.jpg'
    # image_path = '/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/train2017/D7-330-20230531.jpg'
    image_path = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/transformed_image.jpg'
    bidx = 0
    matching_indices = indices[bidx]
    bboxes = outputs['pred_boxes'][bidx]
    scores = outputs['pred_logits'][bidx].sigmoid().squeeze()
    # matching_score = scores[matching_indices[0]]
    targets = targets[bidx]['boxes']
    coord_bboxes = bboxes[:, :2]
    coord_targets = targets[:, :2]

    import mmcv
    import numpy as np
    import cv2
    image = mmcv.imread(image_path, channel_order='bgr')
    image = image.astype(np.uint8)

    pattern = 'rectangle' # 'rectangle' or 'circle'
    if pattern == 'circle':
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
    if pattern == 'rectangle':
        for pred_idx, tgt_idx in zip(matching_indices[0], matching_indices[1]):
            # 随机获取一个颜色
            def random_dark_rgb():
                """生成偏深色的随机RGB"""
                import random
                return tuple(random.randint(0, 120) for _ in range(3))
            score = scores[pred_idx]

            bbox_xywh = bboxes[pred_idx]
            target_xywh = targets[tgt_idx]
            def xywh_to_xyxy(box):
                x, y, w, h = box
                x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                return x1, y1, x2, y2
            bbox_xyxy = xywh_to_xyxy(bbox_xywh)
            target_xyxy = xywh_to_xyxy(target_xywh)
            w, h = image.shape[1], image.shape[0]
            def scale_box(box):
                x1, y1, x2, y2 = box
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                return x1, y1, x2, y2
            bbox_xyxy = scale_box(bbox_xyxy)
            target_xyxy = scale_box(target_xyxy)
            # 绘制矩形框
            bbox_color = random_dark_rgb()
            cv2.rectangle(image, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), thickness=1, color=bbox_color)
            cv2.rectangle(image, (target_xyxy[0], target_xyxy[1]), (target_xyxy[2], target_xyxy[3]), thickness=1, color=bbox_color)
            # 在框上写分数
            cv2.putText(image, f"{score:.2f}", (bbox_xyxy[0], bbox_xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, bbox_color, 2, cv2.LINE_AA)
            print(f'Matched pair - Score: {score:.4f}, BBox: {pred_idx}, Target: {tgt_idx}') # Target: 22 是目标

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

def vis_loss_bar(losses, title="Loss by layer (suffix)", save_path=None):
    """
    参数
    ----
    losses: dict
        形如 {"loss_ce": 1.2, "loss_dice_0": 0.6, "dn_loss_ce": 0.3, ...}
        会自动按后缀分组，例如:
            loss_xxx      -> suffix = "base"
            loss_xxx_0    -> suffix = "0"
            loss_xxx_1    -> suffix = "1"
        同时会把 loss_xxx_0 / loss_xxx_1 合并为同一个 loss type: loss_xxx

    title: str
        图片总标题

    save_path: str or None
        若给定路径，则保存图片到该路径；否则直接 plt.show()
    """
    import re
    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    # 1. 先按 suffix 分组
    grouped_losses = defaultdict(list)

    for loss_key, loss_value in losses.items():
        # 跳过中间层和 dn 相关的 loss（按你的需求）
        if "interm" in loss_key:
            continue
        if "dn" in loss_key:
            continue

        # 以结尾的 _数字 作为后缀，例如 loss_ce_0 -> suffix="0"
        m = re.search(r"_(\d+)$", loss_key)
        suffix = m.group(1) if m else "base"
        grouped_losses[suffix].append((loss_key, float(loss_value)))

    if not grouped_losses:
        print("No valid losses to plot.")
        return

    # 2. 组内按 key 字典序排序（方便阅读）
    for suffix in grouped_losses:
        grouped_losses[suffix].sort(key=lambda x: x[0])

    # 3. 在每个 suffix 内，再按 loss 类型汇总（忽略末尾数字）
    #    agg[suffix][loss_type] = sum(value)
    agg = defaultdict(lambda: defaultdict(float))

    for suffix, items in grouped_losses.items():
        for loss_key, loss_value in items:
            parts = loss_key.split("_")
            # 若最后一段是数字，去掉这段作为 "loss 类型"
            if parts[-1].isdigit():
                base_name = "_".join(parts[:-1])  # loss_ce_0 -> loss_ce
            else:
                base_name = loss_key             # loss_ce -> loss_ce

            agg[suffix][base_name] += float(loss_value)

    # 4. 决定 suffix 的顺序：先 base，再数字，最后其他
    def suffix_sort_key(s):
        if s == "base":
            return (0, 0)
        if s.isdigit():
            return (1, int(s))
        return (2, s)

    suffixes = sorted(agg.keys(), key=suffix_sort_key)
    if not suffixes:
        print("No valid suffixes to plot.")
        return

    # 所有 loss 类型（loss_ce, loss_dice, ...），为了 legend 和颜色统一
    loss_types = sorted({lt for d in agg.values() for lt in d.keys()})
    if not loss_types:
        print("No loss types to plot.")
        return

    x = np.arange(len(suffixes))

    # 计算所有值最大值，用于统一各子图的 y 轴
    max_val = 0.0
    for lt in loss_types:
        vals = [agg[s].get(lt, 0.0) for s in suffixes]
        if vals:
            max_val = max(max_val, max(vals))
    if max_val == 0:
        max_val = 1.0  # 避免全 0 的情况

    # ===== 5. 准备子图布局：每个 loss type 一个 subplot + 1 个堆叠 subplot =====
    n_types = len(loss_types)
    total_plots = n_types + 1  # 多出来的 1 个是堆叠柱状图

    n_cols = min(3, total_plots)   # 每行最多放 3 个
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False
    )

    # ===== 6. 每个 loss type 单独一个柱状图 subplot =====
    for idx, lt in enumerate(loss_types):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        values = np.array([agg[s].get(lt, 0.0) for s in suffixes], dtype=float)

        ax.bar(x, values)
        ax.set_title(lt)
        ax.set_xticks(x)
        ax.set_xticklabels(suffixes, rotation=45, ha="right")
        ax.set_ylim(0, max_val * 1.1)

        if col == 0:
            ax.set_ylabel("Loss value")
        if row == n_rows - 1:
            ax.set_xlabel("Suffix / layer")

    # ===== 7. 堆叠柱状图 subplot（放在最后一个位置） =====
    stack_idx = n_types  # 第 n_types 个 subplot
    row = stack_idx // n_cols
    col = stack_idx % n_cols
    ax = axes[row][col]

    bottoms = np.zeros(len(suffixes), dtype=float)
    cmap = plt.get_cmap("tab10")
    colors = {lt: cmap(i % 10) for i, lt in enumerate(loss_types)}

    for lt in loss_types:
        vals = np.array([agg[s].get(lt, 0.0) for s in suffixes], dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=lt,
            color=colors[lt],
        )
        bottoms += vals

    ax.set_title("Stacked losses")
    ax.set_xticks(x)
    ax.set_xticklabels(suffixes, rotation=45, ha="right")
    ax.set_ylabel("Loss sum")
    ax.legend(fontsize=8)

    # ===== 8. 如果布局里有多余 subplot，把它们关掉 =====
    for idx in range(total_plots, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")

    # ===== 9. 总标题 + 保存 / 显示 =====
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def vis_attention_matrix(attn_matrix,
                          title="Self-Attention Matrix",
                          log_scale=True,
                          clip_percentile=99.8,
                          export_csv=True):
    """
    保存 attention heatmap + 导出 attention 数值 CSV

    参数
    ----
    attn_matrix : torch.Tensor / np.ndarray
        形状 (Q,Q) 或 (H,Q,Q)
    save_path : str
        图像保存路径，如：
        /public/home/zncui/pycharm_project/MaskDINO/myscripts/vis_self_attn_matrix.jpg
    export_csv : bool
        是否导出 CSV
    """

    import os
    import csv
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/vis_self_attn_matrix.jpg"

    # ---- 1. tensor -> numpy ----
    if isinstance(attn_matrix, torch.Tensor):
        attn = attn_matrix.detach().float().cpu().numpy()
    else:
        attn = np.asarray(attn_matrix, dtype=np.float32)

    # ---- 2. 多头 attention -> 平均 ----
    if attn.ndim == 3:
        attn = attn.mean(axis=0)

    assert attn.ndim == 2, f"Expect 2D attention matrix, got {attn.shape}"
    Q, K = attn.shape

    # ---- 3. CSV 导出（原始 attention，不做任何变换） ----
    if export_csv:
        csv_path = save_path.replace(".jpg", ".csv").replace(".png", ".csv")

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # 写 header
            writer.writerow(["query\\key"] + list(range(K)))
            # 写 matrix
            for i in range(Q):
                row = [f"{v:.3f}" for v in attn[i]]  # ⭐ 保留三位小数
                writer.writerow([i] + row)

        print(f"✅ Attention CSV saved to: {csv_path}")

    # ---- 4. heatmap 可视化处理流程 ----

    # 百分位裁剪
    vmax = np.percentile(attn, clip_percentile)
    vmin = np.percentile(attn, 100 - clip_percentile)
    attn_vis = np.clip(attn, vmin, vmax)

    # log 拉伸
    if log_scale:
        attn_vis = np.log1p(attn_vis - attn_vis.min())

    # 归一化
    attn_vis = (attn_vis - attn_vis.min()) / (attn_vis.max() + 1e-8)

    # ---- 5. 绘图 ----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn_vis,
                   cmap="inferno",
                   aspect="auto",
                   interpolation="nearest")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Key index")
    ax.set_ylabel("Query index")

    # 精简刻度
    step = max(Q // 12, 1)
    ax.set_xticks(np.arange(0, K, step))
    ax.set_yticks(np.arange(0, Q, step))

    # 色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized attention", rotation=270, labelpad=12)

    plt.tight_layout()

    # ---- 6. 保存图像 ----
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"✅ Attention heatmap saved to: {save_path}")


def vis_attention_bar_from_csv(
    pred_id,
    csv_path="/public/home/zncui/pycharm_project/MaskDINO/myscripts/vis_self_attn_matrix.csv",
    save_path=None,
):
    """
    从 vis_self_attn_matrix.csv 中读取第 pred_id 行的 attention，
    并绘制一个 bar 图。

    CSV 格式示例:
        query\\key,0,1,2,...,299
        0,0.071,0.004,0.003,...
        1,0.006,0.120,0.008,...
        ...
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch

    # ---- 0. 把 pred_id 转成 Python int ----
    if isinstance(pred_id, torch.Tensor):
        # 支持标量 tensor 或形如 tensor([142])
        pred_id = int(pred_id.detach().cpu().flatten()[0].item())
    else:
        pred_id = int(pred_id)

    # ---- 1. 读 CSV ----
    df = pd.read_csv(csv_path)  # 你的文件是标准逗号分隔

    num_queries, num_cols = df.shape  # (300, 301)

    # ---- 2. 安全检查 ----
    if not (0 <= pred_id < num_queries):
        raise ValueError(f"pred_id={pred_id} 超出范围，合法范围是 [0, {num_queries-1}]")

    # ---- 3. 取出这一行 ----
    row = df.iloc[pred_id]

    # 如果 pred_id 传的是 array-like，df.iloc[...] 会返回 DataFrame，这里兜个底
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    # row 现在应该是一个 Series：
    # query\key      142
    # 0             0.005
    # 1             0.005
    # ...
    # 299           0.003

    # 第一列是 query id，从第二列开始是 attention 值
    attn_values = row.iloc[1:].astype(float).to_numpy()
    key_indices = np.arange(attn_values.shape[0])

    if attn_values.size == 0:
        raise RuntimeError(
            f"pred_id={pred_id} 这一行没有读到任何 attention 数值，row 内容:\n{row}"
        )

    # ---- 4. 画 bar 图 ----
    plt.figure(figsize=(12, 4))
    plt.bar(key_indices, attn_values)

    plt.title(f"Query {pred_id} Attention over Keys", fontsize=12)
    plt.xlabel("Key index")
    plt.ylabel("Attention weight")

    # X 轴刻度稀疏一点
    step = max(len(key_indices) // 20, 1)
    plt.xticks(key_indices[::step])

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # ---- 5. 保存或显示 ----
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"✅ Bar plot saved to: {save_path}")
    else:
        plt.show()


# vis_ms_deform_attn.py
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_ms_deform_attn_sampling(
        sampling_locations: torch.Tensor,
        image_path: str,
        save_path: str,
        batch_idx: int = 0,
        query_idx: int = None,
        max_points_per_level: int = None,
):
    """
    可视化 MSDeformAttn 的采样点，按 feature level 分成多个小图。

    参数
    ----
    sampling_locations: Tensor
        形状为 (N, Len_q, n_heads, n_levels, n_points, 2)，
        一般就是 MSDeformAttn.forward 中算出来的 sampling_locations。
    image_path: str
        背景图片路径，比如你的 transformed_image.jpg。
    save_path: str
        保存可视化结果的路径（.jpg 或 .png 均可）。
    batch_idx: int
        使用哪个 batch 的采样点（一般 N=1，就用 0）。
    query_idx: int or None
        如果是 None：可视化所有 query 的采样点。
        如果是 int：只可视化指定某个 query 的采样点（便于单 query 观察）。
    max_points_per_level: int or None
        为了避免点太多太密，可以随机采样一部分点。
        如果是 None：不过滤。
        比如设为 5000，就每个 level 最多画 5000 个采样点。
    """
    # ---------- 1. 读图 ----------
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"image_path not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_img, W_img = img.shape[:2]

    # ---------- 2. 整理 sampling_locations ----------
    # sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2)
    if sampling_locations.is_cuda:
        sampling_locations = sampling_locations.detach().cpu()
    else:
        sampling_locations = sampling_locations.detach()

    assert sampling_locations.ndim == 6, f"sampling_locations shape error: {sampling_locations.shape}"
    N, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
    assert 0 <= batch_idx < N, f"batch_idx {batch_idx} out of range [0, {N})"

    # 取出某个 batch
    coords = sampling_locations[batch_idx]  # (Len_q, n_heads, n_levels, n_points, 2)

    # 如果只看某一个 query
    if query_idx is not None:
        assert 0 <= query_idx < Len_q, f"query_idx {query_idx} out of range [0, {Len_q})"
        coords = coords[query_idx:query_idx + 1]  # (1, n_heads, n_levels, n_points, 2)
        Len_q_vis = 1
    else:
        Len_q_vis = Len_q

    # ---------- 3. 画图：每个 level 一个子图 ----------
    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4))
    if n_levels == 1:
        axes = [axes]

    for lvl in range(n_levels):
        ax = axes[lvl]
        ax.imshow(img)
        ax.set_title(f"Level {lvl}")
        ax.axis("off")

        # coords: (Len_q_vis, n_heads, n_levels, n_points, 2)
        lvl_coords = coords[:, :, lvl, :, :]  # (Len_q_vis, n_heads, n_points, 2)
        lvl_coords = lvl_coords.reshape(-1, 2)  # (Len_q_vis * n_heads * n_points, 2)

        # 归一化坐标 -> 像素坐标
        xs = lvl_coords[:, 0].numpy()
        ys = lvl_coords[:, 1].numpy()

        # clamp 到 [0, 1]，避免数值略超
        xs = np.clip(xs, 0.0, 1.0)
        ys = np.clip(ys, 0.0, 1.0)

        xs_img = xs * W_img
        ys_img = ys * H_img

        # 如果点太多，可以随机采样一部分
        if max_points_per_level is not None and xs_img.shape[0] > max_points_per_level:
            idx = np.random.choice(xs_img.shape[0], max_points_per_level, replace=False)
            xs_img = xs_img[idx]
            ys_img = ys_img[idx]

        # 画散点
        ax.scatter(xs_img, ys_img, s=2, alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"✅ MSDeformAttn sampling visualization saved to: {save_path}")

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_ms_deform_attn_sampling_by_head(
    sampling_locations: torch.Tensor,
    image_path: str,
    save_path: str,
    batch_idx: int = 0,
    query_idx: int = None,
    max_points_per_head_per_level: int = None,
):
    """
    按 head 区分颜色的 MSDeformAttn 采样点可视化。
    每个 feature level 一个子图，子图中不同 head 使用不同颜色，并带图例。

    参数
    ----
    sampling_locations: Tensor
        形状 (N, Len_q, n_heads, n_levels, n_points, 2)
    image_path: str
        背景图 (transformed_image.jpg)
    save_path: str
        保存路径
    batch_idx: int
        选择第几个 batch
    query_idx: int or None
        None: 所有 query
        int: 指定单个 query
    max_points_per_head_per_level: int or None
        每个 level、每个 head 最多绘制多少个采样点，避免太密。
    """
    # ---------- 1. 读图 ----------
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"image_path not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_img, W_img = img.shape[:2]

    # ---------- 2. 整理 sampling_locations ----------
    if sampling_locations.is_cuda:
        sampling_locations = sampling_locations.detach().cpu()
    else:
        sampling_locations = sampling_locations.detach()

    assert sampling_locations.ndim == 6, f"sampling_locations shape error: {sampling_locations.shape}"
    N, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
    assert 0 <= batch_idx < N, f"batch_idx {batch_idx} out of range [0, {N})"

    coords = sampling_locations[batch_idx]  # (Len_q, n_heads, n_levels, n_points, 2)

    if query_idx is not None:
        assert 0 <= query_idx < Len_q, f"query_idx {query_idx} out of range [0, {Len_q})"
        coords = coords[query_idx:query_idx + 1]  # (1, n_heads, n_levels, n_points, 2)
        Len_q_vis = 1
    else:
        Len_q_vis = Len_q

    # ---------- 3. 颜色：每个 head 一种 ----------
    cmap = plt.get_cmap("tab10")
    head_colors = [cmap(h % cmap.N) for h in range(n_heads)]

    # ---------- 4. 画图 ----------
    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4))
    if n_levels == 1:
        axes = [axes]

    for lvl in range(n_levels):
        ax = axes[lvl]
        ax.imshow(img)
        ax.set_title(f"Level {lvl}")
        ax.axis("off")

        # coords: (Len_q_vis, n_heads, n_levels, n_points, 2)
        # 取当前 level 的所有 query / head / points
        lvl_coords = coords[:, :, lvl, :, :]  # (Len_q_vis, n_heads, n_points, 2)

        # 每个 head 单独画一批点
        for h in range(n_heads):
            # (Len_q_vis, n_points, 2) -> (-1, 2)
            head_coords = lvl_coords[:, h, :, :].reshape(-1, 2)  # (Len_q_vis * n_points, 2)

            xs = head_coords[:, 0].numpy()
            ys = head_coords[:, 1].numpy()

            # clamp 到 [0, 1]
            xs = np.clip(xs, 0.0, 1.0)
            ys = np.clip(ys, 0.0, 1.0)

            xs_img = xs * W_img
            ys_img = ys * H_img

            # 随机采样部分点，避免太密
            if max_points_per_head_per_level is not None and xs_img.shape[0] > max_points_per_head_per_level:
                idx = np.random.choice(xs_img.shape[0], max_points_per_head_per_level, replace=False)
                xs_img = xs_img[idx]
                ys_img = ys_img[idx]

            ax.scatter(
                xs_img,
                ys_img,
                s=4,
                alpha=0.7,
                color=head_colors[h],
                label=f"head {h}" if lvl == 0 else None,  # 避免每个子图都重复 legend
            )

    # 只在第一个子图上放一个 legend（避免太挤）
    axes[0].legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.7,
        title="Heads"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"✅ MSDeformAttn sampling (by head) visualization saved to: {save_path}")

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_ms_deform_attn_sampling_by_query(
    sampling_locations: torch.Tensor,
    image_path: str,
    save_path: str,
    batch_idx: int = 0,
    query_indices=None,
    max_points_per_query_per_level: int = None,
):
    """
    按 query 区分颜色的 MSDeformAttn 采样点可视化。
    每个 feature level 一个子图，子图中不同 query 使用不同颜色。

    参数
    ----
    sampling_locations: Tensor
        形状 (N, Len_q, n_heads, n_levels, n_points, 2)
    image_path: str
        背景图 (transformed_image.jpg)
    save_path: str
        保存路径
    batch_idx: int
        选择第几个 batch
    query_indices: None 或 list[int]
        None: 使用所有 query（颜色可能会重复/很多，不建议太大）
        list: 只可视化这些指定 query（例如 [0, 1, 2, 142]）
    max_points_per_query_per_level: int 或 None
        每个 level、每个 query 最多绘制多少个采样点，避免太密。
    """
    # ---------- 1. 读图 ----------
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"image_path not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_img, W_img = img.shape[:2]

    # ---------- 2. 整理 sampling_locations ----------
    if sampling_locations.is_cuda:
        sampling_locations = sampling_locations.detach().cpu()
    else:
        sampling_locations = sampling_locations.detach()

    assert sampling_locations.ndim == 6, f"sampling_locations shape error: {sampling_locations.shape}"
    N, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
    assert 0 <= batch_idx < N, f"batch_idx {batch_idx} out of range [0, {N})"

    coords = sampling_locations[batch_idx]  # (Len_q, n_heads, n_levels, n_points, 2)

    # 选择要画的 query
    if query_indices is None:
        query_indices = list(range(Len_q))
    else:
        # 过滤出合法的
        query_indices = [q for q in query_indices if 0 <= q < Len_q]
        if len(query_indices) == 0:
            raise ValueError("query_indices 为空或不合法")

    num_queries_vis = len(query_indices)

    # ---------- 3. 为 query 准备颜色 ----------
    # 用连续 colormap，保证很多 query 也有不同颜色
    cmap = plt.get_cmap("hsv")
    query_colors = {}
    for i, q in enumerate(query_indices):
        query_colors[q] = cmap(i / max(1, num_queries_vis - 1))

    # ---------- 4. 画图 ----------
    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4))
    if n_levels == 1:
        axes = [axes]

    for lvl in range(n_levels):
        ax = axes[lvl]
        ax.imshow(img)
        ax.set_title(f"Level {lvl}")
        ax.axis("off")

        # 给 legend 用的 handler
        legend_handles = []

        # 针对每个 query 画点
        for q in query_indices:
            # coords: (Len_q, n_heads, n_levels, n_points, 2)
            # -> (n_heads, n_points, 2) for this query & level
            q_lvl_coords = coords[q, :, lvl, :, :]           # (n_heads, n_points, 2)
            q_lvl_coords = q_lvl_coords.reshape(-1, 2)       # (n_heads * n_points, 2)

            xs = q_lvl_coords[:, 0].numpy()
            ys = q_lvl_coords[:, 1].numpy()

            xs = np.clip(xs, 0.0, 1.0)
            ys = np.clip(ys, 0.0, 1.0)

            xs_img = xs * W_img
            ys_img = ys * H_img

            if max_points_per_query_per_level is not None and xs_img.shape[0] > max_points_per_query_per_level:
                idx = np.random.choice(xs_img.shape[0], max_points_per_query_per_level, replace=False)
                xs_img = xs_img[idx]
                ys_img = ys_img[idx]

            sc = ax.scatter(
                xs_img,
                ys_img,
                s=4,
                alpha=0.7,
                color=query_colors[q],
                label=f"q{q}",
            )

            # 只在第一个 level 收集 legend，避免重复
            if lvl == 0:
                legend_handles.append(sc)

        # 第一个子图放 legend（如果 query 太多你可以自己关掉）
        if lvl == 0 and num_queries_vis <= 20:  # 太多就不显示 legend，避免挤满
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=8,
                framealpha=0.7,
                title="Queries"
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"✅ MSDeformAttn sampling (by query) visualization saved to: {save_path}")





def analyze_query_scores(out, save_path="query_scores_bar.png", topk=None):
    """
    将 out 中所有 pred_logits 做 sigmoid 后，按 Nx1 纵向柱状图形式保存

    参数
    ----
    out : dict
        模型输出
    save_path : str
        保存路径
    topk : int or None
        若指定，只画TopK最大的query（按score排序）
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ===== 1. 收集所有 pred_logits =====
    logits_list = []
    names = []

    if "pred_logits" in out:
        logits_list.append(out["pred_logits"])
        names.append("main/pred_logits")

    if "aux_outputs" in out:
        for i, aux in enumerate(out["aux_outputs"]):
            if "pred_logits" in aux:
                logits_list.append(aux["pred_logits"])
                names.append(f"aux[{i}]/pred_logits")

    if "interm_outputs" in out and isinstance(out["interm_outputs"], dict):
        if "pred_logits" in out["interm_outputs"]:
            logits_list.append(out["interm_outputs"]["pred_logits"])
            names.append("interm/pred_logits")

    assert len(logits_list) > 0, "❌ 没有找到 pred_logits"

    # ===== 2. sigmoid 处理 =====
    scores_list = []
    scores_sum_list = []
    for lg in logits_list:
        lg = torch.as_tensor(lg)

        # 期望 (B, N, C) / (B, N) / (N)
        if lg.dim() == 3:
            lg = lg[..., 0]  # 默认第0类
        if lg.dim() == 2:
            lg = lg[0]
        if lg.dim() != 1:
            raise ValueError(f"不支持 pred_logits 维度: {lg.shape}")

        scores = lg.sigmoid()
        scores_sum = scores.sum()
        scores_list.append(_to_numpy(scores))
        scores_sum_list.append(scores_sum)

    for sc_sum, name in zip(scores_sum_list, names):
        print(f"[INFO] {name}: scores sum = {sc_sum}")
    # python
    import torch

    # 把可能在 GPU 上的 tensor 转为 CPU 上的 float，再去重与统计
    unique_scores = []
    for s in scores_sum_list:
        if torch.is_tensor(s):
            unique_scores.append(float(s.detach().cpu().item()))
        else:
            unique_scores.append(float(s))

    unique_scores_sum_list = list(set(unique_scores))
    mean_score_sum = np.mean(unique_scores_sum_list)
    std_score_sum = np.std(unique_scores_sum_list)
    print(f"[INFO] Unique scores sum: mean = {mean_score_sum:.4f}, std = {std_score_sum:.4f}")
    # ===== 3. 绘制柱状图 =====
    N = len(scores_list)
    fig_h = max(2.2 * N, 3)
    fig, axes = plt.subplots(N, 1, figsize=(12, fig_h), sharex=False)

    if N == 1:
        axes = [axes]

    for i, (sc, name) in enumerate(zip(scores_list, names)):
        ax = axes[i]

        # TopK 可选
        if topk is not None and topk < len(sc):
            idx = np.argsort(sc)[-topk:][::-1]
            sc = sc[idx]
            x = np.arange(len(sc))
            title = f"{name}  (Top-{topk})"
        else:
            x = np.arange(len(sc))
            title = f"{name}  (N={len(sc)})"

        ax.bar(x, sc)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("sigmoid score")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")

        # 标注最大最小
        ax.text(
            0.01, 0.88,
            f"min={sc.min():.3f}  max={sc.max():.3f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            fontsize=9
        )

        # X轴显示密度控制
        if len(sc) > 50:
            ax.set_xticks(np.linspace(0, len(sc)-1, num=10).astype(int))

    axes[-1].set_xlabel("query index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[OK] 柱状图保存完成: {os.path.abspath(save_path)}")
    return save_path

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def _visualize_decoder_outputs(out):
    """
    将所有 decoder 输出中的 query 中心点绘制到同一张背景图上，
    每个 pred_logits / pred_boxes 对应一个子图 (N x 1 布局)。

    假定 pred_boxes 形状为 (B, num_queries, 4)，格式为
        (cx, cy, w, h)，并已在 [0, 1] 范围内归一化。
    """

    # ====== 0. 路径设置 ======
    img_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/transformed_image_v1.jpg"
    save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/decoder_queries_overlay.jpg"

    # ====== 1. 读入背景图 ======
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取背景图: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # ====== 2. 收集 (logits, boxes, name) 对 ======
    pairs = []

    # 主输出
    if "pred_logits" in out and "pred_boxes" in out:
        pairs.append((out["pred_logits"], out["pred_boxes"], "main"))

    # aux_outputs
    if "aux_outputs" in out:
        for i, aux in enumerate(out["aux_outputs"]):
            if "pred_logits" in aux and "pred_boxes" in aux:
                pairs.append((aux["pred_logits"], aux["pred_boxes"], f"aux[{i}]"))

    # interm_outputs
    if "interm_outputs" in out and isinstance(out["interm_outputs"], dict):
        inter = out["interm_outputs"]
        if "pred_logits" in inter and "pred_boxes" in inter:
            pairs.append((inter["pred_logits"], inter["pred_boxes"], "interm"))

    if len(pairs) == 0:
        raise ValueError("在 out 中没有找到任意一对 (pred_logits, pred_boxes)")

    # ====== 3. 工具函数：boxes -> 中心点绝对坐标 ======
    def boxes_to_centers_abs(boxes_tensor, h, w):
        """
        boxes_tensor: (B, N, 4) or (N, 4)
        归一化 cx,cy,wh -> 返回 (N,2) 绝对坐标 (cx_abs, cy_abs)
        """
        boxes = torch.as_tensor(boxes_tensor).detach().float()

        # 只取 batch 的第 0 张
        if boxes.dim() == 3:
            boxes = boxes[0]          # (N, 4)
        elif boxes.dim() != 2:
            raise ValueError(f"不支持的 boxes 维度: {boxes.shape}")

        # 判断是否归一化 (简单判断：最大值 <= 1.0 视为归一化)
        if boxes[..., :2].max() <= 1.0 and boxes[..., 2:].max() <= 1.0:
            # cx, cy, w, h (归一化)
            cx = boxes[:, 0] * w
            cy = boxes[:, 1] * h
        else:
            # 认为是 xyxy 绝对坐标
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

        centers = torch.stack([cx, cy], dim=-1)
        return centers.cpu().numpy()   # (N, 2)

    # ====== 4. 创建子图布局 ======
    num_layers = len(pairs)
    fig_h = max(3 * num_layers, 3)
    fig, axes = plt.subplots(num_layers, 1, figsize=(8, fig_h))

    if num_layers == 1:
        axes = [axes]

    # ====== 5. 逐层画图 ======
    for idx, (logits, boxes, name) in enumerate(pairs):
        ax = axes[idx]

        # 背景图
        ax.imshow(img_rgb)

        # 计算中心点绝对坐标
        centers = boxes_to_centers_abs(boxes, H, W)  # (N, 2)
        xs = centers[:, 0]
        ys = centers[:, 1]

        # 红色点绘制所有 query
        ax.scatter(xs, ys, s=8, c="red", marker="o", alpha=0.8)

        ax.set_title(f"{name}: {centers.shape[0]} queries", fontsize=11)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[OK] decoder query 可视化已保存: {save_path}")
    return save_path

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_decoder_outputs(
    out,
    img_path="/public/home/zncui/pycharm_project/MaskDINO/myscripts/transformed_image_v1.jpg",
    save_path="/public/home/zncui/pycharm_project/MaskDINO/myscripts/decoder_queries_score_buckets.jpg",
):
    """
    可视化 decoder 各层 query 在背景图上的位置，并根据每个 query 自身的 score(=sigmoid(pred_logits))
    分段上色。

    规则：
    - 仅使用 interm_outputs 和 aux_outputs（忽略 main 输出）。
    - 层顺序：interm(如果有) -> aux[0] -> aux[1] -> ...
    - 对每一层：
        1) 从 pred_logits 提取 batch=0、类别0 的 logits，并 sigmoid 得到 score ∈ [0,1]
        2) 每个 query 的颜色由该 score 所在的区间决定：
           score > 0.6                         -> 红色 ("red")
           0.5 <= score <= 0.6                 -> 橙色 ("orange")
           0.4 <= score < 0.5                  -> 金黄 ("gold")
           0.3 <= score < 0.4                  -> 黄绿 ("yellowgreen")
           0.2 <= score < 0.3                  -> 绿 ("green")
           0.1 <= score < 0.2                  -> 浅蓝 ("deepskyblue")
           0.0 <= score < 0.1                  -> 深蓝 ("navy")
        3) 每层一个子图（N x 1 布局），背景为同一张图像，点为该层各 query 的 bbox 中心。
    - pred_boxes 假定为 (B, N, 4)：
        若数值在 [0,1] 内，则视为归一化 (cx, cy, w, h)，否则视为 (x1,y1,x2,y2) 绝对坐标。
    """

    # ===== 0. 读取背景图 =====
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取背景图: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # ===== 1. 工具函数：取 score =====
    def _extract_scores(pred_logits):
        """
        将 pred_logits 转为一维 scores 向量 (N,)：
        - 支持 (B, N, C), (B, N), (N, C), (N,)
        - 默认取 batch=0, 类别维度取第0类，然后过 sigmoid。
        """
        t = torch.as_tensor(pred_logits).detach().float()

        if t.dim() == 3:      # (B, N, C)
            t = t[0]          # (N, C)
            if t.size(-1) > 1:
                # 若想改为取 max 类别，可替换为：t = t.max(-1).values
                t = t[:, 0]
            else:
                t = t[:, 0]
        elif t.dim() == 2:    # (B, N) or (N, C)
            if t.size(0) > 1 and t.size(1) != 1:
                # 视为 (B, N)，取第0个 batch
                t = t[0]
            elif t.size(1) == 1:
                t = t[:, 0]
            else:
                # 视为 (N,)
                pass
        elif t.dim() == 1:    # (N,)
            pass
        else:
            raise ValueError(f"不支持的 pred_logits 维度: {t.shape}")

        scores = t.sigmoid()  # (N,)
        return scores.cpu().numpy()

    # ===== 2. 工具函数：box -> 中心绝对坐标 =====
    def _boxes_to_centers_abs(boxes_tensor, h, w):
        """
        boxes_tensor: (B, N, 4) or (N, 4)
        -> 返回 (N, 2) 绝对坐标 (cx_abs, cy_abs)
        """
        boxes = torch.as_tensor(boxes_tensor).detach().float()

        if boxes.dim() == 3:
            boxes = boxes[0]  # (N, 4)
        elif boxes.dim() != 2:
            raise ValueError(f"不支持的 boxes 维度: {boxes.shape}")

        # 简单判断是否归一化
        normalized = (boxes[..., :4].max() <= 1.0) and (boxes[..., :4].min() >= 0.0)

        if normalized:
            # (cx, cy, w, h) in [0,1]
            cx = boxes[:, 0] * w
            cy = boxes[:, 1] * h
        else:
            # 认为是 (x1, y1, x2, y2)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

        centers = torch.stack([cx, cy], dim=-1)
        return centers.cpu().numpy()  # (N, 2)

    # ===== 3. 按 interm + aux 顺序收集层 =====
    layers = []  # 每个元素: dict(name, scores, centers)

    # interm 在最前
    if "interm_outputs" in out and isinstance(out["interm_outputs"], dict):
        inter = out["interm_outputs"]
        if "pred_logits" in inter and "pred_boxes" in inter:
            scores = _extract_scores(inter["pred_logits"])
            centers = _boxes_to_centers_abs(inter["pred_boxes"], H, W)
            layers.append(
                {
                    "name": "interm",
                    "scores": scores,
                    "centers": centers,
                }
            )

    # 然后 aux[0], aux[1], ...
    if "aux_outputs" in out:
        for i, aux in enumerate(out["aux_outputs"]):
            if "pred_logits" in aux and "pred_boxes" in aux:
                scores = _extract_scores(aux["pred_logits"])
                centers = _boxes_to_centers_abs(aux["pred_boxes"], H, W)
                layers.append(
                    {
                        "name": f"aux[{i}]",
                        "scores": scores,
                        "centers": centers,
                    }
                )

    if len(layers) == 0:
        raise ValueError("没有找到 interm_outputs 或 aux_outputs 中的 (pred_logits, pred_boxes)")

    # ===== 4. 定义 score 分段与颜色映射 =====
    # 注意：score > 0.6 单独设为红色
    #       0.0 ~ 0.6 每 0.1 一个区间，共 6 段
    #       0.6 本身这里归入 [0.5, 0.6] 区间，>0.6 才是红色
    bucket_defs = [
        (0.0, 0.1, "navy",        "[0.0, 0.1)"),
        (0.1, 0.2, "deepskyblue", "[0.1, 0.2)"),
        (0.2, 0.3, "green",       "[0.2, 0.3)"),
        (0.3, 0.4, "yellowgreen", "[0.3, 0.4)"),
        (0.4, 0.5, "gold",        "[0.4, 0.5)"),
        (0.5, 0.6 + 1e-6, "orange", "[0.5, 0.6]"),  # 0.6 归到这里
    ]
    high_color = "red"   # score > 0.6

    # ===== 5. 画图 =====
    num_layers = len(layers)
    fig_h = max(3 * num_layers, 3)
    fig, axes = plt.subplots(num_layers, 1, figsize=(8, fig_h))

    if num_layers == 1:
        axes = [axes]

    # 我们只在第一层画图例，避免太挤
    legend_handles = None

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        ax.imshow(img_rgb)

        centers = layer["centers"]
        scores = layer["scores"]
        # 防止 boxes 数量和 scores 数量不等，截断到最小长度
        n = min(len(centers), len(scores))
        centers = centers[:n]
        scores = scores[:n]

        xs = centers[:, 0]
        ys = centers[:, 1]

        # 先画 score > 0.6 的红点
        mask_high = scores > 0.6
        if mask_high.any():
            h = ax.scatter(xs[mask_high], ys[mask_high], s=16, c=high_color, alpha=0.9, label="> 0.6")
        else:
            h = None

        # 再画各个 bucket
        bucket_handles = []
        for (low, high, color, label) in bucket_defs:
            if high > 0.6:   # 防止和 high_color 冲突，已经让 >0.6 走了高分分支
                high_eff = min(high, 0.6 + 1e-6)
            else:
                high_eff = high

            mask = (scores >= low) & (scores < high_eff)
            if mask.any():
                h_bucket = ax.scatter(xs[mask], ys[mask], s=10, c=color, alpha=0.85, label=label)
                bucket_handles.append(h_bucket)

        n_q = centers.shape[0]
        ax.set_title(f"{layer['name']}  (queries={n_q})", fontsize=11)
        ax.set_axis_off()

        # 第一层添加图例
        if idx == 0:
            handles = []
            labels = []
            # bucket legend
            for (low, high, color, label), h_bucket in zip(bucket_defs, bucket_handles):
                handles.append(h_bucket)
                labels.append(label)
            # high score legend
            if h is not None:
                handles.append(h)
                labels.append("> 0.6")

            if handles:
                legend_handles = (handles, labels)
                ax.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[OK] decoder query 按 score 分段上色可视化已保存: {save_path}")
    return save_path

def vis_decoder_layer_results(outputs, batched_inputs):
    # 获取数据
    # predictions = [('pred_logits', 'pred_boxes', 'pred_masks', 'layer_name')]
    # targets = [{'gt_boxes', 'gt_masks'}]
    # 计算 predictions 中每层与 targets 的 IoU，其中 gt_mask 与 gt_boxs 可选
    img_path = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/transformed_image_v1.jpg'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    predictions = []
    if 'pred_logits' in outputs and 'pred_boxes' in outputs and 'pred_masks' in outputs:
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_masks = outputs['pred_masks']
        predictions.append((pred_logits, pred_boxes, pred_masks, 'main'))
    if 'interm_outputs' in outputs:
        interm = outputs['interm_outputs']
        if 'pred_logits' in interm and 'pred_boxes' in interm and 'pred_masks' in interm:
            pred_logits = interm['pred_logits']
            pred_boxes = interm['pred_boxes']
            pred_masks = interm['pred_masks']
            predictions.append((pred_logits, pred_boxes, pred_masks, 'interm'))
    if 'aux_outputs' in outputs:
        for i, aux in enumerate(outputs['aux_outputs']):
            if 'pred_logits' in aux and 'pred_boxes' in aux and 'pred_masks' in aux:
                pred_logits = aux['pred_logits']
                pred_boxes = aux['pred_boxes']
                pred_masks = aux['pred_masks']
                predictions.append((pred_logits, pred_boxes, pred_masks, f'aux[{i}]'))

    targets = {}
    gt_boxes = batched_inputs[0]['instances'].get('gt_boxes')
    gt_masks = batched_inputs[0]['instances'].get('gt_masks')
    targets.update({'gt_boxes': gt_boxes, 'gt_masks': gt_masks})

    def compute_mask_iou(pred_mask, gt_mask):
        # pred_mask, gt_mask: (H, W), binary mask
        intersection = torch.logical_and(pred_mask, gt_mask).sum()
        union = torch.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            return 0.0
        else:
            return intersection.float() / union.float()

    def compute_box_iou(pred_box, gt_box, H, W):
        # pred_box, gt_box: (cx, cy, w, h) in [0,1]
        # 转为 (x1, y1, x2, y2) 绝对坐标
        pcx, pcy, pw, ph = pred_box
        gx1, gy1, gx2, gy2 = gt_box
        px1 = (pcx - pw / 2.0) * W
        py1 = (pcy - ph / 2.0) * H
        px2 = (pcx + pw / 2.0) * W
        py2 = (pcy + ph / 2.0) * H

        # 计算 IoU
        inter_x1 = max(px1, gx1)
        inter_y1 = max(py1, gy1)
        inter_x2 = min(px2, gx2)
        inter_y2 = min(py2, gy2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_pred = (px2 - px1) * (py2 - py1)
        area_gt = (gx2 - gx1) * (gy2 - gy1)
        union = area_pred + area_gt - intersection

        if union == 0:
            return 0.0
        else:
            return intersection / union

        # # debug 返回 center 距离
        # pred_box_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
        # tgt_box = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)
        # center_distance = abs(pred_box_center[0] - tgt_box[0]) + abs(pred_box_center[1] - tgt_box[1])
        # center_distance = -center_distance
        # return center_distance

    layer_indices_list = []
    max_iou_list = []
    mode = 'gt_boxes'  # 'gt_boxes' or 'gt_masks'
    for l, layer_prediction in enumerate(predictions):
        pred_logits, pred_boxes, pred_masks, layer_name = layer_prediction
        gt_boxes = targets['gt_boxes'].tensor
        instances = gt_boxes
        if 'boxes' in mode:
            W, H = 1024, 1024
            pred_boxes = pred_boxes.squeeze()  # (N, 4)
            iou_matrix = torch.zeros(len(pred_boxes), len(instances))
            for i, box in enumerate(pred_boxes):
                for j, ins in enumerate(instances):
                    iou = compute_box_iou(box, ins, H, W)
                    iou_matrix[i, j] = iou
            iou_sum = iou_matrix.sum(dim=-1)
            zero_iou_sum_indices = torch.where(iou_sum == 0)
            max_iou, indices = iou_matrix.max(dim=-1)
            indices[zero_iou_sum_indices] = -1
            layer_indices_list.append(indices)
            max_iou_list.append(max_iou)
    stacked_layer_indices_list = torch.stack(layer_indices_list, dim=0)
    stacked_max_iou_list = torch.stack(max_iou_list, dim=0)
    # 观察 query id 每层对应的预测结果与最大 iou 匹配实例
    transposed_layer_indices_list = stacked_layer_indices_list[1:].transpose(1, 0)
    transposed_max_iou_list = stacked_max_iou_list[1:].transpose(1, 0)
    layer_pred_logits_list = [item[0].squeeze() for item in predictions] # pred_logits (bs, nq, nc) --> (1, 300, 1) --> (300,)
    transposed_layer_pred_logits_list = torch.stack(layer_pred_logits_list[1:], dim=0).transpose(1, 0) # (10, 300) --> (300, 10)

    information_for_visualization = []
    for i, (transposed_layer_indices, transposed_layer_pred_logits, transposed_max_iou) in \
            enumerate(zip(transposed_layer_indices_list,
                          transposed_layer_pred_logits_list,
                          transposed_max_iou_list)):
        # (300, 10), (300, 10), (300, 10) --> (10,), (10,), (10,)
        transposed_layer_pred_logits_with_sigmoid = transposed_layer_pred_logits.sigmoid().detach().cpu()
        # for query_id in [21, 22]:
        #     if query_id in transposed_layer_indices:
        #         print('query_id: ', query_id)
        #         for layer_idx, layer_logits_with_sigmoid, max_iou in zip(transposed_layer_indices, transposed_layer_pred_logits_with_sigmoid,
        #                                                         transposed_max_iou):
        #             print(layer_idx.item(), ' --> ', layer_logits_with_sigmoid.item(), ' -- ', max_iou.item())
        for tgt_id in range(len(targets['gt_boxes'])):
            # if tgt_id in transposed_layer_indices:
            if tgt_id == transposed_layer_indices[-1]:
                print('query_id: ', tgt_id)
                # print(transposed_layer_indices.tolist())
                # print(transposed_layer_pred_logits_with_sigmoid.tolist())
                # print(transposed_max_iou.tolist())
                information_for_visualization.append((tgt_id,
                                                      transposed_layer_indices.tolist(),
                                                      transposed_layer_pred_logits_with_sigmoid.tolist(),
                                                      transposed_max_iou.tolist()))
    from matplotlib import pyplot as plt
    nrows = 10
    ncols = 5
    fig0, axes0 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes0 = axes0.flatten()

    # 第一列是 max_score, 第二列是 query_id
    # max_score_indices = torch.full((len(targets['gt_boxes']), 2), -1, dtype=torch.float)
    # for i, info in enumerate(information_for_visualization):
    #     tgt_id, _, layer_pred_logits, _ = info
    #     if max_score_indices[tgt_id][0] == -1:
    #         max_score_indices[tgt_id][0] = layer_pred_logits[-1]
    #         max_score_indices[tgt_id][1] = i
    #     curr_max_score = max_score_indices[tgt_id][0]
    #     if layer_pred_logits[-1] > curr_max_score:
    #         max_score_indices[tgt_id][0] = layer_pred_logits[-1]
    #         max_score_indices[tgt_id][1] = i


    for i, info in enumerate(information_for_visualization):
        tgt_id, layer_indices, layer_pred_logits_with_sigmoid, layer_max_iou = info
        ax = axes0[tgt_id]
        x = np.arange(len(layer_indices))
        ax.plot(x, layer_pred_logits_with_sigmoid, marker='x', color='orange', label='Pred Logits (sigmoid)')
        ax.plot(x, layer_max_iou, marker='s', color='green', label='Max IoU')
        # if i == max_score_indices[tgt_id][1]:
        #     ax.plot(x, layer_pred_logits_with_sigmoid, marker='x', color='red', label='Pred Logits (sigmoid)')
        #     ax.plot(x, layer_max_iou, marker='s', color='blue', label='Max IoU')
        ax.set_ylabel('Score / IoU')
        ax.set_title(f'Query ID: {tgt_id}')
        ax.grid(alpha=0.3)
        # lines, labels = ax.get_legend_handles_labels()
        # ax.legend(lines, labels, loc='upper right', fontsize=8)

    # 测试功能
    # lines = axes0[8].get_lines()
    # for line in lines:
    #     xdata = line.get_xdata()
    #     ydata = line.get_ydata()
    #     print("X:", xdata)
    #     print("Y:", ydata)

    # 将分数最高的 score-line 和对应的 iou-line 绘制特殊颜色
    from collections import defaultdict
    line_style_dict = {
        'Pred Logits (sigmoid)': {'marker': 'o', 'color': 'red'},
        'Max IoU': {'marker': 's', 'color': 'blue'},
    }
    for ax in axes0:
        lines = ax.get_lines() # a list of Line2D
        if lines:
            label_ydata_dict = defaultdict(list)
            for line in lines:
                label_ydata_dict[line.get_label()].append(line.get_ydata())
            target_label_ydata = {}
            for label, ydata_list in label_ydata_dict.items():
                max_score = -1
                for ydata in ydata_list:
                    if ydata[-1] > max_score:
                        max_score = ydata[-1]
                        target_label_ydata[label] = ydata
                x = lines[0].get_xdata()
                marker = line_style_dict[label]['marker']
                color = line_style_dict[label]['color']
                ax.plot(x, target_label_ydata[label], marker=marker, color=color)

    plt.tight_layout()
    plt.savefig('/public/home/zncui/pycharm_project/MaskDINO/myscripts/query_layer_instance_iou_logits.jpg', dpi=200)
    plt.close()

    # 观察变动 1. 哪些实例经常遭遇变动 2. 发生变动的是哪些 query
    total_indice_list = []
    changed_list = []
    for i in range(2, len(stacked_layer_indices_list)):
        prev_indices = stacked_layer_indices_list[i - 1]
        curr_indices = stacked_layer_indices_list[i]
        changed = (prev_indices != curr_indices)
        changed_prev_indices = prev_indices[changed].tolist()
        changed_curr_indices = curr_indices[changed].tolist()
        # changed_indices = set(changed_curr_indices + changed_prev_indices)
        # print(f'Layer {i - 1} to Layer {i}: Changed Indices: {changed_indices}')
        total_indice_list = total_indice_list + changed_prev_indices + changed_curr_indices
        changed_list.append(changed)
    unique, counts = np.unique(total_indice_list, return_counts=True)
    for k, v in zip(unique, counts):
        print(f'Instance {k} changed {v} times.')
    img_rgb = img_rgb.copy()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    for i, box in enumerate(targets['gt_boxes']):
        x1, y1, x2, y2 = map(lambda x: int(x.item()), box)
        color = (128, 128, 128)
        if i in unique:
            count = counts[unique == i]
            if count >= 1:
                color = (255, 0, 0)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = str(i)
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        # 初始位置：框的左上
        tx, ty = x1, y1 - 5
        # 边界修正（统一写法，最大/最小裁剪）
        tx = max(2, min(tx, W - tw - 2))
        ty = max(th + 2, min(ty, H - 2))
        cv2.putText(img_bgr, text, (tx, ty),
                    font, scale, color, thick, cv2.LINE_AA)
    cv2.imwrite('/public/home/zncui/pycharm_project/MaskDINO/myscripts/layer_vis_input.jpg', img_bgr)

    # 保存为 csv 文件
    arr = stacked_layer_indices_list.detach().cpu().numpy()
    df = pd.DataFrame(arr)
    df.to_csv('/public/home/zncui/pycharm_project/MaskDINO/myscripts/layer_indices.csv', index=False, header=False)

    unique_layer_indices = torch.unique(stacked_layer_indices_list)

    def get_random_color(seed):
        import colorsys
        hue = (seed * 0.618033988749895) % 1.0  # 黄金比例避免重复
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        return (int(r * 255), int(g * 255), int(b * 255))
    color_map_dict = {f'{index}' : get_random_color(index) for index in unique_layer_indices}
    nrows = 2
    ncols = len(predictions)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    import matplotlib.patches as mpatches
    import math

    # 1) 统一管理 score 区间和颜色
    # hi = None 表示无上界（最后一个 > 0.6）
    SCORE_BINS = [
        ("[0.0, 0.1)", 0.0, 0.1, (0, 0, 128)),  # Navy
        ("[0.1, 0.2)", 0.1, 0.2, (0, 191, 255)),  # DeepSkyBlue
        ("[0.2, 0.3)", 0.2, 0.3, (0, 128, 0)),  # Green
        ("[0.3, 0.4)", 0.3, 0.4, (154, 205, 50)),  # YellowGreen
        ("[0.4, 0.5)", 0.4, 0.5, (255, 215, 0)),  # Gold
        ("[0.5, 0.6]", 0.5, 0.6000001, (255, 165, 0)),  # Orange（闭区间用一点点 trick）
        ("> 0.6", 0.6, None, (255, 0, 0)),  # Red
    ]

    # 2) 分数 -> 颜色
    def score_to_color(score: float):
        for _, lo, hi, color in SCORE_BINS:
            if hi is None:
                if score >= lo:
                    return color
            else:
                if lo <= score < hi:
                    return color
        return (0, 0, 0)  # fallback（理论上到不了）

    # 3) 往某个 ax 上加图例（只调一次就行）
    def add_score_legend(ax):
        legend_items = [
            (label, color) for (label, _, _, color) in reversed(SCORE_BINS)
        ]  # 反转一下：高分在上

        patches = [
            mpatches.Patch(color=tuple(c / 255 for c in color), label=label)
            for label, color in legend_items
        ]
        ax.legend(
            handles=patches,
            loc="lower right",
            fontsize=8,
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            title="Score",
            title_fontsize=9,
        )
    a_zero_cx_cy_counts = torch.zeros(len(predictions))
    b_zero_cx_cy_counts = torch.zeros(len(predictions))
    for l, layer_prediction in enumerate(predictions):
        pred_logits, pred_boxes, pred_masks, layer_name = layer_prediction

        # ========= 画框部分（原逻辑不动） =========
        img_rgb_box = img_rgb.copy()
        layer_indices = layer_indices_list[l]
        pred_boxes = pred_boxes.squeeze()

        # H, W = img_rgb_box.shape[:2]
        # font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        # for i, box in enumerate(targets['gt_boxes']):
        #     # if i != 21 or i != 22:
        #     #     continue
        #     x1, y1, x2, y2 = map(lambda x: int(x.item()), box)
        #     color = (255, 255, 255)
        #     cv2.rectangle(img_rgb_box, (x1, y1), (x2, y2), color, 2)
        #     text = str(i)
        #     (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        #     # 初始位置：框的左上
        #     tx, ty = x1, y1 - 5
        #     # 边界修正（统一写法，最大/最小裁剪）
        #     tx = max(2, min(tx, W - tw - 2))
        #     ty = max(th + 2, min(ty, H - 2))
        #     cv2.putText(img_rgb_box, text, (tx, ty),
        #                 font, scale, color, thick, cv2.LINE_AA)

        # changed = None
        # if l == 1:
        #     changed = changed_list[l - 1]
        # if 2 <= l:
        #     changed1 = changed_list[l - 2] # 当前哪些 query 变动了
        #     if l == len(predictions) - 1:
        #         changed2 = torch.zeros_like(changed1, dtype=torch.bool)
        #     else:
        #         changed2 = changed_list[l - 1]
        #     changed = torch.logical_or(changed1, changed2)

        changed = None
        # if l == 2:
        #     changed = changed_list[l - 1]
        #     print('1: ', changed)
        # if l == 3:
        #     changed = changed_list[l - 2]
        #     print('2: ', changed)

        for i, box in enumerate(pred_boxes):
            cx, cy, w, h = box
            cx = int(cx * W)
            cy = int(cy * H)
            if cx + cy == 0:
                a_zero_cx_cy_counts[l] += 1
            color = (
                color_map_dict[str(layer_indices[i].detach().cpu().numpy())]
                if layer_indices[i] != -1
                else (128, 128, 128)
            )
            # if changed is not None and changed[i]:
            #     color = get_random_color(i)
            #     print(f'layer {l} -- query {i} -- cx() cy(): ({cx}, {cy}) changed color to {color}.')
            #     x1, y1 = int(cx - w / 2.0 * W), int(cy - h / 2.0 * H)
            #     x2, y2 = int(cx + w / 2.0 * W), int(cy + h / 2.0 * H)
            #     area = w * W * h * H
            #     if area == 0:
            #         print(f'layer {l} -- query {i} has zero area box.')
            #     cv2.circle(img_rgb_box, (cx,  cy), 10, color, thickness=-1)
            #     cv2.rectangle(img_rgb_box, (x1, y1), (x2, y2), color, 2)
            #     cv2.putText(img_rgb_box, f'{i}', (x1, y1 - 5), font, scale, color, thick, cv2.LINE_AA)

            x1, y1 = int(cx - w / 2.0 * W), int(cy - h / 2.0 * H)
            x2, y2 = int(cx + w / 2.0 * W), int(cy + h / 2.0 * H)
            # cv2.circle(img_rgb_box, (cx, cy), 10, color, thickness=-1)
            cv2.rectangle(img_rgb_box, (x1, y1), (x2, y2), color, 2)

        ax0 = axes[0][l]
        ax0.imshow(img_rgb_box)
        ax0.set_title(f"{layer_name}: {len(pred_boxes)} queries", fontsize=11)
        ax0.set_axis_off()

        # ========= 画 score 点 + 图例部分 =========
        img_rgb_pts = img_rgb.copy()

        # 一次性算好所有 score
        scores = pred_logits.sigmoid().view(-1)


        for score, box in zip(scores, pred_boxes):
            cx, cy, w, h = box
            cx = int(cx * W)
            cy = int(cy * H)
            if cx + cy == 0:
                b_zero_cx_cy_counts[l] += 1
            color = score_to_color(float(score))
            cv2.circle(img_rgb_pts, (cx, cy), 2, color, 2)

        ax1 = axes[1][l]
        ax1.imshow(img_rgb_pts)
        ax1.set_title(f"{layer_name}: {len(pred_boxes)} queries", fontsize=11)
        ax1.set_axis_off()

        # 只在第一列加一次图例即可
        if l == 0:
            add_score_legend(ax1)
        print('l: ', l)
        print('a_zero_cx_cy_counts: ', a_zero_cx_cy_counts)
        print('b_zero_cx_cy_counts: ', b_zero_cx_cy_counts)

    plt.tight_layout()
    save_path = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/decoder_layers_queries_overlay.jpg'
    plt.savefig(save_path, dpi=200)
    plt.close()












