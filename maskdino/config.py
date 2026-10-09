# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from detectron2.config import CfgNode as CN


def add_maskdino_config(cfg):
    """
    Add config for MaskDINO.
    """
    # NOTE: configs from original mask2former
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "MaskDINO_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # MaskDINO model config
    cfg.MODEL.MaskDINO = CN()
    cfg.MODEL.MaskDINO.LEARN_TGT = False

    # loss
    cfg.MODEL.MaskDINO.PANO_BOX_LOSS = False
    cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS = False
    cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
    cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = 2.

    # cost weight
    cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT = 2.

    # transformer config
    cfg.MODEL.MaskDINO.NHEADS = 8
    cfg.MODEL.MaskDINO.DROPOUT = 0.1
    cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MaskDINO.ENC_LAYERS = 0
    cfg.MODEL.MaskDINO.DEC_LAYERS = 6
    cfg.MODEL.MaskDINO.INITIAL_PRED = True
    cfg.MODEL.MaskDINO.PRE_NORM = False
    cfg.MODEL.MaskDINO.BOX_LOSS = True
    cfg.MODEL.MaskDINO.HIDDEN_DIM = 256
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MaskDINO.TWO_STAGE = True
    cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']
    cfg.MODEL.MaskDINO.DN="seg"
    cfg.MODEL.MaskDINO.DN_NOISE_SCALE=0.4
    cfg.MODEL.MaskDINO.DN_NUM=100
    cfg.MODEL.MaskDINO.PRED_CONV=False

    cfg.MODEL.MaskDINO.EVAL_FLAG = 1

    # ===== 小穗点回归（逐实例点预测）配置 =====
    # 每个实例 query 预测的小穗点数量上限。全库实测单穗小穗数：
    # train min=1 / max=24 / mean=13.9 / p95=18，val min=7 / max=22。
    # 原注释写"最多 23"有误，实测是 24；P=25 即刚好够覆盖，无需留余量。
    cfg.MODEL.MaskDINO.NUM_POINTS = 50
    # 点损失权重
    cfg.MODEL.MaskDINO.POINT_COORD_WEIGHT = 5.0   # 点坐标 L1 回归损失
    cfg.MODEL.MaskDINO.POINT_CLASS_WEIGHT = 2.0   # 点存在性 focal 分类损失
    # 内层匈牙利匹配的代价权重
    cfg.MODEL.MaskDINO.COST_POINT_COORD_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_POINT_CLASS_WEIGHT = 2.0
    # 点存在性 focal 的 alpha。与 loss_labels 的 self.focal_alpha 解耦：
    # 检测头是 300 query 配约 39 个正样本，点头是 P 个 slot 配约 14 个正样本，
    # 正负比温和得多，0.25（来自 RetinaNet 稠密检测）在这里把正样本压得过狠。
    cfg.MODEL.MaskDINO.POINT_FOCAL_ALPHA = 0.25
    # Ni==0（无小穗的 wheat）实例的存在性 focal 损失权重。1.0 = 现状。
    # 推理时 instance_inference 按 pred_classes 门控，wheat 实例本就不输出小穗点，
    # 这部分监督只在误分类时才生效，故可降权。0.0 等价于「只对 wheat_2 算点损失」。
    cfg.MODEL.MaskDINO.POINT_EMPTY_CLS_WEIGHT = 1.0
    # 点坐标误差（内层匹配代价与 L1 损失共用）的归一化方式：
    # "box_axis" = 现状，框相对坐标按 (w, h) 分轴归一化。框长宽比实测 p50 2.3 / p90 3.7，
    #              短轴 1 px 误差的权重是长轴的 2.3~3.7 倍，而小穗最挤的恰是沿轴方向
    #              （相邻约 9.5 px，逼近 σ=8）。
    # "iso"      = 各向同性：误差统一除以 sqrt(w·h)（像素纵横比还原后），两轴同权。
    cfg.MODEL.MaskDINO.POINT_COORD_NORM = "box_axis"
    # ---- GroupPose 式分组 decoder（mini/group-decoder）----
    # True：每个匹配 query 挂 NUM_POINTS 个小穗 token，decoder 每层做「实例内 SA → 实例 token 走原层 →
    # 小穗 token 的跨实例 SA / 可变形 cross-attn / FFN」；小穗点 = 各 token 自己的参考点逐层细化，
    # 存在性由各 token 自己判别。现有 MLP 点头只保留为 interm 与 layer-0 的点预测，兼作初始参考点。
    # 所有层的 pred_point_coords 变为图像归一化绝对坐标（criterion / 推理据此换算）。
    # False：原 MLP 点头，模型逐位不变。
    cfg.MODEL.MaskDINO.POINT_GROUP_DECODER = False
    # 小穗 token 的跨实例 SA："slot" = 第 k 号 token 与其它实例的第 k 号交流（GroupPose 原样）；
    # "none" = 不做（只经实例内 SA 间接获得跨实例信息）
    cfg.MODEL.MaskDINO.POINT_ACROSS_ATTN = "slot"
    # 小穗 token 的 FFN 宽度（实例 token 仍用 DIM_FEEDFORWARD）
    cfg.MODEL.MaskDINO.POINT_DIM_FEEDFORWARD = 1024
    # 实例 token 接收实例内 SA 的更新时乘一个零初始化的标量门：开训时实例路径与原模型一致
    cfg.MODEL.MaskDINO.POINT_WITHIN_GATE = True
    # 小穗 token 输出绝对坐标、在 GT 框系里算误差。开训时实例框还很差，匹配上的 token 可离 GT
    # 几十个框宽（实测 L1 初值 57/层，基线 1.8）。L1 对每个远点都给满额梯度，小穗分支会去追这些
    # 框级别的误差（开训时该分支梯度范数 L1 290 vs ρ 48；对共享主干的梯度两者都约 14，
    # 远小于检测/mask 损失的 2400~3900，不存在「压过其它损失」）。
    # >0 时这些输出的坐标误差（内层代价与损失）改用 ρ(e) = c·log(1 + |e|/c)：|e|<<c 时 ≈ L1
    # （c=1、|e|=0.05 处差 2.5%），远处梯度 1/(1+|e|/c)，方向由已定位好的实例主导。
    # 0 = 纯 L1。只作用于绝对坐标输出，layer-0 / interm 的 MLP 点头不受影响。
    cfg.MODEL.MaskDINO.POINT_COORD_ROBUST_C = 0.0
    # ---- 小穗 token 的点 NMS（mini/group-nms-match、mini/group-nms-prune）----
    # 来源：DDQ（Zhang et al., "Dense Distinct Query for End-to-End Object Detection", CVPR 2023）：
    # 每个 refine 阶段前做 class-agnostic NMS，被压掉的 query 不进匹配、不算任何损失，推理时同样做。
    # 这里只作用于分组 decoder 的小穗 token 输出（第 1~9 层，pred_point_abs=True）：内层匈牙利匹配前，
    # 按存在性分数对 K 个 token 做贪心 NMS（坐标与分数 detach），中心距 < κ·sqrt(w·h) 算重复。
    # 被压掉的 token 存在性与坐标损失都不算；留下但没配上的照常判负；损失分母（GT 小穗数）不变；
    # 不做「留下的少于 Ni 时补回」。layer-0 / interm 的 MLP 点头不受影响。
    # "none"  = 不做（与不加此功能时逐位一致）；
    # "match" = 只在匹配与损失侧做，前向不变：criterion 用所配 GT 框在 iso 空间逐层做 NMS，
    #           某层被压的 token 下一层照常参与注意力。
    # 输出里带 pred_point_keep（[B, Q, K] bool 存活掩码）时，criterion 直接用它，不再自己做 NMS。
    # 每次迭代把被压比例、饿死比例（留下的 token 少于 Ni 的实例占比）写进 metrics.json 的 point_nms/*。
    cfg.MODEL.MaskDINO.POINT_NMS_MODE = "none"
    # NMS 半径 κ：中心距 < κ·sqrt(w·h) 算重复（w、h 为框的像素宽高；iso 空间里半径就是 κ）。
    # 数据依据（mini 标注、原生尺度）：GT 小穗最近邻距离 val p1 8.8 / 中位 16.7 px；
    # κ=0.07 在中位 val 穗上约等于 8 px，只有 0.77% 的 GT 小穗在该半径内有真邻居。
    # LSJ 训练把图缩放 0.1~2.0 倍，像素半径不能写死，所以按框尺度归一化。
    # 按中心距理解：「两个半径 8 px 的圆有重叠」等于中心距 16 px，会把约一半相邻的真小穗并掉。
    cfg.MODEL.MaskDINO.POINT_NMS_KAPPA = 0.07
    # True = 存在性头 point_class_embed 用 3 层 MLP（与坐标头 point_coord_embed 对齐），
    # False = 保持单层 nn.Linear。点头参数 12,850 -> 144,434。
    # 【口径】08-25 在 overfit10 上三向同时改善，但 08-26 的 val 检验判定 mF1 优势是记忆
    # （+0.061 -> -0.0012），仅计数存活；且该结论测于「冻结 trunk」，而 08-26~08-28
    # 已证明那道闸门本身才是瓶颈。故此项为待重测的开关，不是已生效的配方。
    # 【注意】开关会改变参数名（point_class_embed.weight/bias <-> .layers.{0,1,2}.*），
    # 两种设置下训出的 Stage2 checkpoint 不能互相加载（会被静默跳过）。
    cfg.MODEL.MaskDINO.POINT_CLASS_MLP = False
    # True = 存在性头改成 CORAL 秩一致结构（maskdino/modeling/coral.py）：P 个 slot 的 logit
    # = 共享 1 维打分 + P 个有序 bias，目标是有序前缀「slot k 亮 <=> k < Ni」。配套地，内层匹配只让前
    # min(Ni, P) 个 slot 参与匈牙利（代价只用坐标 L1），存在性损失由 focal 换成 CORAL 的 BCE，
    # 推理时 K = sum_k 1[sigmoid(l_k) > POINT_THRESHOLD]、取前 K 个 slot。
    # 打开后 POINT_CLASS_MLP 不起作用（打分固定是 3 层 MLP），POINT_FOCAL_ALPHA / COST_POINT_CLASS_WEIGHT
    # 也不再参与点头。False 时模型与基线逐位相同。预注册见 experiments-bank/ideas/coral-exist.md。
    cfg.MODEL.MaskDINO.POINT_CORAL_EXIST = False
    # FREEZE_EXCEPT_POINT_HEAD=True 时，除点头外额外解冻的参数名子串。
    # 空 = 只训点头（Stage2 原设计）。例：["predictor.decoder"] 解冻 transformer decoder，
    # ["sem_seg_head"] 解冻整个分割头（decoder + pixel decoder + 各预测头），backbone 仍冻结。
    cfg.MODEL.MaskDINO.UNFREEZE_KEYS = []
    # 两阶段迁移：Stage2 冻结除小穗点头（point_coord_embed / point_class_embed）以外的全部参数，
    # 只训练点头。冻结的 trunk 在训练时也强制保持 eval（固定 BN/dropout），与推理一致。
    cfg.MODEL.MaskDINO.FREEZE_EXCEPT_POINT_HEAD = False

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level

    #####################

    # MaskDINO inference config
    cfg.MODEL.MaskDINO.TEST = CN()
    cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = True
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE = 0.06
    # 小穗点存在性置信度阈值（sigmoid 后 > 该值的点才输出）
    cfg.MODEL.MaskDINO.TEST.POINT_THRESHOLD = 0.35
    # 推理侧小穗点 NMS 的半径 κ（含义与数据依据见 POINT_NMS_KAPPA）。> 0 时 instance_inference 在最后一层
    # 对每个 wheat_2 实例的全部 K 个点做贪心 NMS（分数降序，中心距 < κ·sqrt(w·h) 算重复，w、h 为预测框的
    # 像素宽高），再按 POINT_THRESHOLD 过滤；先 NMS 后过滤与先过滤后 NMS 结果相同。
    # 0 = 关（与不加此功能时逐位一致）。内置评估的点指标就是按这里的设置出的。
    cfg.MODEL.MaskDINO.TEST.POINT_NMS_KAPPA = 0.0
    # 评估时是否额外加挂小穗点评估器（SpikeletPointEvaluator，见 maskdino/evaluation/）
    cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL = False
    # 点匹配的距离阈值（原图像素，多档一起报）。GT 实测同穗内小穗最近邻间距中位 16px，
    # 故 8px 已是不产生歧义匹配的上限。
    cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL_SIGMAS = [3.0, 5.0, 8.0]
    # 只有分数不低于该阈值的 wheat_2 预测实例才参与点评估。0.4 是 Stage1 检测器在带点 val 上
    # 实例配对 F1 的峰值（recall 0.76 / precision 0.76）；取 0.5 会让 recall 掉到 0.53，
    # 相当于先丢掉一半 GT 穗，点级 recall 也就被封顶了。
    cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL_SCORE_THRESH = 0.4
    # GT 穗与预测穗配对所需的最小 mask IoU
    cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL_IOU_THRESH = 0.5
    # cfg.MODEL.MaskDINO.TEST.EVAL_FLAG = 1

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"

    # transformer module
    cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # ---- LSJ 增强造成的 wheat_2 标签噪声修复 ----
    # ResizeScale(MIN~MAX)+FixedSizeCrop 在 scale>1 时会把完整的 wheat_2 切到 crop 边界，
    # 但标签仍是 wheat_2，与标注规则「截断的穗子归 wheat」矛盾。抽样实测 100 张增强图的
    # 200 个 wheat_2 中有 45 个（22.5%）贴边。开启后在 mapper 里把贴边实例降级为 wheat，
    # 并清空其 gt_points（截断穗的 GT 点数对应的是完整穗，留着即系统性 GT-图像不一致；
    # 清空后该实例 Ni=0，自然走 POINT_EMPTY_CLS_WEIGHT 的降权路径）。
    # 只作用于训练（mapper 对 is_train=False 提前返回），不影响评估。
    cfg.INPUT.DEMOTE_BORDER_TRUNCATED = False
    cfg.INPUT.DEMOTE_FROM_TO = (1, 0)   # (源 contiguous id, 目标 id)；wheat cat2 为 wheat_2->wheat
    cfg.INPUT.DEMOTE_BORDER_MARGIN = 2  # 贴边判定容差（像素）

    # ---- wheat_2 专项 copy-paste 增强（只作用于训练）----
    # 从离线实例库随机取 wheat_2 贴到当前图，直到该图 wheat_2 数达到 TARGET_PER_IMG。
    # 两条几何约束：wheat_2 之间零重叠；wheat_2 可贴在 wheat 上面，被覆盖的 wheat
    # 掩码减掉重叠区、残余面积比低于 MIN_KEEP 则整个丢弃。
    # 实例库用 myscripts/backup/build_wheat2_bank.py 离线构建（含三条质量闸门）。
    # 【风险】本项目有前科：overfit10 把 wheat_2 占比抬到 14.0%（真实 7.7%），
    # val 上过检 +39%、spike_precision 从 0.7588 崩到 0.5462。调 TARGET_PER_IMG 时
    # 必须盯 spike_precision（判否线 0.70）与预测穗数/GT 穗数（上限 1.2）。
    cfg.INPUT.COPY_PASTE_ENABLED = False
    cfg.INPUT.COPY_PASTE_BANK = ""              # 实例库 npz 路径
    cfg.INPUT.COPY_PASTE_TARGET_PER_IMG = 6     # 补到每图几个 wheat_2（现状 3.1）
    cfg.INPUT.COPY_PASTE_MAX_TRIES = 30         # 单次贴图的位置重试上限
    cfg.INPUT.COPY_PASTE_MARGIN = 8             # 距图边最小像素，须 > DEMOTE_BORDER_MARGIN
    cfg.INPUT.COPY_PASTE_MIN_KEEP = 0.2         # 被遮挡 wheat 的残余面积比下限
    cfg.INPUT.COPY_PASTE_FLIP_PROB = 0.5        # 贴图前随机水平翻转的概率

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = 0.75

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # dinov3_sta backbone
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

    cfg.Default_loading=True  # a bug in my d2. resume use this; if first time ResNet load, set it false

    cfg.SOLVER.GRAD_ACCUM_STEPS = 1