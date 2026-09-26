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
    # True = 存在性头 point_class_embed 用 3 层 MLP（与坐标头 point_coord_embed 对齐），
    # False = 保持单层 nn.Linear。点头参数 12,850 -> 144,434。
    # 【口径】08-25 在 overfit10 上三向同时改善，但 08-26 的 val 检验判定 mF1 优势是记忆
    # （+0.061 -> -0.0012），仅计数存活；且该结论测于「冻结 trunk」，而 08-26~08-28
    # 已证明那道闸门本身才是瓶颈。故此项为待重测的开关，不是已生效的配方。
    # 【注意】开关会改变参数名（point_class_embed.weight/bias <-> .layers.{0,1,2}.*），
    # 两种设置下训出的 Stage2 checkpoint 不能互相加载（会被静默跳过）。
    cfg.MODEL.MaskDINO.POINT_CLASS_MLP = False
    # True = 在 decoder 入口的多尺度特征 src_flatten 上做全局均值池化，接一个 CORAL 序数头
    # img_count_embed（maskdino/modeling/coral.py），预测【整张图的 wheat_2 穗数】min(n, IMG_COUNT_MAX)。
    # 多一项辅助损失 loss_img_count，只作用于主输出（不做深监督）。False 时不建参数，模型与基线逐位相同。
    # 预注册见 experiments-bank/ideas/coral-img.md。
    cfg.MODEL.MaskDINO.IMG_COUNT_HEAD = False
    cfg.MODEL.MaskDINO.IMG_COUNT_WEIGHT = 2.0
    # 序数类 0..IMG_COUNT_MAX，共 IMG_COUNT_MAX 个二分类任务。mini train 增广前每图最多 13 个、val 最多 8 个。
    cfg.MODEL.MaskDINO.IMG_COUNT_MAX = 15
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
    # 评估时是否额外加挂小穗点评估器（SpikeletPointEvaluator，见 maskdino/evaluation/）
    cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL = False
    # SpikeletPointEvaluator 里哪些 wheat_2 预测算作「报出的穗」：
    #   "thresh" = 现状，分数 >= SPIKELET_EVAL_SCORE_THRESH 的全部算；
    #   "topn"   = 按分数取前 N 个，N 是图级 CORAL 头给的该图 wheat_2 穗数（需要 IMG_COUNT_HEAD=True）。
    # 只影响点评估与计数，不影响 COCO AP。同一 checkpoint 两种读法都能评。
    cfg.MODEL.MaskDINO.TEST.IMG_COUNT_READOUT = "thresh"
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