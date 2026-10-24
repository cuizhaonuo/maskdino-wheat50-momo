# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskDINO Training Script based on Mask2Former.
"""
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
except:
    pass

import copy
import itertools
import logging
import os
import time

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger

# MaskDINO
from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    SpikeletPointEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    DetrDatasetMapper,
)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref

# 注册数据集
from data_register import wheat_data, maize_rice_data, xinxiang_data, gwhd2021_data, wheat_data_cat1_quality, wheat_data_cat2, wheat_data_cat3


# 小穗点头的参数名子串（decoder 中的 point_coord_embed / point_class_embed）。
_POINT_HEAD_KEYS = ("point_coord_embed", "point_class_embed")


def freeze_all_but_point_head(model, extra_keys=()):
    """两阶段迁移的 Stage2：冻结除小穗点头以外的全部参数，只训练点头。

    除 point_coord_embed / point_class_embed 外的参数 requires_grad=False，
    因此 build_optimizer 会自动只把点头参数加入优化器（它按 requires_grad 过滤）。

    注意：这里【不能】再去改各模块的 training 标志把 trunk 拉成 eval。本模型里
    training 标志同时承担了语义分支的开关：maskdino.py 的 forward 靠它选训练/推理分支，
    maskdino_decoder.py 靠它决定是否做 DN、以及是否给非最后一层算 mask 预测。把它置 False
    会让 AMPTrainer 的 `assert model.training` 直接失败，或让 aux 的 mask 损失拿到 None。
    而"固定 BN/dropout"这个初衷本就无需处理：backbone 是 FrozenBN、pixel decoder 是 GN、
    MaskDINO.DROPOUT=0.0，没有任何对 train/eval 敏感的层。
    """
    keys = tuple(_POINT_HEAD_KEYS) + tuple(extra_keys)
    trainable, frozen = 0, 0
    for name, p in model.named_parameters():
        if any(k in name for k in keys):
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()

    logger = logging.getLogger("detectron2")
    logger.info(
        f"[FREEZE_EXCEPT_POINT_HEAD] 可训练参数 {trainable:,} | 冻结参数 {frozen:,}"
        + (f" | 额外解冻 {list(extra_keys)}" if extra_keys else "")
    )
    if trainable == 0:
        raise RuntimeError(
            "FREEZE_EXCEPT_POINT_HEAD=True 但未匹配到任何点头参数；"
            f"请检查参数名是否包含 {_POINT_HEAD_KEYS}"
        )
    return model


# ---------------------------------------------------------------------------------------------
# mini/coral-ecount-bbstop：E 损失（loss_point_ecount*）的梯度不进 backbone。
# 开关 MODEL.MaskDINO.POINT_ECOUNT_STOP_BACKBONE（默认 False），预注册见 experiments-bank/ideas/coral-ecount-bbstop.md。
# ---------------------------------------------------------------------------------------------
_ECOUNT_LOSS_PREFIX = "loss_point_ecount"   # 主输出、_0~_8、_interm 共 11 项（_dn 不会被产生）
_BACKBONE_PARAM_PREFIX = "backbone."


def ecount_backbone_stop_backward(loss_dict, scale, pass_params, accum_steps=1, stop_prefix=_ECOUNT_LOSS_PREFIX):
    """把 AMPTrainer 的一次 `scale(sum(loss_dict) / accum).backward()` 拆成两次，梯度累加进 .grad：

      1. E 损失（键以 stop_prefix 开头）：只对 pass_params（非 backbone 的可训练参数）反传，保留计算图；
      2. 其余损失：照常反传到全部参数（含 backbone），释放计算图。

    两份都乘同一个 scale（GradScaler.scale；fp32 验证时传恒等函数）、都除以 accum_steps。
    合并后：backbone 参数的 .grad = 只有其余损失的梯度；非 backbone 参数的 .grad = 全部损失的梯度
    （与一次整体反传只差浮点求和顺序）。先做 E 损失这一遍：它只经过 decoder / pixel decoder，
    保留计算图期间多占的显存比先做整网反传少；两份梯度相加的结果与顺序无关。
    返回 E 损失的键（按 loss_dict 的顺序）。
    """
    stop_keys = [k for k in loss_dict if k.startswith(stop_prefix)]
    if not stop_keys:
        raise RuntimeError(f"POINT_ECOUNT_STOP_BACKBONE=True 但 loss_dict 里没有 {stop_prefix}* 项")
    losses_stop = sum(loss_dict[k] for k in stop_keys) / accum_steps
    losses_rest = sum(v for k, v in loss_dict.items() if not k.startswith(stop_prefix)) / accum_steps
    if losses_stop.requires_grad:
        torch.autograd.backward(scale(losses_stop), inputs=pass_params, retain_graph=True)
    scale(losses_rest).backward()
    return stop_keys


class EcountBackboneStopAMPTrainer(AMPTrainer):
    """POINT_ECOUNT_STOP_BACKBONE=True 时替换 AMPTrainer。

    run_step 照搬 detectron2/engine/train_loop.py 的 AMPTrainer.run_step（含本项目加的 grad_accum_steps：
    只在累积周期开头 zero_grad、loss 除以 grad_accum_steps、累积满才 grad_scaler.step / update），
    只把 backward 换成 ecount_backbone_stop_backward。前向、loss_dict 的记录（_write_metrics）、
    GradScaler 的 unscale_ 与 inf/nan 检查、FullModelGradientClippingOptimizer 的全模型裁剪、
    AdamW 都不变，后几项在 grad_scaler.step 里作用于合并后的梯度。state_dict / resume 同 AMPTrainer。
    只支持单进程（不包 DDP）。
    """

    def __init__(self, model, data_loader, optimizer, **kwargs):
        super().__init__(model, data_loader, optimizer, **kwargs)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self._ecount_pass_params = [p for n, p in named if not n.startswith(_BACKBONE_PARAM_PREFIX)]
        n_stop = len(named) - len(self._ecount_pass_params)
        if n_stop == 0 or not self._ecount_pass_params:
            raise RuntimeError(
                f"POINT_ECOUNT_STOP_BACKBONE：可训练张量 {len(named)} 个，其中 backbone {n_stop} 个；两边都不能为空"
            )
        self._ecount_logged = False
        logging.getLogger("detectron2").info(
            f"[POINT_ECOUNT_STOP_BACKBONE] {_ECOUNT_LOSS_PREFIX}* 的梯度不进 backbone："
            f"可训练张量 {len(named)} 个，挡住 {n_stop} 个（{_BACKBONE_PARAM_PREFIX}*），"
            f"E 损失只反传到其余 {len(self._ecount_pass_params)} 个"
        )

    def run_step(self):
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # 只在一个累积周期开始时清空梯度
        if self.iter % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=self.precision):
            loss_dict = self.model(data)
            if not isinstance(loss_dict, dict):
                raise TypeError("POINT_ECOUNT_STOP_BACKBONE 需要模型返回 loss dict")

        # 与 AMPTrainer 相同：两份 loss 都乘 grad_scaler 的 scale、都除以 grad_accum_steps
        stop_keys = ecount_backbone_stop_backward(
            loss_dict, self.grad_scaler.scale, self._ecount_pass_params, self.grad_accum_steps
        )
        if not self._ecount_logged:
            logging.getLogger("detectron2").info(
                f"[POINT_ECOUNT_STOP_BACKBONE] 挡住 backbone 的 loss 项 {len(stop_keys)} 个：{stop_keys}"
            )
            self._ecount_logged = True

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        # 只有累积满了，才真正更新参数
        if (self.iter + 1) % self.grad_accum_steps == 0:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        # Stage2：只训练小穗点头时，先冻结其余参数（必须在 build_optimizer 之前，
        # 因为优化器按 requires_grad 过滤参数）。
        if cfg.MODEL.MaskDINO.FREEZE_EXCEPT_POINT_HEAD:
            freeze_all_but_point_head(model, cfg.MODEL.MaskDINO.UNFREEZE_KEYS)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)

        if cfg.MODEL.MaskDINO.POINT_ECOUNT_STOP_BACKBONE:
            # mini/coral-ecount-bbstop：E 损失的梯度不进 backbone（见 EcountBackboneStopAMPTrainer）
            if cfg.MODEL.MaskDINO.POINT_ECOUNT_WEIGHT <= 0:
                raise ValueError("POINT_ECOUNT_STOP_BACKBONE=True 需要 POINT_ECOUNT_WEIGHT > 0（没有 E 损失可挡）")
            if not cfg.SOLVER.AMP.ENABLED:
                raise ValueError("POINT_ECOUNT_STOP_BACKBONE=True 只实现了 AMPTrainer 的版本，需要 SOLVER.AMP.ENABLED=True")
            if comm.get_world_size() != 1:
                raise ValueError("POINT_ECOUNT_STOP_BACKBONE=True 只支持单卡（两次反传与 DDP 的梯度同步不兼容）")
            self._trainer = EcountBackboneStopAMPTrainer(
                model,
                data_loader,
                optimizer,
                grad_accum_steps=cfg.SOLVER.GRAD_ACCUM_STEPS,
            )
        elif cfg.SOLVER.AMP.ENABLED:
            self._trainer = AMPTrainer(
                model,
                data_loader,
                optimizer,
                grad_accum_steps=cfg.SOLVER.GRAD_ACCUM_STEPS,
            )
        else:
            self._trainer = SimpleTrainer(
                model,
                data_loader,
                optimizer,
            )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        kwargs = {
            "trainer": weakref.proxy(self),
        }

        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
            # Stage2：额外评估每个 wheat_2 实例内部的小穗点定位与计数（COCOEvaluator 不评这个）
            if cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL:
                evaluator_list.append(
                    SpikeletPointEvaluator(
                        dataset_name,
                        output_dir=output_folder,
                        sigmas=cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL_SIGMAS,
                        score_thresh=cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL_SCORE_THRESH,
                        iou_thresh=cfg.MODEL.MaskDINO.TEST.SPIKELET_EVAL_IOU_THRESH,
                    )
                )
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # coco instance segmentation lsj new baseline
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Semantic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def main(args):
    cfg = setup(args)
    print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
