# -*- coding: utf-8 -*-
"""跑 N 个 iter 并把每个 iter 的 loss 打出来（train_net.py 的 loss 日志周期写死 20，
3 iter 的冒烟看不出数值）。

复用 train_net.py 的 setup / Trainer，不训练完就退，不写评估。

用法：
  CUDA_VISIBLE_DEVICES=1 python tools/smoke_3iter.py --num-gpus 1 \
    --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_mini_sa_order.yaml \
    --iters 3 OUTPUT_DIR output/sa-order-smoke
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from detectron2.engine import default_argument_parser  # noqa: E402
from detectron2.engine.hooks import HookBase  # noqa: E402
from detectron2.utils.events import get_event_storage  # noqa: E402

from train_net import setup, Trainer  # noqa: E402


class LossPrinter(HookBase):
    @staticmethod
    def _fmt(v):
        if isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    def after_step(self):
        storage = get_event_storage()
        items = dict(storage.latest())
        keep = [k for k in items if k in ("total_loss", "lr")
                or k.startswith("loss_point_coral") or k.startswith("loss_point_knn")]
        # 非有限的 loss 一律列出（含那种偶发 inf 的层损失）
        bad = [k for k, v in items.items() if "loss" in k and isinstance(v, (float, int))
               and not (v == v and abs(v) != float("inf"))]
        txt = "  ".join(f"{k}={self._fmt(items[k])}" for k in sorted(keep))
        print(f"[iter {self.trainer.iter + 1}] {txt}"
              + (f"  | NON-FINITE: {bad}" if bad else "  | all losses finite"), flush=True)


def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.register_hooks([LossPrinter()])
    trainer.train()
    print(f"[done] last_iter={trainer.iter + 1}", flush=True)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()
    # --iters 用 SOLVER.MAX_ITER 传下去（yacs 在 setup 里 merge，方便复现）
    args.opts = list(args.opts) + ["SOLVER.MAX_ITER", str(args.iters)]
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
    main(args)
