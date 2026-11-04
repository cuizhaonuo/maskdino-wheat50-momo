# -*- coding: utf-8 -*-
"""逐 iter 打印 loss 的冒烟驱动（detectron2 只在每 20 iter 写一次日志，3 iter 看不到任何数）。

用法：
  CUDA_VISIBLE_DEVICES=1 python tools/probe_sa_order_train.py --iters 3 \
    --config-file configs/coco/instance-segmentation/wheat_joint_single_stage_mini_sa_order.yaml \
    --output-dir output/sa-order-smoke --opts SEED 0

输出每个 iter 的全部 loss_* 键（含 loss_point_coral / loss_point_knn），以及首末 iter 的 total_loss。
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from detectron2.engine import HookBase  # noqa: E402

import train_net as T  # noqa: E402  （import 时顺带注册数据集）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()

    # 脚本自己的默认值放在前面，--opts 里的同名键可以覆盖（merge_from_list 后者优先）
    opts = [
        "SOLVER.MAX_ITER", str(a.iters),
        "SOLVER.CHECKPOINT_PERIOD", str(max(a.iters, 1)),
        "TEST.EVAL_PERIOD", "0",
        "OUTPUT_DIR", a.output_dir,
    ] + list(a.opts)
    args = argparse.Namespace(config_file=a.config_file, opts=opts, num_gpus=1,
                              num_machines=1, machine_rank=0, dist_url="tcp://127.0.0.1:12345",
                              eval_only=False, resume=False, EVAL_FLAG=1)
    cfg = T.setup(args)
    trainer = T.Trainer(cfg)
    trainer.resume_or_load(resume=False)

    first, last = {}, {}

    class PerIter(HookBase):
        def after_step(self):
            st = {k: float(v[0]) for k, v in self.trainer.storage.latest().items()}
            it = int(self.trainer.iter)
            keys = sorted(k for k in st if k.startswith("loss"))
            rec = {k: st[k] for k in keys}
            rec["total_loss"] = st.get("total_loss", float("nan"))
            if not first:
                first.update(rec)
            last.clear()
            last.update(rec)
            print(f"[iter {it}] " + " ".join(f"{k}={rec[k]:.4f}" for k in keys + ["total_loss"]),
                  flush=True)

    trainer.register_hooks([PerIter()])
    trainer.train()

    print("\n[first iter] " + " ".join(f"{k}={v:.6f}" for k, v in first.items()))
    print("[last  iter] " + " ".join(f"{k}={v:.6f}" for k, v in last.items()))
    nan = [k for k, v in last.items() if not torch.isfinite(torch.tensor(v))]
    # loss_point_knn 没有顶层聚合键，只有逐层 loss_point_knn_{i} 与 loss_point_knn_interm
    have = {p: any(k.startswith(p) for k in last) for p in ("loss_point_coral", "loss_point_knn")}
    print(f"[check] losses all finite: {len(nan) == 0}   "
          f"keys present: {have}")


if __name__ == "__main__":
    main()
