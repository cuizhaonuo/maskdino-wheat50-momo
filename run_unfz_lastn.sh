#!/usr/bin/env bash
# Stage2 · 按 decoder 层数细分解冻面：解冻最后 N 层 + decoder 内全部非层模块 + 全部预测头。
# 与 exp-0902-combo 逐项相同，唯一变量是 UNFREEZE_KEYS。
# 预注册文档：experiments-bank/ideas/unfreeze-decoder-last-n.md
#
# 前台阻塞运行，日志写进 $OUT/train.log。GPU 由调用方的 CUDA_VISIBLE_DEVICES 指定。
#   CUDA_VISIBLE_DEVICES=1 setsid nohup bash run_unfz_lastn.sh last1 >/dev/null 2>&1 &
#   串行排队： bash -c 'bash run_unfz_lastn.sh last7; bash run_unfz_lastn.sh last9'
set -euo pipefail

ARM="${1:?用法: bash run_unfz_lastn.sh <last1|last3|last5|last7|last9>}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/public/home/zncui/pycharm_project/maskdino-wheat50
PY=/public/home/zncui/anaconda3/envs/MaskDINO-310/bin/python
STAGE1=$REPO/output_v2/wheat_stage1_detector_v2_cosine/model_final.pth

# FIRST=最靠前的解冻层号（层号 0..8，8 是最后一层）；EXPECT=dry-run 实测的可训练参数
case "$ARM" in
  last1) FIRST=8; EXPECT=2356433  ;;
  last3) FIRST=6; EXPECT=5448145  ;;
  last5) FIRST=4; EXPECT=8539857  ;;
  last7) FIRST=2; EXPECT=11631569 ;;
  last9) FIRST=0; EXPECT=14723281 ;;   # == 组合口径的 arm C
  *) echo "未知 arm: $ARM" >&2; exit 2 ;;
esac

# 五个 arm 共有的常量部分：decoder 内全部非层模块 + 全部预测头（810,577）。
# 点头 point_coord_embed / point_class_embed 由 train_net.py 恒置为可训练，不必列。
HEADS=(
  sem_seg_head.predictor.decoder_norm            # 每层输出进头前的 LN（与 decoder.norm 同一张量）
  sem_seg_head.predictor.decoder.ref_point_head  # 逐层 query 位置编码
  sem_seg_head.predictor.decoder.bbox_embed      # 框头，9 层共享一份
  sem_seg_head.predictor.class_embed             # 分类头
  sem_seg_head.predictor.mask_embed              # 掩码头
)
KEYS=""
for i in $(seq "$FIRST" 8); do KEYS+="'sem_seg_head.predictor.decoder.layers.$i',"; done
for h in "${HEADS[@]}";      do KEYS+="'$h',"; done
KEYS="[${KEYS%,}]"

OUT=$REPO/output_v2/exp-0904-unfz-$ARM
mkdir -p "$OUT"
echo "$EXPECT" > "$OUT/EXPECTED_TRAINABLE"
cd "$WT"

echo "[$(date '+%F %T')] $ARM 起跑 | 层 $FIRST-8 | 期望可训练 $EXPECT | GPU=${CUDA_VISIBLE_DEVICES:-all} | $OUT"
PYTHONPATH=. exec $PY train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/wheat_stage2_pointhead.yaml \
  MODEL.WEIGHTS "$STAGE1" \
  OUTPUT_DIR "$OUT" \
  MODEL.MaskDINO.UNFREEZE_KEYS "$KEYS" \
  >> "$OUT/train.log" 2>&1
