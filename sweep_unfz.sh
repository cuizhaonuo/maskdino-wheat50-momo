#!/usr/bin/env bash
# 小穗点阈值扫描：POINT_THRESHOLD 进的是模型推理（maskdino.py:545），改它必须重跑推理，
# 不是对同一批输出做后过滤。故每档一次完整 --eval_only。
# 用法: CUDA_VISIBLE_DEVICES=<g> bash sweep_unfz.sh <arm> [阈值...]
set -euo pipefail
ARM="${1:?用法: bash sweep_unfz.sh <last1|...|last9> [阈值...]}"; shift
THRS=("$@"); [ ${#THRS[@]} -eq 0 ] && THRS=(0.05 0.10 0.15 0.20 0.28 0.35 0.40 0.50 0.60 0.70)
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/public/home/zncui/pycharm_project/maskdino-wheat50
PY=/public/home/zncui/anaconda3/envs/MaskDINO-310/bin/python
OUT=$REPO/output_v2/exp-0904-unfz-$ARM
cd "$WT"
for t in "${THRS[@]}"; do
  D=$OUT/sweep/T$t
  if [ -f "$D/log.txt" ] && grep -aq 'copypaste: Task: spikelet' "$D/log.txt"; then
    echo "[$ARM] T=$t 已有结果，跳过"; continue
  fi
  mkdir -p "$D"
  echo "[$(date '+%H:%M:%S')] $ARM T=$t ..."
  PYTHONPATH=. $PY train_net.py --eval_only \
    --config-file "$OUT/config.yaml" \
    MODEL.WEIGHTS "$OUT/model_final.pth" \
    OUTPUT_DIR "$D" \
    MODEL.MaskDINO.TEST.POINT_THRESHOLD "$t" > "$D/eval.log" 2>&1 \
    || { echo "[$ARM] T=$t 失败"; tail -15 "$D/eval.log"; }
done
echo "[$(date '+%H:%M:%S')] $ARM 扫描完成"
