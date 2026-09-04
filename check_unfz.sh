#!/usr/bin/env bash
# 核对五个 run 的实际可训练参数是否等于 dry-run 预期，并汇报进度。
REPO=/public/home/zncui/pycharm_project/maskdino-wheat50
printf "%-8s %12s %12s %-6s %8s %s\n" arm 期望 实际 一致 iter 最近一行
for a in last1 last3 last5 last7 last9; do
  OUT=$REPO/output_v2/exp-0904-unfz-$a
  [ -f "$OUT/train.log" ] || { printf "%-8s %12s %12s\n" "$a" "-" "未启动"; continue; }
  exp=$(cat "$OUT/EXPECTED_TRAINABLE" 2>/dev/null || echo "?")
  got=$(grep -m1 FREEZE_EXCEPT_POINT_HEAD "$OUT/train.log" | grep -oE '[0-9][0-9,]*' | head -1 | tr -d ',')
  it=$(grep -oE 'iter: [0-9]+' "$OUT/train.log" | tail -1 | awk '{print $2}')
  ok="等待"; [ -n "$got" ] && { [ "$got" = "$exp" ] && ok="OK" || ok="不符!"; }
  printf "%-8s %12s %12s %-6s %8s %s\n" "$a" "$exp" "${got:--}" "$ok" "${it:--}" \
    "$(grep -E 'total_loss|Error|error|Traceback' "$OUT/train.log" | tail -1 | cut -c1-60)"
done
