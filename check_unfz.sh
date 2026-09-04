#!/usr/bin/env bash
# 核对五个 run 的实际可训练参数是否等于 dry-run 预期，并汇报进度。
REPO=/public/home/zncui/pycharm_project/maskdino-wheat50
STRIP='s/\x1b\[[0-9;]*m//g'
printf "%-7s %11s %11s %-6s %6s %-9s %s\n" arm 期望 实际 一致 iter 显存 最近
for a in last1 last3 last5 last7 last9; do
  OUT=$REPO/output_v2/exp-0904-unfz-$a
  [ -f "$OUT/train.log" ] || { printf "%-7s %11s %11s\n" "$a" "$(cat "$OUT/EXPECTED_TRAINABLE" 2>/dev/null||echo -)" "未启动"; continue; }
  exp=$(cat "$OUT/EXPECTED_TRAINABLE" 2>/dev/null || echo "?")
  got=$(grep -a 'detectron2\]: \[FREEZE_EXCEPT_POINT_HEAD\]' "$OUT/train.log" | head -1 \
        | sed "$STRIP" | sed 's/.*FREEZE_EXCEPT_POINT_HEAD\]//' \
        | grep -oE '[0-9][0-9,]{4,}' | head -1 | tr -d ',')
  line=$(grep -a 'eta:' "$OUT/train.log" | tail -1 | sed "$STRIP")
  it=$(sed -n 's/.*iter: \([0-9]*\).*/\1/p' <<<"$line")
  mem=$(sed -n 's/.*max_mem: \([0-9]*\)M.*/\1M/p' <<<"$line")
  loss=$(sed -n 's/.*total_loss: \([0-9.]*\).*/loss \1/p' <<<"$line")
  err=$(grep -aE 'Error|Traceback|out of memory' "$OUT/train.log" | tail -1 | sed "$STRIP" | cut -c1-40)
  ok="等待"; [ -n "$got" ] && { [ "$got" = "$exp" ] && ok="OK" || ok="不符!"; }
  printf "%-7s %11s %11s %-6s %6s %-9s %s\n" "$a" "$exp" "${got:--}" "$ok" "${it:--}" "${mem:--}" "${err:-$loss}"
done
