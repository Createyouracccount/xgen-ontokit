#!/bin/sh
# 메모리 가드 — Mac 여유 메모리를 30초마다 본다(memory_pressure 의 free %).
#   < 20% : 측정 클라이언트(harness.hybrid)를 일시정지(SIGSTOP) — 새 판독 요청이 쌓이지 않게
#   ≥ 30% : 재개(SIGCONT)
#   < 10% : MLX 서버 종료(캐시 회수) — 감독 스크립트가 측정을 이어서 재시작
# 모든 조치는 로그에 남긴다. 사용: harness/memguard.sh <log>
log=${1:-harness/results/memguard.log}
paused=0
while true; do
  free=$(memory_pressure | tail -1 | sed -E 's/.*: ([0-9]+)%.*/\1/')
  if [ "$free" -lt 10 ]; then
    echo "$(date '+%F %T') free=${free}% < 10 — MLX 서버 종료" >> "$log"; pkill -f mlx_lm.server
  fi
  if [ "$free" -lt 20 ] && [ $paused -eq 0 ]; then
    pkill -STOP -f "harness.hybrid"; paused=1; echo "$(date '+%F %T') free=${free}% < 20 — 측정 일시정지" >> "$log"
  elif [ "$free" -ge 30 ] && [ $paused -eq 1 ]; then
    pkill -CONT -f "harness.hybrid"; paused=0; echo "$(date '+%F %T') free=${free}% ≥ 30 — 재개" >> "$log"
  fi
  sleep 30
done
