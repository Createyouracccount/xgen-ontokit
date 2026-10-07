#!/bin/sh
# 장시간 측정 감독 — 비정상 종료 시 이어하기로 재시작(최대 N회), 재시작마다 로그에 남긴다.
# 사용: harness/supervise.sh <최대재시작> <log> -- <명령...>
max=$1; log=$2; shift 3
n=0
until "$@" >> "$log" 2>&1; do
  n=$((n+1)); echo "[supervise] $(date '+%F %T') 비정상 종료 — 재시작 $n/$max" >> "$log"
  [ "$n" -ge "$max" ] && { echo "[supervise] 재시작 한도 도달 — 중단" >> "$log"; exit 1; }
  sleep 30
done
echo "[supervise] $(date '+%F %T') 정상 종료" >> "$log"
