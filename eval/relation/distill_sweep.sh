#!/bin/bash
# 증류 T·α 스윕 러너 — 캐시된 teacher logits 로 student 학습 + 게이트1/2 채점.
# 각 조합: distill_train.py 학습 → scaling_eval.py holdout 채점(KLUE F1 + colleagues).
# 결과는 distill_sweep_results.tsv 에 append (조합·F1·colleagues·dev).
#
# 사용: (venv 활성 + DISTILL_CACHE 존재 전제)
#   cd ~/distill/xgen-ontokit/eval/relation && source ~/venv/bin/activate
#   DISTILL_STUDENT=klue/roberta-base DISTILL_CACHE=distill_logits.pt bash distill_sweep.sh
set -u
STUDENT="${DISTILL_STUDENT:?DISTILL_STUDENT 필요}"
CACHE="${DISTILL_CACHE:?DISTILL_CACHE 필요}"
RESULTS="${DISTILL_RESULTS:-distill_sweep_results.tsv}"
DEVICE="${DISTILL_DEVICE:-cuda}"

# 설계서 스윕: T {2,4} × α {0.5,0.7,0.9}
TS="${SWEEP_T:-2 4}"
ALPHAS="${SWEEP_ALPHA:-0.5 0.7 0.9}"

echo -e "combo\tKLUE_F1\tcolleagues_F1\tdev\tout_dir" >> "$RESULTS"
for T in $TS; do
  for A in $ALPHAS; do
    tag="T${T}a${A//./}"
    out="model_re_distill_${tag}"
    echo "=== [$(date +%H:%M:%S)] 학습 $tag → $out ==="
    DISTILL_STUDENT="$STUDENT" DISTILL_CACHE="$CACHE" DISTILL_OUT="$out" \
      DISTILL_T="$T" DISTILL_ALPHA="$A" DISTILL_DEVICE="$DEVICE" \
      python3 distill_train.py 2>&1 | tee "train_${tag}.log"

    if [ ! -d "$out" ]; then
      echo "=== $tag 학습 산출 없음(실패) — 건너뜀 ==="
      echo -e "${tag}\tTRAIN_FAIL\t-\t-\t-" >> "$RESULTS"
      continue
    fi

    echo "=== [$(date +%H:%M:%S)] 채점 $tag (holdout) ==="
    DISTILL_DEVICE="$DEVICE" python3 scaling_eval.py "$out" holdout 2>&1 | tee "eval_${tag}.log"

    # 파싱 (eval_re.report/per_class 규약):
    #   "[...@holdout] micro-F1(excl no_rel) = 0.6274  P=... R=..." → '= ' 뒤 첫 숫자
    #   per-class colleagues 행: ('per:colleagues', sup, tp, fp, prec, rec, f1) → 마지막 숫자 = f1
    f1=$(grep 'micro-F1' "eval_${tag}.log" | grep -oE '= [0-9.]+' | grep -oE '[0-9.]+' | head -1)
    col=$(grep 'colleagues' "eval_${tag}.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    dev=$(grep -oE 'dev [0-9.]+' "train_${tag}.log" | grep -oE '[0-9.]+' | tail -1)
    echo -e "${tag}\t${f1:-NA}\t${col:-NA}\t${dev:-NA}\t${out}" >> "$RESULTS"
    echo "=== [$(date +%H:%M:%S)] $tag done: F1=${f1:-NA} colleagues=${col:-NA} ==="
  done
done
echo "DISTILL-SWEEP-DONE"
