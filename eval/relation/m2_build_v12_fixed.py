"""re-ko-aug-v12 고정 학습셋 생성 (0719 심판 90/100 채택분의 재현 스크립트 — 보존 조건 이행).

원리(캡 재표집 교란 차단 — v11 반려 사유): v1 증강파일(m2_prep_sredfm.py 의 P112 필터
비활성 산출물, 결정론)에서 v1 학습과 동일한 선착순 캡(8k/라벨)으로 67,760행을 재현한 뒤
**P112 무근거행 721건만 제자리 제거** — 재표집 없음, 순서 보존 부분수열.
사용: /opt/miniconda3/bin/python m2_build_v12_fixed.py <v1_aug.jsonl> <out.jsonl>
학습: M2_AUG_PATH=<out.jsonl> M2_OUT=model_re_aug_v12 python m2_train_aug.py
"""
import collections
import json
import sys

P112_EVIDENCE = ("설립", "창립", "창설", "창업", "창시", "세운", "세웠", "만든", "만들었")
CAP = 8000

src, dst = sys.argv[1], sys.argv[2]
rows, cnt = [], collections.Counter()
with open(src, encoding="utf-8") as f:
    for line in f:  # v1 학습(load_aug)과 동일: 파일 순서 선착순 캡
        r = json.loads(line)
        if cnt[r["label"]] >= CAP:
            continue
        cnt[r["label"]] += 1
        rows.append(r)
kept = [r for r in rows
        if not (r.get("pid") == "P112"
                and not any(k in r["sentence"] for k in P112_EVIDENCE))]
with open(dst, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"v1 캡 선별 {len(rows)} → P112 무근거 제거 {len(rows)-len(kept)} → 고정셋 {len(kept)}")
