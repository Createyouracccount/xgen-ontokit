"""경로2 2차 ablation — 채택 쌍 라벨 밸런싱(과다 라벨 상한캡).

근거: DebiasPL(CVPR 2022, arXiv 2201.01490) — 균형 소스에서도 pseudo-label 은 특정
라벨로 쏠려 false-majority 편향. Noisy Student 식 클래스 상한캡 + confidence 상위 유지.

1차 결과: per:title 9,822(21%) 등 과다 → 정밀도 하락(0.641→0.573). 상한캡으로 완화 시도.

입력: distill_tag_pairs 산출(label(int) + teacher_conf). 라벨별 CAP 초과분은
  teacher_conf 낮은 순으로 폐기(고확신 우선 유지).

재현:
  BAL_SRC=tagged_all.jsonl BAL_OUT=tagged_balanced.jsonl BAL_CAP=2000 \
  /path/venv/bin/python distill_balance.py
env: BAL_SRC(필수), BAL_OUT(기본 tagged_balanced.jsonl), BAL_CAP(라벨별 상한, 기본 2000)
"""
import json
import os
from collections import defaultdict

from labels import LABELS


def main():
    SRC = os.environ["BAL_SRC"]
    OUT = os.getenv("BAL_OUT", "tagged_balanced.jsonl")
    CAP = int(os.getenv("BAL_CAP") or "2000")

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]

    # 라벨별로 묶고 confidence 내림차순 정렬 → 상위 CAP 만 유지
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    kept = []
    for lab, group in by_label.items():
        group.sort(key=lambda r: r.get("teacher_conf", 0), reverse=True)
        kept.extend(group[:CAP])

    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    before = Counter(LABELS[r["label"]] for r in rows)
    after = Counter(LABELS[r["label"]] for r in kept)
    print(f"밸런싱 CAP={CAP}: {len(rows):,} → {len(kept):,}", flush=True)
    print("과다 라벨 변화:", flush=True)
    for lab, n in before.most_common(6):
        print(f"  {lab}: {n} → {after[lab]}", flush=True)
    print(f"saved → {OUT}", flush=True)
    print("BALANCE-DONE", flush=True)


if __name__ == "__main__":
    main()
