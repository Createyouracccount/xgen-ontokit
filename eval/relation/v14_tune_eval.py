#!/usr/bin/env python3
"""v14 — tune 스플릿 per-class 평가 (K 선택 규율·arm 선별용, 홀드아웃 미접촉).

사용: python3 v14_tune_eval.py model_re_v13c model_re_v14_T1 [...]
출력: 모델별 전체 micro-F1(no_rel 제외 표준) + colleagues + watchlist +
PER-PER 블록(alternate_names·친족 묶음) per-class P/R/F1.
"""
import json
import sys

sys.path.insert(0, ".")
from eval_re import report, per_class          # noqa: E402
from m2_eval_aug import load, predict          # noqa: E402
from labels import LABEL2ID                    # noqa: E402

ID2 = {v: k for k, v in LABEL2ID.items()}
WATCH = ["per:colleagues", "org:member_of", "org:top_members/employees",
         "org:founded_by", "per:employee_of", "per:title"]
PERPER = ["per:alternate_names", "per:spouse", "per:siblings", "per:children",
          "per:parents", "per:other_family"]


def main():
    models = sys.argv[1:] or ["model_re_v13c"]
    rows = load("tune")
    gold = [r["label"] for r in rows]
    for md in models:
        preds = predict(md, rows)
        report(f"{md} @ tune", gold, preds)
        # per_class 반환: (label명, support, tp, fp, P, R, F1)
        rows_pc = per_class(gold, preds, top=999)
        print("  class                              sup    P      R      F1")
        for name, sup, tp, fp, p, rec, f1 in rows_pc:
            if name in WATCH + PERPER:
                print(f"  {name:34s} {sup:4d}  {p:.3f}  {rec:.3f}  {f1:.3f}")


if __name__ == "__main__":
    main()
