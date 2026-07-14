"""패턴 분류기 실행 + ablation.

  patterns  : 통사·구조 패턴만 (미발화 = no_relation) — 순수 우리 채널
  prior     : train 타입쌍 최빈 (baseline)
  combined  : 패턴 우선, 미발화 시 prior 폴백

사용: python3 run_eval.py tune [--errors N] [--holdout-final]
holdout 은 --holdout-final 명시 시에만 접촉(누수 차단 프로토콜).
"""
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from baselines import b1_prior
from eval_re import report, per_class
from extractor_rules import predict
from labels import LABELS


def load(name):
    with open(f"data/{name}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    which = "holdout" if "--holdout-final" in sys.argv else "tune"
    rows = load(which)
    golds = [r["label"] for r in rows]

    import pyarrow.parquet as pq
    t = pq.read_table("data/klue_re_train.parquet")
    train = [{c: t.column(c)[i].as_py() for c in t.column_names} for i in range(t.num_rows)]

    pat = [predict(r) for r in rows]
    pri = b1_prior(train, rows)
    comb = [p if p != 0 else q for p, q in zip(pat, pri)]

    report(f"patterns @{which}", golds, pat)
    report(f"prior    @{which}", golds, pri)
    report(f"combined @{which}", golds, comb)

    fired = sum(1 for p in pat if p != 0)
    print(f"\n패턴 발화율: {fired}/{len(rows)} = {fired/len(rows):.1%}")
    print("\nper-class (patterns):")
    for row in per_class(golds, pat):
        print("  ", row)

    if "--errors" in sys.argv:
        n = int(sys.argv[sys.argv.index("--errors") + 1])
        errs = [(r, p) for r, p in zip(rows, pat) if p != 0 and p != r["label"]]
        cnt = Counter((LABELS[r["label"]], LABELS[p]) for r, p in errs)
        print("\nFP 혼동 top:", cnt.most_common(12))
        for r, p in errs[:n]:
            s, o = r["subject_entity"], r["object_entity"]
            print(f"  gold={LABELS[r['label']]} pred={LABELS[p]} "
                  f"S[{s['word']}|{s['type']}] O[{o['word']}|{o['type']}] :: {r['sentence'][:90]}")


if __name__ == "__main__":
    main()
