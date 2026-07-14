"""KLUE-RE 공식 지표 — micro-F1 (no_relation 제외, TACRED 방식).

정의: no_relation 은 '클래스'에서 제외하되,
  - gold=관계, pred=no_relation → FN
  - gold=no_relation, pred=관계 → FP
  - gold=관계A, pred=관계B → FP(B) + FN(A)
즉 관계 예측의 정밀·재현을 모두 벌한다. KLUE 논문·리더보드와 동일.
"""
from collections import Counter

from labels import LABELS


def micro_f1(golds, preds):
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        if g == 0 and p == 0:
            continue
        if p != 0 and p == g:
            tp += 1
        else:
            if p != 0:
                fp += 1
            if g != 0:
                fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "micro_f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def per_class(golds, preds, top=15):
    stat = {}
    for g, p in zip(golds, preds):
        if g != 0:
            stat.setdefault(g, Counter())["support"] = stat.setdefault(g, Counter()).get("support", 0) + 1
            if p == g:
                stat[g]["tp"] += 1
        if p != 0 and p != g:
            stat.setdefault(p, Counter())["fp"] += 1
    rows = []
    for lid, c in sorted(stat.items(), key=lambda kv: -kv[1].get("support", 0)):
        sup, tp, fp = c.get("support", 0), c.get("tp", 0), c.get("fp", 0)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / sup if sup else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((LABELS[lid], sup, tp, fp, round(prec, 3), round(rec, 3), round(f1, 3)))
    return rows[:top]


def report(name, golds, preds):
    m = micro_f1(golds, preds)
    print(f"[{name}] micro-F1(excl no_rel) = {m['micro_f1']:.4f}  "
          f"P={m['precision']:.3f} R={m['recall']:.3f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    return m
