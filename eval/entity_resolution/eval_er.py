"""ER(개체정규화) 평가 지표 — 동의어쌍 이진 분류 + 순가치 분해.

gold: {"positives": [[a,b],...], "negatives": [[a,b],...]}
  positives = 동의어(같은 개체), negatives = 하드 네거티브(표면 유사하나 다른 개체).
system: 쌍 → bool(동의어라고 판정하는가).

순가치: 형태소 baseline 이 이미 병합하는 표면변이 쌍은 크레딧에서 제외
(계층 '순가치 분해'와 동일 — baseline 이 잡는 건 개선 아님).
"""


def _pair(p):
    """gold 항목(dict{a,b} 또는 [a,b]) → 정렬 튜플."""
    a, b = (p["a"], p["b"]) if isinstance(p, dict) else (p[0], p[1])
    return tuple(sorted((a, b)))


def prf(gold, predict_fn):
    """predict_fn(a,b)->bool 로 P/R/F1. gold=positives/negatives 리스트."""
    pos = {_pair(p) for p in gold["positives"]}
    neg = {_pair(p) for p in gold["negatives"]}
    tp = sum(1 for a, b in pos if predict_fn(a, b))
    fn = len(pos) - tp
    fp = sum(1 for a, b in neg if predict_fn(a, b))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_pos": len(pos), "n_neg": len(neg)}


def net_value(gold, baseline_fn, system_fn):
    """baseline 이 이미 잡는 positives 를 제외한, system 의 순증 recall.

    의미변이(표면 다름)에서만 크레딧 — baseline(형태소키)이 병합하는 표면변이는
    system 이 잡아도 순가치 아님.
    """
    pos = [_pair(p) for p in gold["positives"]]
    residual = [(a, b) for a, b in pos if not baseline_fn(a, b)]  # baseline 미해결
    if not residual:
        return {"residual": 0, "system_hits": 0, "net_recall": 0.0}
    hits = sum(1 for a, b in residual if system_fn(a, b))
    return {"residual": len(residual), "system_hits": hits,
            "net_recall": hits / len(residual)}


def report(name, gold, predict_fn):
    m = prf(gold, predict_fn)
    print(f"[{name}] F1={m['f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
          f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} | pos={m['n_pos']} neg={m['n_neg']})")
    return m
