"""ER 채널 판별력 = AUC(불균형 무관) + 균형 P/R. 심판 R2 수정요구 반영.

단일 threshold F1은 5:1 불균형에 부풀려짐(R2: F1 0.887은 착시). 대신:
  - AUC = P(sim(pos) > sim(neg)) — threshold·불균형 무관 판별력. chance 0.5.
  - 균형 P/R = negatives 를 positives 와 1:1 로 맞춰 재측.
동의어(pos)와 하드네거티브(neg) 분포가 겹치면 AUC≈0.5 = 채널 무익.
"""
import itertools
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../../src")
from baselines import _b2_fn, load


def auc(pos_sims, neg_sims):
    """P(pos > neg) — Mann-Whitney U 정규화. 동점 0.5."""
    if not pos_sims or not neg_sims:
        return 0.0
    wins = ties = 0
    for p in pos_sims:
        for n in neg_sims:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos_sims) * len(neg_sims))


def evaluate(sim_fn, gold, b2=None):
    """sim_fn(a,b)->float. B2(형태소)가 잡는 표면변이는 제외(순수 의미변이 판별력)."""
    if b2 is None:
        b2 = _b2_fn()
    pos = [(p["a"], p["b"]) for p in gold["positives"] if not b2(p["a"], p["b"])]
    neg = [(n["a"], n["b"]) for n in gold["negatives"]]
    ps = [sim_fn(a, b) for a, b in pos]
    ns = [sim_fn(a, b) for a, b in neg]
    a = auc(ps, ns)
    import statistics as st
    print(f"  의미변이 pos {len(ps)} (median sim {st.median(ps):.3f}) / "
          f"neg {len(ns)} (median {st.median(ns):.3f})")
    print(f"  ★ AUC = {a:.3f}  (chance 0.5, 1.0=완전분리)")
    # 균형 P/R: 여러 threshold 에서 pos=neg 동수 기준
    print(f"  {'th':>5} {'P':>6} {'R':>6} {'F1':>6}  (neg 동수 균형)")
    kbal = min(len(ps), len(ns))
    import random
    rng = random.Random(7)
    ns_bal = rng.sample(ns, kbal) if len(ns) > kbal else ns
    ps_bal = rng.sample(ps, kbal) if len(ps) > kbal else ps
    for th in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        tp = sum(1 for s in ps_bal if s >= th)
        fp = sum(1 for s in ns_bal if s >= th)
        fn = len(ps_bal) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {th:>5.2f} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
    return a


if __name__ == "__main__":
    from er_embed import EmbeddingER
    gold = load()
    if "--kure" in sys.argv:
        er = EmbeddingER(model="nlpai-lab/KURE-v1", model_kind="st")
        name = "KURE-v1"
    else:
        er = EmbeddingER()
        name = "klue/roberta-small(날것)"
    terms = {x for p in gold["positives"] + gold["negatives"] for x in (p["a"], p["b"])}
    er.embed_all(sorted(terms))
    print(f"=== {name} ===")
    evaluate(er.similarity, gold)
