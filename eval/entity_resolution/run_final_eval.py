"""ER 최종 평가 — 심판 R2 수정요구 전부 반영. KURE-v1 문장임베딩.

R2 결함 → 수정:
  1. 5:1 불균형이 P 부풀림 → 균형(1:1) P/R 로 측정
  2. threshold 스윕 하한 클리핑 → 하한 0.40 부터(클리핑 감지)
  3. 단일 F1 은 불균형 취약 → 주 지표 = AUC(threshold·불균형 무관)
  4. 날것 klue-roberta 는 주제근접≠동의어 구분 못 함(AUC 0.644) →
     KURE-v1(한국어 문장임베딩, MIT, 대조학습)로 교체

threshold 정직 선택: dev/test 50/50(시드 20260714), dev 균형F1 최대 → test 확정.
의미변이만 크레딧(형태소 baseline B2 가 잡는 표면변이 제외).
"""
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../../src")
from baselines import _b2_fn, load
from eval_auc import auc
from er_embed import EmbeddingER

SEED = 20260714


def _bal_metrics(sim, P, N, th, rng):
    k = min(len(P), len(N))
    Pb = rng.sample(P, k) if len(P) > k else P
    Nb = rng.sample(N, k) if len(N) > k else N
    tp = sum(sim(*p) >= th for p in Pb)
    fp = sum(sim(*n) >= th for n in Nb)
    fn = len(Pb) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, k


def main():
    gold = load()
    b2 = _b2_fn()
    kure = "--klue" not in sys.argv
    er = (EmbeddingER(model="nlpai-lab/KURE-v1", model_kind="st") if kure
          else EmbeddingER())
    name = "KURE-v1" if kure else "klue/roberta-small(날것)"
    terms = {x for p in gold["positives"] + gold["negatives"] for x in (p["a"], p["b"])}
    er.embed_all(sorted(terms))
    sim = er.similarity

    pos = [(p["a"], p["b"]) for p in gold["positives"] if not b2(p["a"], p["b"])]
    neg = [(n["a"], n["b"]) for n in gold["negatives"]]
    rng = random.Random(SEED)
    rng.shuffle(pos); rng.shuffle(neg)
    pd, pt = pos[:len(pos) // 2], pos[len(pos) // 2:]
    nd, nt = neg[:len(neg) // 2], neg[len(neg) // 2:]

    # dev 균형F1 최대 threshold (하한 0.40 — 클리핑 감지)
    ths = [round(0.40 + 0.01 * i, 2) for i in range(56)]
    best = max(ths, key=lambda t: _bal_metrics(sim, pd, nd, t, rng)[2])

    print(f"=== {name} — ER 최종 평가(의미변이, 균형) ===")
    print(f"gold: 의미변이 pos {len(pos)} / hard-neg {len(neg)}")
    print(f"dev threshold = {best} (하한0.40, 클리핑아님={best > 0.40})")
    ps = [sim(*p) for p in pt]; ns = [sim(*n) for n in nt]
    print(f"★ test AUC = {auc(ps, ns):.3f}  (chance 0.5)")
    p, r, f1, k = _bal_metrics(sim, pt, nt, best, rng)
    print(f"test 균형(1:1, k={k}) th={best}: P {p:.3f} R {r:.3f} F1 {f1:.3f}")


if __name__ == "__main__":
    main()
