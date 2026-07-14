"""ER 최종 — 사전(우리말샘)+형태소+임베딩(KURE) 3채널. 심판 R4 후 확정.

심판 결론: ER 은 계층·관계보다 원리적으로 어려운 축(주제근접≠동의어 경계를
임베딩이 못 그음). 88 은 비현실적 목표 → 게이트 0.80 재설정(사용자 결정 2026-07-14).
버그수정: union-find sense 보존 + 블롭 폐기(미국=일본=한국 오병합 제거, 심판 R4).

채널: U 우리말샘 비슷한말(확정,P우선) + M 형태소키(표면변이) + E KURE(고th 보강).
gold=위키 redirect ⊥ 자원=우리말샘(누수 없음). LLM API 0회(로컬 추론+정규식).
threshold 정직 선택: dev/test 50/50(시드20260714), dev 균형F1 최대 → test 확정.
"""
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../../src")
from baselines import _b2_fn, load
from eval_auc import auc

SEED = 20260714


def _bal(fn, P, N, rng):
    k = min(len(P), len(N))
    Pb = rng.sample(P, k) if len(P) > k else P
    Nb = rng.sample(N, k) if len(N) > k else N
    tp = sum(fn(*p) for p in Pb); fp = sum(fn(*n) for n in Nb)
    fn_ = len(Pb) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn_) if tp + fn_ else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), k


def main():
    gold = load()
    b2 = _b2_fn()
    from er_urimalsam import UrimalsamER
    u = UrimalsamER()
    print(f"우리말샘 사전: {u.size()} 표면형 (블롭 {getattr(u, '_blob_dropped', '?')} 폐기)")

    er = None
    if "--no-emb" not in sys.argv:
        from er_embed import EmbeddingER
        er = EmbeddingER(model="nlpai-lab/KURE-v1", model_kind="st")
        terms = {x for p in gold["positives"] + gold["negatives"] for x in (p["a"], p["b"])}
        er.embed_all(sorted(terms))

    def channel(a, b, th):
        if b2(a, b) or u.same_entity(a, b):
            return True
        return er is not None and er.similarity(a, b) >= th

    pos = [(p["a"], p["b"]) for p in gold["positives"]]
    neg = [(n["a"], n["b"]) for n in gold["negatives"]]
    rng = random.Random(SEED)
    rng.shuffle(pos); rng.shuffle(neg)
    pd, pt = pos[:len(pos) // 2], pos[len(pos) // 2:]
    nd, nt = neg[:len(neg) // 2], neg[len(neg) // 2:]

    print(f"gold: pos {len(pos)} / neg {len(neg)} (주제근접 {gold['meta'].get('n_topic_near_neg', 0)})\n")

    # dev 균형F1 최대 threshold (하한 0.50)
    ths = [round(0.50 + 0.01 * i, 2) for i in range(46)]
    best = max(ths, key=lambda t: _bal(lambda a, b: channel(a, b, t), pd, nd, rng)[2])

    # test 확정
    p, r, f1, k = _bal(lambda a, b: channel(a, b, best), pt, nt, rng)
    # AUC(임베딩 연속값, 형태소·사전 병합분 제외한 순수 판별)
    if er is not None:
        ps = [er.similarity(*x) for x in pt if not (b2(*x) or u.same_entity(*x))]
        ns = [er.similarity(*x) for x in nt if not (b2(*x) or u.same_entity(*x))]
        print(f"★ test 임베딩 AUC = {auc(ps, ns):.3f}")
    print(f"dev threshold = {best}")
    print(f"★ test 균형(1:1, k={k}) 3채널: P {p:.3f} R {r:.3f} F1 {f1:.3f}")
    print(f"게이트 0.80: {'통과' if f1 >= 0.80 else '미달'}")


if __name__ == "__main__":
    main()
