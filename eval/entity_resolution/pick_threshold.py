"""threshold 정직 선택 — dev 에서 고르고 test 로 확정(과적합 방지).

gold 를 dev/test 로 분할(시드 고정). dev 에서 F1 최대 threshold 선택 → test 성능 보고.
test 로 threshold 를 고르면 과적합(허수아비) — 계층 R1 교훈.
"""
import json
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../../src")
from baselines import _b2_fn, load
from eval_er import prf, net_value
from er_embed import EmbeddingER


def split(gold, seed=20260714, dev_ratio=0.5):
    rng = random.Random(seed)
    def sp(items):
        items = list(items)
        rng.shuffle(items)
        k = int(len(items) * dev_ratio)
        return items[:k], items[k:]
    pd, pt = sp(gold["positives"])
    nd, nt = sp(gold["negatives"])
    return ({"positives": pd, "negatives": nd, "meta": gold["meta"]},
            {"positives": pt, "negatives": nt, "meta": gold["meta"]})


def main():
    gold = load()
    dev, test = split(gold)
    er = EmbeddingER()
    b2 = _b2_fn()
    terms = {x for p in gold["positives"] + gold["negatives"] for x in (p["a"], p["b"])}
    er.embed_all(sorted(terms))

    # dev 에서 최고 F1 threshold (형태소 결합 기준 — 실제 배선 형태)
    best_th, best_f1 = None, -1
    for th in [round(0.70 + 0.01 * i, 2) for i in range(30)]:
        er._threshold = th
        comb = lambda a, b: b2(a, b) or er.same_entity(a, b)
        f1 = prf(dev, comb)["f1"]
        if f1 > best_f1:
            best_f1, best_th = f1, th
    print(f"dev 선택 threshold = {best_th} (dev F1 {best_f1:.3f})")

    # test 로 확정 보고
    er._threshold = best_th
    comb = lambda a, b: b2(a, b) or er.same_entity(a, b)
    mt = prf(test, comb)
    me = prf(test, er.same_entity)
    nv = net_value(test, b2, er.same_entity)
    mb2 = prf(test, b2)
    print(f"\n=== test 확정(threshold={best_th}) ===")
    print(f"B2 형태소키    : F1 {mb2['f1']:.3f} P {mb2['precision']:.3f} R {mb2['recall']:.3f}")
    print(f"임베딩 단독    : F1 {me['f1']:.3f} P {me['precision']:.3f} R {me['recall']:.3f}")
    print(f"임베딩+형태소  : F1 {mt['f1']:.3f} P {mt['precision']:.3f} R {mt['recall']:.3f}")
    print(f"순가치(의미변이): {nv['system_hits']}/{nv['residual']} = {nv['net_recall']:.1%}")


if __name__ == "__main__":
    main()
