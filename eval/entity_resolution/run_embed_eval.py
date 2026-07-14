"""ER 임베딩 채널 평가 — threshold 스윕 + 형태소/WD 앙상블.

주력 = 임베딩(심판 권고). B2 형태소키 = 표면변이 보강. WD = 정밀도 보조(옵션).
사용: python3 run_embed_eval.py [--model klue/roberta-small] [--st] [--kure]
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../../src")
from baselines import _b2_fn, load
from eval_er import prf, net_value
from er_embed import EmbeddingER


def main():
    gold = load()
    b2 = _b2_fn()

    if "--kure" in sys.argv:
        er = EmbeddingER(model="nlpai-lab/KURE-v1", model_kind="st")
        tag = "KURE-v1"
    elif "--st" in sys.argv:
        i = sys.argv.index("--model") + 1 if "--model" in sys.argv else None
        er = EmbeddingER(model=sys.argv[i] if i else "klue/roberta-small",
                         model_kind="st")
        tag = er._name
    else:
        i = sys.argv.index("--model") + 1 if "--model" in sys.argv else None
        er = EmbeddingER(model=sys.argv[i] if i else "klue/roberta-small")
        tag = er._name

    # 캐시 워밍
    terms = set()
    for p in gold["positives"] + gold["negatives"]:
        terms.add(p["a"]); terms.add(p["b"])
    er.embed_all(sorted(terms))

    print(f"=== 임베딩 모델: {tag} ===")
    print(f"gold: {gold['meta']}")
    print(f"B2 형태소키(표면 baseline): 순가치 대상 residual 계산용\n")

    print(f"{'th':>5} | {'emb F1':>7} {'P':>6} {'R':>6} | {'+morph F1':>9} {'P':>6} {'R':>6} | 순가치(의미변이)")
    for th in [0.80, 0.83, 0.85, 0.87, 0.90, 0.92, 0.95]:
        er._threshold = th
        emb_fn = er.same_entity
        m = prf(gold, emb_fn)
        comb = lambda a, b: b2(a, b) or emb_fn(a, b)
        mc = prf(gold, comb)
        nv = net_value(gold, b2, emb_fn)  # 형태소가 못 잡는 것 중 임베딩이 잡은 것
        print(f"{th:>5.2f} | {m['f1']:>7.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} | "
              f"{mc['f1']:>9.3f} {mc['precision']:>6.3f} {mc['recall']:>6.3f} | "
              f"{nv['system_hits']:>3}/{nv['residual']} = {nv['net_recall']:.1%}")


if __name__ == "__main__":
    main()
