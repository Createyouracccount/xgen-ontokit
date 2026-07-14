"""ER 평가 — baseline(형태소키) + 사전 채널(Wikidata QID) + 결합, 순가치 분해.

  B2 morph      : 현 본체 형태소키(표면변이만)
  D  wd-dict    : Wikidata QID 일치(의미변이)
  C  combined   : B2 or D (표면 + 의미)
사용: python3 run_eval.py  (Wikidata API 캐시 빌드 후 재실행은 빠름)
"""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../../src")
from baselines import _b2_fn, load
from eval_er import report, net_value, prf
from er_dict import WikidataERDict


def main():
    gold = load()
    print("gold:", gold["meta"])
    b2 = _b2_fn()
    report("B2 morph(현 본체)", gold, b2)

    wd = WikidataERDict(online="--offline" not in sys.argv)
    # 캐시 워밍 — 모든 표기 미리 조회(진행 표시)
    terms = set()
    for p in gold["positives"] + gold["negatives"]:
        terms.add(p["a"]); terms.add(p["b"])
    terms = sorted(terms)
    for i, t in enumerate(terms):
        wd._qids(t)
        if (i + 1) % 25 == 0:
            print(f"  wd 조회 {i+1}/{len(terms)}", flush=True)
            wd.save()
    wd.save()

    d_fn = wd.same_entity
    report("D wd-dict(의미변이)", gold, d_fn)
    combined = lambda a, b: b2(a, b) or d_fn(a, b)
    report("C combined(표면+의미)", gold, combined)

    nv = net_value(gold, b2, d_fn)
    print(f"\n순가치(B2 미해결 {nv['residual']}건 중 사전이 잡은 것): "
          f"{nv['system_hits']} = {nv['net_recall']:.1%}")
    # semantic 태깅 한정 순가치
    sem_pos = {(min(p['a'],p['b']), max(p['a'],p['b']))
               for p in gold["positives"] if p["type"] == "semantic"}
    sem_hit = sum(1 for a, b in sem_pos if d_fn(a, b))
    print(f"semantic 태깅({len(sem_pos)}) 중 사전 적중: {sem_hit} = {sem_hit/len(sem_pos):.1%}")


if __name__ == "__main__":
    main()
