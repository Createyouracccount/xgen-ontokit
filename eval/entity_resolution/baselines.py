"""ER 정직 baseline — 현 본체 형태소 정규화(DeterministicDedup).

B0 exact       : 문자열 완전 일치(= 항상 별개, floor 의 floor)
B1 surface-norm: 공백·하이픈·괄호 제거 + 소문자(형태소 baseline 근사, 무학습 floor)
B2 본체 형태소키: DeterministicDedup._noun_key 동일 판정 — 현 프로덕션 실제 능력

핵심: B2 가 표면변이는 잡고 의미변이는 못 잡음을 gold 로 실측. 개선의 순가치는
B2 가 못 잡는 semantic 에서만 발생(eval_er.net_value).
"""
import json
import re
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../../src")
from eval_er import report, net_value

_NORM = re.compile(r"[\s\-·・_()（）]")


def load():
    with open("data/gold.json", encoding="utf-8") as f:
        return json.load(f)


def b0_exact(a, b):
    return a == b


def b1_surface(a, b):
    n = lambda s: _NORM.sub("", s).lower()
    return n(a) == n(b)


def _b2_fn():
    from kiwipiepy import Kiwi
    from ontokit.dedup.deterministic import DeterministicDedup
    d = DeterministicDedup(Kiwi())
    return lambda a, b: (d._noun_key(a) == d._noun_key(b)
                         and bool(d._noun_key(a)))


if __name__ == "__main__":
    gold = load()
    print("gold:", gold["meta"])
    report("B0 exact", gold, b0_exact)
    report("B1 surface-norm", gold, b1_surface)
    b2 = _b2_fn()
    m2 = report("B2 본체 형태소키", gold, b2)
    # 순가치 분해 — B2 가 못 잡는 positives(의미변이)가 개선 여지
    residual = [(p["a"], p["b"]) for p in gold["positives"]
                if not b2(p["a"], p["b"])]
    sem = [p for p in gold["positives"] if p["type"] == "semantic"]
    print(f"\nB2 미해결 positives: {len(residual)}/{len(gold['positives'])} "
          f"(이게 ER 개선의 타깃 = 사전 조회로 잡을 의미변이)")
    print(f"gold semantic 태깅: {len(sem)}")
