"""계층 추출 평가 하네스 — 외부 Wikidata gold(한국어 P279) 기준. 심판 검증용.

두 태스크:
 T1 정의문→상위어 발견(Hypernym Discovery, SemEval-2018 T9 프로토콜): 각 클래스의
    한국어 정의문에서 상위어 후보를 랭킹 추출 → gold 상위어와 대조(P@1/MRR/MAP).
 T2 클래스집합→계층 유도(pairwise is-a detection): 전체 클래스 집합에서 subClassOf
    쌍을 유도 → gold 쌍과 P/R/F1(이질계층 R 별도).

채점은 결정적·재현가능(심판이 독립 실행 가능). gold = wd_gold.json(Wikidata CC0).
사용법: python3 eval_hierarchy.py <method>   method ∈ {baseline, improved}
"""
from __future__ import annotations
import sys, json, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

# r0 gold = 누수 있는 초기본(wd_gold_r0.json). r0 재현·심판검증 전용.
GOLD = json.load(open(os.path.join(_HERE, "data", "wd_gold_r0.json")))
PAIRS = [tuple(p) for p in GOLD["pairs"]]
DESCS = GOLD["descs"]
# gold: child -> set(parents)
GOLD_MAP: dict[str, set] = {}
for c, p in PAIRS:
    GOLD_MAP.setdefault(c, set()).add(p)
ALL_CLASSES = set(c for c, p in PAIRS) | set(p for c, p in PAIRS)
HETERO = [(c, p) for c, p in PAIRS if not (c.endswith(p) and c != p)]


# 공유 Kiwi 인스턴스 — 매 호출 재생성(수천 회 = 분 단위 지연) 방지.
_ke = None
def _get_ke():
    global _ke
    if _ke is None:
        from ontokit.morphology.kiwi_nouns import KiwiNounExtractor
        _ke = KiwiNounExtractor()
    return _ke


# ══ T1: 정의문 → 상위어 발견 (랭킹) ══
def t1_baseline(child: str, desc: str) -> list[str]:
    """현 ontokit hearst_ko.definitional_pairs — 따옴표 정의문만. 정의문에 따옴표
    거의 없어 대부분 빈 결과 예상(baseline 약점 노출)."""
    from ontokit.hierarchy.hearst_ko import definitional_pairs
    ke = _get_ke()
    # definitional_pairs 는 문장에서 (child,parent). 여기선 desc 를 문장처럼 투입.
    pairs = definitional_pairs(f'"{child}"이란 {desc}를 말한다', ke.last_noun)
    return [p["parent"] for p in pairs]


def _last_nouns_ranked(desc: str) -> list[str]:
    """정의문에서 상위어 후보 랭킹 — head 명사 우선(한국어 정의문은 '…는 X이다/X'로
    끝나 head=상위어). 뒤에서부터 명사를 후보로."""
    toks = _get_ke().kiwi.tokenize(desc)
    nouns = [t.form for t in toks if t.tag in ("NNG", "NNP")]
    # head-final: 마지막 명사가 최상위 후보. 복합명사 결합도 시도.
    ranked = []
    seen = set()
    for n in reversed(nouns):
        if n not in seen and len(n) >= 2:
            seen.add(n); ranked.append(n)
    return ranked


def t1_improved(child: str, desc: str) -> list[str]:
    """개선: 정의문 head 명사 랭킹(따옴표 불요). 한국어 정의문 head-final 활용."""
    return _last_nouns_ranked(desc)


def eval_t1(method):
    """SemEval-2018 T9 프로토콜: P@1, MRR, MAP over classes with gold+desc."""
    items = [(c, DESCS[c]) for c in GOLD_MAP if c in DESCS]
    p1 = mrr = mapv = 0.0
    n = 0
    for c, desc in items:
        gold = GOLD_MAP[c]
        preds = method(c, desc)[:15]  # top-15
        if not preds:
            n += 1; continue
        # P@1
        p1 += 1.0 if preds[0] in gold else 0.0
        # MRR
        rr = 0.0
        for i, pr in enumerate(preds):
            if pr in gold:
                rr = 1.0 / (i + 1); break
        mrr += rr
        # AP
        hits = 0; ap = 0.0
        for i, pr in enumerate(preds):
            if pr in gold:
                hits += 1; ap += hits / (i + 1)
        mapv += ap / len(gold) if gold else 0.0
        n += 1
    return {"P@1": p1/n, "MRR": mrr/n, "MAP": mapv/n, "n": n}


# ══ T2: 클래스집합 → 계층 유도 (pairwise) ══
def t2_baseline(classes: set) -> set:
    """현 ontokit: 접미공유만."""
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    return {(h["child"], h["parent"]) for h in induce_suffix_hierarchy(classes)}


def t2_improved(classes: set) -> set:
    """개선: 접미공유 ∪ 정의문 Hearst(head 명사). 정의문 있는 클래스는 head 상위어 추가."""
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    out = {(h["child"], h["parent"]) for h in induce_suffix_hierarchy(classes)}
    for c in classes:
        if c in DESCS:
            heads = _last_nouns_ranked(DESCS[c])
            # top-1 head 명사가 클래스집합에 있으면 계층 후보
            for h in heads[:1]:
                if h in classes and h != c:
                    out.add((c, h))
    return out


def eval_t2(method):
    pred = method(ALL_CLASSES)
    gold = set(PAIRS)
    tp = len(pred & gold)
    p = tp/len(pred) if pred else 0.0
    r = tp/len(gold) if gold else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    # 이질계층 recall
    hset = set(HETERO)
    htp = len(pred & hset)
    hr = htp/len(hset) if hset else 0.0
    return {"P": p, "R": r, "F1": f, "hetero_R": hr, "pred": len(pred), "gold": len(gold)}


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    print(f"=== 외부 gold: Wikidata 한국어 P279 ===")
    print(f"클래스 {len(ALL_CLASSES)}, gold쌍 {len(PAIRS)}, 이질 {len(HETERO)}"
          f"({len(HETERO)/len(PAIRS)*100:.0f}%), 정의문 {len(DESCS)}\n")

    t1m = t1_baseline if method == "baseline" else t1_improved
    t2m = t2_baseline if method == "baseline" else t2_improved

    print(f"[{method}] T1 정의문→상위어 발견 (SemEval T9):")
    r1 = eval_t1(t1m)
    print(f"  P@1={r1['P@1']:.3f}  MRR={r1['MRR']:.3f}  MAP={r1['MAP']:.3f}  (n={r1['n']})")

    print(f"[{method}] T2 클래스집합→계층 유도 (pairwise):")
    r2 = eval_t2(t2m)
    print(f"  P={r2['P']:.3f}  R={r2['R']:.3f}  F1={r2['F1']:.3f}  "
          f"이질R={r2['hetero_R']:.3f}  (pred={r2['pred']} gold={r2['gold']})")

    # 종합 점수(100점): T1 MAP 40 + T2 F1 40 + 이질R 20 (심판 참고용, 심판이 재판정)
    score = r1["MAP"]*40 + r2["F1"]*40 + r2["hetero_R"]*20
    print(f"\n[{method}] 참고 점수: {score:.1f}/100 "
          f"(T1 MAP×40 + T2 F1×40 + 이질R×20)")
