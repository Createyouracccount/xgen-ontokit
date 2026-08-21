"""재현율 측정 — ② 채점.

⚠️ **이 파일은 심판 스폰 전에 커밋된다.** git 해시를 사전 공시에 박는다.
   R6 에서 채점기 mtime 이 판정보다 9분 늦어 "봉인" 주장이 거짓으로 적발됐다
   (판례 3 · K-1 · 30). mtime 은 증거가 되지 못한다 — git 해시만이 된다.

계산하는 것(실행 전 고정):

- **gold**: 3인 중 **2인 이상**이 열거한 개체. 1인만 열거한 것은 gold 아님(별도 집계).
- **재현율(엄격)**: 정확일치만 분자. `|gold ∩ 시스템(정확)| / |gold|`
- **재현율(관대)**: 부분일치도 분자. 경계가 틀려도 "찾긴 찾았다"로 본다.
- **경계결함률**: 부분일치 / (정확일치 + 부분일치). 결함 축의 57.5%가 여기다.
- **과잉추출**: 시스템에는 있는데 어느 심판도 안 적은 것. 정밀도 축의 교차검증.

판례 19 — 모든 비율에 Wilson 95% CI. `0%`·`100%` 단독 표기 금지.
판례 21 — 심판 간 쌍별 일치(Jaccard)를 공시한다. 열거 과제는 범주형이 아니라
          집합이므로 κ 대신 Jaccard 를 쓴다. 사전에 고정한다.
"""
from __future__ import annotations

import json
import math
import os
import re
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGES = ("A", "B", "C")
CLASSES = ("인물", "기관", "지역")

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    """표층형 정규화 — 공백 축약 + 양끝 구두점 제거. 그 외는 건드리지 않는다.

    ⚠️ 약어 확장(한전→한국전력공사)은 **하지 않는다.** 심판에게 원문 그대로
       적으라 지시했으므로, 표층형이 다르면 그것은 실제 불일치다.
    """
    s = _WS.sub(" ", (s or "").strip())
    return s.strip(" .,·'\"“”‘’()[]")


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def load_judges(d: str) -> dict:
    """{judge: {chunk_id: {(label, class), ...}}}"""
    out = {}
    for j in JUDGES:
        p = os.path.join(d, f"recall_gold_{j}.json")
        raw = json.load(open(p, encoding="utf-8"))
        per = {}
        for row in raw["chunks"]:
            per[row["chunk_id"]] = {
                (norm(e["label"]), e["class"])
                for e in row.get("entities", [])
                if norm(e.get("label", "")) and e.get("class") in CLASSES
            }
        out[j] = per
    return out


def system_labels(path: str) -> dict:
    """{chunk_id: {label, ...}} — 시스템이 그 청크에서 뽑은 라벨.

    ⚠️ 클래스는 대조하지 않는다. 시스템 클래스는 세분류('정치인')라 심판의 3분류와
       단위가 다르다(판례 16). 클래스 정확도는 정밀도 축이 이미 재고 있으므로,
       재현율 축은 **라벨 문자열만** 본다. 이 결정을 실행 전에 고정한다.
    """
    raw = json.load(open(path, encoding="utf-8"))
    ents = [e for v in raw.values() for e in v] if isinstance(raw, dict) else raw
    out = {}
    for e in ents:
        lb = norm(e.get("entity", ""))
        if not lb:
            continue
        for cid in e.get("source_chunks") or []:
            out.setdefault(cid, set()).add(lb)
    return out


def match(gold_label: str, sysset: set):
    """정확일치 / 부분일치 / 미탐 판정. 부분일치는 짝을 함께 돌려준다."""
    if gold_label in sysset:
        return "정확", gold_label
    for s in sysset:
        if len(s) >= 2 and (s in gold_label or gold_label in s):
            return "부분", s
    return "미탐", None


def main(work: str, roster: str):
    pack = json.load(open(os.path.join(work, "recall_packet.json"), encoding="utf-8"))
    cids = [c["chunk_id"] for c in pack["chunks"]]
    V = load_judges(work)
    SYS = system_labels(roster)

    gold, solo, per_chunk = [], [], []
    for cid in cids:
        votes = {}
        for j in JUDGES:
            for item in V[j].get(cid, set()):
                votes.setdefault(item, set()).add(j)
        g = {it for it, js in votes.items() if len(js) >= 2}
        s1 = {it for it, js in votes.items() if len(js) == 1}
        sysset = SYS.get(cid, set())

        rows = []
        for lb, cl in sorted(g):
            kind, paired = match(lb, sysset)
            rows.append({"label": lb, "class": cl, "판정": kind, "시스템라벨": paired})
        gold.extend((cid, r) for r in rows)
        solo.extend((cid, lb, cl, sorted(votes[(lb, cl)])) for lb, cl in sorted(s1))

        golds = {lb for lb, _ in g}
        extra = sorted(x for x in sysset
                       if x not in golds and not any(x in y or y in x for y in golds))
        per_chunk.append({"chunk_id": cid, "gold": len(g), "시스템": len(sysset),
                          "과잉": len(extra), "과잉목록": extra[:8]})

    n = len(gold)
    exact = sum(1 for _, r in gold if r["판정"] == "정확")
    part = sum(1 for _, r in gold if r["판정"] == "부분")
    miss = sum(1 for _, r in gold if r["판정"] == "미탐")

    r_str, lo_s, hi_s = wilson(exact, n)
    r_len, lo_l, hi_l = wilson(exact + part, n)
    b_p, b_lo, b_hi = wilson(part, exact + part) if (exact + part) else (0, 0, 0)

    bycls = {}
    for c in CLASSES:
        sub = [r for _, r in gold if r["class"] == c]
        e = sum(1 for r in sub if r["판정"] == "정확")
        p_ = sum(1 for r in sub if r["판정"] == "부분")
        v, l, h = wilson(e, len(sub))
        bycls[c] = {"gold": len(sub), "정확": e, "부분": p_,
                    "미탐": len(sub) - e - p_, "재현율(엄격)": round(v, 4),
                    "CI": [round(l, 4), round(h, 4)]}

    pair = {}
    for a, b in combinations(JUDGES, 2):
        sa = {(c, x) for c in cids for x in V[a].get(c, set())}
        sb = {(c, x) for c in cids for x in V[b].get(c, set())}
        pair[f"{a}-{b}"] = {"jaccard": round(jaccard(sa, sb), 4),
                            "A만": len(sa - sb), "B만": len(sb - sa)}

    res = {
        "표본": {"seed": pack["seed"], "청크": len(cids), "문자": pack["총문자수"]},
        "gold": {"2인이상": n, "1인만(gold아님)": len(solo)},
        "재현율(엄격·정확일치만)": {"값": round(r_str, 4), "k": exact, "n": n,
                                    "CI95": [round(lo_s, 4), round(hi_s, 4)]},
        "재현율(관대·부분일치포함)": {"값": round(r_len, 4), "k": exact + part, "n": n,
                                      "CI95": [round(lo_l, 4), round(hi_l, 4)]},
        "경계결함률": {"값": round(b_p, 4), "k": part, "n": exact + part,
                       "CI95": [round(b_lo, 4), round(b_hi, 4)],
                       "주": "찾긴 찾았으나 경계가 틀린 비율"},
        "미탐": {"n": miss, "비율": round(miss / n, 4) if n else None},
        "클래스별": bycls,
        "판례21_심판간일치(Jaccard)": pair,
        "과잉추출": {"총": sum(c["과잉"] for c in per_chunk),
                     "청크당평균": round(sum(c["과잉"] for c in per_chunk) / len(cids), 2)},
    }
    detail = {
        "미탐목록": [{"chunk": c, **r} for c, r in gold if r["판정"] == "미탐"],
        "부분일치목록": [{"chunk": c, **r} for c, r in gold if r["판정"] == "부분"],
        "1인만열거": [{"chunk": c, "label": l, "class": k, "심판": j} for c, l, k, j in solo],
        "청크별": per_chunk,
    }
    with open(os.path.join(work, "recall_result.json"), "w", encoding="utf-8") as f:
        json.dump({"요약": res, "상세": detail}, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
