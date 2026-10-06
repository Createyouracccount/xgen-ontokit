"""타입 사실 재현율 — 정답 타입 사실(s, C) 중 그래프에서 s 의 타입이 C 이거나 C 로 올라가는 비율.

정답 쪽 C 는 위키데이터 클래스 라벨(및 그 P279 상위). 그래프 쪽은 ontokit 타입 + 계층(subClassOf 폐포).
  python -m harness.typerecall <ontokit_raw.json> [dev|eval|all]
"""
import collections
import json
import re
import sys

from harness import schema as S
from harness.relrecall import split_of


def type_recall(raw, facts, which="all"):
    sup = collections.defaultdict(set)
    for h in raw["concepts"].get("class_hierarchy", []):
        sup[S.norm(h["child"])].add(S.norm(h["parent"]))

    def closure(c):
        seen, st = {c}, [c]
        while st:
            for p in sup.get(st.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    st.append(p)
        return seen
    # 문서별 표제 개체의 타입: 표제와 같은 라벨의 NER 개체 클래스 + per:title 목적어
    tps = collections.defaultdict(set)
    for d, ents in raw["ner_entities"].items():
        for e in ents:
            tps[(d, S.norm(e["entity"]))].add(S.norm(e["class"]))
    for r in raw["relations"]:
        if (r.get("relation_label") or r["predicate"]) == "per:title":
            for c in r.get("source_chunks", []):
                tps[(c.split("#")[0], S.norm(r["subject"]))].add(S.norm(r["object"]))
    E, C = facts["entities"], facts["classes"]
    st = collections.Counter()
    for s, ts in facts["types"].items():
        for d in E[s]["docs"]:
            if which != "all" and split_of(d) != which:
                continue
            got = set()
            for n in E[s]["names"]:
                for t in tps.get((d, S.norm(n)), ()):
                    got |= closure(t)
            for c in ts:
                st["facts"] += 1
                st["hit"] += S.norm((C.get(c) or {}).get("label") or "") in got
    return {"facts": st["facts"], "hit": st["hit"], "recall": st["hit"] / st["facts"] if st["facts"] else 0}


if __name__ == "__main__":
    raw = json.load(open(sys.argv[1]))
    facts = json.load(open("harness/bench/wiki2_oracle_facts.json"))
    print(type_recall(raw, facts, sys.argv[2] if len(sys.argv) > 2 else "all"))
