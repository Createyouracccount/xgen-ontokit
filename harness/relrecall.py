"""관계 사실 재현율 — 정답 사실(본문 근거) 중 ontokit 이 같은 문서에서 맞는 관계로 낸 비율.

  python -m harness.relrecall <ontokit_raw.json> [dev|eval|all]
dev = doc_id 해시 mod 6 == 0 (기제 개발 전용), eval = 나머지.
"""
import collections
import hashlib
import json
import sys

from harness import schema as S
from harness.graph import KLUE_MAP


def _in(d, which):
    """which: "all" | "dev" | "eval" | 문서 id 집합."""
    if isinstance(which, (set, frozenset)):
        return d in which
    return which == "all" or split_of(d) == which


def split_of(doc_id):
    return "dev" if int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 6 == 0 else "eval"


def recall(raw, facts, which="all"):
    E = facts["entities"]

    def keys(q):
        return {S.norm(n) for n in E.get(q, {}).get("names", []) if S.norm(n)}
    pair = collections.defaultdict(set)
    n_rel = 0
    for r in raw["relations"]:
        m = KLUE_MAP.get(r.get("relation_label") or r["predicate"])
        sk, ok = S.norm(r["subject"]), S.norm(r["object"])
        for c in r.get("source_chunks", []):
            d = c.split("#")[0]
            if not _in(d, which):
                continue
            n_rel += 1
            if m and m[0] == "rel":
                pair[(d, sk, ok)].add(m[1])
            if m and m[0] == "inv":
                pair[(d, ok, sk)].add(m[1])
    st = collections.defaultdict(collections.Counter)
    for f in facts["relations"]:
        s, p, o = f["s"], f["p"], f["o"]
        for d in E[s]["docs"]:
            if not _in(d, which):
                continue
            st[p]["facts"] += 1
            st[p]["hit"] += any(p in pair.get((d, a, b), ()) for a in keys(s) for b in keys(o))
    tot = sum(c["facts"] for c in st.values())
    hit = sum(c["hit"] for c in st.values())
    return {"facts": tot, "hit": hit, "recall": hit / tot if tot else 0, "relations_emitted": n_rel,
            "by_rel": {p: round(c["hit"] / c["facts"], 3) for p, c in st.items()}}


if __name__ == "__main__":
    raw = json.load(open(sys.argv[1]))
    facts = json.load(open("harness/bench/wiki2_oracle_facts.json"))
    print(json.dumps(recall(raw, facts, sys.argv[2] if len(sys.argv) > 2 else "all"), ensure_ascii=False, indent=1))
