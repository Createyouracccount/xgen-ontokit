"""손실 분해 — 정답 항목이 그래프 어느 단계에서 끊기는가.

각 정답 주어(x)에 대해: ① 표제 노드 존재(describes) ② 질문의 타입/관계 조건 충족
상수(const)에 대해: ③ 정규화 키로 그래프 노드가 존재하는가
  python -m harness.diagnose <run> <arm> harness/bench/wiki2_bench.json
"""
import asyncio
import collections
import json
import sys

from harness import schema as S
from harness.graph import graph_name, store
from harness.query import PFX


async def main():
    run, arm, bench = sys.argv[1:4]
    g = graph_name(run, arm)
    qs = json.load(open(bench))
    st = store()

    async def ask(q):
        r = await st.sparql_query(PFX + q)
        return r["results"]["bindings"]

    desc = await ask(f"SELECT ?d ?x WHERE {{ GRAPH <{g}> {{ ?d oh:describes ?x }} }}")
    anchored_docs = {b["d"]["value"].rsplit("/", 1)[1] for b in desc}
    keys = {b["k"]["value"] for b in await ask(f"SELECT DISTINCT ?k WHERE {{ GRAPH <{g}> {{ ?n oh:nkey ?k }} }}")}
    ckeys = {b["k"]["value"] for b in await ask(
        f"SELECT DISTINCT ?k WHERE {{ GRAPH <{g}> {{ ?i a ?c . ?c oh:nkey ?k }} }}")}
    out = collections.defaultdict(collections.Counter)
    for q in qs:
        f = q["form"]
        for c in q["plan"]["where"]:
            if c["t"] == "isa":
                out[f]["class_exists" if S.norm(c["class"]) in ckeys else "class_missing"] += 1
            for t in (c.get("s"), c.get("o")):
                if isinstance(t, dict) and "const" in t:
                    out[f]["const_exists" if S.norm(t["const"]) in keys else "const_missing"] += 1
        if q["plan"]["return"] == "x":
            for it in q["gold"]["items"]:
                out[f]["gold_anchored" if set(it["docs"]) & anchored_docs else "gold_unanchored"] += 1
    await st.close()
    print(f"표제 연결 문서 {len(anchored_docs)}")
    for f, c in out.items():
        print(f, dict(c))


if __name__ == "__main__":
    asyncio.run(main())
