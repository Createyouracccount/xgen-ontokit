"""백엔드 동치 검증 — 같은 그래프·같은 IR 을 RDF(Fuseki/SPARQL)와 LPG(Neo4j/Cypher)로 돌려
문항별 답 노드 집합이 같은지 본다. "어떤 DB 를 붙여도 같은 답" 의 측정.

  python -m harness.parity <run> <arm> <src.json> harness/bench/wiki2_bench.json
"""
import asyncio
import json
import os
import sys

from harness.graph import graph_name, ontokit_graph, oracle_graph, store
from harness.lpg import load as lpg_load, run_plan as cypher_plan
from harness.query import run_plan as sparql_plan
from harness.run import score_nodes


def neo4j():
    from xgen_graphstore import create_store
    pw = open(os.path.join(os.path.dirname(__file__), "data", ".neo4j_pass")).read().strip()
    return create_store({"backend": "neo4j", "uri": "bolt://localhost:7687",
                         "username": "neo4j", "password": pw})


async def main():
    run, arm, src, bench = sys.argv[1:5]
    data = json.load(open(src))
    docs = [json.loads(l) for l in open("harness/data/wiki2/docs.jsonl")]
    g = oracle_graph(data, docs) if arm in ("oracle", "placebo") else ontokit_graph(data, docs)[0]
    if arm == "placebo":
        g = oracle_graph(data, docs, placebo=True)
    fz = store()   # store() 가 graphstore 경로를 sys.path 에 올린다
    nj = neo4j()
    n_nodes, n_edges, _ = await lpg_load(g, arm, nj)
    qs = json.load(open(bench))
    same = 0
    diffs = []
    s_sum = c_sum = 0.0
    for q in qs:
        a = await sparql_plan(fz, q["plan"], graph_name(run, arm))
        b = await cypher_plan(nj, q["plan"], arm)
        ua = {n["uri"] for n in a["nodes"]}
        ub = {n["uri"].split("|", 1)[1] for n in b["nodes"]}
        s_sum += score_nodes(q, a["nodes"])["score"]
        c_sum += score_nodes(q, b["nodes"])["score"]
        if ua == ub:
            same += 1
        else:
            diffs.append({"id": q["id"], "sparql_only": len(ua - ub), "cypher_only": len(ub - ua)})
    await fz.close()
    await nj.close()
    out = {"arm": arm, "lpg_nodes": n_nodes, "lpg_edges": n_edges, "questions": len(qs),
           "identical_answer_sets": same, "score_sparql": round(s_sum / len(qs), 4),
           "score_cypher": round(c_sum / len(qs), 4), "diffs": diffs[:20]}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
