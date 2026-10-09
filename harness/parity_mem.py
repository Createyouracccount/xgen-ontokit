"""인메모리(rdflib) ↔ 하네스 저장소(harness.graph.store) 동치 검증 — 같은 그래프·같은 계획의 답 노드(URI·이름)가
문항마다 같은가. S01 §5 는 저장소가 Fuseki 이던 때, §6 은 Oxigraph 로 바꾼 뒤의 결과다.

  python -m harness.parity_mem <run> <arm> <ontokit_raw.json|facts.json> <docs.jsonl> <bench.json> [결과.json(plans)]

계획 출처: 벤치 정답 계획(q["plan"]) + 결과 파일의 LLM 저장 계획(있으면). 실패(예외)·빈 결과·동일을 따로 센다 —
양쪽이 똑같이 실패하거나 똑같이 비어도 "동일"로 보이므로.
"""
import asyncio
import json
import sys
import time

from harness.graph import graph_name, oracle_graph, store
from ontokit.graph import project
from ontokit.query import run, run_plan


def _answer(out):
    return {n["uri"]: sorted(n["names"]) for n in out["nodes"]}


async def _fuseki(st, p, gn):
    try:
        return _answer(await run_plan(st, p, gn))
    except ValueError as e:
        return ("ValueError", str(e))


def _mem(g, p):
    try:
        return _answer(run(g, p))
    except ValueError as e:
        return ("ValueError", str(e))


async def main():
    run_, arm, src, docs_path, bench = sys.argv[1:6]
    docs = [json.loads(line) for line in open(docs_path)]
    data = json.load(open(src))
    if arm in ("oracle", "placebo2"):
        g = oracle_graph(data, docs, placebo=(arm == "placebo2"))
    else:
        g = project(data, docs)[0]
    qs = json.load(open(bench))
    sources = {"bench_plan": {q["id"]: q.get("plan") for q in qs}}
    if len(sys.argv) > 6:
        sources["saved_plan"] = json.load(open(sys.argv[6]))["plans"]
    st = store()
    gn = graph_name(run_, arm)
    report = {"arm": arm, "triples": len(g)}
    for name, plans in sources.items():
        c = {"plans": 0, "identical": 0, "differ": 0, "both_error": 0, "nonempty": 0, "diff_ids": []}
        t_mem = t_max = 0.0
        for qid, p in plans.items():
            if not p:
                continue
            c["plans"] += 1
            a = await _fuseki(st, p, gn)
            t0 = time.time()
            b = _mem(g, p)
            dt = time.time() - t0
            t_mem += dt
            t_max = max(t_max, dt)
            if a == b:
                c["identical"] += 1
                c["both_error"] += isinstance(a, tuple)
                c["nonempty"] += isinstance(a, dict) and bool(a)
            else:
                c["differ"] += 1
                c["diff_ids"].append(qid)
        c["diff_ids"] = c["diff_ids"][:10]
        c["mem_sec_total"], c["mem_sec_max"] = round(t_mem, 1), round(t_max, 2)
        report[name] = c
    await st.close()
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
