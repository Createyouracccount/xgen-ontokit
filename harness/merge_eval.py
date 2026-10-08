"""2바퀴 처치 MERGE 오프라인 채점 — 판독 답(저장분) ∪ 그래프 결과(저장된 계획으로 재질의). LLM 호출 0.

  python -m harness.merge_eval <run> <bench.json> <results.json> --pairs HYB:ontokit_r5,HYB:placebo2 [--out merged.json]

각 HYB:<graph> 팔에 대해 MRG:<graph> 팔을 만든다(같은 결과 파일의 VLLM 과 쌍 비교).
- 목록: 판독 답 ∪ 그래프 결과 노드
- 개수(A): 그래프 결과가 있으면 그 노드 수, 없으면 판독 개수
"""
import argparse
import asyncio
import json

from harness.run import score_nodes


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("bench")
    ap.add_argument("results")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    from harness.graph import graph_name, store
    from harness.query import run_plan
    qs = {q["id"]: q for q in json.load(open(a.bench))}
    res = json.load(open(a.results))
    plans = res.get("plans", {})
    st = store()
    for hyb in a.pairs.split(","):
        garm = hyb.split(":", 1)[1]
        mrg = "MRG:" + garm
        for row in res["rows"]:
            if hyb not in row["arms"] or row["id"] not in qs:
                continue
            q, p = qs[row["id"]], plans.get(row["id"])
            nodes = []
            if p:
                try:
                    nodes = (await run_plan(st, p, graph_name(a.run, garm)))["nodes"]
                except ValueError:
                    nodes = []
            reader = [{"names": [n]} for n in row["arms"][hyb].get("answers", [])]
            sc = score_nodes(q, reader + nodes)
            if q["form"] == "A":
                cnt = len(nodes) if nodes else row["arms"][hyb].get("count_pred")
                sc["count_pred"] = cnt
                sc["score"] = 1.0 if cnt == q["gold"]["count"] else 0.0
            sc["graph_items"] = len(nodes)
            row["arms"][mrg] = sc
        if mrg not in res["arms"]:
            res["arms"].append(mrg)
    await st.close()
    json.dump(res, open(a.out or a.results, "w"), ensure_ascii=False)
    from harness.stats import summarize
    print(summarize(res, base="VLLM"))


if __name__ == "__main__":
    asyncio.run(main())
