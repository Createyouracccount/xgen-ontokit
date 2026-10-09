"""동등성 검사(S01) — 측정 당시의 그래프·결과와 지금 코드의 출력을 문항·트리플 단위로 비교한다.

  python -m harness.equiv rows <저장 결과.json> <재계산 결과.json> <팔,...>
      문항 × 팔의 채점 기록(점수·tp·fp·개수·그래프 항목 수)이 모두 같은지. 집계가 같아도 문항이 다를 수 있다.
  python -m harness.equiv graph <run> <arm> <ontokit_raw.json> <docs.jsonl>
      저장소(Fuseki)에 적재된 그래프 vs ontokit.graph.project — 트리플 집합 비교(개수만이 아니라 내용).
"""
import asyncio
import json
import sys


def rows(old, new, arms):
    a = {r["id"]: r for r in json.load(open(old))["rows"]}
    b = {r["id"]: r for r in json.load(open(new))["rows"]}
    ok = True
    for arm in arms:
        ids = [i for i in a if arm in a[i]["arms"]]
        diff = [i for i in ids if i not in b or a[i]["arms"][arm] != b[i]["arms"].get(arm)]
        used = sum(1 for i in ids if a[i]["arms"][arm].get("graph_items", 0) > 0)
        print(f"{arm}: 문항 {len(ids)} · 동일 {len(ids) - len(diff)} · 다름 {len(diff)} "
              f"(그래프 결과가 들어간 문항 {used}) {diff[:5]}")
        ok &= not diff and bool(ids)
    return ok


def _term(t):
    return ("uri", str(t)) if t.__class__.__name__ == "URIRef" else ("literal", str(t))


async def graph(run, arm, raw, docs):
    from harness.graph import graph_name, store
    from ontokit.graph import project
    g, _ = project(json.load(open(raw)), [json.loads(line) for line in open(docs)])
    lib = {tuple(_term(x) for x in t) for t in g}
    st = store()
    res = await st.sparql_query(f"SELECT ?s ?p ?o WHERE {{ GRAPH <{graph_name(run, arm)}> {{ ?s ?p ?o }} }}")
    await st.close()
    if "results" not in res:
        raise RuntimeError(f"조회 실패: {str(res)[:200]}")
    kind = {"uri": "uri", "literal": "literal", "typed-literal": "literal"}
    stored = {tuple((kind[b[v]["type"]], b[v]["value"]) for v in "spo") for b in res["results"]["bindings"]}
    print(f"{run}/{arm}: 라이브러리 {len(lib)} · 저장소 {len(stored)} · 라이브러리에만 {len(lib - stored)} · "
          f"저장소에만 {len(stored - lib)}")
    return lib == stored


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    ok = rows(args[0], args[1], args[2].split(",")) if cmd == "rows" else asyncio.run(graph(*args))
    print("동일" if ok else "다름")
    sys.exit(0 if ok else 1)
