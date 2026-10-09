"""Oxigraph 저장소 — 서버·JVM 없는 내장 SPARQL 저장소(RocksDB). extras[oxigraph]=pyoxigraph.

Fuseki 대체: ontokit.query 가 컴파일한 SPARQL(명명 그래프 GRAPH <..> 포함)을 그대로 실행한다.
path 를 주면 그 디렉터리에 저장되고(프로세스를 넘어 유지), 없으면 메모리에만 둔다.
여러 프로세스가 같은 디렉터리를 동시에 읽으려면 read_only=True 로 연다(쓰기 인스턴스는 하나).

    from ontokit.backends.oxigraph import OxigraphStore
    st = OxigraphStore("graphs/")
    st.load(g, "urn:my-graph")                       # g = ontokit.graph.project(...)[0]
    nodes = (await ontokit.query.run_plan(st, plan, "urn:my-graph"))["nodes"]
"""
from __future__ import annotations

import json

try:
    import pyoxigraph as ox
except ImportError as e:
    raise ImportError("OxigraphStore 는 pyoxigraph 가 필요합니다: pip install 'xgen-ontokit[oxigraph]'") from e


class OxigraphStore:
    def __init__(self, path=None, *, read_only=False):
        self.store = ox.Store.read_only(str(path)) if read_only else ox.Store(None if path is None else str(path))

    def load(self, g, graph):
        """rdflib.Graph 를 명명 그래프 graph 로 싣는다(같은 이름의 기존 내용은 지운다). 실린 트리플 수를 돌려준다."""
        gn = ox.NamedNode(graph)
        self.store.remove_graph(gn)
        self.store.load(g.serialize(format="nt").encode(), format=ox.RdfFormat.N_TRIPLES, to_graph=gn)
        self.store.flush()
        n = self.count(graph)
        if n != len(g):   # 적재 손실을 조용히 넘기지 않는다
            raise RuntimeError(f"트리플 수 불일치: 로컬 {len(g)} vs 저장소 {n}")
        return n

    def count(self, graph):
        return sum(1 for _ in self.store.quads_for_pattern(None, None, None, ox.NamedNode(graph)))

    def query(self, q):
        """SPARQL SELECT → SPARQL JSON 결과(dict) — Fuseki 응답과 같은 형태."""
        return json.loads(self.store.query(q).serialize(format=ox.QueryResultsFormat.JSON))

    async def sparql_query(self, q):   # ontokit.query.run_plan 이 쓰는 비동기 인터페이스
        return self.query(q)

    async def close(self):
        pass
