"""Cypher(LPG) 백엔드 — 같은 그래프·같은 질의계획을 Neo4j 5 에서. extras[neo4j]=neo4j 드라이버.

투영 규칙(RDF → LPG)
  개체/클래스/문서 = 노드 (:R {g, uri, key, nkeys[], names[]}) — g 는 그래프 이름(격리), key = g|uri (유일)
  rdf:type → [:TYPE], rdfs:subClassOf → [:SUBCLASS_OF], oh:describes → [:DESCRIBES]
  관계 → [:REL {key}]  (관계 키는 속성 — 라벨 폭발 방지)

컴파일 결과는 Neo4j 5 문법(EXISTS { } 서브쿼리, 가변 길이 경로)을 쓴다. Memgraph 등 다른 Cypher 엔진은 확인하지 않았다.
측정 하네스(harness/lpg.py)에서 옮겨 왔다 — 인메모리 SPARQL 과 문항별 동일함은 harness/docs/S01 §8.

    st = Neo4jStore("bolt://localhost:7687", "neo4j", "<비밀번호>")
    await st.load(g, "my-graph")                      # g = ontokit.graph.project(...)[0]
    nodes = (await run_plan(st, plan, "my-graph"))["nodes"]
"""
from __future__ import annotations

from rdflib import URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from ..graph import DESCRIBES, NKEY, RELATIONS, norm, rel_uri

_DESC = URIRef(DESCRIBES)


def project(g, gname):
    """rdflib.Graph → (노드 목록, 간선 목록[(s, 종류, o, 관계키)])."""
    nodes, edges = {}, []

    def n(u):
        return nodes.setdefault(str(u), {"uri": str(u), "g": gname, "nkeys": [], "names": []})

    rel_by_uri = {rel_uri(k): k for k in RELATIONS}
    for s, p, o in g:
        sp = str(p)
        if p == RDFS.label or p == SKOS.altLabel:
            n(s)["names"].append(str(o))
        elif p == NKEY:
            n(s)["nkeys"].append(str(o))
        elif p == RDF.type:
            edges.append((str(s), "TYPE", str(o), None))
            n(s), n(o)
        elif p == RDFS.subClassOf:
            edges.append((str(s), "SUBCLASS_OF", str(o), None))
            n(s), n(o)
        elif p == _DESC:
            edges.append((str(s), "DESCRIBES", str(o), None))
            n(s), n(o)
        elif sp in rel_by_uri:
            edges.append((str(s), "REL", str(o), rel_by_uri[sp]))
            n(s), n(o)
    return list(nodes.values()), edges


class CypherCompiler:
    def __init__(self, gname):
        self.n, self.params = 0, {"g": gname}

    def fresh(self):
        self.n += 1
        return f"k{self.n}"

    def term(self, t, parts):
        if isinstance(t, str):
            return t
        if "var" in t:
            return t["var"]
        v = self.fresh()
        self.params[v] = norm(t["const"])
        parts.append(f"MATCH ({v}:R {{g:$g}}) WHERE ${v} IN {v}.nkeys")
        return v

    def compile(self, plan):
        parts = []
        for c in plan["where"]:
            if c["t"] == "isa":
                cv = self.fresh()
                self.params[cv] = norm(c["class"])
                parts.append(f"MATCH ({c['v']}:R {{g:$g}})-[:TYPE]->(:R)-[:SUBCLASS_OF*0..]->({cv}:R) "
                             f"WHERE ${cv} IN {cv}.nkeys")
            elif c["t"] == "rel":
                if c["p"] not in RELATIONS:
                    raise ValueError(f"알 수 없는 관계: {c['p']}")
                s, o = self.term(c["s"], parts), self.term(c["o"], parts)
                p, inv = c["p"], RELATIONS[c["p"]].inverse
                pk, ik = self.fresh(), self.fresh()
                self.params[pk] = p
                if inv:
                    self.params[ik] = inv
                    parts.append(f"MATCH ({s}:R {{g:$g}}), ({o}:R {{g:$g}}) WHERE "
                                 f"EXISTS {{ ({s})-[e:REL]->({o}) WHERE e.key = ${pk} }} OR "
                                 f"EXISTS {{ ({o})-[e:REL]->({s}) WHERE e.key = ${ik} }}")
                else:
                    parts.append(f"MATCH ({s}:R {{g:$g}})-[{pk}e:REL]->({o}:R {{g:$g}}) WHERE {pk}e.key = ${pk}")
            else:
                raise ValueError(f"알 수 없는 조건: {c}")
        used = {c["v"] for c in plan["where"] if c["t"] == "isa"}
        for c in plan["where"]:
            for t in (c.get("s"), c.get("o")):
                if isinstance(t, str):
                    used.add(t)
                elif isinstance(t, dict) and "var" in t:
                    used.add(t["var"])
        if "x" in used:
            parts.append("MATCH (:R {g:$g})-[:DESCRIBES]->(x)")
        r = plan["return"]
        return "\n".join(parts) + f"\nRETURN DISTINCT {r}.uri AS uri, {r}.names AS names", self.params


class Neo4jStore:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password=None, *, database="neo4j"):
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as e:
            raise ImportError("Neo4jStore 는 neo4j 드라이버가 필요합니다: pip install 'xgen-ontokit[neo4j]'") from e
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    async def run(self, cypher, **params):
        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher, **params)
            return [record async for record in result]

    async def load(self, g, gname):
        """그래프를 gname 으로 싣는다(같은 이름의 기존 노드는 지운다). (노드 수, 간선 수) — 저장소와 어긋나면 예외."""
        await self.run("CREATE CONSTRAINT ontokit_r_key IF NOT EXISTS FOR (n:R) REQUIRE n.key IS UNIQUE")
        await self.run("CREATE INDEX ontokit_r_g IF NOT EXISTS FOR (n:R) ON (n.g)")
        while (await self.run("MATCH (n:R {g:$g}) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS c",
                              g=gname))[0]["c"]:
            pass
        nodes, edges = project(g, gname)
        for i in range(0, len(nodes), 5000):
            await self.run("UNWIND $rows AS r CREATE (n:R) SET n = r, n.key = r.g + '|' + r.uri", rows=nodes[i:i + 5000])
        for t in ("TYPE", "SUBCLASS_OF", "DESCRIBES", "REL"):
            rows = [{"s": gname + "|" + s, "o": gname + "|" + o, "k": k} for s, tt, o, k in edges if tt == t]
            for i in range(0, len(rows), 5000):
                await self.run(f"UNWIND $rows AS r MATCH (a:R {{key:r.s}}), (b:R {{key:r.o}}) "
                               f"CREATE (a)-[e:{t}]->(b) SET e.key = r.k", rows=rows[i:i + 5000])
        nn = (await self.run("MATCH (n:R {g:$g}) RETURN count(n) AS c", g=gname))[0]["c"]
        ne = (await self.run("MATCH (n:R {g:$g})-[e]->() RETURN count(e) AS c", g=gname))[0]["c"]
        if (nn, ne) != (len(nodes), len(edges)):   # 적재 손실을 조용히 넘기지 않는다
            raise RuntimeError(f"적재 불일치: 로컬 노드 {len(nodes)}·간선 {len(edges)} vs 저장소 {nn}·{ne}")
        return nn, ne

    async def close(self):
        await self.driver.close()


async def run_plan(st, plan, gname):
    q, params = CypherCompiler(gname).compile(plan)
    rows = await st.run(q, **params)
    return {"nodes": [{"uri": r["uri"], "names": r["names"]} for r in rows], "cypher": q}
