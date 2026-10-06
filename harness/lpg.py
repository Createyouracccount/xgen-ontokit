"""LPG(Neo4j) 경로 — 같은 RDF 그래프를 LPG 로 투영하고, 같은 IR 을 Cypher 로 컴파일한다.

투영 규칙(RDF → LPG)
  개체/클래스/문서 = 노드(:R {uri, g, nkeys[], names[]}) — g 는 팔 이름(그래프 격리)
  rdf:type → [:TYPE], rdfs:subClassOf → [:SUBCLASS_OF], oh:describes → [:DESCRIBES]
  관계 → [:REL {key}]  (관계 키는 속성 — 라벨 폭발 방지)

graphstore Neo4jBackend 의 드라이버·스키마(uri 유니크 제약)를 그대로 쓴다.
"""
from rdflib import URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from harness import schema as S
from harness.graph import NKEY

DESC = URIRef(S.DESCRIBES)


def project(g, arm):
    nodes, edges = {}, []

    def n(u):
        return nodes.setdefault(str(u), {"uri": str(u), "g": arm, "nkeys": [], "names": []})

    rel_by_uri = {S.rel_uri(k): k for k in S.RELATIONS}
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
        elif p == DESC:
            edges.append((str(s), "DESCRIBES", str(o), None))
            n(s), n(o)
        elif sp in rel_by_uri:
            edges.append((str(s), "REL", str(o), rel_by_uri[sp]))
            n(s), n(o)
    return list(nodes.values()), edges


async def load(g, arm, backend):
    await backend._ensure_schema_once()
    await backend._run("MATCH (n:R {g:$g}) DETACH DELETE n", g=arm)
    nodes, edges = project(g, arm)
    for i in range(0, len(nodes), 5000):
        await backend._run("UNWIND $rows AS r CREATE (n:R:Resource) SET n = r, n.uri = r.g + '|' + r.uri",
                           rows=nodes[i:i + 5000])
    await backend._run("CREATE INDEX oh_nkeys IF NOT EXISTS FOR (n:R) ON (n.g)")
    for t in ("TYPE", "SUBCLASS_OF", "DESCRIBES", "REL"):
        rows = [{"s": arm + "|" + s, "o": arm + "|" + o, "k": k} for s, tt, o, k in edges if tt == t]
        for i in range(0, len(rows), 5000):
            await backend._run(
                f"UNWIND $rows AS r MATCH (a:Resource {{uri:r.s}}), (b:Resource {{uri:r.o}}) "
                f"CREATE (a)-[e:{t}]->(b) SET e.key = r.k", rows=rows[i:i + 5000])
    cnt = await backend._run("MATCH (n:R {g:$g}) RETURN count(n) AS c", g=arm)
    return len(nodes), len(edges), cnt


class CypherCompiler:
    def __init__(self, arm):
        self.arm, self.n, self.params = arm, 0, {"g": arm}

    def fresh(self):
        self.n += 1
        return f"k{self.n}"

    def term(self, t, parts):
        if isinstance(t, str):
            return t
        if "var" in t:
            return t["var"]
        v = self.fresh()
        self.params[v] = S.norm(t["const"])
        parts.append(f"MATCH ({v}:R {{g:$g}}) WHERE ${v} IN {v}.nkeys")
        return v

    def compile(self, plan):
        parts = []
        for c in plan["where"]:
            if c["t"] == "isa":
                cv = self.fresh()
                self.params[cv] = S.norm(c["class"])
                parts.append(f"MATCH ({c['v']}:R {{g:$g}})-[:TYPE]->(:R)-[:SUBCLASS_OF*0..]->({cv}:R) "
                             f"WHERE ${cv} IN {cv}.nkeys")
            else:
                s, o = self.term(c["s"], parts), self.term(c["o"], parts)
                p, inv = c["p"], S.RELATIONS[c["p"]][4]
                pk, ik = self.fresh(), self.fresh()
                self.params[pk] = p
                if inv:
                    self.params[ik] = inv
                    parts.append(f"MATCH ({s}:R {{g:$g}}), ({o}:R {{g:$g}}) WHERE "
                                 f"EXISTS {{ ({s})-[e:REL]->({o}) WHERE e.key = ${pk} }} OR "
                                 f"EXISTS {{ ({o})-[e:REL]->({s}) WHERE e.key = ${ik} }}")
                else:
                    parts.append(f"MATCH ({s}:R {{g:$g}})-[{pk}e:REL]->({o}:R {{g:$g}}) WHERE {pk}e.key = ${pk}")
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


async def run_plan(backend, plan, arm):
    q, params = CypherCompiler(arm).compile(plan)
    rows = await backend._run(q, **params)
    return {"nodes": [{"uri": r["uri"], "names": r["names"]} for r in rows], "cypher": q}
