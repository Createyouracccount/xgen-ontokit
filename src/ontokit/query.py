"""질의계획(IR) → SPARQL 컴파일·실행. 온톨로지 의미를 질의 시점에 쓴다. LLM 0회.

IR:
  {"op": "list"|"count", "return": "x"|"y",
   "where": [{"t": "isa", "v": "x", "class": "가수"},                       # 타입(하위 클래스 폐포)
             {"t": "rel", "s": "x"|{"const": "이름"}, "p": "per:...", "o": {"const": ..}|{"var": "y"}}]}

규칙
- 변수 x 는 항상 '이 문서 모음에 수록된 표제 개체'(oh:describes) 로 한정한다.
  ⚠️ 문서 하나 = 개체 하나인 자료(백과사전)에 맞춘 규칙이다. 제목이 개체 이름이 아닌 문서에서는
  x 로 묻는 목록이 비기 쉽다(docs/STANDALONE.md 한계).
- isa 는 rdf:type/rdfs:subClassOf* — 상위 클래스로 물어도 하위 클래스 인스턴스가 나온다.
- 상수는 정규화 키(oh:nkey)로 개체·클래스에 연결한다(라벨·별칭 모두).
- 역관계가 선언된 속성은 양방향을 UNION 으로 묻는다(per:parents ↔ per:children).

실행
- 인메모리(서버 없음): run(g, plan) — g 는 ontokit.graph.project 결과 또는 그 TTL 을 읽은 rdflib.Graph
- 원격 저장소: await run_plan(store, plan, graph_name) — store 는 async sparql_query(q) → SPARQL JSON 결과
  (예: xgen-graphstore 의 Fuseki 백엔드). 같은 컴파일러·같은 결과 해석을 쓴다.
"""
from __future__ import annotations

import json

from .graph import NS, RELATIONS, norm, rel_uri

PFX = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX oh: <{NS}>
"""


def _lit(s):
    return json.dumps(s, ensure_ascii=False)


class Compiler:
    def __init__(self):
        self.n = 0

    def fresh(self):
        self.n += 1
        return f"?k{self.n}"

    def term(self, t, lines):
        if isinstance(t, str):
            return f"?{t}"
        if "var" in t:
            return f"?{t['var']}"
        v = self.fresh()
        lines.append(f"{v} oh:nkey {_lit(norm(t['const']))} .")
        return v

    def compile(self, plan, graph=None):
        """graph=None 이면 기본 그래프(인메모리), 있으면 GRAPH <graph> 로 감싼다(원격 저장소의 명명 그래프)."""
        lines, vars_ = [], set()
        for c in plan["where"]:
            if c["t"] == "isa":
                cv = self.fresh()
                lines.append(f"?{c['v']} rdf:type/rdfs:subClassOf* {cv} . {cv} oh:nkey {_lit(norm(c['class']))} .")
                vars_.add(c["v"])
            elif c["t"] == "rel":
                s, o = self.term(c["s"], lines), self.term(c["o"], lines)
                for t in (c["s"], c["o"]):
                    if isinstance(t, str):
                        vars_.add(t)
                    elif "var" in t:
                        vars_.add(t["var"])
                p = c["p"]
                if p not in RELATIONS:
                    raise ValueError(f"알 수 없는 관계: {p}")
                inv = RELATIONS[p].inverse
                fwd = f"{s} <{rel_uri(p)}> {o} ."
                if inv and inv != p:
                    lines.append(f"{{ {fwd} }} UNION {{ {o} <{rel_uri(inv)}> {s} . }}")
                elif inv == p:  # 대칭
                    lines.append(f"{{ {fwd} }} UNION {{ {o} <{rel_uri(p)}> {s} . }}")
                else:
                    lines.append(fwd)
            else:
                raise ValueError(f"알 수 없는 조건: {c}")
        if "x" in vars_:
            lines.append("?xd oh:describes ?x .")
        r = plan["return"]
        body = "\n    ".join(lines)
        tail = f"    ?{r} (rdfs:label|skos:altLabel) ?lab .\n"
        if graph is None:
            return PFX + f"SELECT DISTINCT ?{r} ?lab WHERE {{\n    {body}\n{tail}}}"
        return PFX + f"SELECT DISTINCT ?{r} ?lab WHERE {{ GRAPH <{graph}> {{\n    {body}\n{tail}}} }}"


def _nodes(res, ret):
    """SPARQL JSON 결과 → [{"uri", "names"}]. 빈 응답은 실패로 올린다(결과 0 과 구분)."""
    if res is None or "results" not in res:
        raise RuntimeError(f"SPARQL 실패(빈 응답) — 그래프 결과 0 과 구분: {str(res)[:200]}")
    nodes = {}
    for b in res["results"]["bindings"]:
        nodes.setdefault(b[ret]["value"], []).append(b["lab"]["value"])
    return [{"uri": u, "names": ls} for u, ls in nodes.items()]


async def run_plan(st, plan, graph):
    q = Compiler().compile(plan, graph)
    res = await st.sparql_query(q)
    return {"nodes": _nodes(res, plan["return"]), "sparql": q}


def run(g, plan):
    """인메모리 실행 — 서버 없이 rdflib 로. 반환 형태는 run_plan 과 같다."""
    q = Compiler().compile(plan)
    res = json.loads(g.query(q).serialize(format="json"))
    return {"nodes": _nodes(res, plan["return"]), "sparql": q}
