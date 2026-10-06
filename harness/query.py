"""질의계획(IR) → SPARQL 컴파일·실행. 온톨로지 의미를 질의 시점에 쓴다.

IR:
  {"op": "list"|"count", "return": "x"|"y",
   "where": [{"t": "isa", "v": "x", "class": "가수"},                       # 타입(하위 클래스 폐포)
             {"t": "rel", "s": "x"|{"const": "이름"}, "p": "per:...", "o": {"const": ..}|{"var": "y"}}]}

규칙
- 변수 x 는 항상 '이 문서 모음에 수록된 표제 개체'(oh:describes) 로 한정한다.
- isa 는 rdf:type/rdfs:subClassOf* — 상위 클래스로 물어도 하위 클래스 인스턴스가 나온다.
- 상수는 정규화 키(oh:nkey)로 개체·클래스에 연결한다(라벨·별칭 모두).
- 역관계가 선언된 속성은 양방향을 UNION 으로 묻는다(per:parents ↔ per:children).
"""
import asyncio
import json
import sys

from harness import schema as S

PFX = f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX oh: <{S.NS}>
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
        lines.append(f"{v} oh:nkey {_lit(S.norm(t['const']))} .")
        return v

    def compile(self, plan, graph):
        lines, vars_ = [], set()
        for c in plan["where"]:
            if c["t"] == "isa":
                cv = self.fresh()
                lines.append(f"?{c['v']} rdf:type/rdfs:subClassOf* {cv} . {cv} oh:nkey {_lit(S.norm(c['class']))} .")
                vars_.add(c["v"])
            elif c["t"] == "rel":
                s, o = self.term(c["s"], lines), self.term(c["o"], lines)
                for t in (c["s"], c["o"]):
                    if isinstance(t, str):
                        vars_.add(t)
                    elif "var" in t:
                        vars_.add(t["var"])
                p = c["p"]
                if p not in S.RELATIONS:
                    raise ValueError(f"알 수 없는 관계: {p}")
                inv = S.RELATIONS[p][4]
                fwd = f"{s} <{S.rel_uri(p)}> {o} ."
                if inv and inv != p:
                    lines.append(f"{{ {fwd} }} UNION {{ {o} <{S.rel_uri(inv)}> {s} . }}")
                elif inv == p:  # 대칭
                    lines.append(f"{{ {fwd} }} UNION {{ {o} <{S.rel_uri(p)}> {s} . }}")
                else:
                    lines.append(fwd)
            else:
                raise ValueError(f"알 수 없는 조건: {c}")
        if "x" in vars_:
            lines.append("?xd oh:describes ?x .")
        r = plan["return"]
        body = "\n    ".join(lines)
        return (PFX + f"SELECT DISTINCT ?{r} ?lab WHERE {{ GRAPH <{graph}> {{\n    {body}\n"
                f"    ?{r} (rdfs:label|skos:altLabel) ?lab .\n}} }}")


async def run_plan(st, plan, graph):
    q = Compiler().compile(plan, graph)
    res = await st.sparql_query(q)
    if res is None or "results" not in res:
        raise RuntimeError(f"SPARQL 실패(빈 응답) — 그래프 결과 0 과 구분: {str(res)[:200]}")
    nodes = {}
    for b in res["results"]["bindings"]:
        nodes.setdefault(b[plan["return"]]["value"], []).append(b["lab"]["value"])
    return {"nodes": [{"uri": u, "names": ls} for u, ls in nodes.items()], "sparql": q}


if __name__ == "__main__":
    from harness.graph import store, graph_name
    plan = json.loads(sys.argv[3])

    async def _m():
        st = store()
        out = await run_plan(st, plan, graph_name(sys.argv[1], sys.argv[2]))
        await st.close()
        print(out["sparql"])
        for n in out["nodes"][:30]:
            print(n["names"])
        print(len(out["nodes"]), "nodes")
    asyncio.run(_m())
