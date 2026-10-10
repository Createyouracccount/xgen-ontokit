"""질의계획(IR) → SPARQL 컴파일·실행 — 구현은 `ontokit.query`. 여기는 하네스 CLI 만 남는다.

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

# 컴파일러·실행은 라이브러리가 정본이다(harness/docs/S01 동등성 증명).
from ontokit.query import PFX, Compiler, run_plan  # noqa: F401


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
