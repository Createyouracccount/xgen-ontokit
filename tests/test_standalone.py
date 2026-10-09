"""독립 활용 경로(ontokit.graph → ontokit.query → ontokit.merge) 단위 테스트 — 서버·모델·LLM 없이."""
import pytest

pytest.importorskip("rdflib")

from ontokit.graph import project  # noqa: E402
from ontokit.merge import graph_block, merge  # noqa: E402
from ontokit.query import Compiler, run  # noqa: E402

RAW = {
    "concepts": {"class_hierarchy": [{"child": "가수", "parent": "인물"}]},
    "ner_entities": {
        "d1": [{"entity": "김가수", "class": "가수"}],
        "d2": [{"entity": "서울", "class": "도시"}],
        "d3": [{"entity": "이배우", "class": "배우"}, {"entity": "서울", "class": "도시"}],
    },
    "relations": [
        {"subject": "김가수", "predicate": "출생지", "relation_label": "per:place_of_birth", "object": "서울"},
        {"subject": "ㄱ기획", "predicate": "대표", "relation_label": "org:top_members/employees", "object": "김가수"},
        {"subject": "김가수", "predicate": "형제", "relation_label": "per:siblings", "object": "김동생"},  # 어휘 밖
    ],
}
DOCS = [{"doc_id": "d1", "title": "김가수"}, {"doc_id": "d2", "title": "서울 (도시)"},
        {"doc_id": "d3", "title": "어느 영화 촬영기"}]


def _names(out):
    return sorted(n["names"][0] for n in out["nodes"])


def test_project_counts_and_drops_unmapped_relation():
    g, st = project(RAW, DOCS)
    assert st == {"entities": 4, "types": 4, "rels": 3, "rels_mapped": 2, "aliases": 0, "subclass": 1}
    assert "김동생" not in {str(o) for o in g.objects()}   # KLUE_MAP 밖 라벨(per:siblings)은 버린다


def test_run_isa_uses_subclass_closure_and_title_anchor():
    g, _ = project(RAW, DOCS)
    # 가수 ⊑ 인물 → 인물로 물어도 김가수가 나온다
    assert _names(run(g, {"op": "list", "return": "x", "where": [{"t": "isa", "v": "x", "class": "인물"}]})) == ["김가수"]
    # x 는 표제 개체(문서 제목 = 이름)로 한정 — 이배우는 d3 에 언급만 되어 빠진다(STANDALONE 한계)
    assert _names(run(g, {"op": "list", "return": "x", "where": [{"t": "isa", "v": "x", "class": "배우"}]})) == []


def test_run_relation_const_and_inverse_mapping():
    g, _ = project(RAW, DOCS)
    born = {"op": "list", "return": "x",
            "where": [{"t": "rel", "s": "x", "p": "per:place_of_birth", "o": {"const": "서울(특별시)"}}]}
    assert _names(run(g, born)) == ["김가수"]   # 상수는 정규화 키로 — 괄호 수식어 무시
    # org:top_members/employees(조직→인물)는 per:employee_of(인물→조직)의 역으로 실린다
    emp = {"op": "list", "return": "y",
           "where": [{"t": "rel", "s": {"const": "김가수"}, "p": "per:employee_of", "o": {"var": "y"}}]}
    assert _names(run(g, emp)) == ["ㄱ기획"]


def test_run_rejects_unknown_relation():
    g, _ = project(RAW, DOCS)
    with pytest.raises(ValueError):
        run(g, {"op": "list", "return": "x", "where": [{"t": "rel", "s": "x", "p": "per:siblings", "o": {"var": "y"}}]})


def test_compile_named_graph_matches_measured_form():
    """원격 저장소용 문자열은 측정(L02)에 쓰인 형태 그대로 — 바뀌면 S01 동등성 증명을 다시 해야 한다."""
    plan = {"op": "list", "return": "y", "where": [
        {"t": "rel", "s": {"const": "ㄱ"}, "p": "per:parents", "o": {"var": "y"}},
        {"t": "isa", "v": "y", "class": "가수"}]}
    q = Compiler().compile(plan, "urn:g")
    body = q.split("SELECT", 1)[1]
    assert body == (
        ' DISTINCT ?y ?lab WHERE { GRAPH <urn:g> {\n'
        '    ?k1 oh:nkey "ㄱ" .\n'
        '    { ?k1 <https://w3id.org/ontoharness/r/per_parents> ?y . } UNION '
        '{ ?y <https://w3id.org/ontoharness/r/per_children> ?k1 . }\n'
        '    ?y rdf:type/rdfs:subClassOf* ?k2 . ?k2 oh:nkey "가수" .\n'
        '    ?y (rdfs:label|skos:altLabel) ?lab .\n} }')
    # 인메모리 형태는 GRAPH 감싸기만 다르다
    assert Compiler().compile(plan) == q.replace(" { GRAPH <urn:g> {\n", " {\n", 1).replace("\n} }", "\n}")


def test_merge_list_and_count_rules():
    nodes = [{"uri": "u1", "names": ["가"]}, {"uri": "u2", "names": ["나", "나별칭"]}]
    m = merge(["가", "다"], nodes)
    assert m == {"items": [{"names": ["가"]}, {"names": ["다"]}] + nodes}   # 중복 제거 없이 이어 붙임
    assert merge(["가"], nodes, op="count", reader_count=7)["count"] == 2   # 그래프 결과가 있으면 그 수
    assert merge(["가"], [], op="count", reader_count=7)["count"] == 7      # 없으면 판독 개수


def test_graph_block_text_and_truncation():
    assert graph_block([]) == ""
    b = graph_block([f"n{i}" for i in range(152)])
    assert b.startswith("[지식그래프 질의 결과 — 문서 모음 전체에서 조건에 맞는 항목 152개]\nn0, n1,")
    assert "n149 외 2개\n(그래프는 자동 추출이라" in b and "n150" not in b


def test_harness_uses_library_objects():
    """하네스가 사본이 아니라 라이브러리 객체를 쓴다 — 경로가 둘로 갈라지지 않게."""
    import harness.graph as hg
    import harness.hybrid as hh
    import harness.merge_eval as hm
    import harness.query as hq
    import ontokit.graph as og
    import ontokit.merge as om
    import ontokit.query as oq
    assert hg.ontokit_graph is og.project and hg.KLUE_MAP is og.KLUE_MAP
    assert hq.Compiler is oq.Compiler and hq.run_plan is oq.run_plan
    assert hm.merge is om.merge and hh.graph_block is om.graph_block


def test_cli_query_roundtrip_ttl_with_sources(tmp_path, capsys):
    """build 산출물(TTL)을 다시 읽어도 같은 답 + 출처 문서가 붙는다."""
    import json
    from ontokit.cli import main
    g, _ = project(RAW, DOCS)
    ttl = tmp_path / "g.ttl"
    g.serialize(ttl, format="turtle")
    main(["query", str(ttl), json.dumps({"op": "list", "return": "x", "where": [{"t": "isa", "v": "x", "class": "인물"}]})])
    out = json.loads(capsys.readouterr().out)
    assert out["count"] == 1 and out["nodes"][0]["name"] == "김가수" and out["nodes"][0]["docs"] == ["d1"]


def test_cli_ask_fails_loudly_without_llm(tmp_path, monkeypatch):
    from ontokit.cli import main
    monkeypatch.delenv("ONTOKIT_LLM_URL", raising=False)
    monkeypatch.delenv("ONTOKIT_LLM_MODEL", raising=False)
    g, _ = project(RAW, DOCS)
    ttl = tmp_path / "g.ttl"
    g.serialize(ttl, format="turtle")
    with pytest.raises(SystemExit, match="LLM 이 필요"):
        main(["ask", str(ttl), "가수는?"])


def test_planner_parse_strict():
    from ontokit.planner import PROMPT, parse
    assert "per:place_of_birth: 출생지" in PROMPT and PROMPT.endswith("질문: ")
    assert parse('<think>x</think>```json\n{"op":"list","where":[]}\n```') == {"op": "list", "where": [], "return": "x"}
    with pytest.raises(ValueError):
        parse('{"op": "list", "where": [')   # 잘린 JSON — 빈 계획으로 바꾸지 않는다


def test_chunk_text_respects_ner_window():
    from ontokit.pipeline import CHUNK_CHARS, chunk_text
    assert chunk_text("") == [] and chunk_text("  \n\n ") == []
    long_para = "가나다라마바사. " * 400            # 한 문단 3,200자
    chunks = chunk_text("짧은 문단\n\n" + long_para)
    assert all(len(c) <= CHUNK_CHARS for c in chunks) and len(chunks) >= 3
    assert "".join(chunks).replace("\n", "").replace(" ", "") == ("짧은 문단" + long_para).replace(" ", "")


def test_oxigraph_store_matches_memory():
    pytest.importorskip("pyoxigraph")
    import asyncio
    from ontokit.backends.oxigraph import OxigraphStore
    from ontokit.query import run_plan
    g, _ = project(RAW, DOCS)
    st = OxigraphStore()
    assert st.load(g, "urn:t") == len(g)
    for p in ({"op": "list", "return": "x", "where": [{"t": "isa", "v": "x", "class": "인물"}]},
              {"op": "list", "return": "y",
               "where": [{"t": "rel", "s": {"const": "김가수"}, "p": "per:employee_of", "o": {"var": "y"}}]}):
        assert asyncio.run(run_plan(st, p, "urn:t"))["nodes"] == run(g, p)["nodes"]
    assert st.load(g, "urn:t") == len(g)   # 같은 이름으로 다시 실으면 교체(누적 아님)
