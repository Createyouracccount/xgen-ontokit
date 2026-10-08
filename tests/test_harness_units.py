"""하네스 순수 함수 단위 테스트 — 측정 도구 결함(파서·정규화·정답 의미)이 조용히 재발하지 않게.

네트워크·저장소 없이 돈다. 하네스는 패키지 밖(harness/)이라 리포 루트에서 실행한다.
"""
from harness import schema as S
from harness.extract_llm import parse_salvage
from harness.gold import Evaluator, Facts, mentions


def test_parse_salvage_truncated_string_array():
    # L1 사고: 답 목록이 출력 상한에 잘리면 '복구 불가' 로 0점 처리됐다(137칸)
    assert parse_salvage('{"answers": ["김수경", "김미정", "김태') == {
        "answers": ["김수경", "김미정"], "_salvaged": True}


def test_parse_salvage_truncated_object_array_and_failure():
    out = parse_salvage('{"relations": [{"p": "per:origin", "o": "미국"}, {"p": "per:sch')
    assert out["relations"] == [{"p": "per:origin", "o": "미국"}]
    import pytest
    with pytest.raises(ValueError):
        parse_salvage("완전히 JSON 이 아님")


def test_norm_strips_parens_spaces_punct():
    assert S.norm("박정희 (1917년)") == S.norm("박정희") == "박정희"
    assert S.norm("A·B-C") == "abc"


def test_mentions_two_char_josa_boundary():
    # '배우' 는 '배우자' 안에서 인정하지 않는다(정답 근거 오탐 방지)
    assert mentions("그는 미국의 배우이다.", "배우")
    assert not mentions("그의 배우자는 화가였다.", "배우")


def _toy_facts():
    docs = [{"doc_id": "d1", "title": "김가수", "text": "김가수는 서울에서 태어난 가수이다."},
            {"doc_id": "d2", "title": "이작곡", "text": "이작곡은 작곡가이다. 아버지는 이부모이다."}]
    subjects = {
        "김가수": {"qid": "Q1", "label": "김가수", "aliases": [], "claims": {"P106": ["C1"], "P19": ["Q9"]}},
        "이작곡": {"qid": "Q2", "label": "이작곡", "aliases": [], "claims": {"P106": ["C2"], "P22": ["Q8"]}},
    }
    objects = {"C1": {"label": "가수", "aliases": [], "claims": {"P279": ["C3"]}},
               "C2": {"label": "작곡가", "aliases": [], "claims": {"P279": ["C3"]}},
               "C3": {"label": "음악가", "aliases": [], "claims": {}},
               "Q9": {"label": "서울특별시", "aliases": ["서울"], "claims": {}},
               "Q8": {"label": "이부모", "aliases": [], "claims": {}}}
    return Facts(docs, subjects, objects)


def test_evaluator_semantics_closure_alias_inverse():
    ev = Evaluator(_toy_facts())
    # 하위 클래스 폐포: 본문엔 가수·작곡가만 적혔어도 '음악가' 로 둘 다
    assert set(ev.solve({"op": "list", "return": "x", "where": [{"t": "isa", "v": "x", "class": "음악가"}]})) == {"Q1", "Q2"}
    # 별칭: '서울' 로 물어도 서울특별시
    assert set(ev.solve({"op": "list", "return": "x", "where": [
        {"t": "rel", "s": "x", "p": "per:place_of_birth", "o": {"const": "서울"}}]})) == {"Q1"}
    # 역관계: 부모(P22) 사실로 '자녀' 를 물을 수 있다
    assert set(ev.solve({"op": "list", "return": "y", "where": [
        {"t": "rel", "s": {"const": "이부모"}, "p": "per:children", "o": {"var": "y"}}]})) == {"Q2"}
