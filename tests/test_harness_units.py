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


# ── 메모리 가드(OS 무관) ──
from harness.guard import decide, parse_cgroup, parse_meminfo, parse_memory_pressure


def test_guard_parsers_each_os():
    assert round(parse_meminfo("MemTotal: 1000 kB\nMemFree: 100 kB\nMemAvailable: 250 kB\n")) == 25   # Linux
    assert parse_cgroup("max", "123", 10**12) is None                                                 # 한도 없음 → 호스트 지표로
    assert parse_cgroup("1000", "900", 10**12) == 10.0                                                # 컨테이너 한도 기준
    assert parse_cgroup("2000", "100", 1000) is None                                                  # 한도 ≥ 호스트 = 무의미
    assert parse_memory_pressure("System-wide memory free percentage: 33%") == 33.0                   # macOS


def test_guard_decide_hysteresis():
    acts, p = decide(15, False)            # 20 미만 → 정지
    assert acts == ["pause"] and p
    acts, p = decide(25, True)             # 20~30 사이 → 그대로(진동 방지)
    assert acts == [] and p
    acts, p = decide(35, True)             # 30 이상 → 재개
    assert acts == ["resume"] and not p
    acts, p = decide(5, False)             # 10 미만 → LLM 서버 회수 + 정지
    assert acts == ["kill", "pause"] and p


def test_placebo_changes_every_fact():
    # L01: 목적어가 겹치는 관계에서 위약이 사실을 그대로 남겨 음성 대조가 무너졌다
    from harness.graph import placebo_facts
    types = [("s1", ["역"]), ("s2", ["역"]), ("s3", ["산"]), ("s4", ["대학"])]
    rels = [{"s": "s1", "p": "loc:located_in", "o": "구A"}, {"s": "s2", "p": "loc:located_in", "o": "구A"},
            {"s": "s3", "p": "loc:located_in", "o": "구A"}, {"s": "s4", "p": "loc:located_in", "o": "구B"},
            {"s": "s1", "p": "per:origin", "o": "나라X"}]
    nt, nr = placebo_facts(types, rels)
    orig = {(r["s"], r["p"], r["o"]) for r in rels}
    assert len(nr) == len(rels) and not ({(r["s"], r["p"], r["o"]) for r in nr} & orig)
    assert all(not (set(t) & set(dict(types)[s])) for s, t in nt)
