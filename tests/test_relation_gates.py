"""rel2 방출 게이트 판례표 (설계 rel2_design_v2.md, 패널 채택 86/100 + 재심 라이더).

판례 출처: eval_runs/typing/rel1_relation_audit_ui100.md(감사) + rel2_gate_sim.py(시뮬)
+ 렌즈B 반례(미상 통과·국가 주어·날짜/약어 비분할). G5 어휘는 코드 상수 동결 기준.
"""
import pytest

from ontokit.extractors.relation_encoder_ko import (
    KoreanRelationEncoder, _gate, _SENT_SPLIT_FALLBACK, DEFAULT_MIN_SCORE,
)

S = "삼성전자는 올해 초 이재용 부회장을 선임했다."  # 트리거(선임) 보유 기본 문장


# ── 차단 판례 (감사 실측 병리) ─────────────────────────────────────────
BLOCK = [
    # S게이트: 술어 접두-주어 타입 교차 오류
    ("org:top_members/employees", "인물", "인물", "구본혁", S, "S"),   # 이인구→구본혁
    ("per:employee_of", "기관", "기관", "서울대", S, "S"),             # 대한상의→서울대
    ("per:title", "기관", "문화·제도", "연구원", "엔비디아 연구원 이 대표를 만났다.", "S"),
    # T게이트: 상대시간 목적어(앵커불능)
    ("org:founded", "기관", "날짜", "올해", "유안타증권은 올해 설립 10년을 맞았다.", "T"),
    ("org:founded", "기관", "날짜", "지난해", "삼성전자는 지난해 창립했다.", "T"),
    ("per:employee_of", "인물", "날짜", "이날", "김경택은 이날 회사에 재직 중이라고 밝혔다.", "T"),
    ("org:dissolved", "기관", "날짜", "분기", "정부는 분기 중 해산을 검토했다.", "T"),
    # O게이트: 목적어 서명 위반(확정 클래스 모순)
    ("org:top_members/employees", "기관", "문화·제도", "사장", "아큐론 사장이 취임했다.", "O"),
    ("org:top_members/employees", "기관", "수량", "270여명", "KB금융 임직원 270여명이 근무한다.", "O"),
    ("org:dissolved", "기관", "지역", "우크라이나", "혼다는 우크라이나 해산 절차를 밟았다.", "O"),
    ("per:siblings", "인물", "동물", "모기", "이명순은 동생 모기와 산다.", "O"),
    ("org:founded", "기관", "수량", "27세", "카스는 27세에 설립됐다.", "O"),
    # G5: 술어 트리거 어휘 부재(동시출현 날조)
    ("org:dissolved", "기관", "날짜", "5월", "현대차 넥쏘가 5월 도요타·혼다보다 많이 팔렸다.", "G5"),
    ("org:member_of", "기관", "기관", "구글", "던파M이 구글플레이 매출 톱3를 탈환했다.", "G5"),
    ("per:origin", "인물", "지역", "미국", "미국 석학이 최태원 회장 의지를 평가했다.", "G5"),
]

# ── 통과 판례 (정탐 + 렌즈B 반례) ──────────────────────────────────────
PASS = [
    # 정탐 6종 (시뮬 판례표)
    ("per:schools_attended", "인물", "기관", "우석대학교", "임충식은 우석대학교를 졸업했다.", None),
    ("org:place_of_headquarters", "기관", "지역", "서울", "이데일리 본사는 서울에 있다.", None),
    ("per:employee_of", "정치인", "기관", "기획재정부", "추경호는 기획재정부에 재직했다.", None),
    ("per:title", "인물", "문화·제도", "대통령", "윤석열 대통령이 발표했다.", None),
    ("org:founded", "기관", "날짜", "1969년", "삼성전자는 1969년 설립됐다.", None),
    ("per:employee_of", "인물", "기관", "삼성전자", "이재용은 삼성전자에 합류했다.", None),
    # 렌즈B R1: 미상 타입은 통과(확증된 모순만 차단)
    ("per:date_of_birth", "인물", "", "1545년", "이순신은 1545년 태어났다.", None),
    ("per:employee_of", "인물", "미지클래스", "티맥스", "이희상은 티맥스에 근무한다.", None),
    # 렌즈B R2: 국가(지역) 주어 org:* 허용
    ("org:member_of", "지역", "기관", "UN", "대한민국은 UN에 가입했다.", None),
    # 렌즈B 재심: 고유어 동사 트리거
    ("org:founded", "기관", "날짜", "1947년", "정주영이 1947년 현대건설을 세웠다.", None),
    ("per:employee_of", "인물", "기관", "삼성전자", "그는 1998년부터 삼성전자에서 근무했다.", None),
    # 미정의 술어는 통과(보수)
    ("per:religion", "인물", "용어", "불교", "그는 불교 신자다.", None),
]


@pytest.mark.parametrize("label,s_cls,o_cls,obj,sent,want", BLOCK)
def test_gate_blocks_pathology(label, s_cls, o_cls, obj, sent, want):
    assert _gate(label, s_cls, o_cls, obj, sent) == want


@pytest.mark.parametrize("label,s_cls,o_cls,obj,sent,want", PASS)
def test_gate_keeps_legit(label, s_cls, o_cls, obj, sent, want):
    assert _gate(label, s_cls, o_cls, obj, sent) is want


def test_fallback_split_preserves_dates_and_abbrev():
    # 렌즈B R3: 공문서 날짜·영문 약어에서 폴백 분할이 파괴하지 않는다.
    t1 = "그는 1969. 5. 4. 출생했다. 이후 서울에서 자랐다."
    parts = [p for p in _SENT_SPLIT_FALLBACK.split(t1) if p and p.strip()]
    assert any("1969. 5. 4." in p for p in parts)
    t2 = "Apple Inc. 는 1976년 설립됐다. 본사는 쿠퍼티노다."
    parts2 = [p for p in _SENT_SPLIT_FALLBACK.split(t2) if p and p.strip()]
    assert any("Inc. 는 1976년" in p for p in parts2)


def test_pairs_sentence_scope():
    # G4: 다른 문장의 개체끼리는 쌍이 생성되지 않는다.
    enc = KoreanRelationEncoder.__new__(KoreanRelationEncoder)
    ents = [{"entity": "삼성전자", "class": "기관"}, {"entity": "김철수", "class": "인물"}]
    sents = ["삼성전자는 실적을 발표했다.", "김철수는 서울에 산다."]
    pairs = enc._pairs(ents, sents)
    assert pairs == []
    # 같은 문장이면 쌍 생성
    sents2 = ["삼성전자 김철수 사장이 발표했다."]
    pairs2 = enc._pairs(ents, sents2)
    assert any(p[0] == "삼성전자" and p[2] == "김철수" for p in pairs2)


def test_default_min_score_raised():
    assert DEFAULT_MIN_SCORE == 0.5


def test_min_score_env_override(monkeypatch):
    # conf 스윕 라운드(0722): env ONTOKIT_RELATION_CONF_MIN 배선.
    # 우선순위 = 명시 인자 > env > 기본 0.5. 파싱 실패는 기본값 폴백.
    monkeypatch.setenv("ONTOKIT_RELATION_CONF_MIN", "0.7")
    enc = KoreanRelationEncoder(model="dummy")
    assert enc._min_score == 0.7
    enc2 = KoreanRelationEncoder(model="dummy", min_score=0.55)
    assert enc2._min_score == 0.55  # 명시 인자가 env 를 이긴다
    monkeypatch.setenv("ONTOKIT_RELATION_CONF_MIN", "not-a-float")
    assert KoreanRelationEncoder(model="dummy")._min_score == DEFAULT_MIN_SCORE
    monkeypatch.delenv("ONTOKIT_RELATION_CONF_MIN")
    assert KoreanRelationEncoder(model="dummy")._min_score == DEFAULT_MIN_SCORE


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "-inf", "-0.3"])
def test_min_score_env_rejects_nonfinite(monkeypatch, bad):
    # 주심 D5(0722): float("nan") 파싱 성공 → score<nan 항상 False → 게이트 조용한
    # 무력화. nan/inf/음수는 거부하고 기본값 폴백해야 한다.
    monkeypatch.setenv("ONTOKIT_RELATION_CONF_MIN", bad)
    assert KoreanRelationEncoder(model="dummy")._min_score == DEFAULT_MIN_SCORE
