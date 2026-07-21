"""sg1 어절 경계 게이트 판례 테스트.

차단 판례는 gate3 개발 코퍼스 79건의 대표 유형(설계서 §1), 보존 판례는
조사 체계 연역(격·보조·접속·서술격 '이' 개재형 + 스태킹)에서 구성.
"""
import pytest

from ontokit.ner.span_gate import boundary_ok, gate_spans, _josa_only


# ── 차단 판례 (경계 절단) ───────────────────────────────────────────
BLOCK = [
    # (text, span) — span 은 text 내 첫 출현
    ("프란시스쿠 드 알메이다가 인도 총독으로", "알메이"),      # 우측 '다가' 비조사열
    ("마이클 패러데이는 영국의", "패러"),                      # 우측 한글 연속 '데이는' 실패
    ("독일 헤센주의 마르부르크 부근", "부르크"),               # 좌측 한글 '르'
    ("FRS, 1831년 6월", "183"),                               # 우측 숫자
    ("Waleran IV가 아들을", "ale"),                           # 좌우 라틴
    # ('1919년'의 '1919'는 수리 1회로 숫자+단위 면제 — REPAIR_KEEP 참조)
    ("남연의 초대 황제 모용덕의 묘호이다", "용덕"),            # 좌측 한글 '모'
    ("일본의 성인 비디오 여배우", "배우"),                     # 좌측 한글 '여'
]

# ── 보존 판례 (정상 어절 경계) ──────────────────────────────────────
KEEP = [
    ("삼성전자는 반도체를 생산한다", "삼성전자"),              # 조사 '는'
    ("이재용 부회장이 발표했다", "이재용"),                    # 공백 경계
    ("서울에서는 행사가 열렸다", "서울"),                      # 스태킹 에서+는
    ("한국은행으로부터 자료를 받았다", "한국은행"),            # 스태킹 으로+부터...
    ("박정희 후보가 당선됐다.", "박정희"),                     # 공백
    ("김주현입니다", "김주현"),                                # 서술격 '입니다'
    ("대한민국의 수도", "대한민국"),                           # 관형격 '의'
    ("맥스웰(1831년생)은 물리학자이다", "맥스웰"),             # 구두점 경계
    ("소크라테스이다", "소크라테스"),                          # 서술격 '이다'
    ("문서 말미의 홍길동", "홍길동"),                          # 우측 EDGE
]


@pytest.mark.parametrize("text,span", BLOCK)
def test_block(text, span):
    i = text.index(span)
    assert boundary_ok(text, i, i + len(span)) is False


@pytest.mark.parametrize("text,span", KEEP)
def test_keep(text, span):
    i = text.index(span)
    assert boundary_ok(text, i, i + len(span)) is True


def test_josa_stacking():
    assert _josa_only("에서는")
    assert _josa_only("으로부터")
    assert _josa_only("만으로도")
    assert not _josa_only("데이는")   # '데이' 비조사
    assert not _josa_only("다가")     # 서술격 축약 '다' 는 예외 아님(설계 §2-1)


def test_known_limit_son_jeongui():
    # 공시된 구조적 한계: 조사 동형 말미('손정의' 인명) — 게이트가 놓친다(통과).
    text = "소프트뱅크 손정의 회장이"
    i = text.index("손정")
    assert boundary_ok(text, i, i + 2) is True  # '의' 가 관형격 동형이라 통과


# ── sg1 수리 1회(0722) 판례 — G1 오살 실측 유형의 연역 보강 ─────────
REPAIR_KEEP = [
    ("밍크에서 1998년 6월부터 연재됐다", "1998"),        # 숫자+단위 '년' 관행
    ("시즌에는 16승을 거두었다", "16"),                   # 숫자+'승'
    ("현재는 32비트 및 64비트", "32"),                    # 숫자+'비트'
    ("바젤에서 남쪽으로 58km 떨어져", "58"),              # 숫자+라틴 단위
    ("급진 군인들의 사회주의", "군인"),                    # 복수접미 들+의
    ("과학자들과 기술자들은", "과학자"),                   # 들+과
    ("진압책임자는 조병옥이였고 그는", "조병옥"),          # 비표준 '이였고'
    ("하동 정씨였던 정치순과", "하동 정씨"),               # '였던'
    ("마르세유는 겨울엔 온난다습하다", "겨울"),            # 축약 조사 '엔'
]

REPAIR_BLOCK = [
    ("FRS, 1831년 6월", "183"),          # 숫자 스팬+우측 숫자 = 절단 유지
    ("프란시스쿠 드 알메이다가 인도", "알메이"),  # '다가' 계속 차단(축약 '다' 미포함)
    ("봉기는 11월에 일어났다", "11"),     # 숫자+한글 면제의 공시 비용 — G2 실측
]


@pytest.mark.parametrize("text,span", REPAIR_KEEP)
def test_repair_keep(text, span):
    i = text.index(span)
    assert boundary_ok(text, i, i + len(span)) is True


def test_repair_block_unchanged():
    # 수리가 완화 전용임을 고정 — 대표 절단 판례는 계속 차단
    for text, span in [REPAIR_BLOCK[0], REPAIR_BLOCK[1]]:
        i = text.index(span)
        assert boundary_ok(text, i, i + len(span)) is False
    # 공시 비용: '[11]월' 은 면제로 통과(G2 17→16, 94% 실측 유지)
    text, span = REPAIR_BLOCK[2]
    i = text.index(span)
    assert boundary_ok(text, i, i + len(span)) is True


def test_gate_spans_offsets():
    text = "마이클 패러데이는 영국의 과학자이다"
    ents = [
        {"entity": "패러", "start": 4, "end": 6},            # 절단 → 차단
        {"entity": "영국", "start": 10, "end": 12},          # '의' 조사 → 보존
        {"entity": "무오프셋", "start": None, "end": None},  # 검사 불가 → 통과
    ]
    out = gate_spans(text, ents)
    assert [e["entity"] for e in out] == ["영국", "무오프셋"]
