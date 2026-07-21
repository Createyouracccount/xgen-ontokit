"""sg1 — NER 스팬 어절 경계 게이트 (결정론, LLM 0콜). ⚠️ 패널 기각·파킹(0722).

**채택 기각 확정** (주심 34/100 + 렌즈B/C, eval_runs/typing/sg1_judge_verdicts.md):
신선 표본 G1 오살 v1 54.0%→수리 후 v2 24.3%(Wilson 상단 31.1%) — 게이트(≤10%)
3배 초과. 근본 원인 = 우측 명사 합성 연속('[삿포로]시'·'[고려]시대')과 절단
('[패러]데이')이 표면 문자로 구분 불가. 좌측도 접두 합성('반정부'→'정부')이
뉴스·법령 문체에서 오살(렌즈B 10/15). G2(절단 회수)는 양 라운드 100%.

**재도입 금지 조건**(위반 시 파울로 판례 3회째): 판별 기제(우측 연속의 독립
명사성 또는 스팬 완전 형태소성 — Kiwi) 없는 표면 규칙 재도전 금지. 재도전 시
①축소 스코프 사전 등록 ②새 신선 표본(main 400·reserve 100 소진) ③수리 0회
④렌즈C 회귀 게이트 RG1~RG4 필수. env ONTOKIT_NER_SPAN_GATE 는 opt-in 파킹
자산으로만 존속(기본 off 유지 — on 승격 금지).


wiki2 오류의 ~50%가 MALFORMED(스팬 절단·오염)이고 그중 절반이 스크립트 연속
절단형('알메이'다·'패러'데이·'183'1·마르'부르크')임이 gate3 개발 코퍼스 79건에서
실측됨(eval_runs/typing/sg1_rule_design.md — 개발 코퍼스는 규칙 설계에만 사용,
채택 게이트는 봉인된 신선 표본 sg1_fresh_sample_ids.json 전용).

규칙(어절 경계 정합):
- 좌측 인접 문자가 한글/숫자/라틴 → 어절 중간 시작 = 절단 → 차단(예외 없음).
- 우측 인접이 숫자/라틴 → 절단 → 차단. 한글이면 연속 run 을 조사 열로
  최장일치 반복 분해 — 전부 조사면 정상(개체명+조사 어절), 실패 시 차단.
- 조사 목록은 학교문법 폐집합 연역(설계서 §2 — gate3 79건 귀납 금지).
  서술격 모음-뒤 축약형('다' 등)·파생 접미사는 예외 아님(정밀도 우선, 비용은
  오살 게이트가 실측). 알려진 한계: 조사 동형 고유명사 말미('손정의')는 놓침.
- 오프셋 없는 스팬(2패스 등)은 검사 불가 → 통과.

동작 = 방출 차단(수리 아님 — align_spans 확장 이후 잔여 절단만, 확장 재시도는
'한국측' 오결합 계열 위험으로 별도 라운드). env ONTOKIT_NER_SPAN_GATE 로 토글.
"""
from __future__ import annotations

import re

_HANGUL = re.compile(r"[가-힣]")
_WORDCHAR = re.compile(r"[가-힣0-9A-Za-z]")

# 조사 폐집합 — 학교문법 연역(격·보조·접속 + 서술격 '이' 개재형). 설계서 §2.
_JOSA = frozenset({
    # 격조사
    "이", "가", "께서", "을", "를", "의", "에", "에서", "에게", "에게서",
    "한테", "한테서", "께", "더러", "보고", "로", "으로", "로서", "으로서",
    "로써", "으로써", "라고", "이라고", "와", "과", "랑", "이랑", "하고",
    "처럼", "만큼", "보다", "같이", "아", "야", "여", "이여", "이시여",
    # 보조사
    "은", "는", "도", "만", "뿐", "까지", "마저", "조차", "부터", "마다",
    "밖에", "커녕", "은커녕", "는커녕", "나마", "이나마", "야말로", "이야말로",
    "라도", "이라도", "인들", "은들", "든지", "이든지", "든", "이든",
    "나", "이나", "이야", "요", "마는", "만은", "은요", "는요",
    # 접속조사 (격과 중복 제외분)
    "며", "이며", "에다", "에다가",
    # 서술격조사 '이다' 활용 — '이' 개재형만(모음-뒤 축약형 제외, 설계서 §2-1)
    "이다", "이자", "이란", "이라", "이라는", "인", "이던", "이었다",
    "이었으며", "이었고", "입니다", "이에요", "이고", "이지만", "이거나",
    # ── sg1 수리 1회(0722, G1 오살 실측 후 연역 보강 — 설계서 §5 추록) ──
    # 서술격 과거 활용 '였-' 계열 + 통용 표기 '이였-' 계열 (단음절 축약
    # 다/라/자/란 은 계속 제외 — '알메이다가' 보호)
    "였다", "였고", "였던", "였으며", "였지만", "이였다", "이였고", "이였던",
    # 축약 조사 (에+는→엔, 께+는→껜)
    "엔", "껜",
})

# 생산적 복수 접미 — 체언+들+조사 는 표준 결합(수리 1회: '군인들의' 오살 해소).
# 조사열 분해 전에 선행 1회만 소비 허용. 폐집합 유지(다른 접미사 확장 금지).
_PLURAL_SUFFIX = "들"

# 숫자 스팬(수사) 판정 — 수사+단위명사(년·월·명·승·km·g …)는 표준 결합이고
# modu-ner 의 날짜/수량 스팬은 숫자만 잡는 게 모델 관행(신선 표본 실측:
# 오살 108건 중 ~68건이 '[1998]년' 유형) — 절단이 아니라 관행이므로 숫자
# 스팬의 우측 한글/라틴 연속은 면제. 우측 숫자 연속('[183]1')은 차단 유지.
_NUMERIC_SPAN = re.compile(r"[0-9][0-9.,\s]*")
_JOSA_MAXLEN = max(len(j) for j in _JOSA)


def _josa_only(run: str) -> bool:
    """한글 run 이 [복수접미 '들' ≤1회 +] 조사 열로 완전 분해되는가 — 최장일치."""
    if run.startswith(_PLURAL_SUFFIX):  # 수리 1회: 체언+들(+조사) 표준 결합
        rest = run[len(_PLURAL_SUFFIX):]
        if not rest or _josa_only_strict(rest):
            return True
    return _josa_only_strict(run)


def _josa_only_strict(run: str) -> bool:
    i, n = 0, len(run)
    while i < n:
        for length in range(min(_JOSA_MAXLEN, n - i), 0, -1):
            if run[i:i + length] in _JOSA:
                i += length
                break
        else:
            return False
    return True


def boundary_ok(text: str, start: int, end: int) -> bool:
    """스팬이 어절 경계에 정합하면 True, 절단이면 False."""
    if start > 0 and _WORDCHAR.match(text[start - 1]):
        return False  # 어절 중간 시작
    if end < len(text):
        ch = text[end]
        numeric = _NUMERIC_SPAN.fullmatch(text[start:end]) is not None
        if _HANGUL.match(ch):
            if numeric:
                return True  # 수리 1회: 수사+단위명사 관행('[1998]년') 면제
            j = end
            while j < len(text) and _HANGUL.match(text[j]):
                j += 1
            return _josa_only(text[end:j])
        if _WORDCHAR.match(ch):
            # 숫자 스팬 + 라틴 단위('[30]g'·'[58]km')는 관행 면제,
            # 숫자 연속('[183]1')·라틴 연속은 차단 유지.
            if numeric and not ch.isdigit():
                return True
            return False
    return True


def gate_spans(text: str, ents: list[dict]) -> list[dict]:
    """경계 절단 스팬 차단. 오프셋 없는 스팬은 통과(검사 불가 — 공시)."""
    out = []
    for e in ents:
        st, en = e.get("start"), e.get("end")
        if isinstance(st, int) and isinstance(en, int) and 0 <= st < en <= len(text):
            if not boundary_ok(text, st, en):
                continue
        out.append(e)
    return out
