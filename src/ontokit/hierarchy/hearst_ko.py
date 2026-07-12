"""한국어 Hearst 패턴 (정의문/열거) — subClassOf 보조 유도.

finreg 489 실측: 정의문("X이란…말한다") 96건, "X의 종류" 55건이 법령문 신호.
단 "마지막 명사=상위"는 서술구조라 오탐 많음 → 현재 접미공유가 주 엔진이고
이 모듈은 확장 여지(정의문 정밀화, KorLex 검증 결합)로 보류. 장르=정의문일 때 89.7% 계열.
"""
from __future__ import annotations
import re

# "생명보험업"이란 ... 을 말한다  (정의 대상=따옴표 안이 클래스)
DEF_QUOTED = re.compile(
    r'["“]([가-힣A-Za-z]{2,20})["”]\s*(?:이란|란|이라 함은|라 함은|이라고|은|는)\s+'
    r'(.{5,80}?)(?:을|를|이|가)?\s*말한다')
# "X의 종류/구분/유형" 표제
KIND_HEADER = re.compile(r'([가-힣A-Za-z]{2,20})의?\s*(?:종류|구분|유형)')


# 정의 본문 꼬리의 의존명사(닫힌 문법 집합) — "…정하는 것/등록된 자/기타 등"으로
# 끝나는 정의는 분류학적 상위어가 아니라 조건절이다. last_noun 이 앞의 비-head
# 명사("대통령령", "등록")로 폴백해 거짓 상위어를 만들던 주 오류원(finreg 35쌍
# 실측: 순도 ~1/3 → 의존명사 꼬리 드랍 후 재측정으로 결정).
_DEP_NOUN_TAIL = re.compile(r"(?:것|자|등|바|수|곳|때)\s*$")


def definitional_pairs(text: str, last_noun_fn) -> list[dict]:
    """정의문에서 (child=정의대상, parent=피정의 상위개념) 추출.
    ⚠️ 실측상 오탐 있어 보수적 사용 권장(KorLex 검증 결합 시 정밀도↑).
    last_noun_fn: 구절→마지막 명사 (KiwiNounExtractor.last_noun 주입).

    이 함수는 법령체(따옴표 정의문 `"X"이란 … 말한다`)만 커버한다.
    본문 꼬리가 의존명사(것/자/등…)면 조건절 정의라 쌍을 만들지 않는다."""
    out = []
    for m in DEF_QUOTED.finditer(text):
        child = m.group(1).strip()
        body = m.group(2)
        if _DEP_NOUN_TAIL.search(body.strip()):
            continue
        parent = last_noun_fn(body)
        if child and parent and child != parent and len(parent) >= 2:
            out.append({"parent": parent, "child": child})
    return out

