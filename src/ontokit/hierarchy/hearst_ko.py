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


def definitional_pairs(text: str, last_noun_fn) -> list[dict]:
    """정의문에서 (child=정의대상, parent=피정의 상위개념) 추출.
    ⚠️ 실측상 오탐 있어 보수적 사용 권장(KorLex 검증 결합 시 정밀도↑).
    last_noun_fn: 구절→마지막 명사 (KiwiNounExtractor.last_noun 주입)."""
    out = []
    for m in DEF_QUOTED.finditer(text):
        child = m.group(1).strip()
        parent = last_noun_fn(m.group(2))
        if child and parent and child != parent and len(parent) >= 2:
            out.append({"parent": parent, "child": child})
    return out
