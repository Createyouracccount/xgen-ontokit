"""복합명사 접미 공유 기반 subClassOf 계층 유도.

finreg 489 실측 핵심 발견: 한국어는 head-final이라 복합명사의 뒤가 상위 개념.
`생명보험업`의 뒤 `보험업`이 상위 → 생명보험업 ⊂ 보험업. 대주주⊂주주, 자회사⊂회사.
실측 순도 높음(825→1710건). 정의문 "마지막명사=상위"는 서술구조라 오탐 많아 제외.
"""
from __future__ import annotations
from ..morphology.kiwi_nouns import STOP_HEAD


def induce_suffix_hierarchy(class_names: set[str]) -> list[dict]:
    """클래스 집합에서 접미 공유 subClassOf 유도.
    child가 parent로 끝나고 더 길면 child subClassOf parent.
    전역 1회 호출(청크 경계 무관)이 정확.
    """
    cls_list = [c for c in class_names if 2 <= len(c) <= 20 and c not in STOP_HEAD]
    seen, out = set(), []
    for child in cls_list:
        for parent in cls_list:
            if (child != parent and len(parent) >= 2 and child.endswith(parent)
                    and len(child) > len(parent) and parent not in STOP_HEAD):
                key = (parent, child)
                if key not in seen:
                    seen.add(key)
                    out.append({"parent": parent, "child": child})
    return out
