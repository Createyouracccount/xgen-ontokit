"""복합명사 접미 공유 기반 subClassOf 계층 유도.

finreg 489 실측 핵심 발견: 한국어는 head-final이라 복합명사의 뒤가 상위 개념.
`생명보험업`의 뒤 `보험업`이 상위 → 생명보험업 ⊂ 보험업. 대주주⊂주주, 자회사⊂회사.
실측 순도 높음(825→1710건). 정의문 "마지막명사=상위"는 서술구조라 오탐 많아 제외.

성능·정밀도(v0.4):
- **인덱스화**: 이전 O(N²) pairwise endswith 를 접미 인덱스 조회 O(N·L²)로 교체
  (L=이름 길이 ≤ MAX_LEN=20 이라 사실상 선형). 800만 문서 → 수십만 클래스에서
  이전 이중 루프는 수조 비교로 멈췄음 — 라이브러리의 존재 이유(대용량 LLM-free)를
  살리려면 계층 유도도 O(N) 여야 한다.
- **허브 필터**: parent 후보가 MIN_CHILDREN 개 이상 child 의 접미로 등장할 때만 상위로
  인정. `대학⊂학`, `국가⊂가` 같은 형태소 경계 무시 오탐(순수 문자열 endswith 의 약점)을
  빈도 임계로 제거. `보험업`처럼 여러 하위어를 거느린 진짜 상위개념만 남는다.
"""
from __future__ import annotations
from collections import defaultdict
from ..morphology.kiwi_nouns import STOP_HEAD

MAX_LEN = 20        # 클래스 이름 최대 길이 (접미 인덱스 상한 — L² 항 방어)
MIN_SUFFIX_LEN = 2  # 상위개념 후보 최소 길이 (1글자 접미 '학'/'가' 배제)
MIN_CHILDREN = 2    # 허브 임계 — 이만큼의 child 접미로 등장해야 상위로 인정


def induce_suffix_hierarchy(class_names: set[str],
                            min_children: int = MIN_CHILDREN) -> list[dict]:
    """클래스 집합에서 접미 공유 subClassOf 유도 (인덱스화 + 허브 필터).

    child 가 parent 로 끝나고 더 길면 child subClassOf parent — 단, parent 가
    min_children 개 이상의 서로 다른 child 의 접미일 때만(허브) 인정.
    전역 1회 호출(청크 경계 무관)이 정확.

    복잡도: O(N·L²) — 각 이름의 접미 후보(≤L개)를 집합 조회. L≤MAX_LEN 이라 실질 선형.
    """
    names = {c for c in class_names
             if MIN_SUFFIX_LEN <= len(c) <= MAX_LEN and c not in STOP_HEAD}

    # 1) parent 후보별로 그 후보를 접미로 갖는 child 수집 (인덱스 조회, pairwise 아님).
    parent_to_children: dict[str, set[str]] = defaultdict(set)
    for child in names:
        n = len(child)
        # child 의 가능한 진접미(자기보다 짧은 뒤쪽 부분문자열)만 조회.
        for i in range(1, n - MIN_SUFFIX_LEN + 1):
            parent = child[i:]
            if parent in names and parent not in STOP_HEAD:
                parent_to_children[parent].add(child)

    # 2) 허브 필터 — min_children 이상 거느린 parent 만 상위로 인정.
    out = []
    for parent, children in parent_to_children.items():
        if len(children) < min_children:
            continue
        for child in children:
            out.append({"parent": parent, "child": child})
    return out
