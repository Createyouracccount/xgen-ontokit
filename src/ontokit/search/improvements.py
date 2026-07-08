"""검색 개선 — XGEN multi_turn_rag.py에 이식된 개선을 라이브러리로 추출.

이번 세션 커밋(ontology-search 브랜치):
- 07f6405: _seed_classes subClassOf* 이행폐포 (계층 전수열거)
- aa282e8: 근거블록 재랭킹 vscore 결측 floor guard

XGEN은 이 함수들을 호출해 개선을 적용한다(인라인 하드코딩 대신 라이브러리 단일소스).
검색 A/B 실측: subClassOf* 회귀 0, floor guard로 vscore 결측 청크 랭킹 복원.
"""
from __future__ import annotations


# ── 개선 #1: subClassOf* 이행폐포 SPARQL (07f6405) ──
# 매칭 클래스의 하위클래스 인스턴스까지 전수 포함. `*`=zero-length path 포함이라
# flat 온톨로지는 기존과 동일(순수 superset, 회귀 0). "보험" 하위 16종 전수열거 실증.
def class_instances_triple(*, transitive: bool = True) -> str:
    """_seed_classes의 인스턴스 매칭 트리플 절 반환.
    transitive=True: subClassOf* 이행폐포(개선). False: 직접 인스턴스만(구버전)."""
    if transitive:
        return "?sub rdfs:subClassOf* ?c . ?i rdf:type ?sub . ?i rdfs:label ?il"
    return "?i rdf:type ?c . ?i rdfs:label ?il"


# ── 개선 #2: vscore 결측 floor guard (aa282e8) ──
# vscore 없는 청크는 vnorm=0 고정 시 블렌드의 70%를 통째로 잃고 랭킹 바닥에 가라앉음.
# 결측 시 knorm으로 대체해 키워드 exact-match 청크가 묻히지 않게 함.
def blend_score(vscore, knorm: float, vmin: float, vrng: float,
                *, w_vec: float = 0.7, w_key: float = 0.3) -> float:
    """근거블록 재랭킹 블렌드 점수. vscore 결측 시 knorm으로 floor."""
    if isinstance(vscore, (int, float)):
        vnorm = (vscore - vmin) / vrng if vrng else 0.0
    else:
        vnorm = knorm   # floor guard: 결측 → keyword norm
    return w_vec * vnorm + w_key * knorm
