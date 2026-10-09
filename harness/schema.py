"""하네스 공통 스키마(TBox) — 오라클·ontokit 그래프와 질의 컴파일러가 같은 어휘를 쓴다.

관계 어휘·URI·정규화는 라이브러리(`ontokit.graph`)가 정본이다(추출기가 내는 KLUE-RE 라벨).
여기서는 정답 쪽 Wikidata 속성 매핑만 더한다 — 벤치 채점용이라 라이브러리에 두지 않는다.
"""
from ontokit.graph import (  # noqa: F401 — 하네스 모듈들이 S.<이름> 으로 쓴다
    CLS, DESCRIBES, DOC, ENT, MENTIONED_IN, NS, RDF_TYPE, RDFS_LABEL, RDFS_SUBCLASS, REL, SKOS_ALT,
    cls_uri, ent_uri, norm, rel_uri,
)
from ontokit.graph import RELATIONS as _REL

# 관계 key → 정답 쪽 Wikidata 속성
WIKIDATA = {
    "per:place_of_birth": ["P19"],
    "per:place_of_death": ["P20"],
    "per:schools_attended": ["P69"],
    "per:employee_of": ["P108", "P54"],
    "per:origin": ["P27"],
    "per:spouse": ["P26"],
    "per:parents": ["P22", "P25"],
    "per:children": ["P40"],
    "org:founded_by": ["P112"],
    "org:place_of_headquarters": ["P159"],
    "loc:located_in": ["P131"],
    "loc:country": ["P17"],
}
# key: (한국어 이름, 설명, Wikidata 속성들, 추이적?, 역관계 key)
RELATIONS = {k: (r.ko, r.desc, WIKIDATA[k], r.transitive, r.inverse) for k, r in _REL.items()}
WD_TO_REL = {p: k for k, v in RELATIONS.items() for p in v[2]}
TYPE_PROPS = ("P31", "P106")   # rdf:type 으로 모델링(직업도 클래스 — 상위 직업으로 폐포 가능)
