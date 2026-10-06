"""하네스 공통 스키마(TBox) — 오라클·ontokit 그래프와 질의 컴파일러가 같은 어휘를 쓴다.

관계 키는 ontokit 관계 인코더의 KLUE-RE 라벨을 정본으로 삼고(추출기가 내는 어휘),
Wikidata 속성은 정답 쪽 매핑으로만 쓴다. KLUE 에 없는 위치 관계는 `loc:` 로 둔다.
"""
import re
import unicodedata

NS = "https://w3id.org/ontoharness/"
ENT = NS + "e/"     # 개체
CLS = NS + "c/"     # 클래스(타입)
REL = NS + "r/"     # 관계
DOC = NS + "d/"     # 문서

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
SKOS_ALT = "http://www.w3.org/2004/02/skos/core#altLabel"
DESCRIBES = NS + "describes"     # 문서 → 그 문서의 표제 개체
MENTIONED_IN = NS + "mentionedIn"

# key: (한국어 이름, 설명, Wikidata 속성들, 추이적?, 역관계 key)
RELATIONS = {
    "per:place_of_birth": ("출생지", "인물이 태어난 장소", ["P19"], False, None),
    "per:place_of_death": ("사망지", "인물이 사망한 장소", ["P20"], False, None),
    "per:schools_attended": ("출신학교", "인물이 다니거나 졸업한 학교", ["P69"], False, None),
    "per:employee_of": ("소속", "인물이 일하거나 소속된 조직·팀", ["P108", "P54"], False, None),
    "per:origin": ("국적", "인물의 국적·출신 국가", ["P27"], False, None),
    "per:spouse": ("배우자", "인물의 배우자", ["P26"], False, "per:spouse"),
    "per:parents": ("부모", "인물의 부모", ["P22", "P25"], False, "per:children"),
    "per:children": ("자녀", "인물의 자녀", ["P40"], False, "per:parents"),
    "org:founded_by": ("설립자", "조직을 설립한 인물·조직", ["P112"], False, None),
    "org:place_of_headquarters": ("본사소재지", "조직의 본부·본사 위치", ["P159"], False, None),
    "loc:located_in": ("위치", "장소·조직이 속한 행정구역(상위로 추이적)", ["P131"], True, None),
    "loc:country": ("소속국가", "장소·조직이 속한 나라", ["P17"], False, None),
}
WD_TO_REL = {p: k for k, v in RELATIONS.items() for p in v[2]}
TYPE_PROPS = ("P31", "P106")   # rdf:type 으로 모델링(직업도 클래스 — 상위 직업으로 폐포 가능)

_PAREN = re.compile(r"\s*\([^)]*\)")
_PUNCT = re.compile(r"[\s·\-_.,'\"「」『』<>《》〈〉]+")


def norm(s: str) -> str:
    """라벨 정규화 — 괄호 수식어·공백·구두점 제거, NFC, 소문자."""
    s = unicodedata.normalize("NFC", s or "")
    s = _PAREN.sub("", s)
    return _PUNCT.sub("", s).lower()


def rel_uri(key: str) -> str:
    return REL + key.replace(":", "_").replace("/", "_")


def ent_uri(key: str) -> str:
    from urllib.parse import quote
    return ENT + quote(key, safe="")


def cls_uri(key: str) -> str:
    from urllib.parse import quote
    return CLS + quote(key, safe="")
