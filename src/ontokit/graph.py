"""추출 결과 → 질의 가능한 RDF 그래프(투영). LLM 0회. extras[owl]=rdflib.

`OntologyBuilder.build()` / `DeterministicKoreanExtractor.extract()` 의 4-tuple 을 하나의 그래프로 싣는다.
관계 어휘는 추출기가 내는 KLUE-RE 라벨이 정본이다(KLUE 에 없는 위치 관계만 `loc:`).
`ontokit.query` 가 같은 어휘로 질의계획을 SPARQL 로 컴파일한다.

    from ontokit.graph import project
    g, stats = project({"concepts": c, "ner_entities": e, "relations": r}, docs)
    g.serialize("graph.ttl")

docs = [{"doc_id", "title"}, ...] — 문서 제목과 정규화 이름이 같은 개체를 그 문서의 표제 개체(describes)로 잇는다.
측정 하네스(harness/)에서 옮겨 온 코드다. 동작을 바꾸면 harness/docs/S01 의 동등성 증명을 다시 해야 한다.
"""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple
from urllib.parse import quote

try:
    from rdflib import Graph, Literal, URIRef
    from rdflib.namespace import RDF, RDFS, SKOS
except ImportError as e:  # 코어 의존성 0 — 그래프 기능만 extra 로
    raise ImportError("ontokit.graph 는 rdflib 가 필요합니다: pip install 'xgen-ontokit[owl]'") from e

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
NKEY = URIRef(NS + "nkey")       # 정규화 이름 키 — 질의 상수가 라벨·별칭에 연결되는 지점


class Rel(NamedTuple):
    ko: str            # 한국어 이름
    desc: str          # 설명(계획기 카탈로그에 그대로 쓰인다)
    transitive: bool
    inverse: str | None


RELATIONS = {
    "per:place_of_birth": Rel("출생지", "인물이 태어난 장소", False, None),
    "per:place_of_death": Rel("사망지", "인물이 사망한 장소", False, None),
    "per:schools_attended": Rel("출신학교", "인물이 다니거나 졸업한 학교", False, None),
    "per:employee_of": Rel("소속", "인물이 일하거나 소속된 조직·팀", False, None),
    "per:origin": Rel("국적", "인물의 국적·출신 국가", False, None),
    "per:spouse": Rel("배우자", "인물의 배우자", False, "per:spouse"),
    "per:parents": Rel("부모", "인물의 부모", False, "per:children"),
    "per:children": Rel("자녀", "인물의 자녀", False, "per:parents"),
    "org:founded_by": Rel("설립자", "조직을 설립한 인물·조직", False, None),
    "org:place_of_headquarters": Rel("본사소재지", "조직의 본부·본사 위치", False, None),
    "loc:located_in": Rel("위치", "장소·조직이 속한 행정구역(상위로 추이적)", True, None),
    "loc:country": Rel("소속국가", "장소·조직이 속한 나라", False, None),
}

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
    return ENT + quote(key, safe="")


def cls_uri(key: str) -> str:
    return CLS + quote(key, safe="")


def doc_uri(doc_id: str) -> str:
    return DOC + doc_id


def add_names(g, node, names):
    names = [n for n in dict.fromkeys(names) if n]
    if not names:
        return
    g.add((node, RDFS.label, Literal(names[0])))
    for n in names[1:]:
        g.add((node, SKOS.altLabel, Literal(n)))
    for n in names:
        k = norm(n)
        if k:
            g.add((node, NKEY, Literal(k)))


def tbox(g):
    for key, r in RELATIONS.items():
        p = URIRef(rel_uri(key))
        g.add((p, RDFS.label, Literal(r.ko)))
        g.add((p, RDFS.comment, Literal(r.desc)))


# 추출기 KLUE-RE 라벨 → 그래프 어휘. 주석이 근거다.
KLUE_MAP = {
    "per:place_of_birth": ("rel", "per:place_of_birth"),
    "per:place_of_death": ("rel", "per:place_of_death"),
    "per:schools_attended": ("rel", "per:schools_attended"),
    "per:employee_of": ("rel", "per:employee_of"),
    "per:origin": ("rel", "per:origin"),
    "per:spouse": ("rel", "per:spouse"),
    "per:parents": ("rel", "per:parents"),
    "per:children": ("rel", "per:children"),
    "org:founded_by": ("rel", "org:founded_by"),
    "org:place_of_headquarters": ("rel", "org:place_of_headquarters"),
    "org:top_members/employees": ("inv", "per:employee_of"),   # 조직→인물 = 인물 소속의 역
    "loc:located_in": ("rel", "loc:located_in"),   # ontokit 위치 서술 채널(4차, opt-in)
    "loc:country": ("rel", "loc:country"),
    "per:title": ("type", None),          # KLUE per:title = 직업·직위 → 타입으로
    "per:alternate_names": ("alias", None),
    "org:alternate_names": ("alias", None),
}


def project(raw, docs):
    """추출 결과(raw = {"concepts", "ner_entities", "relations"}) → (rdflib.Graph, 통계).

    - 개체: 정규화 이름 하나당 노드 하나(같은 이름은 문서가 달라도 합쳐진다). 언급 문서는 oh:mentionedIn
    - 타입: NER 클래스와 per:title 목적어 → rdf:type, class_hierarchy → rdfs:subClassOf
    - 관계: KLUE_MAP 에 있는 라벨만. 나머지 라벨은 버리고 stats["rels"] − stats["rels_mapped"] 로 센다
    """
    g = Graph()
    tbox(g)
    title_key = {}
    for d in docs:
        title_key.setdefault(norm(re.sub(r"\s*\([^)]*\)", "", d["title"])), []).append(d["doc_id"])

    def node(label):
        return URIRef(ent_uri("ok:" + norm(label)))

    def cls(label):
        return URIRef(cls_uri("ok:" + norm(label)))

    seen = set()

    def ensure(label):
        n = node(label)
        if n not in seen:
            seen.add(n)
            add_names(g, n, [label])
            for d in title_key.get(norm(label), []):
                g.add((URIRef(doc_uri(d)), URIRef(DESCRIBES), n))
        return n

    stats = {"entities": 0, "types": 0, "rels": 0, "rels_mapped": 0, "aliases": 0, "subclass": 0}
    for doc_id, ents in raw["ner_entities"].items():
        for e in ents:
            if not norm(e["entity"]):
                continue
            n = ensure(e["entity"])
            stats["entities"] += 1
            g.add((n, URIRef(MENTIONED_IN), URIRef(doc_uri(doc_id))))
            if e.get("class"):
                g.add((n, RDF.type, cls(e["class"])))
                add_names(g, cls(e["class"]), [e["class"]])
                stats["types"] += 1
    for h in raw["concepts"].get("class_hierarchy", []):
        if norm(h["child"]) and norm(h["parent"]):
            g.add((cls(h["child"]), RDFS.subClassOf, cls(h["parent"])))
            add_names(g, cls(h["child"]), [h["child"]])
            add_names(g, cls(h["parent"]), [h["parent"]])
            stats["subclass"] += 1
    for r in raw["relations"]:
        stats["rels"] += 1
        m = KLUE_MAP.get(r.get("relation_label") or r["predicate"])
        if not m or not norm(r["subject"]) or not norm(r["object"]):
            continue
        kind, key = m
        sn, on = ensure(r["subject"]), ensure(r["object"])
        if kind == "rel":
            g.add((sn, URIRef(rel_uri(key)), on))
        elif kind == "inv":
            g.add((on, URIRef(rel_uri(key)), sn))
        elif kind == "type":
            g.add((sn, RDF.type, cls(r["object"])))
            add_names(g, cls(r["object"]), [r["object"]])
        elif kind == "alias":
            g.add((sn, SKOS.altLabel, Literal(r["object"])))
            g.add((sn, NKEY, Literal(norm(r["object"]))))
            stats["aliases"] += 1
        stats["rels_mapped"] += 1
    return g, stats
