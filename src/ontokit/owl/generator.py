"""결정적 OWL/TTL 생성 — 번역 없이 한국어 URI. LLM 0회. extras[owl]=rdflib.

XGEN OWLGenerator 이식(262줄) — 단, _translate_batch(LLM 번역) 완전 제거.
원본의 _get_english_uri fallback(한글 그대로)이 이미 있어 번역 없이 완결.
대용량서 번역 O(N) 콜(800만=103만콜) 병목 제거. concepts → OWL/TTL 순수 변환.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List

XSD_TYPE_MAP = {
    "xsd:string": "string", "xsd:decimal": "decimal", "xsd:integer": "integer",
    "xsd:float": "float", "xsd:double": "double", "xsd:boolean": "boolean",
    "xsd:date": "date", "xsd:dateTime": "dateTime",
    "string": "string", "decimal": "decimal", "integer": "integer",
}

_HANGUL_RE = re.compile(r"[가-힣]")


def label_lang(text: str) -> str:
    """라벨 언어태그 자동판정 — 한글 포함이면 ko, 아니면 en.

    v0.5 까지 전 라벨 lang="ko" 하드코딩이라 영어 클래스도 "…"@ko 로 출력 —
    RDF 의미론 오류 + FILTER(lang(?l)='en') 질의 전멸(0710 실측). 이중언어 필수."""
    return "ko" if _HANGUL_RE.search(text or "") else "en"


def clean_korean_name(raw: str) -> str:
    if not raw:
        return ""
    name = re.sub(r"(range|domain|type|class):", "", raw) if ":" in raw else raw
    name = re.sub(r"<[^>]*>", "", name)
    name = re.sub(r"[():/,\[\]{}<>\"'\\]", "", name)
    return name.strip()


class DeterministicOWLGenerator:
    """rdflib 기반 OWL/TTL 생성 — 번역 없이 한국어 URI(safe)."""

    NAMESPACE_URI = "https://w3id.org/xgen-domain#"

    def __init__(self, namespace_uri: str | None = None,
                 disjoint_max_siblings: int = 12):
        self.NAMESPACE_URI = namespace_uri or self.NAMESPACE_URI
        # disjoint 는 O(형제²) — 대용량 허브서 폭발 방지 상한. 0=disjoint 전면 생략.
        self.disjoint_max_siblings = disjoint_max_siblings

    def _uri(self, korean_name: str) -> str:
        """번역 없이 한글 그대로 안전 URI(원본 _get_english_uri fallback)."""
        safe = re.sub(r"[^가-힣a-zA-Z0-9]", "", korean_name)
        return safe if safe else "Unknown"

    def generate(self, concepts: Dict[str, Any], domain_name: str = "xgen-domain") -> Dict[str, Any]:
        from rdflib import Graph, Namespace, Literal, URIRef  # extras[owl]
        from rdflib.namespace import RDF, RDFS, OWL, XSD

        classes = concepts.get("classes", [])
        obj_props = concepts.get("object_properties", [])
        data_props = concepts.get("datatype_properties", [])
        hierarchy = concepts.get("class_hierarchy", [])

        g = Graph()
        ns = Namespace(self.NAMESPACE_URI)
        g.bind("", ns); g.bind("owl", OWL); g.bind("rdfs", RDFS); g.bind("xsd", XSD)

        onto = URIRef(self.NAMESPACE_URI.rstrip("#"))
        g.add((onto, RDF.type, OWL.Ontology))
        g.add((onto, RDFS.label, Literal(domain_name, lang=label_lang(domain_name))))

        class_uris = {}
        for cls in classes:
            name = clean_korean_name(cls.get("name", ""))
            if not name:
                continue
            uri = ns[self._uri(name)]
            class_uris[name] = uri
            g.add((uri, RDF.type, OWL.Class))
            g.add((uri, RDFS.label, Literal(name, lang=label_lang(name))))
            if cls.get("description"):
                g.add((uri, RDFS.comment, Literal(cls["description"], lang=label_lang(cls["description"]))))

        for h in hierarchy:
            p = class_uris.get(clean_korean_name(h.get("parent", "")))
            c = class_uris.get(clean_korean_name(h.get("child", "")))
            if p and c:
                g.add((c, RDFS.subClassOf, p))
        for cls in classes:
            name = clean_korean_name(cls.get("name", ""))
            parent = cls.get("parent")
            if parent and name in class_uris:
                pu = class_uris.get(clean_korean_name(parent))
                if pu:
                    g.add((class_uris[name], RDFS.subClassOf, pu))

        # Disjoint (형제) — O(형제²) 라 대용량 허브(자식 수백)서 트리플 폭발/OOM.
        # 형제 수 상한으로 방어(disjoint_max_siblings, 기본 12). 초과 허브는 disjoint 생략.
        pc: Dict[str, List] = {}
        for h in hierarchy:
            cu = class_uris.get(clean_korean_name(h.get("child", "")))
            p = clean_korean_name(h.get("parent", ""))
            if p and cu:
                pc.setdefault(p, []).append(cu)
        for sibs in pc.values():
            if 1 < len(sibs) <= self.disjoint_max_siblings:
                for i, s1 in enumerate(sibs):
                    for s2 in sibs[i + 1:]:
                        g.add((s1, OWL.disjointWith, s2))

        for prop in obj_props:
            name = clean_korean_name(prop.get("name", ""))
            if not name:
                continue
            uri = ns[self._uri(name)]
            g.add((uri, RDF.type, OWL.ObjectProperty))
            g.add((uri, RDFS.label, Literal(name, lang=label_lang(name))))
            d = clean_korean_name(prop.get("domain", "")); r = clean_korean_name(prop.get("range", ""))
            if d in class_uris:
                g.add((uri, RDFS.domain, class_uris[d]))
            if r in class_uris:
                g.add((uri, RDFS.range, class_uris[r]))

        for prop in data_props:
            name = clean_korean_name(prop.get("name", ""))
            if not name:
                continue
            uri = ns[self._uri(name)]
            g.add((uri, RDF.type, OWL.DatatypeProperty))
            g.add((uri, RDFS.label, Literal(name, lang=label_lang(name))))
            d = clean_korean_name(prop.get("domain", ""))
            if d in class_uris:
                g.add((uri, RDFS.domain, class_uris[d]))
            g.add((uri, RDFS.range, XSD[XSD_TYPE_MAP.get(prop.get("range", "xsd:string"), "string")]))

        return {
            "owl_content": g.serialize(format="xml"),
            "ttl_content": g.serialize(format="turtle"),
            "class_count": len(classes),
            "property_count": len(obj_props) + len(data_props),
            "translations": {},   # 번역 없음(한국어 URI)
        }

    async def generate_async(self, concepts: Dict[str, Any],
                             domain_name: str = "xgen-domain") -> Dict[str, Any]:
        """XGEN owl_generator.generate(async) 인터페이스 호환 래퍼."""
        return self.generate(concepts, domain_name=domain_name)
