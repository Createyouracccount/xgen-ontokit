"""한국어 LLM-free 온톨로지 추출기 — Extractor 프로토콜 구현.

Kiwi 복합명사(클래스) + KoELECTRA NER(엔티티) + 접미공유(subClassOf 계층). LLM 0회.
finreg 489 실측: 4.5초/$0, 클래스 3156·subClassOf 1710. 검색 A/B에서 gpt-4o와 동일(0.947).

XGEN pipeline은 이것을 gpt-4o DocumentOntologyExtractor 대신 주입 가능(같은 4-tuple 계약).
"""
from __future__ import annotations
from typing import Optional

from ..morphology.kiwi_nouns import KiwiNounExtractor
from ..hierarchy.suffix_share import induce_suffix_hierarchy
from .base import merge_concepts


class DeterministicKoreanExtractor:
    def __init__(self, kiwi=None, ner=None, domain_words: Optional[list[str]] = None):
        """kiwi: Kiwi 인스턴스(없으면 생성, extras[korean]).
        ner: KoElectraNER 인스턴스(None이면 엔티티 추출 생략, extras[ner]).
        domain_words: 사용자사전 도메인 용어."""
        self.nouns = KiwiNounExtractor(kiwi, domain_words)
        self.ner = ner

    async def extract(
        self,
        documents: dict[str, list[dict]],
        *,
        domain: str = "",
        existing: Optional[dict] = None,
    ) -> tuple[dict, dict, list, list]:
        merged = existing or {"classes": [], "object_properties": [],
                              "datatype_properties": [], "class_hierarchy": []}
        all_entities: dict[str, list] = {}
        all_relations: list = []
        all_data_props: list = []

        for doc_name, chunks in documents.items():
            for ch in chunks:
                cid = ch.get("chunk_id")
                text = ch.get("chunk_text", "")
                if not text.strip():
                    continue
                sc = [cid] if cid else []
                # ① 복합명사 → 클래스
                nouns = self.nouns.compound_nouns(text)
                doc_classes = [{"name": n, "description": "", "parent": None,
                                "source_chunks": sc} for n in nouns]
                # ② NER → 인스턴스 엔티티
                if self.ner is not None:
                    ents = self.ner.entities(text, source_chunks=sc)
                    if ents:
                        all_entities.setdefault(doc_name, []).extend(ents)
                merged = merge_concepts(merged, {
                    "classes": doc_classes, "object_properties": [],
                    "datatype_properties": [], "class_hierarchy": []})

        # ③ 계층: 전체 클래스에 접미공유 1회 (청크 경계 무관)
        all_names = {c["name"] for c in merged["classes"]}
        merged = merge_concepts(merged, {
            "classes": [], "object_properties": [], "datatype_properties": [],
            "class_hierarchy": induce_suffix_hierarchy(all_names)})

        return merged, all_entities, all_relations, all_data_props
