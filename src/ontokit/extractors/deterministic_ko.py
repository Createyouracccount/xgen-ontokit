"""한국어 LLM-free 온톨로지 추출기 — Extractor 프로토콜 구현.

Kiwi 복합명사(클래스) + KoELECTRA NER(엔티티) + 접미공유(subClassOf 계층). LLM 0회.
finreg 489 실측: 4.5초/$0, 클래스 3156·subClassOf 1710. 검색 A/B에서 gpt-4o와 동일(0.947).

XGEN pipeline은 이것을 gpt-4o DocumentOntologyExtractor 대신 주입 가능(같은 4-tuple 계약).
"""
from __future__ import annotations
from typing import Optional

from ..morphology.kiwi_nouns import KiwiNounExtractor
from ..hierarchy.suffix_share import induce_suffix_hierarchy
from ..utils.lang_detect import detect_lang
from .base import merge_concepts


class DeterministicKoreanExtractor:
    def __init__(self, kiwi=None, ner=None, domain_words: Optional[list[str]] = None,
                 en_nouns=None, en_ner=None):
        """kiwi: Kiwi 인스턴스(없으면 생성, extras[korean]).
        ner: KoElectraNER 인스턴스(None이면 한국어 엔티티 추출 생략, extras[ner]).
        domain_words: 사용자사전 도메인 용어.
        en_nouns: EnglishNounExtractor(None이면 영어 명사추출 생략, extras[english]).
        en_ner: EnglishNER(None이면 영어 엔티티 추출 생략, extras[ner]).

        혼합 코퍼스(한국어+영어)에서 청크 언어를 감지해 언어별 도구로 라우팅.
        영어 도구 미주입 시 영어 청크의 클래스/인스턴스는 스킵(하위호환 —
        기존 한국어 전용 동작 유지)."""
        self.nouns = KiwiNounExtractor(kiwi, domain_words)
        self.ner = ner
        self.en_nouns = en_nouns
        self.en_ner = en_ner

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
                # 청크 언어 감지 → 언어별 도구 라우팅(형태소·NER)
                lang = detect_lang(text)
                if lang == "en" and self.en_nouns is not None:
                    nouns = self.en_nouns.compound_nouns(text)
                    ner = self.en_ner
                else:
                    # 한국어(또는 영어 도구 미주입 시 폴백) — 영어 도구 없으면
                    # 영어 청크는 Kiwi 가 한글 0개라 사실상 빈 결과(안전).
                    nouns = self.nouns.compound_nouns(text)
                    ner = self.ner
                # ① 복합명사 → 클래스
                doc_classes = [{"name": n, "description": "", "parent": None,
                                "source_chunks": sc} for n in nouns]
                # ② NER → 인스턴스 엔티티 (언어별 NER)
                if ner is not None:
                    ents = ner.entities(text, source_chunks=sc)
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
