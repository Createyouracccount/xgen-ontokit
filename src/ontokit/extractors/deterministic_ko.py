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


class DeterministicKoreanExtractor:
    def __init__(self, kiwi=None, ner=None, domain_words: Optional[list[str]] = None,
                 en_nouns=None, en_ner=None, relation_extractor=None,
                 enable_relations: bool = True):
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
        # 한국어 관계(objectProperty) 추출 — 조사 기반 SVO. Kiwi 인스턴스 공유.
        self.relations = None
        if enable_relations:
            if relation_extractor is not None:
                self.relations = relation_extractor
            else:
                from .relation_ko import KoreanRelationExtractor
                self.relations = KoreanRelationExtractor(kiwi=self.nouns.kiwi)

    async def extract(
        self,
        documents: dict[str, list[dict]],
        *,
        domain: str = "",
        existing: Optional[dict] = None,
    ) -> tuple[dict, dict, list, list]:
        all_entities: dict[str, list] = {}
        all_relations: list = []
        all_data_props: list = []

        # 클래스 이름 → source_chunks(set) 딕셔너리 누적 — 매 청크 merge_concepts(O(T·C),
        # 내부 리스트 선형탐색까지 겹쳐 사실상 제곱)를 폐기. 청크당 O(1) dict 갱신 후
        # 루프 밖에서 1회 리스트화. existing(이어서 빌드) 클래스도 이 dict 로 흡수한다.
        class_chunks: dict[str, set] = {}
        if existing:
            for c in existing.get("classes", []):
                nm = c.get("name")
                if nm:
                    class_chunks.setdefault(nm, set()).update(c.get("source_chunks", []))

        # en 라우팅 침묵 방지(#5) — 영어 청크인데 영어 도구 미주입이면 조용히 스킵되던 것을
        # stats 로 노출. "문서 500 조용한 누락" 류 함정 재발 차단.
        skipped_en_chunks = 0

        for doc_name, chunks in documents.items():
            for ch in chunks:
                cid = ch.get("chunk_id")
                text = ch.get("chunk_text", "")
                if not text.strip():
                    continue
                sc = [cid] if cid else []
                # 청크 언어 감지 → 언어별 도구 라우팅(형태소·NER)
                lang = detect_lang(text)
                if lang == "en":
                    if self.en_nouns is None:
                        skipped_en_chunks += 1
                        continue  # 영어 도구 없음 — Kiwi 로 폴백해봐야 빈 결과, 명시적 스킵
                    nouns = self.en_nouns.compound_nouns(text)
                    ner = self.en_ner
                else:
                    nouns = self.nouns.compound_nouns(text)
                    ner = self.ner
                # ① 복합명사 → 클래스 (dict 누적, O(1))
                for n in nouns:
                    class_chunks.setdefault(n, set()).update(sc)
                # ② NER → 인스턴스 엔티티 (언어별 NER)
                if ner is not None:
                    ents = ner.entities(text, source_chunks=sc)
                    if ents:
                        all_entities.setdefault(doc_name, []).extend(ents)
                # ③ 관계(objectProperty) — 조사 기반 SVO. 한국어 청크만(영어는 조사 없음).
                if self.relations is not None and lang != "en":
                    rels = self.relations.extract(text, source_chunks=sc)
                    if rels:
                        all_relations.extend(rels)

        # 루프 밖 1회 리스트화 — merged 스키마 구성.
        merged = {
            "classes": [{"name": nm, "description": "", "parent": None,
                         "source_chunks": list(chunks)}
                        for nm, chunks in class_chunks.items()],
            "object_properties": list(existing.get("object_properties", [])) if existing else [],
            "datatype_properties": list(existing.get("datatype_properties", [])) if existing else [],
            "class_hierarchy": list(existing.get("class_hierarchy", [])) if existing else [],
        }
        if skipped_en_chunks:
            merged["skipped_en_chunks"] = skipped_en_chunks

        # ④ 계층: 전체 클래스에 접미공유 1회 (청크 경계 무관). 인덱스화+허브필터(O(N·L²)).
        #   한국어 head-final 특성으로 복합명사 접미가 상위 개념(생명보험업⊂보험업).
        #   정의문(Hearst) 계층은 실측상 노이즈가 이득을 상쇄해 미채택.
        merged["class_hierarchy"].extend(induce_suffix_hierarchy(set(class_chunks.keys())))

        return merged, all_entities, all_relations, all_data_props
