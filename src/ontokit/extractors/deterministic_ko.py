"""한국어 LLM-free 온톨로지 추출기 — Extractor 프로토콜 구현.

Kiwi 복합명사(클래스) + KoELECTRA NER(엔티티) + 접미공유(subClassOf 계층). LLM 0회.
finreg 489 실측: 4.5초/$0, 클래스 3156·subClassOf 1710. 검색 A/B에서 gpt-4o와 동일(0.947).

XGEN pipeline은 이것을 gpt-4o DocumentOntologyExtractor 대신 주입 가능(같은 4-tuple 계약).
"""
from __future__ import annotations
from typing import Optional

from ..morphology.kiwi_nouns import KiwiNounExtractor
from ..hierarchy.suffix_share import induce_suffix_hierarchy
from ..hierarchy.hearst_ko import definitional_pairs, copula_pairs
from ..utils.lang_detect import detect_lang
from .base import merge_concepts


class DeterministicKoreanExtractor:
    def __init__(self, kiwi=None, ner=None, domain_words: Optional[list[str]] = None,
                 en_nouns=None, en_ner=None, relation_extractor=None,
                 enable_relations: bool = True, enable_hearst: bool = False):
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
        # 한국어 Hearst 정의문 계층 — 접미공유가 못 잡는 이질 상위어 보완 목적.
        # ⚠️ 기본 OFF. 실측(위키 30문서 + finreg 489) 결과 두 방식 모두 노이즈가
        #   이득을 상쇄: 계사(위키체)는 정밀도 ~40%(계사가 동일성/은유/부가절도 표현),
        #   따옴표(법령체)는 "정의문 마지막 명사=상위"가 서술구조라 오탐(생명보험업⊂영업).
        #   접미공유(고순도)가 주 엔진이고 Hearst 는 순이득이 안 나 기본 비활성.
        #   향후 KorLex/CoreNet 계층 검증을 결합해 후보를 걸러내면 켤 가치가 생긴다
        #   (설계서 기법 22). 코드는 그때/정형도메인 실험을 위해 보존.
        self.enable_hearst = enable_hearst

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
        # Hearst 정의문 계층 후보(parent/child). 접미공유와 함께 마지막에 병합.
        all_hearst: list = []

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
                # ③ 관계(objectProperty) — 조사 기반 SVO. 한국어 청크만(영어는 조사 없음).
                if self.relations is not None and lang != "en":
                    rels = self.relations.extract(text, source_chunks=sc)
                    if rels:
                        all_relations.extend(rels)
                # ④ Hearst 정의문 계층 — 한국어 청크만. 법령체(따옴표)+위키체(계사).
                #   접미공유(⑤)가 접미 안 겹치는 이질 상위어를 못 잡는 것을 보완.
                if self.enable_hearst and lang != "en":
                    all_hearst.extend(
                        definitional_pairs(text, self.nouns.last_noun))
                    all_hearst.extend(copula_pairs(text, self.nouns.kiwi))
                merged = merge_concepts(merged, {
                    "classes": doc_classes, "object_properties": [],
                    "datatype_properties": [], "class_hierarchy": []})

        # ⑤ 계층 확정: 접미공유(전역 1회) + Hearst 정의문을 함께 병합.
        #   - 접미공유: child/parent 가 이미 클래스인 것끼리 접미 매칭(청크 경계 무관).
        #   - Hearst: 정의문에서 나온 parent(상위어)가 클래스 집합에 없을 수 있으므로,
        #     계층에 쓰인 상위어를 클래스로도 등록해 고아 subClassOf(하위파이프라인
        #     post_build_fixer 가 제거)로 버려지지 않게 한다.
        all_names = {c["name"] for c in merged["classes"]}
        hearst_edges = [
            h for h in all_hearst
            if h.get("parent") and h.get("child")
            and h["parent"] != h["child"]
        ]
        # Hearst 상위어/하위어를 클래스로 편입(누락분만).
        new_cls = []
        for h in hearst_edges:
            for name in (h["parent"], h["child"]):
                if name not in all_names:
                    all_names.add(name)
                    new_cls.append({"name": name, "description": "",
                                    "parent": None, "source_chunks": []})
        merged = merge_concepts(merged, {
            "classes": new_cls, "object_properties": [], "datatype_properties": [],
            "class_hierarchy": induce_suffix_hierarchy(all_names) + hearst_edges})

        return merged, all_entities, all_relations, all_data_props
