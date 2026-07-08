"""OntologyBuilder — 추출 + 결정적 dedup 을 묶은 빌드 정책. LLM 0회.

경계:
- 라이브러리(여기): documents → concepts 4-tuple (추출 + 결정적 dedup). 순수 데이터 변환.
- XGEN 인프라(호출측): OWL 생성·Fuseki 업로드·graph SPARQL 병합·DB·job 상태.

XGEN pipeline 은 이 빌더를 주입해 build() 결과(4-tuple)를 받아 OWL/Fuseki 로 넘긴다.
dedup 정책(LLM 스킵·형태소 정규화)이 전부 라이브러리에 있어 XGEN 오염 0.
"""
from __future__ import annotations
from typing import Optional

from ..extractors.deterministic_ko import DeterministicKoreanExtractor
from ..dedup.deterministic import DeterministicDedup


class OntologyBuilder:
    """LLM-free 온톨로지 빌더 — 추출기 + 결정적 dedup 조립."""

    def __init__(self, extractor=None, dedup=None, *, kiwi=None,
                 domain_words: Optional[list[str]] = None, ner=None,
                 enable_dedup: bool = True):
        self.extractor = extractor or DeterministicKoreanExtractor(
            kiwi=kiwi, ner=ner, domain_words=domain_words)
        self.dedup = dedup or (DeterministicDedup(kiwi=kiwi) if enable_dedup else None)

    async def build(
        self,
        documents: dict[str, list[dict]],
        *,
        domain: str = "",
        existing: Optional[dict] = None,
    ) -> tuple[dict, dict, list, list]:
        """documents → 완성된 (concepts, ner_entities, relations, data_properties).
        추출 후 결정적 dedup 적용. LLM 0회, 대용량 확장 가능(컨텍스트 초과 없음)."""
        concepts, entities, relations, data = await self.extractor.extract(
            documents, domain=domain, existing=existing)

        if self.dedup is not None:
            rename = self.dedup.compute_rename_map(concepts, entities)
            concepts, entities, relations, data = self.dedup.apply(
                rename, concepts, entities, relations, data)

        return concepts, entities, relations, data

    # XGEN pipeline 어댑터 — extract_from_chunk_documents 인터페이스 호환
    async def extract_from_chunk_documents(self, documents, domain_name="",
                                           existing_concepts=None, **kwargs):
        return await self.build(documents, domain=domain_name, existing=existing_concepts)
