"""주입 인터페이스 — structural typing(Protocol), 상속 불필요.

XGEN(호스트)은 이 프로토콜을 만족하는 구현을 주입한다. ontokit은 인프라(Fuseki/Qdrant/
LLM provider)에 결합하지 않고 프로토콜에만 의존한다. omnifuse의 GraphStore/VectorStore/LLM에
더해, 우리 빌드측의 Extractor 프로토콜을 추가해 **빌드+검색을 하나의 키트로** 묶는다.
"""
from __future__ import annotations
from typing import Protocol, Optional, Any, Awaitable


# ── 빌드측: 추출기 (우리 이음새 = extract_from_chunk_documents) ──
class Extractor(Protocol):
    """문서 청크 → 온톨로지 4-tuple. gpt-4o 추출기도, DeterministicKoreanExtractor도
    이 프로토콜을 만족한다. XGEN pipeline은 어느 것이든 주입 가능."""

    async def extract(
        self,
        documents: dict[str, list[dict]],   # {"파일명": [{"chunk_id","chunk_text","chunk_index"}]}
        *,
        domain: str = "",
        existing: Optional[dict] = None,
    ) -> tuple[dict, dict, list, list]:
        """반환 4-tuple (하위단 계약 — 반드시 지킬 것):
        - concepts: {classes, object_properties, datatype_properties, class_hierarchy}
        - ner_entities: {doc: [{entity, class, type, source_chunks}]}
        - relations: [{subject, predicate, object, source_chunks, ...}]
        - data_properties: [{entity, property, value, source_chunks, ...}]
        ⚠️ source_chunks 태깅 필수(grounding), class_hierarchy 생성 필수(OWL/SCS).
        """
        ...


# ── 검색측: omnifuse 호환 3프로토콜 ──
class GraphStore(Protocol):
    def search_labels(self, query: str, *, limit: int = 30) -> list[tuple[Any, float]]: ...
    def class_instances(self, class_id: str, *, limit: int = 1000) -> list[Any]: ...
    def neighbors(self, node_id: str, *, hops: int = 1, limit: int = 100) -> list[tuple[str, str, str]]: ...
    def count_class(self, class_id: str) -> int: ...
    def get_node(self, node_id: str) -> Optional[Any]: ...


class VectorStore(Protocol):
    def search(self, query: str, *, limit: int = 20) -> list[tuple[Any, float]]: ...
    def fetch(self, ids: list[str]) -> list[Any]: ...


class LLM(Protocol):
    def generate(self, prompt: str, *, system: str = "", timeout: Optional[float] = None) -> str | Awaitable[str]: ...
