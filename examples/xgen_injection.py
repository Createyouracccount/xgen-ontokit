"""XGEN 주입 예시 — ExtractorFactory (기존 RerankerFactory 패턴 복제).

이 파일을 XGEN service/ontology/extractor_factory.py 로 두고,
pipeline.py:197 을 팩토리 호출로 바꾸면 config 스위치로 gpt-4o ↔ LLM-free 전환.
"""
import importlib
import logging

logger = logging.getLogger("ontology.extractor_factory")


class ExtractorFactory:
    """온톨로지 추출기 팩토리 — config ONTOLOGY_EXTRACTOR 로 provider 선택.
    XGEN service/reranker/reranker_factory.py 와 동일 패턴."""

    PROVIDER_NAMES = {
        # 기존 gpt-4o 추출기 (XGEN 내장)
        "llm_gpt4o": "service.ontology.document_ontology_extractor.DocumentOntologyExtractor",
        # LLM-free 한국어 추출기 (ontokit 라이브러리)
        "deterministic_ko": "ontokit.extractors.deterministic_ko.DeterministicKoreanExtractor",
    }

    @classmethod
    def _import(cls, path: str):
        module_name, class_name = path.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), class_name)

    @classmethod
    def create(cls, config_composer=None, llm=None, domain_words=None):
        provider = "llm_gpt4o"
        if config_composer is not None:
            try:
                provider = config_composer.get("ONTOLOGY_EXTRACTOR", "llm_gpt4o")
            except Exception:
                pass
        if provider not in cls.PROVIDER_NAMES:
            logger.warning("Unknown extractor '%s' → llm_gpt4o", provider)
            provider = "llm_gpt4o"
        klass = cls._import(cls.PROVIDER_NAMES[provider])

        if provider == "llm_gpt4o":
            return klass(llm)
        # deterministic_ko: LLM 불필요, 도메인 사전만
        # ⚠️ 어댑터: XGEN 은 extract_from_chunk_documents 를 호출하므로 얇은 래퍼 필요
        return _KoAdapter(klass(domain_words=domain_words))


class _KoAdapter:
    """ontokit 의 extract(documents, *, domain, existing) 를
    XGEN 의 extract_from_chunk_documents(documents, domain_name, existing_concepts, ...) 로 어댑트."""
    def __init__(self, ext):
        self._ext = ext

    async def extract_from_chunk_documents(self, documents, domain_name="",
                                           existing_concepts=None, **kwargs):
        return await self._ext.extract(documents, domain=domain_name,
                                       existing=existing_concepts)
