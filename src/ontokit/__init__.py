"""xgen-ontokit — XGEN 온톨로지 빌드·검색 개선 키트.

이번 세션(2026-07)에서 확인·수정한 온톨로지 개선을 하나로 말아 XGEN에 주입하는 라이브러리.
빌드(LLM-free 한국어 추출)와 검색(계층 전수열거·랭킹 floor) 개선을 함께 담는다.

빌드:
    from ontokit import DeterministicKoreanExtractor
    ext = DeterministicKoreanExtractor(domain_words=[...])
    concepts, entities, relations, data = await ext.extract(documents)

검색 개선:
    from ontokit.search import class_instances_triple, blend_score

프로토콜(주입 인터페이스):
    from ontokit import Extractor, GraphStore, VectorStore, LLM
"""
from .protocols import Extractor, GraphStore, VectorStore, LLM
from .extractors.deterministic_ko import DeterministicKoreanExtractor
from .extractors.base import merge_concepts
from .dedup.deterministic import DeterministicDedup
from .owl.generator import DeterministicOWLGenerator
from .builder.ontology_builder import OntologyBuilder
from .morphology.en_nouns import EnglishNounExtractor
from .ner.english import EnglishNER
from .utils.lang_detect import detect_lang
from .citations import (
    CitationCollector, extract_citation_pairs, citations_to_ttl,
    citations_insert_update, doc_uri, KO_LAW_ARTICLE,
)
from .filter.class_promotion import ClassPromotionFilter, PromotionDecision
from .cooccurrence import CooccurrenceCollector, default_label_ok, make_korean_label_ok

__version__ = "0.13.1"
__all__ = [
    "Extractor", "GraphStore", "VectorStore", "LLM",
    "DeterministicKoreanExtractor", "merge_concepts",
    "DeterministicDedup", "DeterministicOWLGenerator", "OntologyBuilder",
    "EnglishNounExtractor", "EnglishNER", "detect_lang",
    "CitationCollector", "extract_citation_pairs", "citations_to_ttl",
    "citations_insert_update", "doc_uri", "KO_LAW_ARTICLE",
    "ClassPromotionFilter", "PromotionDecision",
    "CooccurrenceCollector", "default_label_ok", "make_korean_label_ok",
]
