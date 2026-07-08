"""스모크 — 라이브러리 import·핵심 동작 검증 (코어는 의존성 0, 빌드는 extras[korean])."""
import asyncio


def test_import_core():
    """코어 import — 의존성 0."""
    from ontokit import (Extractor, GraphStore, VectorStore, LLM,
                         DeterministicKoreanExtractor, merge_concepts)
    from ontokit.search import class_instances_triple, blend_score
    assert DeterministicKoreanExtractor is not None


def test_search_improvements():
    """검색 개선 함수 — 의존성 0으로 동작."""
    from ontokit.search import class_instances_triple, blend_score
    # #1 subClassOf* 이행폐포
    assert "subClassOf*" in class_instances_triple(transitive=True)
    assert "subClassOf*" not in class_instances_triple(transitive=False)
    # #2 floor guard: vscore 결측 → knorm으로
    assert blend_score(None, 0.5, 0.0, 1.0) == 0.7 * 0.5 + 0.3 * 0.5
    assert blend_score(0.8, 0.5, 0.0, 1.0) == 0.7 * 0.8 + 0.3 * 0.5


def test_suffix_hierarchy():
    """접미 공유 계층 — 순수 파이썬, 의존성 0."""
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    names = {"보험업", "생명보험업", "손해보험업", "주주", "대주주", "회사", "자회사"}
    hier = induce_suffix_hierarchy(names)
    pairs = {(h["parent"], h["child"]) for h in hier}
    assert ("보험업", "생명보험업") in pairs
    assert ("보험업", "손해보험업") in pairs
    assert ("주주", "대주주") in pairs
    assert ("회사", "자회사") in pairs


def test_extract_korean():
    """한국어 추출 E2E — extras[korean] 필요."""
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(
            domain_words=["여신전문금융업", "신용카드업", "보험업", "생명보험업"])
    except ImportError:
        return  # kiwipiepy 없으면 skip
    docs = {"보험업법": [{"chunk_id": "c1", "chunk_index": 0,
             "chunk_text": "생명보험업과 손해보험업은 보험업의 종류이다. 신용카드업은 여신전문금융업에 속한다."}]}
    concepts, ents, rels, dps = asyncio.run(ext.extract(docs))
    names = {c["name"] for c in concepts["classes"]}
    assert "보험업" in names
    # 계층 유도 확인
    pairs = {(h["parent"], h["child"]) for h in concepts["class_hierarchy"]}
    assert ("보험업", "생명보험업") in pairs
