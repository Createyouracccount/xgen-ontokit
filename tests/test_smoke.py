"""스모크 — 라이브러리 import·핵심 동작 검증 (코어는 의존성 0, 빌드는 extras[korean])."""
import asyncio

import pytest


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
    """접미 공유 계층 — 인덱스화 + 허브필터. 순수 파이썬, 의존성 0."""
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    names = {"보험업", "생명보험업", "손해보험업", "자동차보험업",
             "회사", "자회사", "모회사"}
    hier = induce_suffix_hierarchy(names)
    pairs = {(h["parent"], h["child"]) for h in hier}
    # 허브(자식 ≥2)만 상위로 인정 — 보험업(3자식)·회사(2자식) 인정
    assert ("보험업", "생명보험업") in pairs
    assert ("보험업", "손해보험업") in pairs
    assert ("회사", "자회사") in pairs
    assert ("회사", "모회사") in pairs


def test_suffix_hub_filter():
    """허브 필터 — 자식 1개나 1글자 접미 오탐은 제외(#2)."""
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    # '학'(1글자)·'가' 접미 오탐 + 자식 1개(주주/대주주)는 걸러져야
    names = {"대학", "과학", "학", "국가", "전문가", "주주", "대주주"}
    pairs = {(h["parent"], h["child"]) for h in induce_suffix_hierarchy(names)}
    assert ("학", "대학") not in pairs      # 1글자 접미 오탐
    assert ("가", "국가") not in pairs      # 형태소 경계 무시 오탐
    assert ("주주", "대주주") not in pairs  # 허브 미달(자식 1개)


def test_suffix_hierarchy_scale():
    """수십만 클래스 선형 — O(N²) 재발 방지 회귀(#1)."""
    import time
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    names = {f"{p}보험업" for p in
             ("생명", "손해", "자동차", "화재", "해상", "재", "여행", "상해")}
    names |= {f"항목{i}단위" for i in range(50000)}  # 대량 잡음
    t = time.time()
    induce_suffix_hierarchy(names)
    assert time.time() - t < 3.0  # 5만+ 클래스가 초 단위(구 O(N²)면 분 단위)


def test_lang_detect():
    """언어 감지 — 한글/영문 비율, 의존성 0."""
    from ontokit.utils.lang_detect import detect_lang
    assert detect_lang("삼성전자는 서울에 있다") == "ko"
    assert detect_lang("Samsung is in Seoul") == "en"
    assert detect_lang("") == "en"


def test_extract_korean():
    """한국어 추출 E2E — extras[korean] 필요."""
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(
            domain_words=["여신전문금융업", "신용카드업", "보험업", "생명보험업"],
            enable_relations=False)
    except ImportError:
        pytest.skip("kiwipiepy 없으면 skip")
    docs = {"보험업법": [{"chunk_id": "c1", "chunk_index": 0,
             "chunk_text": "생명보험업과 손해보험업은 보험업의 종류이다. 신용카드업은 여신전문금융업에 속한다."}]}
    concepts, ents, rels, dps = asyncio.run(ext.extract(docs))
    names = {c["name"] for c in concepts["classes"]}
    assert "생명보험업" in names  # 복합명사 클래스 추출


def test_extract_dict_accumulation():
    """merge dict 누적 — 같은 클래스가 여러 청크에 오면 source_chunks 합산(#3)."""
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(enable_relations=False)
    except ImportError:
        pytest.skip("의존성 미설치")
    docs = {"d": [
        {"chunk_id": "c1", "chunk_text": "생명보험업은 중요하다"},
        {"chunk_id": "c2", "chunk_text": "생명보험업을 감독한다"},
    ]}
    concepts, *_ = asyncio.run(ext.extract(docs))
    found = False
    for c in concepts["classes"]:
        if c["name"] == "생명보험업":
            assert set(c["source_chunks"]) == {"c1", "c2"}  # 두 청크 합산
            found = True
    assert found


def test_extract_en_skip_stats():
    """영어 청크 침묵 스킵 방지 — skipped_en_chunks 노출(#5)."""
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(en_nouns=None, en_ner=None,
                                           enable_relations=False,
                                           auto_english=False)  # auto-wire 끄고 스킵경로 검증
    except ImportError:
        pytest.skip("의존성 미설치")
    docs = {"d": [{"chunk_id": "c1", "chunk_text": "This is an English chunk"}]}
    concepts, *_ = asyncio.run(ext.extract(docs))
    assert concepts.get("skipped_en_chunks") == 1  # 조용히 사라지지 않음


def test_relation_carryover_tag():
    """관계 주어 캐리오버 태깅 — 생략주어 추정 구분(#4)."""
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        pytest.skip("의존성 미설치")
    tris = r.extract("금융위원회는 은행을 감독하고 증권사를 관리한다",
                     source_chunks=["c1"])
    assert len(tris) == 2
    assert not tris[0].get("inferred_subject")   # 명시 주어
    assert tris[1].get("inferred_subject")        # 캐리오버(생략주어)


def test_dedup_rename_map():
    """결정적 dedup — 형태소 키 동일한 클래스 병합 맵(#6 커버)."""
    try:
        from ontokit.dedup.deterministic import DeterministicDedup
        d = DeterministicDedup()
    except ImportError:
        pytest.skip("의존성 미설치")
    concepts = {"classes": [{"name": "보험업"}, {"name": "보험업"}],
                "object_properties": [], "datatype_properties": [],
                "class_hierarchy": []}
    rename = d.compute_rename_map(concepts, {})
    assert isinstance(rename, dict)  # 실패 없이 맵 반환


def test_suffix_hierarchy_english():
    """영어 계층 — 공백형은 단어 접미·대소문자 무시, 단어경계 보호(v0.6 #1). 의존성 0."""
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    names = {"Life insurance business", "General Insurance Business",
             "insurance business", "Reinsurance business"}
    pairs = {(h["parent"], h["child"]) for h in induce_suffix_hierarchy(names)}
    assert ("insurance business", "Life insurance business") in pairs
    assert ("insurance business", "General Insurance Business") in pairs  # 대소문자 무시
    # 단어경계 보호 — reinsurance 는 insurance 의 문자접미지만 단어접미 아님
    assert ("insurance business", "Reinsurance business") not in pairs


def test_extract_mixed_language_chunk():
    """혼합 청크 이중 추출 — 한국어 우세 청크의 영어 용어 보존(v0.6 #2)."""
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(enable_relations=False)  # auto-wire(#4)
    except ImportError:
        pytest.skip("kiwipiepy 미설치")
    if ext.en_nouns is None:
        pytest.skip("nltk 미설치 — 스킵")
    docs = {"d": [{"chunk_id": "c1",
                   "chunk_text": "금융위원회는 Basel Committee 권고에 따라 생명보험업을 검토한다"}]}
    concepts, *_ = asyncio.run(ext.extract(docs))
    names = {c["name"] for c in concepts["classes"]}
    assert "생명보험업" in names                 # 한국어 유지
    assert any("Basel" in n for n in names)     # 영어 용어 생존 (v0.5=통째 소실)


def test_owl_label_lang():
    """OWL 라벨 언어태그 — 한글=@ko, 영어=@en 자동판정(v0.6 #3). extras[owl]."""
    try:
        import rdflib  # noqa: F401
    except ImportError:
        pytest.skip("의존성 미설치")
    from ontokit.owl.generator import DeterministicOWLGenerator
    gen = DeterministicOWLGenerator()
    concepts = {"classes": [
        {"name": "생명보험업", "description": ""},
        {"name": "Life insurance business", "description": ""}],
        "object_properties": [], "datatype_properties": [], "class_hierarchy": []}
    ttl = gen.generate(concepts)["ttl_content"]
    assert '"생명보험업"@ko' in ttl
    assert '"Life insurance business"@en' in ttl


def test_auto_english_wiring():
    """auto-wire — nltk 설치 시 en_nouns 자동 배선, auto_english=False 로 끔(v0.6 #4)."""
    import importlib.util
    if importlib.util.find_spec("nltk") is None:
        pytest.skip("nltk 미설치 — 스킵")
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(enable_relations=False)
    except ImportError:
        pytest.skip("kiwipiepy 미설치")
    assert ext.en_nouns is not None
    ext_off = DeterministicKoreanExtractor(enable_relations=False, auto_english=False)
    assert ext_off.en_nouns is None


def test_suffix_case_collision_deterministic():
    """대소문자 충돌 결정성 — 사전순 최소 표면형 승자 + 정렬 방출(v0.6.1).

    v0.6.0 결함: set 순회(해시시드 의존)라 실행마다 다른 표면형이 parent 로 살아남음."""
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    names = {"Insurance Business", "insurance business",
             "Life insurance business", "General Insurance Business"}
    pairs = [(h["parent"], h["child"]) for h in induce_suffix_hierarchy(names)]
    assert pairs and all(p == "Insurance Business" for p, _ in pairs)  # 사전순 최소 승자
    assert pairs == sorted(pairs)  # 방출 순서도 결정적


def test_hierarchy_no_dup_on_incremental_rebuild():
    """이어빌드 hierarchy 중복 증식 방지 — 패스마다 2→4→6 선형증식하던 결함(v0.6.1).

    중복 pair 는 OWL disjoint 형제 리스트를 오염시켜 자기-disjoint(unsatisfiable)까지 유발."""
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(enable_relations=False, auto_english=False)
    except ImportError:
        pytest.skip("의존성 미설치")
    existing = {
        "classes": [{"name": n, "description": "", "parent": None, "source_chunks": ["c1"]}
                    for n in ("생명보험업", "손해보험업", "보험업")],
        "object_properties": [], "datatype_properties": [],
        "class_hierarchy": [{"parent": "보험업", "child": "생명보험업"},
                            {"parent": "보험업", "child": "손해보험업"}],
    }
    docs = {"e": [{"chunk_id": "c9", "chunk_text": "무관한 화학 물질 문서 내용."}]}
    m, *_ = asyncio.run(ext.extract(docs, existing=existing))
    m2, *_ = asyncio.run(ext.extract(docs, existing=m))
    pairs2 = [(h["parent"], h["child"]) for h in m2["class_hierarchy"]]
    assert len(pairs2) == len(set(pairs2)) == 2  # 2패스 후에도 중복 0


def test_owl_no_self_disjoint():
    """OWL 자기-disjoint 방어 — 중복 pair 유입에도 self-disjointWith 미생성(v0.6.1)."""
    try:
        import rdflib  # noqa: F401
    except ImportError:
        pytest.skip("의존성 미설치")
    from ontokit.owl.generator import DeterministicOWLGenerator
    concepts = {"classes": [{"name": n, "description": ""} for n in ("보험업", "생명보험업", "손해보험업")],
                "object_properties": [], "datatype_properties": [],
                "class_hierarchy": [{"parent": "보험업", "child": "생명보험업"},
                                    {"parent": "보험업", "child": "손해보험업"},
                                    {"parent": "보험업", "child": "생명보험업"}]}  # 중복 주입
    ttl = DeterministicOWLGenerator().generate(concepts)["ttl_content"]
    import re as _re
    for m in _re.finditer(r"(:\S+)\s+owl:disjointWith\s+(:\S+)", ttl):
        assert m.group(1) != m.group(2)  # 자기-disjoint 없음


def test_suffix_no_latin_char_suffix():
    """라틴 단일토큰 문자접미 오탐 차단 — placement⊂cement 류(v0.6.1)."""
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    pairs = {(h["parent"], h["child"]) for h in induce_suffix_hierarchy(
        {"cement", "placement", "replacement", "insurance", "reinsurance", "coinsurance"})}
    assert not pairs  # 순수 라틴 단일토큰끼리는 문자접미 계층 생성 금지


def test_relations_survive_en_dominant_chunk():
    """en-지배 혼합청크의 한국어 문장 관계 보존 — lang 게이트→한글존재 게이트(v0.6.1)."""
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(enable_relations=True, auto_english=False)
    except ImportError:
        pytest.skip("의존성 미설치")
    en_pad = ("The Financial Services Commission supervises banks under the applicable "
              "banking statutes and issues binding guidance for licensed institutions.")
    docs = {"d": [{"chunk_id": "c1", "chunk_text": en_pad + " 금융위원회는 은행을 감독한다."}]}
    _, _, rels, _ = asyncio.run(ext.extract(docs))
    assert ("금융위원회", "감독", "은행") in {(r["subject"], r["predicate"], r["object"]) for r in rels}


def test_en_nouns_rejects_hangul_tail():
    """en_nouns 토큰 fullmatch — 'Basel규제' 라틴+한글 꼬리 토큰 배출 금지(v0.6.1)."""
    import importlib.util
    if importlib.util.find_spec("nltk") is None:
        pytest.skip("의존성 미설치")
    from ontokit.morphology.en_nouns import EnglishNounExtractor
    res = EnglishNounExtractor().compound_nouns("The Basel규제 framework and Basel규제 rules")
    assert not any("규제" in n for n in res)  # 한글 혼입 클래스명 없음
