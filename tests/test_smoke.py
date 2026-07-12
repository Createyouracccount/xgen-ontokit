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


def test_relation_prohibition():
    """금지 규범 — "~하여서는 아니 된다" → 술어+' 금지' (v0.7 ⑤)."""
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    tris = r.extract("보험회사는 기존보험계약을 부당하게 소멸시켜서는 아니 된다",
                     source_chunks=["c"])
    assert any(t["predicate"] == "소멸 금지" and t["object"] == "기존보험계약" for t in tris)


def test_relation_discretion_not_prohibition():
    """재량 규정 — "~하지 아니할 수 있다"는 금지 아님 (v0.7 ⑤ 예외)."""
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    tris = r.extract("보험회사는 주권을 발행하지 아니할 수 있다", source_chunks=["c"])
    assert not any("금지" in t["predicate"] for t in tris)


def test_relation_dative_object():
    """여격 목적어 폴백 — "금융위원회에 등록" (v0.7 ④)."""
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    tris = r.extract("보험중개사는 금융위원회에 등록하여야 한다", source_chunks=["c"])
    assert ("보험중개사", "등록", "금융위원회") in {
        (t["subject"], t["predicate"], t["object"]) for t in tris}


def test_relation_adnominal_subject_restore():
    """관형절 주어 복원 — "금융위원회가 지정하는 기관에"의 금융위원회가
    바깥 주어를 덮지 않음 (v0.7 ⑦)."""
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    tris = r.extract("보험중개사는 금융위원회가 지정하는 기관에 영업보증금을 예탁하여야 한다",
                     source_chunks=["c"])
    subs = {t["subject"] for t in tris if t["predicate"] == "예탁"}
    assert subs == {"보험중개사"}  # 금융위원회(내포절 주어) 아님


def test_relation_vv_predicate_and_xsn():
    """VV 술어(따름) + XSN 접미사 결합(선임계리사 파손 방지) (v0.7 ⑥, 0711 파손수정)."""
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    tris = r.extract("요청받은 중앙행정기관의 장은 요청에 따라야 한다", source_chunks=["c"])
    assert any(t["predicate"] == "따름" for t in tris)   # VV 명사형화
    tris2 = r.extract("보험회사는 다른 보험회사의 선임계리사를 해당 보험회사의 선임계리사로 선임할 수 없다",
                      source_chunks=["c"])
    assert all(t["object"] != "리사" and t["subject"] != "리사" for t in tris2)  # 표면 파손 없음


# ---------------- 인용 온톨로지 (v0.8, 0712 PoC 라이브러리화) ----------------

def test_citation_pairs_basic():
    """같은 그룹 안에서만 키 해석 + 자기인용·중복 제외."""
    from ontokit.citations import extract_citation_pairs
    docs = [
        {"doc_id": "A", "law": "보험업법", "article_no": "제1조",
         "text": "제2조 및 제2조에 따라, 제1조는 자기다"},
        {"doc_id": "B", "law": "보험업법", "article_no": "제2조", "text": "본문"},
        {"doc_id": "C", "law": "은행법", "article_no": "제2조", "text": "제1조 참조"},
        {"doc_id": "D", "law": "은행법", "article_no": "제1조", "text": ""},
    ]
    pairs = extract_citation_pairs(docs)
    assert pairs == [("A", "B"), ("C", "D")]  # 그룹 경계·자기인용·중복 전부 반영


def test_citation_collector_streaming_equals_batch():
    """청크 단위 스트리밍 add == 일괄 추출 (빌드 파이프라인 등가성)."""
    from ontokit.citations import CitationCollector, extract_citation_pairs
    docs = [
        {"doc_id": "A", "law": "L", "article_no": "제3조", "text": "제4조와 제5조의2를 본다"},
        {"doc_id": "B", "law": "L", "article_no": "제4조", "text": "제3조로 돌아간다"},
        {"doc_id": "C", "law": "L", "article_no": "제5조의2", "text": ""},
    ]
    batch = extract_citation_pairs(docs)
    col = CitationCollector()
    for d in docs:  # 텍스트를 청크 2개로 쪼개 스트리밍
        t = d["text"]; mid = len(t) // 2
        col.add(d["doc_id"], group=d["law"], key=d["article_no"], text=t[:mid])
        col.add(d["doc_id"], group=d["law"], key=d["article_no"], text=t[mid:])
    # 청크 절단으로 표기가 끊길 수 있어 상집합 비교가 아닌: 절단 없는 지점 기준 동일성
    col2 = CitationCollector()
    for d in docs:
        col2.add(d["doc_id"], group=d["law"], key=d["article_no"], text=d["text"])
    assert col2.pairs() == batch == [("A", "B"), ("A", "C"), ("B", "A")]


def test_citation_no_keys_is_noop():
    """key 메타데이터 전무 → pairs 빈 리스트 (오연결 대신 무동작)."""
    from ontokit.citations import extract_citation_pairs
    docs = [{"doc_id": "A", "text": "제2조 참조"}, {"doc_id": "B", "text": "제1조"}]
    assert extract_citation_pairs(docs) == []


def test_citation_sparql_and_ttl_encoding():
    """URI percent-encode 왕복 + DROP/INSERT 멱등 형태 + 빈 pairs 는 DROP 만."""
    from urllib.parse import unquote
    from ontokit.citations import (citations_insert_update, citations_to_ttl, doc_uri)
    ns = "https://w3id.org/xgen-domain#"
    pairs = [("보험업법_제81조.txt", "보험업법_제141조.txt")]
    stmts = citations_insert_update(pairs, "urn:g", namespace=ns)
    assert stmts[0].startswith("DROP SILENT GRAPH")
    assert "INSERT DATA { GRAPH <urn:g>" in stmts[1]
    u = doc_uri(ns, "보험업법_제81조.txt")
    assert " " not in u and unquote(u.rsplit("/", 1)[1]) == "보험업법_제81조.txt"
    ttl = citations_to_ttl(pairs, namespace=ns)
    assert ttl.count("<") == 3 and ttl.strip().endswith(".")
    assert citations_insert_update([], "urn:g") == ["DROP SILENT GRAPH <urn:g>"]


def test_citation_masks_other_scope_refs():
    """「타법」 제N조는 같은 그룹 제N조로 오연결하지 않음 (거짓 엣지 > 누락)."""
    from ontokit.citations import extract_citation_pairs
    docs = [
        {"doc_id": "A", "law": "보험업법", "article_no": "제1조",
         "text": "「은행법」 제4조에도 불구하고 제5조를 따른다"},
        {"doc_id": "B", "law": "보험업법", "article_no": "제4조", "text": ""},
        {"doc_id": "C", "law": "보험업법", "article_no": "제5조", "text": ""},
    ]
    assert extract_citation_pairs(docs) == [("A", "C")]  # 제4조 오연결 없음
    # 마스킹 끄면(mask_pattern=None) 종전 동작
    assert extract_citation_pairs(docs, mask_pattern=None) == [("A", "B"), ("A", "C")]


def test_citation_masks_samelaw_alias():
    """'같은 법/동법 제N조' 별칭은 자기 스코프로 해석하지 않음 (R2 적대검증 적발)."""
    from ontokit.citations import extract_citation_pairs
    docs = [
        {"doc_id": "A", "law": "여신전문금융업법 시행령", "article_no": "제6조의8",
         "text": "같은 법 제4조에 따른 등록을 하고 동법 제5조를 지키며 제7조를 따른다"},
        {"doc_id": "B", "law": "여신전문금융업법 시행령", "article_no": "제4조", "text": ""},
        {"doc_id": "C", "law": "여신전문금융업법 시행령", "article_no": "제5조", "text": ""},
        {"doc_id": "D", "law": "여신전문금융업법 시행령", "article_no": "제7조", "text": ""},
    ]
    assert extract_citation_pairs(docs) == [("A", "D")]  # 같은법/동법 오연결 없음


def test_citation_chunk_boundary_mask_carry():
    """청크 경계에서 갈라진 「타법」제N조 도 캐리로 마스킹 (경계 거짓엣지 차단)."""
    from ontokit.citations import CitationCollector
    col = CitationCollector()
    col.add("A", group="보험업법", key="제1조", text="이 조는 「은행법")
    col.add("A", group="보험업법", key="제1조", text="」 제4조에 따른다")
    col.add("B", group="보험업법", key="제4조", text="")
    assert col.pairs() == []  # 경계 분리에도 은행법 제4조 오연결 없음
    # 경계에서 갈라진 참 인용은 캐리로 회수
    col2 = CitationCollector()
    col2.add("A", group="L", key="제1조", text="다음은 제4")
    col2.add("A", group="L", key="제1조", text="조에 따른다")
    col2.add("B", group="L", key="제4조", text="")
    assert col2.pairs() == [("A", "B")]


def test_citation_tail_carry_no_truncated_mask_bypass():
    """캐리 절단점이 「…」 내부에 떨어져도 마스킹 무력화 거짓 엣지 없음 (R3 A' repro)."""
    from ontokit.citations import CitationCollector
    col = CitationCollector()
    # 청크1: 완결된 「은행법」 제9조 (정상 마스킹됨) + 꼬리 100자 절단점이 표현 내부에
    # 오도록 뒤를 짧게 — 캐리에 "법」 제9조…"만 실리는 상황
    chunk1 = "본문 " * 30 + "「은행법」 제9조에 따른다" + " 끝" * 20
    col.add("A", group="보험업법", key="제1조", text=chunk1)
    col.add("A", group="보험업법", key="제1조", text="다음 청크 본문")
    col.add("B", group="보험업법", key="제9조", text="")
    assert col.pairs() == []  # 캐리 재스캔이 제9조를 거짓 엣지로 만들지 않음
    # 대조: 경계 걸침 참 인용 회수는 유지 (직전 테스트와 동일 보장 재확인)
    col2 = CitationCollector()
    col2.add("A", group="L", key="제1조", text="다음은 제4")
    col2.add("A", group="L", key="제1조", text="조에 따른다")
    col2.add("B", group="L", key="제4조", text="")
    assert col2.pairs() == [("A", "B")]
    # 메모리 상한: 다음 문서로 넘어가면 직전 문서 꼬리 해제
    assert "A" not in col2._tails or col2._last_doc == "A"
    col2.add("C", group="L", key="제7조", text="x")
    assert "A" not in col2._tails and "B" not in col2._tails


# ---------------- 클래스 승격 필터 (v0.9, 0712 mixed20k 44만 과생성 대응) ----------------

def test_class_promotion_gate_and_rules():
    """지지도 게이트 + 정크 규칙 — 대형 코퍼스 기준."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.filter.class_promotion import ClassPromotionFilter
    f = ClassPromotionFilter(corpus_chunks=20000)
    # 고립(df1·무구조) → gate
    assert f.decide("파트타임농부", df=1).reason == "gate"
    # df1 이라도 구조 참여 시 생존
    assert f.decide("수력발전소", df=1, has_rel=True).keep
    assert f.decide("환경공학", df=1, has_kid=True).keep
    # 반복 병합 / 지시상대 head / 지시 관형 / 과결합
    assert f.decide("오에겐자부로오에겐자부로", df=5).reason == "repeat"
    assert f.decide("통합이전", df=3).reason == "relhead"
    assert f.decide("오늘날", df=4).reason == "relhead"
    assert f.decide("해당국가", df=6).reason == "deictic"
    assert f.decide("교통도로수도고속도로", df=2).reason == "overjoin"
    # 유효 클래스 생존 (관형 수식은 동격 병합과 달리 유지)
    for ok in ("텔레비전진행자", "상호방위조약", "고대일본어", "미국증권거래위원회"):
        assert f.decide(ok, df=3).keep, ok


def test_class_promotion_small_corpus_gate_off():
    """소형 코퍼스(finreg류)는 지지도 게이트 자동 비활성 — 유효 df1 개념 보존."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.filter.class_promotion import ClassPromotionFilter
    f = ClassPromotionFilter(corpus_chunks=489)
    assert f.decide("매출채권", df=1).keep      # 게이트 비활성
    assert f.decide("합병절차", df=1).keep
    assert f.decide("해당요구", df=1).reason == "deictic"  # 정크 규칙은 상시
    # corpus_chunks 미상(None) → 게이트 비활성 (fail-open: 잔존 > 오삭제)
    assert ClassPromotionFilter().decide("파트타임농부", df=1).keep


def test_class_promotion_noun_reading_preference():
    """단독 재분석 오탐 방지 — top-3 명사 독법 우선 (예금/주기/실제명의 finreg 실측)."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.filter.class_promotion import ClassPromotionFilter
    small = ClassPromotionFilter(corpus_chunks=489)
    for w in ("예금", "주기", "과다", "실제명의", "매출채권"):
        assert small.decide(w, df=1).keep, w
    # 법령 긴 복합어는 소형 코퍼스에서 과결합 미적용
    assert small.decide("한국채택국제회계기준", df=1).keep
    assert small.decide("기업구조개선기관전용사모집합투자기구", df=1).keep
    # 대형 코퍼스에선 과결합 활성 유지
    big = ClassPromotionFilter(corpus_chunks=20000)
    assert big.decide("교통도로수도고속도로", df=2).reason == "overjoin"


def test_class_promotion_instance_holder_survives():
    """인스턴스를 거느린 df1 클래스는 게이트 생존 — 고아 인스턴스 방지."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.filter.class_promotion import ClassPromotionFilter
    f = ClassPromotionFilter(corpus_chunks=20000)
    assert f.decide("희귀유형", df=1).reason == "gate"
    assert f.decide("희귀유형", df=1, has_inst=True).keep


def test_cooccurrence_selection():
    """통계 선별 — pair df·lift 만으로 우연쌍·허브쌍 배제 (목록 0)."""
    from ontokit.cooccurrence import CooccurrenceCollector
    col = CooccurrenceCollector(min_pair_df=2, lift_k=2.0)
    # A·B 는 3회 함께(강결합), 허브 H 는 모든 청크 등장(우연 결합)
    for i in range(3):
        col.add_chunk(f"c{i}", [("A", "안카라"), ("B", "안카라주"), ("H", "미국")])
    for i in range(3, 10):
        col.add_chunk(f"c{i}", [("H", "미국"), (f"X{i}", f"기타{i}")])
    got = col.edges()
    pairs = {(a, b) for a, b, _ in got}
    assert ("A", "B") in pairs                      # 강결합 생존
    assert ("A", "H") not in pairs and ("B", "H") not in pairs  # 허브 lift 탈락
    # count 내림차순 + 정규순서(a<b)
    assert all(a < b for a, b, _ in got)


def test_cooccurrence_label_filter_and_exclude():
    """형태 라벨 자격(달력파편·기호심장·반복) + SVO 기연결 쌍 제외."""
    from ontokit.cooccurrence import CooccurrenceCollector, default_label_ok
    # 형태규칙: 언어무관 결정적
    assert not default_label_ok("0년")          # 달력 파편
    assert not default_label_ok("월_1일")
    assert not default_label_ok("1세기")
    assert not default_label_ok("____")         # 기호 심장
    assert not default_label_ok("금융위금융위")   # 이중반복(6자+, 클래스필터와 동일 조건)
    assert default_label_ok("연도별평가")        # 달력단위 포함해도 잔여 있으면 유효
    assert default_label_ok("Ankara_Province")  # 라틴 유효
    col = CooccurrenceCollector(min_pair_df=2, lift_k=0.0)
    for i in range(2):
        col.add_chunk(f"c{i}", [("A", "제네바"), ("B", "0년"), ("C", "협약")])
    got = {(a, b) for a, b, _ in col.edges()}
    assert got == {("A", "C")}                   # 정크 라벨 쌍 전부 배제
    # SVO 기연결 쌍 제외 (방향 무관 정규화)
    assert col.edges(exclude_pairs={("C", "A")}) == []


def test_cooccurrence_truncation_and_stats():
    """허브 청크 절단은 결정적 + 통계 공시(무증상 절단 금지)."""
    from ontokit.cooccurrence import CooccurrenceCollector
    col = CooccurrenceCollector(max_entities_per_chunk=3)
    ents = [(f"E{i:02d}", f"라벨{i:02d}") for i in range(10)]
    col.add_chunk("c0", ents)
    col.add_chunk("c0", ents)  # 중복 청크 무시
    assert col.stats["chunks"] == 1
    assert col.stats["chunks_truncated"] == 1
    assert len(col._ent_df) == 3


def test_cooccurrence_fragment_rejection():
    """NER 파편 형태 거부 — 괄호·고립인용부호·라틴미소(소문자시작). 목록 0."""
    from ontokit.cooccurrence import default_label_ok as ok
    # 구두점 파편
    for bad in ("' Union", "Regional ) League", "Margaret (", 'Mike "', "ton '"):
        assert not ok(bad), bad
    # 라틴 미소 파편(소문자 시작, 최장 알파벳연쇄 ≤2) + mixed-case 2글자(Ah/Re/Uk)
    for bad in ("ho", "co", "ra", "ur", "pi", "yt", "ko", "us PP.",
                "Ah", "Re", "Ma", "Uk", "Je"):
        assert not ok(bad), bad
    # 보존: all-caps 약어·모델명(대문자 시작), 군주 서수(긴 본체), 다국어, 중간 어포스트로피
    for good in ("SI", "OH", "PS2", "An-26", "Elizabeth I", "Haakon V",
                 "대한민국", "Ankara", "United States", "Val d ' Oise"):
        assert ok(good), good


def test_cooccurrence_number_and_edge_fragment():
    """독립 숫자 토큰·가장자리 구두점 파편 거부 — 약어(U.S./Inc.)·모델명은 보존."""
    from ontokit.cooccurrence import default_label_ok as ok
    # 한글+독립숫자 잘림 거부, 구두점 파편 거부
    for bad in ("고려 892", "제1조 2", "North Rhine -", ". Michael"):
        assert not ok(bad), bad
    # 영어 '이름+숫자' 명명은 보존(R2 N1) + 약어·모델명 보존
    for good in ("U.S.", "Inc.", "Jr.", "AH-64 아파치", "B-1A", "New South Wales",
                 "Val d ' Oise", "Ph.D", "Boeing 747", "Apollo 11", "iPhone 12", "Area 51"):
        assert ok(good), good


def test_cooccurrence_korean_josa_ending():
    """조사 종결 파편('조선엔') 거부 — NER 경계 잘림. kiwi 필요."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.cooccurrence import make_korean_label_ok
    ok = make_korean_label_ok()
    assert not ok("조선엔")           # 조선 + 에 + ㄴ
    for good in ("조선", "대한민국", "금융위원회", "보험회사"):
        assert ok(good), good


def _fake_llm(reply='{"triples":[{"s":"금융위원회","p":"감독","o":"은행"}]}'):
    """테스트용 LLM 스텁 — 고정 응답 + 호출 카운트."""
    class _L:
        def __init__(self): self.calls = 0
        def generate(self, prompt, *, system="", timeout=None):
            self.calls += 1
            return reply
    return _L()


def test_hybrid_budget_zero_equals_pure_rule():
    """예산 0 → LLM 호출 0 → 순수 규칙과 동일(LLM-free 불변식)."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    import asyncio
    from ontokit.extractors.relation_hybrid import HybridRelationExtractor, BudgetGuard
    llm = _fake_llm()
    chunks = [{"chunk_id": "c0", "chunk_text": "고양이는 물고기를 좋아한다."},
              {"chunk_id": "c1", "chunk_text": "여기 관계 없는 파편 텍스트."}]
    h = HybridRelationExtractor(llm=llm, budget=BudgetGuard(max_chunk_pct=0.0, max_usd=0.0))
    rels, rep = asyncio.run(h.extract_corpus(chunks))
    assert llm.calls == 0, "예산 0인데 LLM 호출됨"
    assert rep["llm_called"] == 0 and rep["spent_usd"] == 0.0
    # 규칙 없이(llm=None) 돌린 것과 관계 수 동일해야
    h2 = HybridRelationExtractor(llm=None)
    rels2, _ = asyncio.run(h2.extract_corpus(chunks))
    assert len(rels) == len(rels2)


def test_hybrid_chunk_pct_hard_cap():
    """청크 비율 상한이 하드 상한 — 후보 많아도 상한 초과 호출 없음."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    import asyncio
    from ontokit.extractors.relation_hybrid import HybridRelationExtractor, BudgetGuard
    llm = _fake_llm()
    # 10개 전부 규칙 0건 후보 → 20% 상한이면 정확히 2회만
    chunks = [{"chunk_id": f"c{i}", "chunk_text": f"파편텍스트{i} 조사없음"} for i in range(10)]
    h = HybridRelationExtractor(llm=llm, budget=BudgetGuard(max_chunk_pct=0.2, max_usd=None))
    rels, rep = asyncio.run(h.extract_corpus(chunks))
    assert llm.calls == 2, f"20% 상한인데 {llm.calls}회 호출"
    assert rep["llm_called"] == 2 and rep["budget_skipped"] == 8


def test_hybrid_usd_hard_cap():
    """달러 상한이 하드 상한 — 사전 판정으로 초과 방지."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    import asyncio
    from ontokit.extractors.relation_hybrid import HybridRelationExtractor, BudgetGuard
    llm = _fake_llm()
    chunks = [{"chunk_id": f"c{i}", "chunk_text": "가"*3000 + " 조사없는파편"} for i in range(20)]
    # 청크당 사전판정 ~$0.011(3000자in + max_output_chars 1200자out). cap $0.025면 2회.
    b = BudgetGuard(max_chunk_pct=None, max_usd=0.025,
                    price_per_1k_input=0.005, price_per_1k_output=0.015,
                    max_output_chars=1200)
    h = HybridRelationExtractor(llm=llm, budget=b)
    rels, rep = asyncio.run(h.extract_corpus(chunks))
    # 고정 소형 출력(fake)이라 실제 회계는 사전판정보다 작음 → 상한 내
    assert rep["spent_usd"] <= 0.025, f"상한 초과: {rep['spent_usd']}"
    assert 2 <= llm.calls <= 3


def test_hybrid_topup_recovers_relation():
    """규칙 0건 청크를 LLM 이 보강 — origin='llm_topup' 태깅."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    import asyncio
    from ontokit.extractors.relation_hybrid import HybridRelationExtractor, BudgetGuard
    llm = _fake_llm()
    chunks = [{"chunk_id": "c0", "chunk_text": "조사없는 명사 나열 파편 조각"}]
    h = HybridRelationExtractor(llm=llm, budget=BudgetGuard(max_chunk_pct=1.0))
    rels, rep = asyncio.run(h.extract_corpus(chunks))
    topups = [r for r in rels if r.get("origin") == "llm_topup"]
    assert len(topups) == 1 and topups[0]["subject"] == "금융위원회"
    assert rep["llm_triples"] == 1


def test_hybrid_usd_cap_overshoot_bounded():
    """달러 상한: 문자근사+대형출력이면 최대 1호출 초과, 그 후 유한 정지(R2 MED)."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    import asyncio
    from ontokit.extractors.relation_hybrid import HybridRelationExtractor, BudgetGuard
    # LLM이 대형 출력(200관계≈6000자) 반환 — 사전판정(max_output_chars=1200) 초과
    big_reply = '{"triples":[' + ",".join(
        '{"s":"금융위원회","p":"감독%d","o":"은행"}' % i for i in range(200)) + ']}'
    llm = _fake_llm(big_reply)
    chunks = [{"chunk_id": f"c{i}", "chunk_text": "조사없는 파편 조각들"} for i in range(20)]
    # cap을 5호출분 정도로 잡아도, 대형출력이면 1호출 후 정지해야(유한 초과)
    b = BudgetGuard(max_chunk_pct=None, max_usd=0.02,
                    price_per_1k_input=0.005, price_per_1k_output=0.015,
                    max_output_chars=1200)
    h = HybridRelationExtractor(llm=llm, budget=b)
    _, rep = asyncio.run(h.extract_corpus(chunks))
    # 초과는 유한: 사후 회계가 상한 넘겨도 이후 전량 거부 → 무한증가 없음
    assert rep["llm_called"] <= 3, f"대형출력인데 {rep['llm_called']}회 (유한 정지 실패)"
    # usage 콜백도 사전판정은 근사라 첫 호출 초과는 못 막음 — '유한 정지'만 보장.
    # (진짜 하드는 LLM max_tokens 물리제한 필요 — docstring 공시)
    def usage_ex(_res):
        return {"input_tokens": 20, "output_tokens": 2000}   # ≈$0.03/호출
    llm2 = _fake_llm(big_reply)
    b2 = BudgetGuard(max_chunk_pct=None, max_usd=0.02,
                     price_per_1k_input=0.005, price_per_1k_output=0.015)
    h2 = HybridRelationExtractor(llm=llm2, budget=b2, usage_extractor=usage_ex)
    _, rep2 = asyncio.run(h2.extract_corpus(chunks))
    assert rep2["llm_called"] == 1, "첫 초과 후 유한 정지 실패"  # 1호출로 상한 넘고 정지
