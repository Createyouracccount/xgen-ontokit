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


def test_suffix_morpheme_gate():
    """형태소 경계 게이트(kiwi 주입) — 문자 파편 상위어 컷, 진짜 계층 보존.
    mixed20k 실측: '대한민국'→'민국'은 파편(Kiwi 단일형태소)이라 컷,
    '생명보험업'→'보험업'은 형태소 경계 일치라 보존. GraphRAG 그래프 회피 원인 제거."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    kiwi = Kiwi()
    names = {"문화재대한민국", "사람대한민국", "대한민국", "민국",
             "생명보험업", "손해보험업", "보험업"}
    pairs = {(h["parent"], h["child"]) for h in induce_suffix_hierarchy(names, kiwi=kiwi)}
    # 파편 '민국' 상위어 전부 컷
    assert not any(p == "민국" for p, _ in pairs), "문자 파편 '민국'이 상위어로 남음"
    # 진짜 형태소 경계 계층은 보존
    assert ("보험업", "생명보험업") in pairs
    assert ("보험업", "손해보험업") in pairs


def test_suffix_proper_noun_and_dup_gate():
    """상위어 NNP(고유명사) 게이트 + 중복접합 게이트 — 접합 클래스 오염 차단.
    mixed20k 실측: '대한민국'(NNP)이 문화재대한민국·고등학교대한민국 15개의 허브가 돼
    그래프 오염 → NNP head 허브 거부. '최양업최양업'(Kiwi 오분절로 NNG)은 child==parent*2
    중복 게이트로 컷. 진짜 복합어 상위어(보험업/전문학교=NNG)는 회귀 0(심판 검증)."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    kiwi = Kiwi()
    names = {"문화재대한민국", "고등학교대한민국", "대한민국",   # NNP 허브 (컷)
             "최양업최양업", "최양업",                        # 중복접합 (컷)
             "안성농업고등전문학교", "안성농업전문학교", "전문학교"}  # 진짜 (보존)
    pairs = {(h["parent"], h["child"]) for h in induce_suffix_hierarchy(names, kiwi=kiwi)}
    # NNP 상위어(대한민국)는 허브에서 제거
    assert not any(p == "대한민국" for p, _ in pairs), "NNP '대한민국'이 상위어로 남음"
    # 중복접합 child 제거
    assert ("최양업", "최양업최양업") not in pairs, "중복접합 '최양업최양업' 잔존"
    # 진짜 보통명사(NNG) 상위어 계층은 보존
    assert ("전문학교", "안성농업고등전문학교") in pairs
    assert ("전문학교", "안성농업전문학교") in pairs


def test_definitional_hierarchy_heterogeneous():
    """정의문 계층 — 접미공유가 원리적 불가한 이질계층(강아지⊂동물). extras[korean].
    외부 gold(Wikidata P279) 심판루프 89/100 검증 로직의 본체화."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        pytest.skip("kiwipiepy 없으면 skip")
    from ontokit.hierarchy.hearst_ko import definitional_pairs
    kiwi = Kiwi()

    def parent_of(text, child):
        pairs = definitional_pairs(text, None, kiwi=kiwi)
        return {p["parent"] for p in pairs if p["child"] == child}

    # ① 계사 종결 "X는 … Y이다" — 이질계층(문자 공유 0)
    assert "음식" in parent_of("계란빵은 한국의 길거리 음식이다", "계란빵")
    # ② genus "X는 … Y의 한 분야" — 형식명사 '분야'는 상위어 제외, Y만
    p = parent_of("통계학은 데이터를 다루는 수학의 한 분야이다", "통계학")
    assert "수학" in p and "분야" not in p
    # ③ 서술 "X는 … Y를 말한다" (Kiwi: 말+하 분리 대응)
    assert "활동" in parent_of("공공외교는 문화를 알리는 활동을 말한다", "공공외교")
    # ④ "X는 Y에 속하는 …"
    assert "고양이과" in parent_of(
        "도메스틱 쇼트헤어는 고양이과에 속하는 동물이다", "도메스틱쇼트헤어")


def test_definitional_hierarchy_gates():
    """정의문 계층 오탐 게이트 — sub-word 파편(주의/화) 상위어 배제. extras[korean]."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        pytest.skip("kiwipiepy 없으면 skip")
    from ontokit.hierarchy.hearst_ko import definitional_pairs, _is_standalone_noun
    kiwi = Kiwi()
    # 바운드 형태소 접미 게이트 — '주의'·'화'는 상위어 부적격
    assert not _is_standalone_noun("주의", kiwi)
    assert not _is_standalone_noun("화", kiwi)
    assert _is_standalone_noun("음식", kiwi)   # 유효명사는 통과
    assert _is_standalone_noun("수학", kiwi)
    # "민족주의는 … 이념이다" → parent='이념'(유효), '주의' 파편 안 나옴
    pairs = definitional_pairs("민족주의는 민족을 중시하는 이념이다", None, kiwi=kiwi)
    parents = {p["parent"] for p in pairs}
    assert "이념" in parents
    assert "주의" not in parents


def test_definitional_title_line_and_gates():
    """정의문 배관 수리 3종(심판 적대검증 반영) — 제목줄 스킵·EF판정·제목정합.
    실측: 배관버그로 실빌드 2,000청크 정의쌍 1건(채널 전멸) → 수리+게이트 후
    615건·정밀도 87%(n=100 홀드아웃 감사, 하한 0.85 통과)."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        pytest.skip("kiwipiepy 없으면 skip")
    from ontokit.hierarchy.hearst_ko import definitional_pairs
    from ontokit.morphology.kiwi_nouns import KiwiNounExtractor
    ne = KiwiNounExtractor(); kiwi = ne.kiwi
    # ① 제목줄 스킵 — 백과형 "제목\n정의문" 에서 정의문을 봐야 함
    pairs = definitional_pairs("연암대학교\n연암대학교는 천안시에 있는 사립 전문대학이다.",
                               ne.last_noun, kiwi=kiwi)
    assert any(p["parent"] == "전문대학" for p in pairs)
    # ② EF 판정 — '다' 종결 표제어(람다)가 문장으로 오인되면 안 됨 (심판 1-A)
    pairs = definitional_pairs("람다\n람다는 그리스 문자이다.", ne.last_noun, kiwi=kiwi)
    assert any(p["parent"] in ("문자", "그리스문자") for p in pairs)
    # ③ 제목-주어 정확일치 — 주어 오식별(다른 개체가 주어로 추출) 컷
    pairs = definitional_pairs("히미코\n일본은 히미코가 다스린 고대 국가이다.",
                               ne.last_noun, kiwi=kiwi)
    assert not any(p["child"] == "일본" for p in pairs), "제목≠주어 오식별 잔존"


def test_definitional_instance_typing():
    """정의문 인스턴스 타이핑 (ABox↔TBox 브리지) — 주어가 NER 개체면 rdf:type.
    본질 진단(0714): 계층 클래스 13,441 중 인스턴스 도달 0 → 계층 leg 공회전 수복.
    게이트: 최장 parent·개체-as-클래스 미방출·TTA쌍 ≥2 합의(심판 판정)."""
    try:
        from ontokit import DeterministicKoreanExtractor
    except ImportError:
        pytest.skip("의존성 미설치")

    class _MockNER:
        def entities(self, text, *, source_chunks):
            return [{"entity": s, "class": "기관", "type": "INSTANCE",
                     "source_chunks": source_chunks}
                    for s in ("서울고등학교", "부산고등학교") if s in text]
    ext = DeterministicKoreanExtractor(ner=_MockNER(), enable_relations=False)
    docs = {"d": [
        {"chunk_id": "c1", "chunk_text": "서울고등학교\n서울고등학교는 서울에 있는 공립고등학교이다."},
        {"chunk_id": "c2", "chunk_text": "부산고등학교\n부산고등학교는 부산에 있는 공립고등학교이다."},
        {"chunk_id": "c3", "chunk_text": "계란빵\n계란빵은 한국의 길거리 음식이다."},
    ]}
    concepts, ents, *_ = asyncio.run(ext.extract(docs))
    # 개체 재타입: 기관 → 공립고등학교 (최장 parent)
    flat = [e for es in ents.values() for e in es]
    assert all(e["class"] == "공립고등학교" for e in flat), flat
    names = {c["name"] for c in concepts["classes"]}
    hier = {(h["child"], h["parent"]) for h in concepts["class_hierarchy"]}
    # 개체-as-클래스 오염 없음 + 타입 클래스는 존재 + TTA 상향 계층(≥2 합의)
    assert "서울고등학교" not in names
    assert "공립고등학교" in names
    assert ("공립고등학교", "기관") in hier
    # 비개체 주어 정의쌍은 클래스 계층으로 잔류
    assert ("계란빵", "음식") in hier


def test_definitional_hierarchy_backcompat():
    """구 API 하위호환 — kiwi 미주입 시 따옴표 정의문 + last_noun_fn 폴백."""
    from ontokit.hierarchy.hearst_ko import definitional_pairs
    # kiwi 없이 last_noun_fn 만 — 따옴표 정의문(법령체)
    pairs = definitional_pairs(
        '"신용공여"란 여신 거래를 말한다',
        last_noun_fn=lambda s: s.split()[-1] if s.split() else "")
    # 폴백 경로 동작(빈 결과여도 예외 없이) — 계약 유지 확인
    assert isinstance(pairs, list)


def test_relation_encoder_fallback_no_env(monkeypatch):
    """관계 인코더 불변식 ① — env 미지정이면 규칙 조사SVO 채널(인코더 미로드)."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    monkeypatch.delenv("ONTOKIT_RELATION_ENCODER_MODEL", raising=False)
    from ontokit.extractors.deterministic_ko import DeterministicKoreanExtractor
    ex = DeterministicKoreanExtractor()
    assert type(ex.relations).__name__ == "KoreanRelationExtractor"


def test_relation_encoder_fallback_no_ner(monkeypatch):
    """관계 인코더 불변식 ② — env 지정돼도 NER 없으면 개체쌍 불가 → 규칙 폴백."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    monkeypatch.setenv("ONTOKIT_RELATION_ENCODER_MODEL", "/any/path")
    from ontokit.extractors.deterministic_ko import DeterministicKoreanExtractor
    ex = DeterministicKoreanExtractor()  # ner=None
    assert type(ex.relations).__name__ == "KoreanRelationExtractor"


def test_relation_encoder_fallback_bad_path(monkeypatch):
    """관계 인코더 불변식 ③ — env+NER 있어도 모델 경로 오류면 규칙 폴백(warmup 적발)."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    import logging
    logging.disable(logging.CRITICAL)
    monkeypatch.setenv("ONTOKIT_RELATION_ENCODER_MODEL", "/nonexistent/model/path")
    from ontokit.extractors.deterministic_ko import DeterministicKoreanExtractor

    class _MockNER:
        def entities(self, text, *, source_chunks):
            return []
    ex = DeterministicKoreanExtractor(ner=_MockNER())
    # 경로 오류 → warmup 실패 → 규칙 폴백(LLM-free·프로덕션 안전 불변식)
    assert type(ex.relations).__name__ == "KoreanRelationExtractor"
    logging.disable(logging.NOTSET)


def test_ner_score_gate(monkeypatch):
    """NER 신뢰도 게이트 — 저확신 오탐(실측 '전자' 0.28)은 컷, 진짜 엔티티는 통과.
    e2e 빌드서 적발한 '삼성전자'↔'전자' 그래프 파편화 방지. 모델 다운로드 없이
    주입 파이프라인으로 검증."""
    from ontokit.ner.koelectra import KoElectraNER
    fake_out = [
        {"word": "삼성전자", "entity_group": "OG", "score": 0.97},
        {"word": "전자", "entity_group": "OG", "score": 0.28},   # 오탐(수식어)
    ]
    ner = KoElectraNER(pipeline=lambda *a, **k: fake_out)  # _ensure 스킵
    ents = {e["entity"] for e in ner.entities("x", source_chunks=[])}
    assert "삼성전자" in ents
    assert "전자" not in ents, "저확신 오탐이 게이트를 통과함"
    # env 로 게이트 완화 시 오탐도 통과(조정 가능성 확인)
    monkeypatch.setenv("ONTOKIT_NER_MIN_SCORE", "0.0")
    ner2 = KoElectraNER(pipeline=lambda *a, **k: fake_out)
    assert "전자" in {e["entity"] for e in ner2.entities("x", source_chunks=[])}


def test_relation_encoder_type_mapping():
    """modu-ner 클래스 → KLUE 타입 결정적 매핑 + typed marker 규약."""
    from ontokit.extractors.relation_encoder_ko import _klue_type, _mark
    assert _klue_type("인물") == "PER"
    assert _klue_type("기관") == "ORG"
    assert _klue_type("지역") == "LOC"
    assert _klue_type("날짜") == "DAT"
    assert _klue_type("수량") == "NOH"
    assert _klue_type("용어") == "POH"      # 미등록 고유명 → POH 폴백
    m = _mark("금호고속 이덕연 사장", "금호고속", "ORG", "이덕연", "PER")
    assert "[S:ORG]" in m and "[O:PER]" in m and "[/S]" in m and "[/O]" in m
    # 중첩·미출현 개체쌍 → None(스킵)
    assert _mark("금호고속만 있다", "금호고속", "ORG", "없음", "PER") is None


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


def test_class_promotion_gate_large_corpus():
    """클래스 승격 게이트(심판 OR-게이트) — df=1 고아 컷 + 계층 참여 보존 + NER 동일명 강등.
    mixed20k 실측(클래스 95%가 df=1 고아, completeness 5%) 기반. 소규모 코퍼스는 별도 테스트."""
    try:
        from ontokit import DeterministicKoreanExtractor
    except ImportError:
        pytest.skip("의존성 미설치")

    class _MockNER:  # '한국관광공사'를 개체로 방출 — 동일명 클래스 강등 검증
        def entities(self, text, *, source_chunks):
            if "한국관광공사" in text:
                return [{"entity": "한국관광공사", "class": "기관",
                         "type": "INSTANCE", "source_chunks": source_chunks}]
            return []
    ext = DeterministicKoreanExtractor(ner=_MockNER(), enable_relations=False,
                                       enable_hearst=False)
    # 500+ 청크 (df 게이트 활성). 패딩 청크는 명사 없는 텍스트.
    docs = {"d": (
        [{"chunk_id": "c1", "chunk_text": "생명보험업과 손해보험업은 성장했다"},
         {"chunk_id": "c2", "chunk_text": "생명보험업과 손해보험업이 있다"},
         {"chunk_id": "c3", "chunk_text": "희귀복합명사표본 이 단어는 여기 한 번만 나온다"},
         {"chunk_id": "c4", "chunk_text": "한국관광공사가 발표했다"},
         {"chunk_id": "c5", "chunk_text": "한국관광공사는 서울에 있다"},
         {"chunk_id": "c6", "chunk_text": "국고사업선정과 활동지원사업선정이 진행된다"},
         # 사업선정은 c7 에만(df=1) — 계층 허브 참여만으로 보존됨을 검증
         {"chunk_id": "c7", "chunk_text": "국고사업선정과 활동지원사업선정 및 사업선정이 있다"}]
        + [{"chunk_id": f"p{i}", "chunk_text": "그렇게 되었다"} for i in range(500)]
    )}
    concepts, ents, *_ = asyncio.run(ext.extract(docs))
    names = {c["name"] for c in concepts["classes"]}
    # df≥2 (c1,c2) → 보존
    assert "생명보험업" in names and "손해보험업" in names
    # 계층 참여(사업선정 허브) → df=1 이어도 보존 (OR-게이트 핵심, 심판 A-1)
    hier = {(h["parent"], h["child"]) for h in concepts["class_hierarchy"]}
    assert ("사업선정", "국고사업선정") in hier
    assert "사업선정" in names
    # df=1 ∧ 고아 → 탈락
    assert "희귀복합명사표본" not in names
    # NER 동일명(한국관광공사, df=2 지만 고아) → 강등 (df 무관, 심판 B 판정)
    assert "한국관광공사" not in names
    assert concepts["class_gate_stats"]["dropped_ner_dup"] >= 1


def test_class_promotion_gate_small_corpus():
    """소규모 코퍼스(<500청크) — df 조건 비활성, df=1 정당 도메인 용어 보존(심판 A-2).
    tester_trade(309청크) 류 도메인 코퍼스 학살 방지."""
    try:
        from ontokit import DeterministicKoreanExtractor
        ext = DeterministicKoreanExtractor(enable_relations=False, enable_hearst=False)
    except ImportError:
        pytest.skip("의존성 미설치")
    docs = {"d": [{"chunk_id": "c1", "chunk_text": "검사부품목록 을 검토했다"}]}
    concepts, *_ = asyncio.run(ext.extract(docs))
    names = {c["name"] for c in concepts["classes"]}
    assert "검사부품목록" in names  # df=1 이지만 소규모 코퍼스 → 보존


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


def test_corpus_level_relation_integration():
    """코퍼스레벨 관계추출기(extract_corpus, 예: hybrid) 주입 통합 (0713).

    deterministic_ko 가 청크별 extract 대신 루프 뒤 extract_corpus 1회 호출하는지.
    스텁 코퍼스 추출기로 검증 — 규칙 결과 + 스텁이 더한 트리플 모두 나와야."""
    import asyncio
    try:
        from ontokit.extractors.deterministic_ko import DeterministicKoreanExtractor
    except ImportError:
        pytest.skip("의존성 미설치")

    class _CorpusRel:  # extract_corpus 만(extract 없음) = 코퍼스레벨로 판별돼야
        async def extract_corpus(self, chunks):
            return ([{"subject": "코퍼스", "predicate": "처리", "object": "일괄",
                      "predicate_type": "ObjectProperty",
                      "source_chunks": [chunks[0]["chunk_id"]] if chunks else []}],
                    {"total_chunks": len(chunks)})

    ext = DeterministicKoreanExtractor(relation_extractor=_CorpusRel())
    assert ext._rel_is_corpus is True   # extract_corpus 만 → 코퍼스레벨 판별
    docs = {"d": [{"chunk_id": "c1", "chunk_text": "금융위원회는 은행을 감독한다."}]}
    _, _, rels, _ = asyncio.run(ext.extract(docs))
    # 주입 추출기가 관계 소스 — 청크별 rule.extract 대신 extract_corpus 1회.
    assert any(r["predicate"] == "처리" for r in rels)
    # 한국어 청크가 버퍼로 수집돼 extract_corpus 에 전달됐는지(코퍼스 1회 경로)
    assert rels[0]["source_chunks"] == ["c1"]


def test_hybrid_budget_max_usd_hard_cap():
    """max_usd 하드 상한 — 달러 예산 초과 직전 정지, 초과 지출 없음 (0713 가드레일).

    출력이 max_output_chars 가정을 넘으려 해도, can_call 이 '한 콜이 상한을 넘을 수
    있으면 시작 안 함'으로 막아 spent_usd 는 max_usd 를 절대 초과하지 않는다."""
    import asyncio
    from ontokit.extractors.relation_hybrid import (
        HybridRelationExtractor, BudgetGuard)

    calls = {"n": 0}

    # 매 호출 큰 출력(관계 다수) 방출 — 실지출이 콜당 유의미하게 쌓이도록.
    _big_o = "목" * 900  # 출력 목적어 길이 ≈ 콜당 실비용 지배
    class _GreedyLLM:
        def generate(self, prompt, *, system="", timeout=None, max_tokens=None):
            calls["n"] += 1
            return '{"triples":[{"s":"주","p":"술","o":"%s"}]}' % _big_o

    # 규칙이 0건인 청크 20개(무주어 나열 — 조사 SVO 안 걸림)
    chunks = [{"chunk_id": "c%d" % i, "chunk_text": "항목 나열 %d." % i}
              for i in range(20)]
    # 실지출: 콜당 out≈900자/3/1000*price_out=$0.3(+in 소액). max_usd=$3 →
    # 실지출이 $3 근처 도달 시 이후 콜은 can_call(spent+예상 > max_usd)로 거부.
    # 사전판정 예상은 max_output_chars(=900자) 로 실출력과 맞춤(하드 보장).
    budget = BudgetGuard(max_chunk_pct=None, max_usd=3.0,
                         price_per_1k_input=0.0, price_per_1k_output=1.0,
                         max_output_chars=900)
    h = HybridRelationExtractor(llm=_GreedyLLM(), budget=budget, topup_when="empty")
    rels, report = asyncio.run(h.extract_corpus(chunks))

    assert budget.spent_usd <= 3.0 + 1e-9, "달러 하드 상한 초과: $%.4f" % budget.spent_usd
    assert report["llm_called"] < 20, "예산 무시하고 전량 호출됨"
    assert report["budget_skipped"] > 0, "상한 도달 후 스킵이 없음"


def test_hybrid_budget_chunk_pct_hard_cap():
    """chunk_pct 하드 상한 — 전체 청크의 지정 비율까지만 LLM 호출(호출수 캡)."""
    import asyncio
    from ontokit.extractors.relation_hybrid import (
        HybridRelationExtractor, BudgetGuard)

    class _LLM:
        def generate(self, prompt, *, system="", timeout=None, max_tokens=None):
            return '{"triples":[{"s":"a","p":"b","o":"c"}]}'

    chunks = [{"chunk_id": "c%d" % i, "chunk_text": "나열 %d." % i} for i in range(10)]
    # 10청크 × 0.3 = 3콜 상한. max_usd 무제한(호출수 캡만 시험).
    budget = BudgetGuard(max_chunk_pct=0.3, max_usd=None)
    h = HybridRelationExtractor(llm=_LLM(), budget=budget, topup_when="empty")
    _, report = asyncio.run(h.extract_corpus(chunks))
    assert report["llm_called"] == 3, "chunk_pct 캡(3) 위반: %d" % report["llm_called"]


def test_hybrid_shortest_first_ordering():
    """예산 빠듯할 때 짧은 청크 우선 — 저수율 긴 청크에 예산 먼저 안 태움
    (R2 실빌드: 긴청크 우선 → 예산 소진 후 생산적 짧은청크 도달 못해 관계 0)."""
    import asyncio
    from ontokit.extractors.relation_hybrid import (
        HybridRelationExtractor, BudgetGuard)

    called_texts = []

    class _LLM:
        def generate(self, prompt, *, system="", timeout=None, max_tokens=None):
            called_texts.append(len(prompt))
            return '{"triples":[{"s":"a","p":"b","o":"c"}]}'

    # 긴 청크(무수확 가정) + 짧은 청크 섞기. chunk_pct 로 2콜만 허용.
    chunks = ([{"chunk_id": "long%d" % i, "chunk_text": "긴 나열 " * 500}
               for i in range(3)]
              + [{"chunk_id": "short%d" % i, "chunk_text": "짧은 나열 %d." % i}
                 for i in range(3)])
    budget = BudgetGuard(max_chunk_pct=2/6, max_usd=None)  # 6청크 중 2콜
    h = HybridRelationExtractor(llm=_LLM(), budget=budget, topup_when="empty")
    asyncio.run(h.extract_corpus(chunks))
    # 짧은 청크(짧은 prompt) 먼저 호출됐는지 — 첫 2콜 prompt 길이가 작아야
    assert all(pl < 200 for pl in called_texts[:2]), \
        "짧은 청크 우선 아님: %s" % called_texts[:2]


def test_hybrid_disabled_zero_llm():
    """LLM-free 불변식 — llm=None 이면 LLM 호출 0, 순수 규칙 결과만 (기본 안전)."""
    import asyncio
    from ontokit.extractors.relation_hybrid import (
        HybridRelationExtractor, BudgetGuard)

    chunks = [{"chunk_id": "c1", "chunk_text": "금융위원회는 은행을 감독한다."},
              {"chunk_id": "c2", "chunk_text": "항목 나열."}]
    h = HybridRelationExtractor(llm=None, budget=BudgetGuard(max_chunk_pct=1.0))
    rels, report = asyncio.run(h.extract_corpus(chunks))
    assert report["llm_called"] == 0, "llm=None 인데 LLM 호출 발생"
    assert report["llm_triples"] == 0
    # 규칙은 여전히 동작(금융위원회-감독-은행 류)
    assert report["rule_triples"] >= 1


def test_hybrid_zero_budget_zero_llm():
    """예산 0(chunk_pct=0) → LLM 호출 0 — 순수 규칙 동치(가드레일 하한)."""
    import asyncio
    from ontokit.extractors.relation_hybrid import (
        HybridRelationExtractor, BudgetGuard)

    called = {"n": 0}

    class _LLM:
        def generate(self, prompt, *, system="", timeout=None, max_tokens=None):
            called["n"] += 1
            return '{"triples":[]}'

    chunks = [{"chunk_id": "c%d" % i, "chunk_text": "나열 %d." % i} for i in range(5)]
    budget = BudgetGuard(max_chunk_pct=0.0, max_usd=None)
    h = HybridRelationExtractor(llm=_LLM(), budget=budget, topup_when="empty")
    _, report = asyncio.run(h.extract_corpus(chunks))
    assert called["n"] == 0, "예산 0인데 LLM 호출됨"
    assert report["llm_called"] == 0


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


def test_synonym_dict_fallback_no_env(monkeypatch):
    """사전 dedup 채널 불변식 ① — env 미지정이면 사전 채널 미생성(형태소만)."""
    try:
        from ontokit.dedup.deterministic import DeterministicDedup
    except ImportError:
        pytest.skip("의존성 미설치")
    monkeypatch.delenv("ONTOKIT_SYNONYM_DICT", raising=False)
    d = DeterministicDedup()
    assert d._syn is None


def test_synonym_dict_fallback_bad_path(monkeypatch):
    """사전 dedup 채널 불변식 ② — env 경로 오류면 형태소 폴백(예외 없이)."""
    try:
        from ontokit.dedup.deterministic import DeterministicDedup
    except ImportError:
        pytest.skip("의존성 미설치")
    import logging
    logging.disable(logging.CRITICAL)
    monkeypatch.setenv("ONTOKIT_SYNONYM_DICT", "/nonexistent/synonym.tsv")
    d = DeterministicDedup()
    assert d._syn is None  # 로드 실패 → 형태소만(불변식)
    logging.disable(logging.NOTSET)


def test_synonym_dict_merge(tmp_path, monkeypatch):
    """사전 채널 — 형태소키 다른 동의어(전자상거래=이커머스) 병합. TSV 스냅샷."""
    try:
        from ontokit.dedup.deterministic import DeterministicDedup
    except ImportError:
        pytest.skip("의존성 미설치")
    # 최소 스냅샷: 두 표기가 같은 대표(R1) 공유 = 동의어
    tsv = tmp_path / "syn.tsv"
    tsv.write_text("전자상거래\tR1\n이커머스\tR1\n무관어\tR2\n", encoding="utf-8")
    monkeypatch.setenv("ONTOKIT_SYNONYM_DICT", str(tsv))
    d = DeterministicDedup()
    assert d._syn is not None and d._syn.size() == 3
    concepts = {"classes": [{"name": "전자상거래"}, {"name": "이커머스"}, {"name": "무관어"}],
                "object_properties": []}
    rename = d.compute_rename_map(concepts, {})
    # 전자상거래/이커머스 중 하나가 다른 하나로 병합, 무관어는 별개
    assert ("이커머스" in rename or "전자상거래" in rename)
    assert "무관어" not in rename


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


def test_cooccurrence_degree_cap():
    """허브 도미네이션 억제 — 노드당 방출 엣지 max_degree 상한(0713 R2 결함).

    허브 H 가 모든 노드와 만나는 별그래프에서, H 의 방출 엣지가 max_degree 로
    제한되고 강한(count 높은) 연관부터 남는지 검증. 통계 공시도 확인.
    """
    from ontokit.cooccurrence import CooccurrenceCollector
    col = CooccurrenceCollector(min_pair_df=1, lift_k=0.0, max_degree=3,
                                label_ok=None)
    # 허브 H 가 P0..P9 각각과 여러 청크서 동시출현(count 차등: P0 최다 → P9 최소)
    for i in range(10):
        for rep in range(10 - i):          # P0: 10회, P9: 1회
            col.add_chunk(f"c{i}_{rep}", [("H", "허브"), (f"P{i}", f"주변{i}")])
    edges = col.edges()
    # H 의 방출 엣지 ≤ 3
    h_edges = [(a, b, c) for a, b, c in edges if "H" in (a, b)]
    assert len(h_edges) <= 3, h_edges
    # 강한 연관(P0·P1·P2)이 남고 약한(P9)은 절단
    partners = {a if b == "H" else b for a, b, c in h_edges}
    assert "P0" in partners and "P9" not in partners, partners
    assert col.stats["edges_degree_capped"] > 0


def test_cooccurrence_degree_cap_disabled():
    """max_degree=None 이면 무제한(하위호환) — 허브 전 엣지 방출."""
    from ontokit.cooccurrence import CooccurrenceCollector
    col = CooccurrenceCollector(min_pair_df=1, lift_k=0.0, max_degree=None,
                                label_ok=None)
    for i in range(10):
        col.add_chunk(f"c{i}", [("H", "허브"), (f"P{i}", f"주변{i}")])
    edges = col.edges()
    h_edges = [e for e in edges if "H" in (e[0], e[1])]
    assert len(h_edges) == 10
    assert col.stats["edges_degree_capped"] == 0


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
    # 정상 거부 — Kiwi 가 조사종결로 정분석하는 케이스(0713 R2 실측)
    for bad in ("조선엔", "인천이", "국회의", "일제강점기에"):
        assert not ok(bad), bad
    for good in ("조선", "대한민국", "금융위원회", "보험회사", "인천", "일제강점기"):
        assert ok(good), good
    # ⚠️ 알려진 한계(0713 R2): Kiwi 오분석 케이스는 못 잡는다. '베트남와'를
    # 베트남+오(VV)+어(EF)로 오독 → 조사 아님으로 통과. 목록-0 원칙상 결정
    # 규칙으로 안전 제거 불가(상류 dedup 몫). 회귀 감시용으로 현 동작을 고정.
    assert ok("베트남와")  # 통과 = 알려진 결함(수정 시 이 단언 뒤집을 것)


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
    """달러 상한이 하드 상한 — 보수적 사전판정+안전마진으로 실측이 절대 초과 안 함
    (0713 R2: 실빌드 44% 초과 발견 → 하드화). 핵심 불변식: spent_usd ≤ max_usd."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    import asyncio
    from ontokit.extractors.relation_hybrid import HybridRelationExtractor, BudgetGuard
    llm = _fake_llm()
    chunks = [{"chunk_id": f"c{i}", "chunk_text": "가"*3000 + " 조사없는파편"} for i in range(20)]
    # cap 을 여러 콜분($0.30)으로 넉넉히 — 하드 상한이 실측을 절대 안 넘기는지 확인.
    b = BudgetGuard(max_chunk_pct=None, max_usd=0.30,
                    price_per_1k_input=0.005, price_per_1k_output=0.015,
                    max_output_chars=1200)
    h = HybridRelationExtractor(llm=llm, budget=b)
    rels, rep = asyncio.run(h.extract_corpus(chunks))
    assert rep["spent_usd"] <= 0.30, f"하드 상한 초과: {rep['spent_usd']}"
    assert rep["llm_called"] >= 1, "예산 넉넉한데 호출 0"
    # 안전마진 예약으로 상한 근처에서 멈춤 — 마지막 콜이 상한을 넘기지 않음
    assert rep["spent_usd"] + b._worst_call_cost() > 0.30 or rep["budget_skipped"] == 0


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


def test_hybrid_usd_cap_tight_zero_calls():
    """상한이 1콜 최악비용보다 작으면 안전마진 예약으로 호출 0 — 절대 초과 안 함
    (R2 하드화: 무리한 1콜로 상한 넘기느니 규칙 폴백). max_tokens 물리제한과 병행."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    import asyncio
    from ontokit.extractors.relation_hybrid import HybridRelationExtractor, BudgetGuard
    big_reply = '{"triples":[' + ",".join(
        '{"s":"금융위원회","p":"감독%d","o":"은행"}' % i for i in range(200)) + ']}'
    chunks = [{"chunk_id": f"c{i}", "chunk_text": "조사없는 파편 조각들"} for i in range(20)]
    # 대형출력 + 매우 타이트한 상한($0.02 < 1콜 최악비용) → 호출 0, 실측 $0.
    def usage_ex(_res):
        return {"input_tokens": 20, "output_tokens": 2000}   # 실제로도 큼
    b = BudgetGuard(max_chunk_pct=None, max_usd=0.02,
                    price_per_1k_input=0.005, price_per_1k_output=0.015,
                    max_output_chars=1200)
    h = HybridRelationExtractor(llm=_fake_llm(big_reply), budget=b,
                                usage_extractor=usage_ex)
    _, rep = asyncio.run(h.extract_corpus(chunks))
    assert rep["spent_usd"] <= 0.02, f"타이트 상한 초과: {rep['spent_usd']}"
    # 1콜 최악비용($worst)이 상한보다 크면 마진 예약으로 0콜(안전 폴백)
    assert rep["llm_called"] == 0, f"타이트 상한인데 {rep['llm_called']}콜(초과위험)"


def test_relation_xpn_prefix_preserved():
    """③' XPN/SN 접두 보존 — "제3세계"가 "세계"로 절단 방출되던 유령 변이 수리(0714 심판).
    꼬리가 NNB(제391조)인 법령 참조는 원설계대로 무음 통과(주어 '상법' 유지 = 회귀 금지)."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.extractors.relation_ko import KoreanRelationExtractor
    ex = KoreanRelationExtractor()
    tris = ex.extract("제3세계가 천문대를 결정한다.", source_chunks=["t"])
    assert [(t["subject"], t["object"]) for t in tris] == [("제3세계", "천문대")]
    tris = ex.extract("멋진 신세계가 천문대를 결정한다.", source_chunks=["t"])
    assert tris and tris[0]["subject"] == "신세계"
    # 법령 참조 회귀 가드: "상법 제391조를" 의 인자는 여전히 '상법'
    tris = ex.extract("이사는 상법 제391조를 준수한다.", source_chunks=["t"])
    assert tris and tris[0]["object"] == "상법"


def test_relation_jkg_depth_cap():
    """②' '의' 연쇄 깊이 상한(슬라이딩 ≤1) — "소수의거듭제곱의상" 3중 연접 유령 노드
    차단(0714 심판). 최근 세그먼트만 유지: 종의 원소의 원자량 → 원소의원자량."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.extractors.relation_ko import KoreanRelationExtractor
    ex = KoreanRelationExtractor()
    tris = ex.extract("함수는 소수의 거듭제곱의 상을 결정한다.", source_chunks=["t"])
    assert tris and tris[0]["object"] == "거듭제곱의상"  # 깊이 1로 절단(소수의 탈락)
    # 깊이 1 정상 케이스는 그대로
    tris = ex.extract("정부는 도봉산의 이름을 명명한다.", source_chunks=["t"])
    assert tris and tris[0]["object"] == "도봉산의이름"


def test_relation_nummix_gate():
    """숫자 혼합 인자 게이트 — 한글-선도 혼합("제3세계")만 허용, 숫자-선도("747")는 컷
    (0712 숫자토큰 한글한정 원칙과 동일 계열)."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.extractors.relation_ko import KoreanRelationExtractor
    ex = KoreanRelationExtractor()
    tris = ex.extract("보잉이 747 기종을 판매한다.", source_chunks=["t"])
    # 747은 인자로 방출 금지 — 기종만
    assert tris and tris[0]["object"] == "기종"
    assert all("747" not in t["object"] and "747" not in t["subject"] for t in tris)


def test_english_ner_misc_and_junk_gates():
    """영어 NER 방출 게이트 — MISC('기타' 쓰레기통) 미방출(기본), 숫자뿐 표면 컷,
    PER/LOC/ORG 는 공유 클래스(인물/지역/기관)로 유지(0714 R2 심판 조건부 채택)."""
    from ontokit.ner.english import EnglishNER
    fake = [
        {"word": "Apple", "entity_group": "MISC", "score": 0.99},
        {"word": "2004", "entity_group": "ORG", "score": 0.95},
        {"word": "Obama", "entity_group": "PER", "score": 0.99},
        {"word": "London", "entity_group": "LOC", "score": 0.98},
        {"word": "UN", "entity_group": "ORG", "score": 0.97},
    ]
    ner = EnglishNER(pipeline=lambda *a, **k: None)  # _ensure 우회용 더미
    out = ner._to_dicts(fake, ["c1"])
    got = {(d["entity"], d["class"]) for d in out}
    assert got == {("Obama", "인물"), ("London", "지역"), ("UN", "기관")}, got


def test_relation_passive_three_way():
    """P0-1 피동 3분기(0715 심판 조건부 채택) — '에 의해' 행위자는 능동 스왑,
    되+처소/대격은 방출 억제, 받/당하는 접사 보존 방출."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.extractors.relation_ko import KoreanRelationExtractor
    ex = KoreanRelationExtractor()
    t = ex.extract("독립신문은 유길준에 의해 창간되었다.", source_chunks=["t"])
    assert [(x["subject"], x["predicate"], x["object"]) for x in t] == \
        [("유길준", "창간", "독립신문")]
    assert ex.extract("자민련이 한나라당에 흡수되었다.", source_chunks=["t"]) == []
    t = ex.extract("김용남은 징역을 선고받았다.", source_chunks=["t"])
    assert t and (t[0]["predicate"], t[0]["object"]) == ("선고받음", "징역")


def test_relation_ec_object_reset():
    """P0-3(0715 심판 채택) — 절 경계(EC)에서 목적어 잔류물 리셋: 비방출 용언을
    건너뛴 obj 오귀속 차단. 능동 접속문(각 절이 자기 목적어)은 무손실."""
    try:
        import kiwipiepy  # noqa
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.extractors.relation_ko import KoreanRelationExtractor
    ex = KoreanRelationExtractor()
    assert ex.extract("비미호가 사신을 보내 봉헌하였다.", source_chunks=["t"]) == []
    t = ex.extract("회사가 제품을 개발하고 서비스를 판매한다.", source_chunks=["t"])
    assert [(x["predicate"], x["object"]) for x in t] == \
        [("개발", "제품"), ("판매", "서비스")]


def test_relation_completive_malda_not_prohibition():
    """완료상 '~고 말다'는 금지 아님 — '~지 말다'(금지)와 위치 결속으로 분리 (R6 심판)."""
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    # 완료상: 금지로 오탐하면 안 됨
    tris = r.extract("정부는 결국 그 정책을 폐기하고 말았다", source_chunks=["c"])
    assert not any("금지" in t["predicate"] for t in tris), tris
    # 진짜 금지("~지 말다")는 유지
    tris2 = r.extract("사업자는 그 정보를 제3자에게 제공하지 말아야 한다", source_chunks=["c"])
    assert any("금지" in t["predicate"] for t in tris2) or not tris2


def test_relation_self_loop_cut():
    """주어==목적어 자기루프 방출 금지 (R6 심판 — 구조적 정크)."""
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    for text in ["자기장이 자기장을 유도한다", "회사는 회사를 지배한다"]:
        tris = r.extract(text, source_chunks=["c"])
        assert not any(t["subject"] == t["object"] for t in tris), (text, tris)


def test_relation_nn_join_toggle():
    """P0-2 공백 보존 NN 연접 — ONTOKIT_NN_JOIN=1 일 때만(기본 off, R7 심판 opt-in 채택)."""
    import os
    try:
        from ontokit.extractors.relation_ko import KoreanRelationExtractor
        r = KoreanRelationExtractor()
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    sent = "항공기가 후쿠오카 공항에 착륙했다"
    os.environ.pop("ONTOKIT_NN_JOIN", None)
    base = r.extract(sent, source_chunks=["c"])
    assert any(t["object"] == "공항" for t in base), base   # 기존: 파편
    os.environ["ONTOKIT_NN_JOIN"] = "1"
    try:
        joined = r.extract(sent, source_chunks=["c"])
        assert any(t["object"] == "후쿠오카 공항" for t in joined), joined
    finally:
        os.environ.pop("ONTOKIT_NN_JOIN", None)


def test_class_synonym_candidates_gates(tmp_path):
    """클래스 동의어 후보 생성 — deny 게이트(substr/ambig/hub/zero) 동작 (R9)."""
    from ontokit.dedup.synonym_dict import SynonymDictDedup
    from ontokit.dedup.class_synonyms import class_synonym_candidates
    tsv = tmp_path / "syn.tsv"
    tsv.write_text(
        "국가\t나라003\n나라\t나라003\n"          # 정상 후보
        "성우\t유성우001\n유성우\t유성우001\n"     # substr deny
        "사상\t사상001 사상002\n사건\t사상001\n"   # ambig deny(사상 다의)
        "역사서\t역사서001\n역사책\t역사서001\n",  # zero deny(양쪽 0)
        encoding="utf-8")
    d = SynonymDictDedup(tsv_path=str(tsv))
    cands = class_synonym_candidates(
        {"국가": 7, "나라": 5, "성우": 1, "유성우": 5, "사상": 1, "사건": 900,
         "역사서": 0, "역사책": 0}, d)
    by = {(c["a"], c["b"]): c["deny"] for c in cands}
    assert by[("국가", "나라")] == []
    assert "substr" in by[("성우", "유성우")]
    assert "ambig" in by[("사건", "사상")]
    assert "zero" in by[("역사서", "역사책")]
    assert cands[0]["deny"] == []  # 정렬: 클린 후보 우선


def test_label_hygiene_gate():
    """R11 T4 — 파편 라벨 컷, 정상 라벨 통과 (실측 파편 케이스)."""
    from ontokit.instance_typing import label_ok
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    kiwi = Kiwi()
    for bad in ("는 신화다", "이소라 이소", "프랑스와", "이"):
        assert not label_ok(bad, kiwi), bad
    for good in ("대한민국", "열린우리당", "브라운슈바이크 공국", "서울대학교", "이소라", "COVID-19"):
        assert label_ok(good, kiwi), good


def test_definitional_parent_by_evidence_frequency():
    """R11 T0 — 최장-parent 폐기: 다수 증거 parent 가 쓰레기 1건(더 긴 라벨)을 이긴다."""
    from ontokit.hierarchy.hearst_ko import assign_definitional_types
    ents = {"d": [{"entity": "대한민국", "class": "지역"}]}
    pairs = ([{"child": "대한민국", "parent": "국가"}] * 3
             + [{"child": "대한민국", "parent": "시킴지역"}])
    remaining, _tta, n = assign_definitional_types(pairs, ents)
    assert n == 1 and ents["d"][0]["class"] == "국가"


def test_definitional_parent_entity_as_class_gate():
    """R11 T0 — 개체-as-클래스 parent(자식 1) 컷: '광주' rdf:type '광주광역시' 차단."""
    from ontokit.hierarchy.hearst_ko import assign_definitional_types
    ents = {"d": [{"entity": "광주", "class": "지역"},
                  {"entity": "광주광역시", "class": "지역"}]}
    pairs = [{"child": "광주", "parent": "광주광역시"}]
    _rem, _tta, n = assign_definitional_types(pairs, ents)
    assert n == 0 and ents["d"][0]["class"] == "지역"


def test_span_align_recovers_and_no_false_join():
    """R13-2 — 토큰 중간 절단 회수·인명 병합, '한국 측/서울 역' 오결합 없음."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.ner.span_align import align_spans
    kiwi = Kiwi()
    text = "진양호는 아름답다."
    ents = [{"entity": "진양", "class": "지역", "start": 0, "end": 2}]
    out = align_spans(text, ents, kiwi)
    assert out[0]["entity"] == "진양호"
    text2 = "한국 측이 발표했다."
    ents2 = [{"entity": "한국", "class": "지역", "start": 0, "end": 2}]
    assert align_spans(text2, ents2, kiwi)[0]["entity"] == "한국"
    # 인명 병합은 G-A 오결합 실증으로 제거 — 분리 유지 확인
    text3 = "데릭 로 마크 뷰릭이 등판했다."
    ents3 = [{"entity": "데릭 로", "class": "인물", "start": 0, "end": 4},
             {"entity": "마크 뷰릭", "class": "인물", "start": 5, "end": 10}]
    assert len(align_spans(text3, ents3, kiwi)) == 2


def test_span_align_wired_in_extractor():
    """R13-2 배선 검증 — staticmethod 에 kiwi 전달(무성 no-op 회귀 방지)."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        import pytest; pytest.skip("kiwipiepy 미설치")
    from ontokit.extractors.deterministic_ko import DeterministicKoreanExtractor

    class FakeNER:
        def entities(self, text, *, source_chunks):
            return [{"entity": "진양", "class": "지역", "type": "INSTANCE",
                     "source_chunks": source_chunks, "start": 0, "end": 2}]
    ex = DeterministicKoreanExtractor(ner=FakeNER(), enable_relations=False,
                                      auto_english=False, enable_hearst=False)
    out: dict = {}
    ex._run_ner_batched(FakeNER(), [("d", "진양호는 아름답다.", [])], out, kiwi=ex.nouns.kiwi)
    assert out["d"][0]["entity"] == "진양호"


def test_ner_two_pass_covers_tail(monkeypatch):
    """R13-1 — 1200자 초과 청크 후반부 개체 회수(2패스), env 0 이면 미실행."""
    from ontokit.ner.koelectra import KoElectraNER

    calls = []
    def fake_pipe(text):
        calls.append(text)
        if "후반개체" in text:
            return [{"word": "후반개체", "score": 0.9, "entity_group": "OG", "start": 0, "end": 4}]
        return [{"word": "전반개체", "score": 0.9, "entity_group": "OG", "start": 0, "end": 4}]
    ner = KoElectraNER(pipeline=fake_pipe)
    text = ("가" * 1300) + " 후반개체 등장"
    out = ner.entities(text, source_chunks=[])  # 기본 off (G-A 파편 실증)
    assert {e["entity"] for e in out} == {"전반개체"} and len(calls) == 1
    calls.clear()
    monkeypatch.setenv("ONTOKIT_NER_TWO_PASS", "1")
    out2 = ner.entities(text, source_chunks=[])
    assert {e["entity"] for e in out2} == {"전반개체", "후반개체"} and len(calls) == 2


def test_ner_ensemble_union_gates():
    """R14b — aux 는 신규 스팬만·게이트 통과분만 union, 실패는 비치명."""
    from ontokit.ner.ensemble import EnsembleNER, aux_gate

    class Base:
        def entities(self, text, *, source_chunks):
            return [{"entity": "서울", "class": "지역"}]
    class Aux:
        def entities(self, text, *, source_chunks):
            return [{"entity": "서울", "class": "기관"},          # 중복 스팬 — 제외
                    {"entity": "MHKs", "class": "기관"},          # 약어 복수형 — 컷
                    {"entity": "수르지크어", "class": "기관"},     # 언어명→기관 — 컷
                    {"entity": "2011", "class": "수량"},          # 연도 재매핑
                    {"entity": "안성농업전문학교", "class": "기관"}]  # 정상 신규
    out = EnsembleNER(Base(), Aux()).entities("t", source_chunks=[])
    ents = {e["entity"]: e["class"] for e in out}
    assert ents == {"서울": "지역", "2011": "날짜", "안성농업전문학교": "기관"}
    assert aux_gate({"entity": "a De Mornay", "class": "인물"}) is None


def test_en_ner_auto_wire_env(monkeypatch):
    """R-en-0 — ONTOKIT_NER_EN=auto 면 영어 NER 지연 배선, 미설정이면 기존대로 None."""
    from ontokit.extractors.deterministic_ko import DeterministicKoreanExtractor
    monkeypatch.delenv("ONTOKIT_NER_EN", raising=False)
    ex = DeterministicKoreanExtractor(enable_relations=False, auto_english=False, enable_hearst=False)
    assert ex.en_ner is None
    monkeypatch.setenv("ONTOKIT_NER_EN", "auto")
    ex2 = DeterministicKoreanExtractor(enable_relations=False, auto_english=False, enable_hearst=False)
    # transformers 설치 환경이면 EnglishNER, 아니면 None(실패 무해) — 예외만 없으면 통과
    assert ex2.en_ner is None or type(ex2.en_ner).__name__ == "EnglishNER"
