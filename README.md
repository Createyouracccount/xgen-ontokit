# xgen-ontokit

XGEN 온톨로지 **빌드·검색 개선 키트**. 우리가 수정·주입하는 온톨로지 개선을 하나로 말아 XGEN에 주입하는 라이브러리. omnifuse(검색 라이브러리)와 상보 — 이쪽은 **빌드(LLM-free 한국어·영어 추출) + 검색 개선**을 함께 담는다.

## 언어 지원 매트릭스 (v0.9, 정직하게)

| 축 | 한국어 | 영어 | 혼합 청크 |
|---|---|---|---|
| 클래스(복합명사) | ✅ Kiwi | ✅ nltk POS (extras[english], **auto-wire**) | ✅ 이중 추출(소수언어 용어 보존) |
| 계층(subClassOf) | ✅ 문자 접미공유 | ✅ 단어 접미공유(대소문자 무시) | ✅ |
| 엔티티(NER) | ✅ KoELECTRA(주입) | ✅ dslim BERT MIT(주입) | ⚠️ 지배언어만(비용) |
| **관계(SVO)** | ✅ 조사 기반 | ❌ **미지원** — 측정 인프라 확보 후 별도 트랙 | 한국어만 |
| OWL 라벨 | `@ko` | `@en` (자동판정) | 혼재 출력 |

⚠️ 품질 검증 범위: 한국어=finreg 489 실측, 영어=구조 테스트만(코퍼스 실측 미완).
관계추출은 규칙 F1 0.309(v0.7.0) vs LLM 0.654(0710~11 pooling 실측) — 정형텍스트 정밀도용, recall 한계 명시.
⚠️ v0.5 대비 동작 변화: `auto_english=True` 기본이라 **라틴 약어가 섞인 한국어 코퍼스에
영어 클래스가 새로 추가**된다(순수 한글 코퍼스는 출력 완전 동일 — finreg 489 실측).
기존 동작 유지가 필요하면 `auto_english=False`.

## 철학
- **코어 의존성 0** — 백엔드·모델(kiwipiepy/transformers/httpx)은 전부 extras.
- **프로토콜 주입** — XGEN은 인프라 결합 없이 프로토콜 구현만 주입. `Extractor`/`GraphStore`/`VectorStore`/`LLM`.
- **단일 소스** — 개선을 XGEN 코드에 인라인 하드코딩하지 않고 라이브러리 한 곳에서 관리. config 스위치로 A/B.

## 설치
```bash
pip install xgen-ontokit                 # 코어 (의존성 0)
pip install "xgen-ontokit[korean]"       # + Kiwi 형태소
pip install "xgen-ontokit[ner]"          # + KoELECTRA NER
pip install "xgen-ontokit[all]"          # 전부
# GitHub 직접:
pip install "git+https://github.com/<org>/xgen-ontokit.git"
```

## 빌드 — LLM-free 한국어·영어 추출
```python
from ontokit import DeterministicKoreanExtractor

ext = DeterministicKoreanExtractor(domain_words=["여신전문금융업", "보험업"])
concepts, entities, relations, data = await ext.extract(documents)
# documents = {"파일명": [{"chunk_id","chunk_text","chunk_index"}, ...]}
# concepts.class_hierarchy 에 subClassOf 계층(접미공유), source_chunks 태깅 포함
```
finreg 489 실측: **4.5초 / $0** (gpt-4o 23분/$2 대비), 클래스 3156·subClassOf 1710.
검색 A/B: gpt-4o 빌드와 Recall@10 동일(0.947). ⚠️ 단 이 지표는 **벡터 leg(임베딩+FTS)가
recall을 캐리**하므로 *빌드 방식 차이를 측정하지 못한다*(LLM/LLM-free 구분 불가). 계층·관계
품질은 검색 recall 이 아니라 계층 카운트/전수열거/관계 GT 로 따로 측정해야 한다(로드맵 참조).

## 인용 온톨로지 (v0.8) — doc-level `:cites`
```python
from ontokit.citations import CitationCollector, citations_insert_update, doc_uri

col = CitationCollector()            # 스트리밍 — 청크경계 캐리(TAIL_CARRY)
col.feed(file_name, chunk_text)      # 「법령명」 제N조 패턴, 같은법·동법 마스킹
sparql = citations_insert_update(col.edges(), graph_uri + "__cites")
```
법령류 상호인용을 문서 레벨 `:cites` 엣지로 방출 → XGEN multi_turn_rag 5번째 leg
(UNION SPARQL, 최소 변위 삽입). mixed 실측: multihop 완전회수 dev 0.842/ho 0.700/te 0.767.
근거: `docs/그래프sources_인용온톨로지_연결_PoC실증_2026_07_12.md`.

## 클래스 승격 필터 (v0.9) — LLM-free 과생성 해소
```python
from ontokit.filter import ClassPromotionFilter

f = ClassPromotionFilter(corpus_chunks=n_chunks)  # 미상(None)·소형(<5000)이면 지지도 게이트 자동 비활성
keep, reason = f.decide(label, df=df, has_rel=..., has_kid=..., has_inst=...)
```
승격 기준(termhood): 재사용(df≥2) 또는 구조 참여(관계·계층부모·인스턴스)시에만 클래스 승격.
정크규칙은 통계+닫힌 문법 기능어만(도메인 블랙리스트 금지). mixed20k 실측: 444,817→70,671
(-84.1%), 관계 트리플 100% 보존. ⚠️고립 df1 유효개념도 지운다(의도된 비용, XGEN 배선은
사이드카 `<graph>__filtered`로 가역). 근거: `docs/클래스승격필터_과생성해소_2026_07_12.md`.

## 검색 개선
```python
from ontokit.search import class_instances_triple, blend_score
# #1 subClassOf* 이행폐포 — 하위클래스 인스턴스 전수열거 (회귀 0)
# #2 vscore 결측 floor guard — 키워드 exact-match 청크 랭킹 복원
```

## XGEN 주입 (수정 지점 1곳)
```python
# service/ontology/pipeline.py
# 기존:  self.doc_extractor = DocumentOntologyExtractor(self.llm)
# 개선:  self.doc_extractor = ExtractorFactory.create(config, llm=self.llm)
#        (config ONTOLOGY_EXTRACTOR=deterministic_ko → LLM-free 전환)
```
`ExtractorFactory`는 XGEN 기존 `RerankerFactory` 패턴(PROVIDER_NAMES + importlib + config) 복제.

## 구조
```
src/ontokit/
├── protocols.py          # 주입 인터페이스 (Extractor/GraphStore/VectorStore/LLM)
├── extractors/           # deterministic_ko(핵심, 한·영 이중추출) + base(merge_concepts)
├── morphology/           # kiwi_nouns(한국어) + en_nouns(영어 nltk POS)
├── hierarchy/            # suffix_share(접미공유·주엔진, ko=문자/en=단어), hearst_ko(확장)
├── ner/                  # koelectra(ko) + english(dslim BERT MIT)
├── citations.py          # doc-level :cites 인용 수집·SPARQL 방출 (v0.8)
├── filter/               # class_promotion — termhood 승격 게이트 (v0.9)
└── search/               # improvements (subClassOf*, floor guard) — ⚠️XGEN 전용
```

## 근거
실측: `docs/LLM-free_추출기_프로토타입_실측_2026_07_08.md`, `docs/온톨로지검색_synaptic_vs_XGEN_실측종합_2026_07_07.md` (xgen-levelup/docs).

## XGEN 배포 결합 (public 리포)

리포가 public이라 **인증 없이** 설치된다:

```bash
pip install "git+https://github.com/Createyouracccount/xgen-ontokit.git"
# 버전 고정(권장): ...xgen-ontokit.git@v0.9.0
```

XGEN `pyproject.toml` dependencies 또는 requirements에 위 URL 추가.
검증됨: 컨테이너에서 인증 없이 git clone → pip install → import·추출 E2E 동작.
