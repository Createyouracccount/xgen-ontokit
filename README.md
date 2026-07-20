# xgen-ontokit

**한국어** · [ENGLISH](README.en.md)

XGEN 온톨로지 **빌드·검색 개선 키트**. 우리가 수정·주입하는 온톨로지 개선을 하나로 말아 XGEN에 주입하는 라이브러리. omnifuse(검색 라이브러리)와 상보 — 이쪽은 **빌드(LLM-free 한국어·영어 추출) + 검색 개선**을 함께 담는다.

## 언어 지원 매트릭스 (v0.13.1, 정직하게)

| 축 | 한국어 | 영어 | 혼합 청크 |
|---|---|---|---|
| 클래스(복합명사) | ✅ Kiwi | ✅ nltk POS (extras[english], **auto-wire**) | ✅ 이중 추출(소수언어 용어 보존) |
| 계층(subClassOf) | ✅ 문자 접미공유 + **정의문(Hearst, 기본 on)** | ✅ 단어 접미공유(대소문자 무시) | ✅ |
| 엔티티(NER) | ✅ KoELECTRA(주입) | ✅ dslim BERT MIT(주입 또는 `ONTOKIT_NER_EN=auto`) | ⚠️ 지배언어만(비용) |
| **관계** | ✅ 조사 SVO(규칙) + KLUE-RE 인코더(opt-in) | ✅ **spaCy 의존 SVO(opt-in, v0.13)** | 한국어만 |
| 인스턴스 타이핑 | ✅ 정의문 + **직업 P106 어휘집(기본 on)** | ❌ 미지원 | 한국어만 |
| OWL 라벨 | `@ko` | `@en` (자동판정) | 혼재 출력 |

⚠️ 품질 검증 범위: 한국어=finreg 489 실측, 영어=구조 테스트만(코퍼스 실측 미완).
관계추출: 규칙 조사SVO는 **가용성 폴백 전용**으로 지위 확정(앙상블 영구 기각, B3) +
**KLUE-RE 인코더 채널(opt-in, holdout micro-F1 0.6274 — 외부 gold KLUE-RE)**.
인코더는 env 로 켜며 미설정 시 규칙 폴백 — 아래 [관계 인코더](#관계-인코더-v013--klue-re--sredfm-ko-증강-holdout-06274) 참조.
⚠️ v0.5 대비 동작 변화: `auto_english=True` 기본이라 **라틴 약어가 섞인 한국어 코퍼스에
영어 클래스가 새로 추가**된다(순수 한글 코퍼스는 출력 완전 동일 — finreg 489 실측).
기존 동작 유지가 필요하면 `auto_english=False`.
⚠️ v0.11 대비 동작 변화: `enable_hearst=True`(정의문 이질계층), `enable_occupation=True`
(직업 타이핑)가 **기본 on** — 순수 접미공유 출력과 다르다. 되돌리려면 각각 `False`.

### 기본값 / env 스위치 한눈에

무인자 생성(`DeterministicKoreanExtractor()`) 시 켜지는 것과, env 로만 켜지는 것의 구분:

| 채널 | 기본 | 스위치 | 모델 로드 |
|---|---|---|---|
| 한국어 클래스·접미공유 계층 | **on** | — | 없음(Kiwi) |
| 영어 클래스 | **on**(nltk 설치 시) | `auto_english=False` | 없음(nltk POS) |
| 정의문 계층·타이핑(Hearst) | **on** | `enable_hearst=False` | 없음(규칙) |
| 직업 타이핑(P106) | **on** | `enable_occupation=False` / `ONTOKIT_OCCUPATION_TYPING=off` | 없음(동봉 어휘집) |
| 한국어 관계(조사 SVO) | **on** | `enable_relations=False` | 없음(Kiwi) |
| 관계 인코더(KLUE-RE) | off | `ONTOKIT_RELATION_ENCODER_MODEL` | transformers(로컬) |
| 영어 NER | off | `ONTOKIT_NER_EN=auto` | transformers(로컬) |
| 영어 관계(spaCy) | off | `ONTOKIT_RELATION_EN=auto` | spaCy |
| 보조 NER union | off | `ONTOKIT_NER_AUX_MODEL` | transformers(로컬) |
| 사전 동의어 병합 | off | `ONTOKIT_SYNONYM_DICT` | 없음(TSV) |

**불변식**: 기본 경로는 **LLM 호출 0회 · transformers 로드 0회**. 모델을 쓰는 채널은
전부 env opt-in 이고, 기본 on 인 신규 채널(직업 타이핑)도 패키지 동봉 어휘집만 읽는다
(네트워크 0). 유일한 LLM 경로는 `relation_hybrid.HybridRelationExtractor` 로,
`relation_extractor=` 로 **명시 주입**해야만 진입하며 예산 0 이면 순수 규칙과 동치다.

## 철학
- **코어 의존성 0** — 백엔드·모델(kiwipiepy/transformers/httpx)은 전부 extras.
- **프로토콜 주입** — XGEN은 인프라 결합 없이 프로토콜 구현만 주입. `Extractor`/`GraphStore`/`VectorStore`/`LLM`.
- **단일 소스** — 개선을 XGEN 코드에 인라인 하드코딩하지 않고 라이브러리 한 곳에서 관리. config 스위치로 A/B.

## 설치
```bash
pip install xgen-ontokit                 # 코어 (의존성 0)
pip install "xgen-ontokit[korean]"       # + Kiwi 형태소
pip install "xgen-ontokit[ner]"          # + KoELECTRA NER
pip install "xgen-ontokit[relation-encoder]"  # + KLUE-RE 관계 인코더(opt-in)
pip install "xgen-ontokit[all]"          # 전부
# 영어 관계(opt-in)는 전용 extra 미정의 — 직접 설치:
#   pip install spacy && python -m spacy download en_core_web_sm
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

## 동시출현 약관계 (v0.10) — LLM-free 관계밀도 확충 (언어무관)
```python
from ontokit.cooccurrence import CooccurrenceCollector, make_korean_label_ok

col = CooccurrenceCollector(min_pair_df=3, lift_k=2.0, label_ok=make_korean_label_ok())
col.add_chunk(chunk_id, [(uri, label), ...])   # 청크 스트리밍
edges = col.edges(exclude_pairs=svo_pairs)      # [(a, b, count)] — SVO 기연결 제외
```
같은 청크 동시출현 엔티티쌍을 `coOccursWith`(함께언급) 약관계로 방출 — SVO(한국어 전용)가
못 채우는 관계밀도·영어권을 결정적으로 확충. 선별은 통계만(pair df≥3 ∧ lift>2, 목록 0),
라벨 자격은 형태·품사(달력·구두점·숫자토큰·라틴 미소·mixed-case 파편·조사 종결·단독
의존명사·충돌 싱크). mixed20k 실측: 1.75%→10.5%, SVO 100% 보존, 표시 정크율 ~18%.
⚠️coarse 관계(종류 없음), 소비측은 SVO 우선 + co-occ 폴백 슬롯. 잘림·병합 파편(`대구광역`)은
형태 판별 밖(상류 NER). 근거: `docs/관계밀도_coOccursWith_확충_2026_07_12.md`.

## 관계 인코더 (v0.13) — KLUE-RE + SREDFM-ko 증강, holdout 0.6274
```bash
pip install "xgen-ontokit[relation-encoder]"          # + transformers·torch
export ONTOKIT_RELATION_ENCODER_MODEL=/path/to/model_re   # 이 env가 on/off 스위치
```
규칙 조사SVO(KLUE 연결력 0.8%)를 넘는 **로컬 RE 인코더** 채널. klue/roberta-small
파인튜닝, **LLM API 호출 0회**(로컬 추론, NER 과 동일 계열).

**채택 이력** (외부 gold = KLUE-RE 공식 validation 7,765, micro-F1):

| 버전 | holdout | 비고 |
|---|---|---|
| re-ko-v1 | 0.5924 | KLUE 공식 roberta-small baseline 60.85 정합 |
| re-ko-aug-v1 | 0.6259 | SREDFM-ko 증강 |
| **re-ko-aug-v12 (현재)** | **0.6274** | P112 무근거행 721 제거 — `founded_by` 0.519→0.696 |

현재 채택본은 `eval/relation/MODEL_LOCK.json` 이 단일 진실원(release_tag·sha256 고정).
⚠️ 모델 크기는 여전히 **small** — 공식 base 0.6666 / large 0.6959 대비 아래이며,
증강이 아닌 **모델 크기 상향은 미소진 레버**다(LLM 없이 가능한 남은 개선 여지).

- **불변식**: env 미지정·extras 미설치·경로오류·NER부재 중 하나라도면 규칙 조사SVO 폴백.
  "설치·설정 안 하면 안 켜진다". NER(KoELECTRA)이 준 개체를 쌍으로 조합→관계 분류.
- **규칙 채널 지위**(B3 확정): 규칙 조사SVO 는 **가용성 폴백 전용**. 인코더와의 앙상블은
  **영구 기각** — holdout 실측에서 규칙 단독 보정 29건 vs 오염 144건으로 순가치 음수.
- **모델 교체**: 특정 모델 미종속. ①env 변경 ②`eval/relation/train_encoder.py` 재학습
  ③`relation_extractor=` 주입 ④`.extract()` 래퍼 — `eval/relation/README.md` '모델 교체'.
- 재현·평가: `eval/relation/`(train_encoder.py·eval_encoder.py·gold). 가중치는
  Release 자산으로 배포(git 미커밋). 근거: `docs/ontokit_관계_KLUE-RE_인코더_심판루프_90_2026_07_14.md`.

## 정의문 계층·타이핑 (v0.12~) — 이질계층 유도 (기본 on)
접미공유가 **원리적으로 불가능한** 이질계층(강아지⊂동물, 신용공여⊂거래)을 정의문
종결패턴(계사/genus/서술/속하는)으로 유도한다. `enable_hearst=True` 가 기본.

```python
ext = DeterministicKoreanExtractor(enable_hearst=True)   # 기본값
```
- **ABox↔TBox 브리지**: 정의문 주어가 NER 엔티티면 `subClassOf` 대신 `rdf:type` 으로
  방출 — 계층 도달률 0% 였던 고립 섬 문제를 수복.
- 전부 규칙(Kiwi 형태소 + 종결패턴). LLM 0콜.
- ⚠️ **근거 수준**: 외부 gold(Wikidata P279) 심판루프 89/100 및 실빌드 615건·정밀도 87%
  는 **개발 라운드의 자체 심판·커밋 기록**이며, `eval/hierarchy/` 에 재현 가능한 산출물로
  아직 랜딩되지 않았다. 재현 하네스 정비는 미완 — 이 수치는 그 전제에서 읽을 것.

## 직업 인스턴스 타이핑 (v0.13) — P106 어휘집 (기본 on)
인물 엔티티에 직업 클래스를 부여하는 빌드타임 채널. 열거(enum) 천장의 병목이
"통로가 아니라 물"(코퍼스 내 직업 타이핑 희박)이라는 진단에서 나온 외부지식 주입.

```python
ext = DeterministicKoreanExtractor(enable_occupation=True)   # 기본값
# 끄기: enable_occupation=False 또는 ONTOKIT_OCCUPATION_TYPING=off
```
- 어휘집 `data/occupation_lexicon_ko.json.gz`(4,121쌍) — SREDFM-ko P106 표면형 +
  Wikidata 후보 중 **블라인드 2인 합의 정탐만**(348, 인간 검증). **패키지 동봉 = 빌드
  네트워크 0**, 모델 로드 0, LLM 0콜.
- 게이트: 인물지배 컷(동음이의 방어) + 증거 게이트(`ONTOKIT_OCCUPATION_EVIDENCE=adj`
  기본). 도메인 오탐 63.6%→0% 실측(자체 측정, 외부 gold 아님).
- 복수 직업(갈릴레이=물리학자·수학자)은 추가 레코드로 방출 → `rdf:type` 복수 자연 지원.

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
│                         #   relation_ko(조사 SVO) / relation_encoder_ko(KLUE-RE, opt-in)
│                         #   relation_en(spaCy 의존 SVO, opt-in) / relation_hybrid(⚠️LLM, 주입 전용)
├── morphology/           # kiwi_nouns(한국어) + en_nouns(영어 nltk POS)
├── hierarchy/            # suffix_share(접미공유·주엔진, ko=문자/en=단어), hearst_ko(정의문, 기본 on)
├── instance_typing/      # occupation(P106 어휘집·기본 on) + evidence + hygiene (v0.13)
├── ner/                  # koelectra(ko) + english(dslim BERT MIT) + ensemble·span_align
├── dedup/                # deterministic(형태소) + synonym_dict(우리말샘, opt-in)
│                         #   class_synonyms(TBox 후보 제안 — 병합 안 함, 오프라인 검토용)
├── citations.py          # doc-level :cites 인용 수집·SPARQL 방출 (v0.8)
├── filter/               # class_promotion — termhood 승격 게이트 (v0.9)
├── cooccurrence.py       # coOccursWith 동시출현 약관계 — 관계밀도 확충 (v0.10)
└── search/               # improvements (subClassOf*, floor guard) — ⚠️XGEN 전용
```

## 근거
실측: `docs/LLM-free_추출기_프로토타입_실측_2026_07_08.md`, `docs/온톨로지검색_synaptic_vs_XGEN_실측종합_2026_07_07.md` (xgen-levelup/docs).

## XGEN 배포 결합 (public 리포)

리포가 public이라 **인증 없이** 설치된다:

```bash
pip install "git+https://github.com/Createyouracccount/xgen-ontokit.git"
# 버전 고정(권장): ...xgen-ontokit.git@v0.13.1
```

XGEN `pyproject.toml` dependencies 또는 requirements에 위 URL 추가.
검증됨: 컨테이너에서 인증 없이 git clone → pip install → import·추출 E2E 동작.
