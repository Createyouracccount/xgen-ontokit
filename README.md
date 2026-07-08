# xgen-ontokit

XGEN 온톨로지 **빌드·검색 개선 키트**. 우리가 수정·주입하는 온톨로지 개선을 하나로 말아 XGEN에 주입하는 라이브러리. omnifuse(검색 라이브러리)와 상보 — 이쪽은 **빌드(LLM-free 한국어 추출) + 검색 개선**을 함께 담는다.

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

## 빌드 — LLM-free 한국어 추출
```python
from ontokit import DeterministicKoreanExtractor

ext = DeterministicKoreanExtractor(domain_words=["여신전문금융업", "보험업"])
concepts, entities, relations, data = await ext.extract(documents)
# documents = {"파일명": [{"chunk_id","chunk_text","chunk_index"}, ...]}
# concepts.class_hierarchy 에 subClassOf 계층(접미공유), source_chunks 태깅 포함
```
finreg 489 실측: **4.5초 / $0** (gpt-4o 23분/$2 대비), 클래스 3156·subClassOf 1710.
검색 A/B: gpt-4o 빌드와 **Recall@10 완전 동일(0.947)** — 벡터 leg가 검색 캐리, 빌드 방식 무관.

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
├── extractors/           # deterministic_ko(핵심) + base(merge_concepts)
├── morphology/           # kiwi_nouns (복합명사)
├── hierarchy/            # suffix_share(접미공유·주엔진), hearst_ko(정의문·확장)
├── ner/                  # koelectra
└── search/               # improvements (subClassOf*, floor guard)
```

## 근거
실측: `docs/LLM-free_추출기_프로토타입_실측_2026_07_08.md`, `docs/온톨로지검색_synaptic_vs_XGEN_실측종합_2026_07_07.md` (xgen-levelup/docs).
