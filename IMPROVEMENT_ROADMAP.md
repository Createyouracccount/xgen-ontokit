# ontokit 개선 로드맵 (2026-07-10)

한국어 LLM-free 온톨로지 추출기(ontokit)의 남은 개선 여지를 정리한다. 두 조사
문서(`docs/프롬프트/ontology_search.txt`, `ontology_prompt_all.txt`)의 39개 기법과
현재 구현을 대조하고, "반영 가능성 × 효과 × 비용"으로 우선순위를 매겼다.

## 현재 구현 상태 (baseline)

`DeterministicKoreanExtractor`가 실제 배선한 것:

| 컴포넌트 | 방식 | 상태 |
|---|---|---|
| 클래스 추출 | Kiwi 복합명사(`kiwi_nouns`) | ✅ 배선 |
| 계층(subClassOf) | 접미공유(`suffix_share`) | ✅ 배선 |
| 엔티티(인스턴스) | KoELECTRA/EnglishNER | ✅ 배선(주입 시) |
| 관계(objectProperty) | 조사 기반 SVO(`relation_ko`) | ✅ 배선(0710 신규) |
| dedup | Kiwi 형태소 정규화(`deterministic`) | ✅ 배선 |
| OWL 생성 | 결정적(`owl/generator`) | ✅ 배선 |

관계 노이즈 정제(띄어쓰기 경계)까지 완료 — 위키 30문서 노이즈 20%→2%.

## 미반영 기법 분류

### 🟢 Tier 1 — 저비용·고효과 (착수 권장)

**1. Hearst 정의문 계층 배선** (기법 19)
- 현황: `hierarchy/hearst_ko.py`에 `definitional_pairs()` **구현돼 있으나 호출처 0**(미배선).
- 효과: 접미공유가 못 잡는 **이질 상위어**(`강아지 ⊂ 동물`처럼 접미 안 겹침) 계층 유도.
  정형 텍스트(법령·규정·위키 정의문)에서 정밀도 89.7% 계열.
- 비용: 매우 낮음. `definitional_pairs(text, self.nouns.last_noun)`을 청크 루프에서
  호출해 `class_hierarchy`에 merge. `last_noun_fn`은 기존 `KiwiNounExtractor.last_noun`
  주입만. 약 5줄.
- 리스크: 자유텍스트 오탐(57%). 정의문 패턴(`X이란…말한다`, `X의 종류`)으로 이미
  제한돼 있어 보수적. KorLex 검증(기법 22)이 이상적이나 자원 부재(아래) → 없이도
  정의문 한정이면 순이득 가능. **A/B로 순도 측정 후 기본 on/off 결정.**

**2. co-occurrence 관계 보완** (기법 12)
- 현황: 없음. 조사규칙(`relation_ko`)이 못 잡는 관계(조사 생략·복잡 문장)는 누락.
- 효과: 같은 청크에 동시출현한 엔티티쌍을 약한 관계(`relatedTo`)로 연결 → 관계 recall↑.
  synaptic/FastGraphRAG의 핵심. "coarse하지만 검색 보완" 실증.
- 비용: 낮음. `all_entities`가 **이미 청크별로 수집됨**(`source_chunks` 보유) —
  같은 청크 엔티티쌍을 `predicate="관련"` ObjectProperty로 생성. 기존 데이터 재사용.
- 리스크: coarse(관계 종류 없음). predicate를 명시적 `relatedTo`로 두어 SVO 관계와
  구분. 폭발 방지 위해 청크당 엔티티쌍 상한(예: top-N) 필요.

### 🟡 Tier 2 — 중효과 or 조건부

**3. KF-DeBERTa 자동 도메인 라우팅** (기법 2)
- 현황: `KoElectraNER`가 `model` 인자로 KF-DeBERTa 수동 교체만 가능. 자동 선택 없음.
- 효과: 금융 컬렉션에서 NER F1 88→91.80. 단 도메인 판정 로직 필요(파일명/키워드).
- 비용: 중. 도메인 감지 + 모델 전환. 금융 특화 코퍼스일 때만 이득 → 범용성 낮음.
- 판정: 금융 컬렉션 실사용이 확정되면 착수. 지금은 보류.

**4. class_deduplicator 스케일 결함 대응** (기법 26)
- 현황: 이건 **ontokit이 아니라 XGEN 본체** 결함(스테이지 2·4 단일프롬프트 O(N)토큰).
  ontokit dedup(`deterministic.py`)은 이미 형태소 규칙 기반(LLM 0).
- 효과: 대량 클래스에서 dedup 병목 제거. 단 ontokit dedup을 쓰면 이미 우회됨.
- 판정: ontokit 경로에선 이미 해결. XGEN 본체 dedup 교체는 별도 이슈.

### 🔴 Tier 3 — 자원/라이선스 막힘 (현재 불가)

**5. KorLex/CoreNet 계층 검증** (기법 22, "최고 레버리지")
- 현황: `korlex.pusan.ac.kr`·`semanticweb.kaist.ac.kr` **둘 다 DNS 실패**(0710 재확인).
  대안 KWN(`catSirup/KorEDA` pickle)은 계층 포함 여부·라이선스 미확정.
- 효과: 계층 오탐 제거의 최고 레버리지(57%→사용가능). 하지만 자원이 없으면 불가.
- 판정: **KWN pickle 실물 확보·검증이 선결**. 없으면 착수 불가.

### ⚫ Tier 4 — 배제 (한국어/라이선스/방향 부적합)

- GLiNER/GLiREL/GLiNER2(한국어 붕괴 or 미지원), mREBEL/REBEL(비상용+저품질),
  gliner_ko(비상용), gleaning(LLM 비용 증가=역방향), Stanza 의존구문 RE(무거운 의존성
  +Kaist 트리뱅크 라이선스, 조사규칙 대비 과투자).

## 착수 방향 (권장 순서)

1. **Hearst 정의문 계층 배선** (Tier 1-1) — 저비용, 계층 recall↑, A/B로 순도 검증
2. **co-occurrence 관계** (Tier 1-2) — 관계 recall↑, 기존 엔티티 재사용
3. (조건부) KWN 확보되면 → Hearst+KorLex 검증으로 계층 정밀도↑

각 착수는 실측(빌드→triple 수·품질 스팟체크) 후 커밋. 검증 축 주의: 검색 recall은
벡터 leg가 캐리(기법 34)하므로 **계층/관계 개선은 카운트·전수열거·관계질의 GT로 측정**
해야 효과가 보인다(검색 A/B로는 0.947 그대로).

## 정직한 한계

- 조사규칙 RE·접미공유·Hearst 모두 **정형 텍스트(법령/규정) sweet spot**, 자유텍스트
  (위키)는 노이즈 상존. gpt-4o 대비 관계/계층 recall 갭은 설계상 존재(벡터 leg가 보완).
- 한국어 triple 품질의 상한은 결정적 규칙의 한계. 고품질이 필수면 hybrid top-up
  (고가치 청크만 LLM, 기법 31)이 다음 단계.
