# 관계추출(RE) 외부 평가 — KLUE-RE 한국어 gold

ontokit 관계추출의 품질을 **외부 신뢰 데이터셋**(KLUE-RE, CC BY-SA 4.0)으로 실증한다.
계층 축(eval/hierarchy/, 심판 89/100)과 같은 프로토콜. 자체 합성 GT 금지
(계층에서 합성 F1 0.96 → 외부 gold 0.33 붕괴 실측).

## 과제

KLUE-RE: (문장, subject 개체, object 개체) → 30개 관계 라벨 분류(no_relation 포함).
공식 지표 = micro-F1 (no_relation 제외, TACRED 방식).

## 데이터 (외부, 재현 가능)

```bash
curl -sL -o data/klue_re_train.parquet \
  "https://huggingface.co/datasets/klue/klue/resolve/main/re/train-00000-of-00001.parquet"
curl -sL -o data/klue_re_validation.parquet \
  "https://huggingface.co/datasets/klue/klue/resolve/main/re/validation-00000-of-00001.parquet"
python3 prep_data.py   # → data/tune.json(6,000) / data/holdout.json(7,765)
```

## 누수 차단 프로토콜

- **tune**(train 표본 6,000): 규칙 튜닝·반복은 여기서만
- **holdout**(공식 validation 7,765): 반복 중 미접촉, 심판 채점 전용
- **인코더 학습 = train − tune**(26,470): tune 평가가 부풀지 않게 제외
- train no_relation 29.4% vs holdout 59.6% — 분포 이동이 있어 holdout 이 진짜 시험

## 시스템 구성 (전부 LLM 호출 0회)

| 채널 | 파일 | 성격 |
|---|---|---|
| 통사·구조 규칙 | `extractor_rules.py` | 괄호 구조·직함 병치·관형격 방향·서술 트리거·계사. 결정적, 설명가능, 고정밀 |
| 로컬 RE 인코더 | `train_encoder.py` → `model_re/` | klue/roberta-small 파인튜닝(로컬, LLM 아님 — NER modu-ner 전례). 재현율 담당 |
| 타입 게이트 | `labels.py` | 라벨 의미 정의 기반 (subj,obj) 타입 제약 — 결정적 후처리 |

## 실행

```bash
python3 baselines.py tune --svo     # 정직 baseline: B1 타입쌍 prior / B2 현 SVO 연결력
python3 run_eval.py tune            # 규칙 채널 + prior 앙상블
python3 run_eval.py --holdout-final # ⚠️심판 전용
python3 train_encoder.py            # 인코더 재현(MPS ~20분, model_re/ 생성)
python3 eval_encoder.py tune        # 인코더/앙상블 ablation
```

## 정직 baseline (허수아비 금지 — 계층 R1 교훈)

- B1 타입쌍 prior(train 최빈) @tune = **0.565** ← 이겨야 할 floor
- B2 현 본체 조사 SVO의 gold쌍 연결력 = **0.8%** — "관계 84% miss" 진단의 KLUE 재확인.
  뉴스·위키 관계는 동사 SVO가 아니라 괄호·병치·관형격 구조에 산다.

## 결과 (심판 R1 90/100 통과, 2026-07-14)

holdout(공식 validation 7,765, no_rel 59.6%) 최종:

| 시스템 | holdout F1 | 비고 |
|---|---|---|
| B1 타입쌍 prior | 0.371 | 정직 floor |
| B1+ 단어쌍암기(심판 구축) | 0.381 | 최강 무학습 floor |
| 규칙 patterns | 0.404 (P 0.673) | ⚠️tune P 0.838에서 열화 — per:title(0.765) 외 대부분 붕괴 |
| **E1 인코더 (채택)** | **0.5924** | 공식 RoBERTa-small baseline 60.89와 정합(−1.7pp, 학습량 적음 감안 정상) |
| E1+타입게이트 | 0.556 | 게이트 −3.6pp — **미적용 확정** |
| 앙상블 E2/E3 | 0.552/0.548 | 규칙은 인코더에 순증 0 — E1 단독 채택 |

심판 적발 사항(정직 공시):
- **타입제약표 과협소**: gold 위반율 tune 3.4% → holdout **15.0%**. "KLUE 타입 노이즈"가
  아니라 제약표가 실분포 대비 좁은 것. 게이트는 규칙 채널 발화 조건으로만 사용.
- **규칙 채널의 holdout 열화**: 고정밀 서사는 per:title 계열에서만 성립. 인코더 없는
  환경의 폴백 가치도 한계적(vs prior +3pp).
- 인코더학습∩holdout 완전동일 (문장,subj,obj) 3건 존재(7,765 중 3, 효과 무시 가능).

## 모델 교체 (나중에 다른 모델로 바꾸는 법)

본체는 특정 모델에 묶이지 않는다. 관계 채널은 `.extract(text, *, source_chunks) -> list[dict]`
계약만 요구하므로, 모델 교체 = 채널 객체 교체다. 4가지 경우:

| 원하는 것 | 방법 | 수정 범위 |
|---|---|---|
| 같은 KLUE 형식 다른 가중치 | env `ONTOKIT_RELATION_ENCODER_MODEL` = 로컬경로 or HF hub id | 없음 |
| 더 큰 모델로 재학습 | `train_encoder.py` MODEL 한 줄(예: klue/roberta-large) + 재학습 | 스크립트 1줄 |
| 코드에서 직접 주입 | `DeterministicKoreanExtractor(relation_extractor=enc, ner=...)` — env보다 우선 | 호출부만 |
| 다른 라벨 체계·방식 | `.extract()` 계약으로 래핑한 새 클래스를 위처럼 주입 | 새 래퍼 1개 |

**교체 조건(경우 1·2)**: 30개 KLUE 라벨 출력 + typed marker 입력
(`[S:ORG] … [/S] [O:PER] … [/O]`) 이해. 라벨 체계가 다르면 경우 4(래퍼).

현재 드롭인 호환 채널 3종(전부 `.extract()` 계약):
`KoreanRelationExtractor`(규칙 조사SVO) / `KoreanRelationEncoder`(KLUE-RE 인코더) /
`HybridRelationExtractor`(규칙+LLM top-up). 새 모델도 같은 계약으로 감싸면 끝.

## 주의

- `model_re/` 가중치(~270MB)는 git 미커밋 — `train_encoder.py`로 재현(시드 고정).
- KLUE gold 스팬이 문장 내 재출현(인용문)을 가리키는 경우가 있어 규칙은
  최근접 출현쌍으로 재정렬(`_best_spans`).
- 라이선스: KLUE 데이터·사전학습모델 CC BY-SA 4.0(상용 OK, 표기+동일조건).
  파인튜닝 산출물도 동일 조건 상속.
