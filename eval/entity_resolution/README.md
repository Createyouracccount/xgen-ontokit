# 개체정규화(ER) 외부 평가 — 한국어 위키피디아 redirect gold

ontokit 개체정규화(동의어 병합)의 품질을 **외부 신뢰 데이터셋**으로 실증한다.
계층(Wikidata P279, 89/100)·관계(KLUE-RE, 90/100)에 이은 3축 로드맵의 마지막.
자체 합성 GT 금지(계층 합성 F1 0.96 → 외부 gold 0.33 붕괴 실측).

## 과제

같은 실세계 개체의 서로 다른 표기를 하나로. 온톨로지에서 동의어가 별개 URI 로
쪼개지면 그래프 파편화("이커머스"·"전자상거래"가 별도 노드).

## 외부 gold — 위키피디아 redirect

`build_gold.py`: 한국어 위키피디아 넘겨주기(redirect) = 사람이 편집한 이명/동의어.
CC BY-SA 4.0(상업 OK), API 키 불필요.
- positives 431 (surface 23 / **semantic 408**), negatives 160(표면충돌 79 + **주제근접 81**).
- surface = 형태소 baseline 이 잡는 표면변이(공백·하이픈), semantic = 표면 다른
  의미변이(우한폐렴↔코로나19, COVID-19↔코로나19) ← **ER 축의 핵심 타깃**.
- 주제근접 하드네거티브(삼성전자‖LG전자 등, 도메인태그 기반 임베딩독립) — 심판 R3/R4
  요구. 임베딩이 주제근접을 동의어로 오병합하는지 시험.

## 시스템 (전부 LLM 호출 0회) — 3채널 최종

| 채널 | 파일 | 잡는 것 | 한계 |
|---|---|---|---|
| 형태소키(현 본체) | `../../src/ontokit/dedup/deterministic.py` | 표면변이(삼성전자=삼성 전자) | 의미변이 불가 |
| 우리말샘 사전 | `er_urimalsam.py` → `../../src/ontokit/dedup/synonym_dict.py` | 확정 동의어(컴퓨터=전산기), P 1.000 | 현대 고유명·신조어 미수록(교집합 9.8%) |
| KURE 임베딩 | `er_embed.py` | 의미근접 | 주제근접≠동의어 미분리(AUC 0.81 천장) |
| ~~Wikidata QID~~ | `er_dict.py`(폐기) | 순가치 9.6% | 라벨검색 부실(no-QID 333/393) |

누수 차단: gold=위키 redirect ⊥ 사전=우리말샘 ⊥ 임베딩=일반코퍼스(독립 소스).

## 실행

```bash
python3 build_gold.py        # 위키 redirect → data/gold.json (API, 백오프)
python3 baselines.py         # 정직 baseline: B0/B1/B2 형태소키
python3 er_urimalsam.py 상속 # 우리말샘 XML(GitHub 미러) → data/urimalsam_syn.txt 스냅샷
python3 run_dict_eval.py     # 3채널(형태소+사전+임베딩) dev/test 정직분할, 게이트 0.80
python3 eval_auc.py --kure   # 판별력 AUC(주제근접 벽 진단)
```

## 결과 (심판 5R, 게이트 0.80 미달)

- 정직 baseline B2 형태소키 F1 0.146(의미변이 397 전부 놓침).
- 3채널 합집합 **test 균형 F1 0.776**(30시드 평균 0.785, 0.80 통과율 37%) — **정직 경계 미달**.
- ⚠️ 임베딩은 주제근접(인공지능~머신러닝 0.714)과 동의어(전자상거래~이커머스 0.690)를
  코사인으로 원리적 미분리. 사전은 현대 고유명 미수록. **LLM-free 로 0.80 안정통과 불가.**
- 심판 이력·근거: `docs/ontokit_ER_심판_텍스트접근_채널교체_2026_07_14.md`.

## 주의

- `data/`(gold·우리말샘 1.9GB·캐시)는 재현 가능(위키API/GitHub미러/스크립트). git 미커밋.
- 우리말샘 자원: spellcheck-ko/korean-dict-nikl (CC BY-SA 2.0 KR, API키 불필요).
- 본체 배선: `synonym_dict.py`(사전 TSV) + env `ONTOKIT_SYNONYM_DICT`. 미설정 시 형태소만.
