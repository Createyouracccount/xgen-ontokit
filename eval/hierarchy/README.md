# 계층(subClassOf) 추출 외부 평가 — Wikidata 한국어 gold

ontokit 계층 유도의 품질을 **외부의 신뢰성 있는 데이터셋**(Wikidata 한국어
subclass-of, CC0)으로 실증한다. 합성 자체제작 GT의 착시(F1 0.96였으나 외부 gold
33점)를 피하기 위한 것. 심판 서브에이전트 검증 루프로 개발됨(2026-07-13).

## 왜 이 데이터인가
- **접미공유(현 ontokit)는 이질계층(강아지⊂동물)을 원리적으로 recall 0** 으로
  못 잡는다. 이 gold 는 **71% 가 이질계층**이라 그 결함을 정면으로 잰다.
- Wikidata P279(subclass of) = 커뮤니티 큐레이션 KB, CC0, 독립.

## ⚠️ 데이터 세대 — 누수 주의 (심판 지적 반영)

| 파일 | 소스 | 상태 |
|---|---|---|
| `data/wd_gold_r0.json` | Wikidata P279 쌍 + **Wikidata schema:description** | 🔴 **누수** — 정의문과 라벨이 같은 엔티티에서 co-authored(정의문에 답이 ~49% 포함). r0 재현 전용, 실성능 측정 부적합 |
| `data/gold_r1.json` | Wikidata P279 **전이폐포** + **한국어 위키피디아 lead**(독립 소스) | ✅ 누수 차단본. 추출 소스(위키피디아)와 라벨(Wikidata)이 분리. **이걸 사용** |

`gold_r1.json` 구조:
- `direct_pairs`: Wikidata 직접 P279 쌍 (1631)
- `gold_closure`: child → {P279* 다중홉 조상} — 층위 불일치 크레딧(기계⊂컴퓨터 인정)
- `leads`: child → 한국어 위키피디아 lead 문단 (추출 입력, 독립 소스)
- `dev`/`test`: held-out 분할 (dev 1043 / test 348). **test 는 튜닝 중 미열람**

## 평가 프로토콜 (SemEval-2018 Task 9 준거 — 출판물 비교가능)
- **T1 상위어 발견**: 위키피디아 lead 에서 상위어 랭킹 추출 → 전이폐포 gold 대조.
  지표 P@1 / MRR / MAP.
- **T2 계층 유도**: 클래스집합(+distractor 열린어휘) → subClassOf 쌍 → P/R/F1 +
  **hetero-F1**(이질계층, 정밀도 가드).
- **substring oracle**: lead 에 gold조상이 substring 인 trivial 매처. 방법이 이걸
  넘어야 유효(심판 요구).

## 실행
```bash
# gold_r1.json 은 커밋돼 있음(재수집 불필요). 없으면 아래 빌드.
/opt/miniconda3/bin/python3 eval_r1.py improved dev     # 개발
/opt/miniconda3/bin/python3 eval_r1.py improved test    # 최종(held-out)
/opt/miniconda3/bin/python3 eval_r1.py baseline dev     # 현 ontokit 기준선
```
`improved` = 위키 lead head-final 상위어 추출 + 접미공유 결합(실험).
`baseline` = 현 ontokit(접미공유 + hearst_ko 따옴표 정의문만).
⚠️ 의존성: `kiwipiepy`, `numpy`, `scipy`. 파이썬 = `/opt/miniconda3/bin/python3`
(kiwipiepy 설치된 환경. 세션 환경 특수사정 — 재설치 시 `pip install kiwipiepy`).

## gold 재빌드 (재수집 필요 시)
```bash
python3 build_gold_r0.py    # Wikidata P279+desc(누수본) → wd_gold.json
python3 build_gold_r1.py    # 전이폐포 조상 회수 (SPARQL)
python3 fill_leads2.py      # 위키피디아 lead 회수 (rate-limit 안전, 0.2s 간격)
```
⚠️ 위키피디아 REST API rate-limit — lead 회수는 항목당 0.2s + timeout 10s + 50개
마다 저장(중단 대비). 연타하면 589 근처에서 hang(실측).

## 결과 이력 (심판 채점)
- **R0** (누수 gold): baseline 19 / improved 33 → **심판 26/100** (누수·게임가능
  지표·gold 층위 6개 결함 지적).
- **R1** (누수 차단): 진행 중. 정본 = `docs/ontokit_처방_PoC실증_2026_07_13.md` 및
  후속 세션 문서.

## 관련 문서 (company/xgen-levelup/docs/)
- `ontokit_SOTA_LLM-free빌더_격차분석_설계_2026_07_13.md` — 격차·처방·로드맵
- `ontokit_처방_PoC실증_2026_07_13.md` — 3처방 PoC 실증
