# ER 축 심판 프로토콜 — 접근 타당성 판정 (수치 게이트 아님)

계층(89)·관계(90)는 "개선 채널이 88 넘었나"를 물었다. ER 축은 **그 전 질문**을
먼저 판정한다: **이 접근 자체가 옳은가.** 개발자가 자체 진단으로 "텍스트 기반 ER은
틀린 프레임, W3C sameAs/entity-linking 이 정석"이라 주장 — 심판은 이를 독립 검증한다.

자기채점 금지(계층에서 심판이 순환누수·허수아비·게이트미작동 3결함 적발한 전례).

## 심판이 직접 실행·판정할 것

### A. 현 접근(텍스트 기반 Wikidata QID 조회)의 실측 재현 (25)
- `cd eval/entity_resolution && python3 baselines.py`, `python3 run_eval.py` 재실행.
- 개발자 주장 재현: B2 형태소키 F1 0.146(표면변이만) / D Wikidata QID F1 0.162 /
  순가치 9.6%(의미변이 397 중 38)가 실제인가.
- gold(위키 redirect) 무결성: 실제 위키피디아 API 산출인가, positives/negatives
  분류(surface/semantic)가 타당한가.

### B. 개발자 진단의 타당성 — "텍스트 기반 ER은 틀린 프레임인가" (35, 핵심)
독립적으로 판단하라(개발자 결론 신뢰 금지):
1. **누수 논증 검증**: 개발자는 "gold=위키 redirect 를 Wikidata alias 로 잡는 건
   사전끼리 베끼기(순환누수)"라 주장. 이 논증이 맞나? aliases 인덱싱이 실제로
   순환인지, 아니면 정당한 독립 신호인지 직접 따져라.
2. **W3C 표준 프레임 검증**: 동의어가 `owl:sameAs`/`skos:altLabel`/`skos:exactMatch`
   로 온톨로지에서 '선언'되는 게 표준인가? ER 이 텍스트 발견 문제가 아니라 외부 KB
   링크(entity linking) 문제라는 재프레이밍이 W3C·지식공학 관점에서 옳은가?
   (근거: W3C OWL/SKOS 명세, 실제 KG 구축 관행)
3. **문서 내 신호 부재 검증**: 문서 텍스트에 "A=B(동의어)"가 실제로 잘 안 쓰이는가?
   계층("X는 Y이다")·관계("X가 Y를 한다")와 달리 동의어는 문서에 명시 신호가
   드문가? 이게 순가치 9.6%의 근본 원인인가, 아니면 방법이 나빠서인가?

### C. 재설계 방향의 건전성 (25)
- 개발자가 준비 중인 "외부 KB entity linking → sameAs/altLabel 병합"이 옳은 방향인가?
- 이 방향의 위험(entity linking 오류 전파, KB 커버리지 병목, 여전한 누수 가능성)을
  심판이 독립적으로 지적하라. 특히 이것도 결국 Wikidata 를 쓰면 gold(위키)와의
  누수가 재발하는지.
- LLM-free 불변 유지되는가(entity linking 을 LLM 없이 결정적으로 할 수 있나).

### D. 정직성 (15)
- 개발자가 낮은 수치(9.6%)를 숨기지 않고 스스로 멈춰 심판에 보낸 것이 정직한가.
- 자기채점 없이 심판에 보낸 절차가 지켜졌나.

## 출력
- 축별 점수 + 근거(실행 로그 인용)
- **핵심 판정**: "텍스트 기반 ER 접근은 (틀렸다/부분적으로 맞다/맞다)" — 이유와 함께.
- **권고**: 재설계 방향이 옳은지, 옳다면 구체적 실행 지침, 위험 요소.
- 이건 88 게이트가 아니라 **방향 판정**이다. 점수는 참고, 판정문이 핵심.

## 환경
- python: /opt/miniconda3/bin/python3 (kiwipiepy·torch·transformers·pyarrow)
- 작업: xgen-ontokit/eval/entity_resolution/
- 파일: DESIGN.md·README.md·eval_er.py·baselines.py·er_dict.py·run_eval.py·
  build_gold.py·data/gold.json·er_eval.log
