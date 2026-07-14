# 관계 축 심판 프로토콜 (계층 축과 동일 규율, 88 게이트)

심판(독립 서브에이전트)은 아래를 **직접 재실행·검증**하고 100점 만점으로 채점한다.
88 미만이면 결함 목록과 함께 반려 → loop 계속. 자기채점 금지(계층에서 순환누수·
허수아비 baseline·게이트 미작동 3결함을 심판이 적발한 전례).

## 채점 축

1. **외부 gold 정당성** (15): KLUE-RE 공식 데이터인가, 라이선스 확인됐는가,
   임의 가공·필터링으로 점수를 부풀리지 않았는가 (parquet 원본 대조).
2. **누수 차단** (20):
   - tune ⊂ train, holdout = 공식 validation 분리 확인 (guid 교집합 0)
   - **인코더 학습셋에 tune·holdout guid가 없는가** (train_encoder.py 재검증)
   - 규칙이 gold 라벨을 직접 참조하지 않는가 (extractor_rules.py 코드 검사)
3. **baseline 정직성** (15): B1 타입쌍 prior가 허수아비가 아닌가(직접 재계산),
   개선 폭이 정당한 floor 대비인가. B2 SVO 연결력 주장 재확인.
4. **holdout 성능** (25): 심판이 직접 `run_eval.py --holdout-final`,
   `eval_encoder.py holdout` 실행. tune 대비 급락(과적합)이 없는가,
   분포 이동(no_rel 29%→60%) 하에서 결론이 유지되는가.
5. **순가치 분해** (15): 규칙/인코더/타입게이트 각각의 기여가 ablation으로
   분리 실증됐는가. "규칙이 인코더에 실제로 무엇을 더하는가"가 정직하게 보고됐는가
   (더하는 게 없으면 없다고 보고해야 함).
6. **LLM-free 준수** (10): 전 파이프라인에 LLM API 호출 0회(코드 검사),
   인코더는 로컬 추론만, 결정적 재현(시드 고정) 가능한가.

## 실행 환경

- python: /opt/miniconda3/bin/python3 (kiwipiepy·torch·transformers 설치됨)
- 작업 디렉토리: xgen-ontokit/eval/relation/
- holdout 실행 권한: 심판만. 개발 loop 는 tune 결과만 인용 가능.
