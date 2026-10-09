"""질문 → 질의계획(JSON) 프롬프트. LLM 은 사용자가 고른다(ontokit 은 호출하지 않는다 — `ontokit ask` 만 예외).

L02 측정의 계획은 이 프롬프트를 로컬 Qwen3-8B(temperature 0)에 넣어 얻었다. 다른 LLM 에서의 계획 품질은 측정되지 않았다.
벤치 문항을 예시로 쓰지 않는다(누수 방지). 예시는 벤치에 없는 관계·이름으로 만든 일반형이다.

    text = my_llm(planner.PROMPT + question)
    plan = planner.parse(text)          # 형식이 틀리면 ValueError — 조용히 빈 계획으로 바꾸지 않는다
    nodes = ontokit.query.run(g, plan)["nodes"]
"""
from __future__ import annotations

import json

from .graph import RELATIONS

_CATALOG = "\n".join(f"- {k}: {v.ko} — {v.desc}" for k, v in RELATIONS.items())

PROMPT = f"""너는 질문을 그래프 질의계획(JSON)으로 바꾼다. 설명 없이 JSON 하나만 출력한다.

## 질의계획 형식
{{"op": "list" 또는 "count", "return": 반환할 변수, "where": [조건...]}}
조건 종류:
- 타입: {{"t": "isa", "v": 변수, "class": "클래스 이름"}}  (직업·분류·종류. 하위 분류는 자동 포함)
- 관계: {{"t": "rel", "s": 주어, "p": 관계키, "o": 목적어}}
  주어/목적어는 변수 이름 문자열("x","y") 또는 {{"const": "이름"}} 또는 {{"var": "y"}}
변수 규칙: "x" 는 문서 모음에 수록된 항목. 다른 대상을 반환하려면 "y" 를 쓴다.

## 관계키 (이 중에서만 고른다)
{_CATALOG}

## 예시
질문: 이 문서 모음에서 국적이 '프랑스'인 인물 중 '화가'에 해당하는 인물·항목을 모두 나열해줘.
{{"op":"list","return":"x","where":[{{"t":"rel","s":"x","p":"per:origin","o":{{"const":"프랑스"}}}},{{"t":"isa","v":"x","class":"화가"}}]}}
질문: 이 문서 모음에서 '부산광역시'에 있는 곳은 모두 몇인가?
{{"op":"count","return":"x","where":[{{"t":"rel","s":"x","p":"loc:located_in","o":{{"const":"부산광역시"}}}}]}}
질문: 'ㄱ회사'의 설립자는?
{{"op":"list","return":"y","where":[{{"t":"rel","s":{{"const":"ㄱ회사"}},"p":"org:founded_by","o":{{"var":"y"}}}}]}}
질문: 이 문서 모음에서 '파리'에서 사망한 인물들의 출신학교를 모두 나열해줘.
{{"op":"list","return":"y","where":[{{"t":"rel","s":"x","p":"per:place_of_death","o":{{"const":"파리"}}}},{{"t":"rel","s":"x","p":"per:schools_attended","o":{{"var":"y"}}}}]}}

질문: """


def parse(text):
    """LLM 출력 → 계획. 사고 블록·코드펜스를 걷어 내고 JSON 하나를 읽는다. 형식 오류는 ValueError."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        p = json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"계획 JSON 파싱 실패: {text[:200]}") from e
    if not isinstance(p, dict) or "where" not in p or "op" not in p:
        raise ValueError(f"계획 형식 오류: {str(p)[:200]}")
    p.setdefault("return", "x")
    return p
