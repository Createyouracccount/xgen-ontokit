"""질의 계획기 — 자연어 질문 → IR(JSON). 로컬 LLM(ollama) 1회 호출.

벤치 문항을 예시로 쓰지 않는다(누수 방지). 예시는 벤치에 없는 관계·이름으로 만든 일반형이다.
"""
import json
import os
import urllib.request

from harness.schema import RELATIONS

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("HARNESS_PLANNER_MODEL", "qwen3:8b")

_CATALOG = "\n".join(f"- {k}: {v[0]} — {v[1]}" for k, v in RELATIONS.items())

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


def plan(question: str, timeout=120):
    body = {"model": MODEL, "prompt": PROMPT + question + "\n/no_think", "stream": False,
            "format": "json", "options": {"temperature": 0, "seed": 0, "num_ctx": 4096}}
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=timeout))["response"]
    p = json.loads(out)
    if not isinstance(p, dict) or "where" not in p or "op" not in p:
        raise ValueError(f"계획 형식 오류: {out[:200]}")
    p.setdefault("return", "x")
    return p
