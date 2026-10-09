"""질의 계획기 — 자연어 질문 → IR(JSON). 로컬 LLM 1회 호출(백엔드는 harness.llm)."""
import os

# 프롬프트는 라이브러리가 정본이다(ontokit.planner — harness/docs/S01). 하네스는 LLM 호출·복구 파서만 가진다.
from ontokit.planner import PROMPT


def plan(question: str, timeout=180):
    from harness.llm import generate_json
    p = generate_json(PROMPT + question, model=os.getenv("HARNESS_PLANNER_MODEL") or None,
                      num_ctx=4096, max_tokens=512, timeout=timeout)
    if not isinstance(p, dict) or "where" not in p or "op" not in p:
        raise ValueError(f"계획 형식 오류: {str(p)[:200]}")
    p.setdefault("return", "x")
    return p
