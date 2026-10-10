"""벡터 RAG 답에 그래프 질의 결과를 합치는 두 단계. LLM 재호출 0, 의존성 0.

측정 하네스에서 검증된 경로(harness/docs/L02 — 구조형 Δ EVAL +0.081, HOLDOUT-1 +0.057)는 두 단계를 모두 쓴다:

1. graph_block(names) — 판독기 프롬프트의 문서 청크 뒤에 붙일 그래프 결과 블록.
   그래프 결과가 0 이면 빈 문자열(= 벡터만 쓴 프롬프트와 같은 입력).
2. merge(reader_answers, nodes, op=...) — 그 판독기의 답 ∪ 그래프 결과. 개수 질문은 그래프 결과가 있으면 그 수.

블록 없이 merge 만 쓰는 조합은 측정되지 않았다.
"""
from __future__ import annotations

GRAPH_MAX = 150


def graph_block(names):
    """그래프 결과 이름 목록 → 판독기 프롬프트 블록(문서 청크 뒤, 질문 앞에 둔다)."""
    if not names:
        return ""
    shown = ", ".join(names[:GRAPH_MAX])
    more = f" 외 {len(names) - GRAPH_MAX}개" if len(names) > GRAPH_MAX else ""
    return (f"[지식그래프 질의 결과 — 문서 모음 전체에서 조건에 맞는 항목 {len(names)}개]\n"
            f"{shown}{more}\n(그래프는 자동 추출이라 누락·오류가 있을 수 있다. 문서와 대조해 판단하라.)\n\n")


def merge(reader_answers, nodes, *, op="list", reader_count=None):
    """판독 답 ∪ 그래프 결과.

    reader_answers: 판독기가 낸 답 이름 목록(str)
    nodes: ontokit.query.run / run_plan 결과의 "nodes" — [{"uri", "names"}]
    op: "list" | "count". count 면 그래프 결과가 있을 때 그 노드 수, 없으면 reader_count.
    반환: {"items": [{"names": [...]}, ...], "count": int | None (op="count" 일 때만)}
    items 는 판독 답 뒤에 그래프 노드를 이어 붙인 것이다 — 같은 대상을 중복 제거하지 않는다(측정된 규칙 그대로).
    """
    out = {"items": [{"names": [a]} for a in reader_answers] + list(nodes)}
    if op == "count":
        out["count"] = len(nodes) if nodes else reader_count
    return out
