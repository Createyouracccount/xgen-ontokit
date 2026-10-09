"""ontokit CLI — 서버·도커 없이 문서 → 그래프 → 질의.

  ontokit build <docs.jsonl | 폴더> -o graph.ttl [--raw raw.json]   LLM 0회 (extras: korean,ner,owl)
  ontokit query graph.ttl '<계획 JSON>' | plan.json                 인메모리 SPARQL (extras: owl)
  ontokit ask graph.ttl "질문"                                      계획만 LLM(OpenAI 호환) — 설정 없으면 실패

ask 의 LLM: env ONTOKIT_LLM_URL(…/v1 까지), ONTOKIT_LLM_MODEL, 선택 ONTOKIT_LLM_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request


def _load_graph(path):
    from rdflib import Graph
    g = Graph()
    g.parse(path)
    return g


def _sources(g, uri):
    """노드의 출처 문서 — 표제 문서(describes)와 언급 문서(mentionedIn)."""
    from rdflib import URIRef
    from .graph import DESCRIBES, DOC, MENTIONED_IN
    n = URIRef(uri)
    docs = {str(d) for d in g.subjects(URIRef(DESCRIBES), n)} | {str(d) for d in g.objects(n, URIRef(MENTIONED_IN))}
    return sorted(d[len(DOC):] for d in docs if d.startswith(DOC))


def _answer(g, plan):
    from .query import run
    nodes = run(g, plan)["nodes"]
    out = {"plan": plan, "count": len(nodes),
           "nodes": [{"name": n["names"][0], "names": n["names"], "uri": n["uri"], "docs": _sources(g, n["uri"])}
                     for n in nodes]}
    return out


def cmd_build(a):
    from .graph import project
    from .pipeline import extract, read_docs, relation_channel_on
    docs = read_docs(a.input)
    if not docs:
        raise SystemExit(f"입력에 문서가 없습니다: {a.input} (.jsonl 또는 .txt·.md 가 든 폴더)")
    if not relation_channel_on():
        print("관계 채널 꺼짐: ONTOKIT_RELATION_ENCODER_MODEL 미설정 → 관계 0건, 타입·클래스만 만든다 "
              "(docs/STANDALONE.md)", file=sys.stderr)
    raw = asyncio.run(extract(docs, hearst=a.hearst))
    if a.raw:
        with open(a.raw, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
    titles = {}
    for d in docs:
        titles.setdefault(d["doc_id"], d["title"])
    g, stats = project(raw, [{"doc_id": k, "title": v} for k, v in titles.items()])
    g.serialize(a.out, format="turtle")
    print(json.dumps({"docs": len(titles), "chunks": len(docs), "triples": len(g), **stats, "out": a.out},
                     ensure_ascii=False))


def cmd_query(a):
    plan = json.load(open(a.plan, encoding="utf-8")) if a.plan.endswith(".json") else json.loads(a.plan)
    print(json.dumps(_answer(_load_graph(a.graph), plan), ensure_ascii=False, indent=1))


def _llm(prompt):
    url, model = os.getenv("ONTOKIT_LLM_URL"), os.getenv("ONTOKIT_LLM_MODEL")
    if not url or not model:
        raise SystemExit("ask 는 LLM 이 필요합니다: ONTOKIT_LLM_URL(OpenAI 호환, …/v1)·ONTOKIT_LLM_MODEL 을 설정하세요. "
                         "LLM 없이 쓰려면 계획을 직접 써서 `ontokit query` 를 쓰세요.")
    headers = {"Content-Type": "application/json"}
    if os.getenv("ONTOKIT_LLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["ONTOKIT_LLM_API_KEY"]
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 512}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"]


def cmd_ask(a):
    from .planner import PROMPT, parse
    plan = parse(_llm(PROMPT + a.question))   # 형식 오류는 ValueError 로 멈춘다(빈 결과로 바꾸지 않는다)
    print(json.dumps(_answer(_load_graph(a.graph), plan), ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ontokit", description="문서 → 그래프 → 질의 (서버 없음)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="문서 → 그래프(Turtle). LLM 0회")
    b.add_argument("input", help=".jsonl(한 줄 = 한 청크: doc_id, text, title?) 또는 .txt·.md 폴더")
    b.add_argument("-o", "--out", required=True, help="출력 .ttl")
    b.add_argument("--raw", help="추출 원출력 JSON 도 저장")
    b.add_argument("--hearst", action="store_true", help="정의문 채널(백과체 전용 권장)")
    b.set_defaults(fn=cmd_build)
    q = sub.add_parser("query", help="질의계획 실행 — 인메모리")
    q.add_argument("graph", help="build 가 만든 .ttl")
    q.add_argument("plan", help="계획 JSON 문자열 또는 .json 파일")
    q.set_defaults(fn=cmd_query)
    k = sub.add_parser("ask", help="질문 → 계획(LLM) → 질의")
    k.add_argument("graph")
    k.add_argument("question")
    k.set_defaults(fn=cmd_ask)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
