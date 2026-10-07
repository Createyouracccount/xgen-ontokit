"""H2 — 로컬 LLM 스키마 유도 추출. 문서마다 1회, 외부 API 0(ollama).

출력은 ontokit 원출력과 같은 모양(ner_entities·relations·concepts)으로 저장해
`merge` 로 ontokit 그래프와 합친다. 주제 개체 = 문서 제목(정의문 주어) 단위 사실만 뽑는다.

  python -m harness.extract_llm <docs.jsonl> <out.json> [model]
  python -m harness.extract_llm merge <ontokit_raw.json> <llm.json> <out.json>
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

from harness.schema import RELATIONS


_CAT = "\n".join(f"- {k}: {v[0]} — {v[1]}" for k, v in RELATIONS.items())
PROMPT = f"""아래 문서의 주제(첫 줄의 표제)에 대해, **본문에 적혀 있는 사실만** JSON 으로 뽑아라.
본문에 없는 내용은 아무리 알고 있어도 쓰지 마라.

형식: {{"types": [주제의 분류·직업·종류 명사...], "relations": [{{"p": 관계키, "o": 대상 이름}}...]}}
- types: "미국의 배우이자 가수" → ["배우", "가수"]. "대한민국의 도시" → ["도시"]. 수식어(국가명 등)는 빼고 명사만.
- relations 의 p 는 아래 키 중에서만. o 는 본문에 적힌 표기 그대로.
- types 는 최대 5개, relations 는 최대 15개. 간결하게.
{_CAT}

문서:
"""


def ask(model, text, timeout=300):
    from harness.llm import generate_json
    return generate_json(PROMPT + text[:2000], model=model, num_ctx=4096, max_tokens=1024, timeout=timeout)


def parse_salvage(txt):
    """JSON 파싱 — 출력 상한에 잘린 경우 마지막으로 완결된 항목까지 살린다(잘린 꼬리만 버림).
    복구 불가면 예외(0건으로 조용히 넘기지 않는다)."""
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    for i in range(len(txt) - 1, 0, -1):
        if txt[i] != "}":
            continue
        for tail in ("]}", "}", "]}}", ""):
            try:
                out = json.loads(txt[:i + 1] + tail)
                if isinstance(out, dict):
                    out["_salvaged"] = True
                    return out
            except json.JSONDecodeError:
                continue
    raise ValueError(f"JSON 복구 불가: {txt[:80]}")


def extract(docs_path, out, model):
    docs = [json.loads(l) for l in open(docs_path)]
    res = json.load(open(out)) if os.path.exists(out) else {"model": model, "docs": {}}
    t0, n_err = time.time(), 0
    from concurrent.futures import ThreadPoolExecutor
    todo = [d for d in docs if d["doc_id"] not in res["docs"]]
    workers = int(os.getenv("HARNESS_LLM_WORKERS", "4"))

    def one(d):
        try:
            return d, ask(model, d["text"]), None
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            raise
        except Exception as e:
            return d, None, e
    with ThreadPoolExecutor(workers) as ex:
        results = ex.map(one, todo)
        for i, (d, a, e) in enumerate(results):
            if e is not None:
                n_err += 1
                res["docs"][d["doc_id"]] = {"title": d["title"], "types": [], "relations": [], "err": str(e)[:120]}
            else:
                rels = a.get("relations") or []
                if isinstance(rels, dict):
                    rels = [{"p": k, "o": v} for k, v in rels.items()]
                res["docs"][d["doc_id"]] = {"title": d["title"], "types": a.get("types") or [], "relations": rels}
            if i % 25 == 0:
                json.dump(res, open(out, "w"), ensure_ascii=False)
                print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s err={n_err}", flush=True)
    res["seconds"] = res.get("seconds", 0) + time.time() - t0
    json.dump(res, open(out, "w"), ensure_ascii=False)
    print(f"done {len(res['docs'])} docs err={n_err}")


def merge(ok_path, llm_path, out):
    """ontokit 원출력 + LLM 사실 → ontokit 원출력 모양. 주제 = 문서 제목(괄호 제거)."""
    import re
    raw = json.load(open(ok_path))
    llm = json.load(open(llm_path))
    n_t = n_r = 0
    for doc_id, a in llm["docs"].items():
        subj = re.sub(r"\s*\([^)]*\)", "", a["title"]).strip()
        ents = raw["ner_entities"].setdefault(doc_id, [])
        for t in a.get("types", []):
            if isinstance(t, str) and t.strip():
                ents.append({"entity": subj, "class": t.strip(), "type": "INSTANCE",
                             "source_chunks": [f"{doc_id}#0"], "channel": "llm"})
                n_t += 1
        for r in a.get("relations", []):
            if not isinstance(r, dict):
                continue
            p, o = r.get("p"), r.get("o")
            if p in RELATIONS and isinstance(o, str) and o.strip():
                raw["relations"].append({"subject": subj, "predicate": p, "object": o.strip(),
                                         "relation_label": p, "source_chunks": [f"{doc_id}#0"],
                                         "channel": "llm"})
                n_r += 1
    json.dump(raw, open(out, "w"), ensure_ascii=False)
    print(f"merged types={n_t} relations={n_r} -> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(*sys.argv[2:5])
    else:
        extract(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "qwen3:8b")
