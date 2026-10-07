"""루프 판정 측정기(L00 §1) — 같은 판독기·같은 벡터 청크에 그래프 블록 유무만 바꿔 실제 답을 채점한다.

  python -m harness.hybrid <run> <bench.json> <vec_coll> <out.json> --arms VLLM,HYB:<graph_arm>[,...]
  예) --arms VLLM,HYB:ontokit_r5,HYB:oracle,HYB:placebo

- VLLM: 상위 40청크(청크당 600자)만
- HYB:<arm>: 같은 청크 + 그래프 질의 결과 블록(계획 실패·결과 0이면 블록 없음 = VLLM 과 같은 입력)
- 판독기 Qwen3-8B(temperature 0, seed 0). 백엔드는 harness.llm(env) — 로컬 MLX·DGX vLLM 을 --shard 로 나눠 병렬.
  환경마다 판독 결과가 다를 수 있으므로 out 에 환경을 기록하고, 겹침 표본으로 환경 차이를 따로 잰다.
  계획은 문항당 1회 계산해 캐시(팔 사이 동일).
- 이어하기: out 파일에 있는 (문항, 팔) 은 건너뛴다.
"""
import argparse
import asyncio
import json
import os
import time
import urllib.error
import urllib.request

from harness import vector
from harness.run import score_nodes

K, CHARS, GRAPH_MAX = 40, 600, 150


def read(q, hits, graph_items):
    ctx = "\n\n".join(f"[문서 {i + 1}] {h['title']}\n{h['text'][:CHARS]}" for i, h in enumerate(hits))
    gblock = ""
    if graph_items:
        names = ", ".join(graph_items[:GRAPH_MAX])
        more = f" 외 {len(graph_items) - GRAPH_MAX}개" if len(graph_items) > GRAPH_MAX else ""
        gblock = (f"[지식그래프 질의 결과 — 문서 모음 전체에서 조건에 맞는 항목 {len(graph_items)}개]\n"
                  f"{names}{more}\n(그래프는 자동 추출이라 누락·오류가 있을 수 있다. 문서와 대조해 판단하라.)\n\n")
    want = ('{"count": 정수, "answers": [근거 항목 이름...]}' if q["form"] == "A" else '{"answers": [이름, ...]}')
    # 그래프 블록을 문서 뒤에 둔다 — 팔 사이 프롬프트 앞부분이 같아 판독기 KV 캐시를 재사용한다
    prompt = (f"아래 자료만 근거로 질문에 답하라. 자료에 없는 것은 쓰지 마라. JSON 하나만 출력: {want}\n\n"
              f"{ctx}\n\n{gblock}질문: {q['q']}")
    from harness.llm import generate_json
    return generate_json(prompt, model=os.getenv("HARNESS_READER_MODEL") or None, num_ctx=32768, max_tokens=1024)

def score(q, a):
    names = [x for x in (a.get("answers") or []) if isinstance(x, str)]
    sc = score_nodes(q, [{"names": [n]} for n in names])
    if q["form"] == "A":
        sc["count_pred"] = a.get("count")
        sc["score"] = 1.0 if a.get("count") == q["gold"]["count"] else 0.0
    sc["answers"] = names[:60]
    return sc


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("bench")
    ap.add_argument("vec")
    ap.add_argument("out")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n — 문항 번호 mod n == i 만(환경 간 병렬 분할)")
    a = ap.parse_args()
    arms = a.arms.split(",")
    si, sn = map(int, a.shard.split("/"))
    qs = [q for k, q in enumerate(json.load(open(a.bench))) if k % sn == si]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"run": a.run, "arms": [], "rows": [], "plans": {}}
    res["arms"] = list(dict.fromkeys(res["arms"] + arms))
    res["llm"] = {"backend": os.getenv("HARNESS_LLM", "ollama"), "url": os.getenv("HARNESS_LLM_URL", ""),
                  "model": os.getenv("HARNESS_LLM_MODEL", "")}
    rows = {r["id"]: r for r in res["rows"]}
    from harness.graph import graph_name, store
    from harness.planner import plan as llm_plan
    from harness.query import run_plan
    st = store()
    t0 = time.time()
    for i, q in enumerate(qs):
        row = rows.setdefault(q["id"], {"id": q["id"], "form": q["form"], "arms": {}})
        todo = [x for x in arms if x not in row["arms"]]
        if not todo:
            continue
        hits = vector.search(a.vec, q["q"], K)
        if q["id"] not in res["plans"]:
            try:
                res["plans"][q["id"]] = llm_plan(q["q"])
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                raise
            except Exception:
                res["plans"][q["id"]] = None
        p = res["plans"][q["id"]]
        for arm in todo:
            items = []
            if arm.startswith("HYB:") and p:
                try:
                    out = await run_plan(st, p, graph_name(a.run, arm.split(":", 1)[1]))
                    items = [n["names"][0] for n in out["nodes"]]
                except ValueError:   # 스키마 밖 계획 → 블록 없음
                    items = []
            try:
                sc = score(q, read(q, hits, items))
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                raise
            except Exception as e:
                sc = {"score": 0.0, "err": str(e)[:160]}
            sc["graph_items"] = len(items)
            row["arms"][arm] = sc
        res["rows"] = list(rows.values())
        json.dump(res, open(a.out, "w"), ensure_ascii=False)
        print(f"{i + 1}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
    await st.close()
    from harness.stats import summarize
    print(summarize(res, base="VLLM"))


if __name__ == "__main__":
    asyncio.run(main())
