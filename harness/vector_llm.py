"""벡터 + LLM 실제 답변(V-LLM) — 상한(V40)이 아니라 실제로 읽고 답하면 얼마나 되는가.

근거: 제품 검색 API 상위 k(기본 40), 청크당 앞 800자(로컬 모델 컨텍스트 한도 — 제품 근거 예산과
같은 '문자 수 기준' 절단). 답: JSON {"answers": [이름...]} 또는 {"count": n}.
로컬 LLM(ollama) 이라 제품 합성 모델보다 약할 수 있다 — 판정은 상한(V40) 기준이고 이건 참고치.

  python -m harness.vector_llm harness/bench/wiki2_bench.json out.json [k] [chars]
"""
import json
import os
import sys
import time
import urllib.request

from harness import vector
from harness.run import score_nodes

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("HARNESS_READER_MODEL", "qwen3:8b")


def answer(q, hits, chars):
    ctx = "\n\n".join(f"[문서 {i + 1}] {h['title']}\n{h['text'][:chars]}" for i, h in enumerate(hits))
    want = ('{"count": 정수, "answers": [근거가 된 항목 이름...]}' if q["form"] == "A"
            else '{"answers": [이름, ...]}')
    prompt = (f"아래 문서들만 근거로 질문에 답하라. 문서에 없는 것은 쓰지 마라. "
              f"JSON 하나만 출력: {want}\n\n{ctx}\n\n질문: {q['q']}\n/no_think")
    body = {"model": MODEL, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": 0, "seed": 0, "num_ctx": 32768}}
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(json.load(urllib.request.urlopen(req, timeout=900))["response"])


def main():
    bench, out = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    chars = int(sys.argv[4]) if len(sys.argv) > 4 else 800
    qs = json.load(open(bench))
    done = {r["id"]: r for r in json.load(open(out))["rows"]} if os.path.exists(out) else {}
    rows = []
    for q in qs:
        if q["id"] in done:
            rows.append(done[q["id"]])
            continue
        t0 = time.time()
        try:
            a = answer(q, vector.search("oh_wiki2_vec", q["q"], k), chars)
            names = [x for x in (a.get("answers") or []) if isinstance(x, str)]
            sc = score_nodes(q, [{"names": [n]} for n in names])
            if q["form"] == "A":
                sc["count_pred"] = a.get("count")
                sc["score"] = 1.0 if a.get("count") == q["gold"]["count"] else 0.0
            sc["answers"] = names[:80]
        except Exception as e:   # 실패는 0점 + 사유 기록
            sc = {"score": 0.0, "err": str(e)[:200]}
        sc["sec"] = round(time.time() - t0, 1)
        rows.append({"id": q["id"], "form": q["form"], "arms": {f"VLLM{k}": sc}})
        json.dump({"run": "vllm", "arms": [f"VLLM{k}"], "rows": rows}, open(out, "w"), ensure_ascii=False)
        print(q["id"], round(sc["score"], 3), sc.get("err", ""), sc["sec"], flush=True)


if __name__ == "__main__":
    main()
