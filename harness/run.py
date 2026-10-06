"""측정기 — 벡터(상한)·그래프 팔을 같은 문항으로 돌리고 채점한다.

점수(문항당 0~1): 열거형은 정답 집합 F1, 집계(A)는 개수 정확 일치.
벡터는 **상한**으로 잰다 — 정답 근거 문서가 상위 k 에 들어온 비율(R)로 F1_up = 2R/(1+R)
(정밀도 1 가정). LLM 이 근거를 완벽히 읽어도 이 값을 넘을 수 없으므로 벡터에 유리한 비교다.

  python -m harness.run <run> harness/bench/wiki2_bench.json <out.json> --arms V40,V100,G:oracle:gold,...
"""
import argparse
import asyncio
import json
import time

from harness import schema as S


def _keys(names):
    return {S.norm(n) for n in names if S.norm(n)}


def score_nodes(q, nodes):
    gold = q["gold"]["items"]
    gk = [_keys(it["names"]) for it in gold]
    hit_items, fp = set(), 0
    for n in nodes:
        nk = _keys(n["names"])
        m = [i for i, k in enumerate(gk) if k & nk]
        if m:
            hit_items.update(m)
        else:
            fp += 1
    tp = len(hit_items)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    out = {"p": p, "r": r, "f1": f1, "n_pred": len(nodes), "tp": tp, "fp": fp}
    if q["form"] == "A":
        out["count_pred"] = len(nodes)
        out["score"] = 1.0 if len(nodes) == q["gold"]["count"] else 0.0
    else:
        out["score"] = f1
    return out


def score_vector(q, hits):
    got = {h["doc_id"] for h in hits}
    gold = q["gold"]["items"]
    cov = [any(d in got for d in it["docs"]) for it in gold]
    r = sum(cov) / len(gold) if gold else 0.0
    out = {"r": r, "covered": sum(cov), "n_gold": len(gold)}
    if q["form"] == "A":
        out["score"] = 1.0 if all(cov) else 0.0   # 근거가 다 와야 셀 수 있다(필요조건)
    else:
        out["score"] = 2 * r / (1 + r) if r else 0.0
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("bench")
    ap.add_argument("out")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--vec", default="oh_wiki2_vec")
    a = ap.parse_args()
    qs = json.load(open(a.bench))
    arms = a.arms.split(",")
    from harness import vector
    from harness.graph import graph_name, store
    from harness.query import run_plan
    st = store()
    plans = {}
    if any(x.endswith(":llm") for x in arms):
        from harness.planner import plan as llm_plan
        for q in qs:
            try:
                plans[q["id"]] = {"plan": llm_plan(q["q"]), "err": None}
            except Exception as e:  # 계획 실패는 0점으로 기록 — 감추지 않는다
                plans[q["id"]] = {"plan": None, "err": str(e)[:200]}
    res = {"run": a.run, "arms": arms, "ts": time.strftime("%Y-%m-%d %H:%M"), "rows": []}
    for q in qs:
        row = {"id": q["id"], "form": q["form"], "n_gold": len(q["gold"]["items"]), "arms": {}}
        for arm in arms:
            if arm.startswith("V"):
                k = int(arm[1:])
                hits = vector.search(a.vec, q["q"], k)
                row["arms"][arm] = score_vector(q, hits)
            else:
                _, garm, pmode = arm.split(":")
                if pmode == "gold":
                    p, perr = q["plan"], None
                else:
                    p, perr = plans[q["id"]]["plan"], plans[q["id"]]["err"]
                if p is None:
                    row["arms"][arm] = {"score": 0.0, "err": f"plan: {perr}"}
                    continue
                try:
                    out = await run_plan(st, p, graph_name(a.run, garm))
                    sc = score_nodes(q, out["nodes"])
                except ValueError as e:   # 계획이 스키마 밖(알 수 없는 관계 등)
                    sc = {"score": 0.0, "err": f"compile: {e}"}
                if pmode == "llm":
                    sc["plan"] = p
                row["arms"][arm] = sc
        res["rows"].append(row)
    await st.close()
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    from harness.stats import summarize
    print(summarize(res))


if __name__ == "__main__":
    asyncio.run(main())
