"""그래프 팔 생성·적재 — 같은 스키마(harness.schema)로 오라클·위약·ontokit 그래프를 만들고
graphstore 를 통해 백엔드에 싣는다. 백엔드 교체는 graphstore 의 create_store 한 곳.

  python -m harness.graph load <run> <arm> <facts|ontokit_raw.json> [docs.jsonl]
    arm ∈ oracle | placebo | ontokit
"""
import asyncio
import json
import os
import random
import sys

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

from harness import schema as S
# ontokit 그래프 투영은 라이브러리가 정본이다(harness/docs/S01 동등성 증명). 하네스는 오라클·위약·적재만 가진다.
from ontokit.graph import KLUE_MAP, NKEY, add_names as _add_names, project as ontokit_graph, tbox as _tbox  # noqa: F401

U = URIRef


def placebo_facts(types, rels, seed=7):
    """음성 대조: 개수·차수는 유지하되 **모든 사실을 반드시 다른 것으로** 바꾼다.

    HOLDOUT-1 에서 드러난 결함 수정(L01 §3): 관계 안 목적어 치환은 목적어가 겹치면(여러 역 → 같은 구)
    사실을 지우지 못했고, 타입 집합 치환은 클래스 개수를 보존했다.
    - 관계: (s, p, o) → (s, p, o') — o' 는 그 관계의 목적어 풀에서, s 의 원래 목적어들과 다른 것.
      그 관계의 목적어가 하나뿐이면 전체 관계 목적어 풀에서 뽑는다.
    - 타입: s 의 타입 집합 → 원래 타입과 **겹치지 않는** 다른 개체의 타입 집합(없으면 전체 클래스 풀에서 겹치지 않게).
    """
    rng = random.Random(seed)
    all_cls = sorted({c for _, ts in types for c in ts})
    tsets = [list(t) for _, t in types]
    new_types = []
    for s_, ts in types:
        orig = set(ts)
        pool = [t for t in tsets if not (set(t) & orig)]
        if pool:
            new_types.append((s_, rng.choice(pool)))
        else:
            cand = [c for c in all_cls if c not in orig]
            new_types.append((s_, rng.sample(cand, min(len(ts), len(cand))) if cand else []))
    true_objs = {}
    by_p = {}
    for r in rels:
        true_objs.setdefault((r["s"], r["p"]), set()).add(r["o"])
        by_p.setdefault(r["p"], set()).add(r["o"])
    all_objs = sorted({r["o"] for r in rels})
    new_rels = []
    for r in rels:
        bad = true_objs[(r["s"], r["p"])]
        cand = [o for o in sorted(by_p[r["p"]]) if o not in bad] or [o for o in all_objs if o not in bad]
        if cand:
            new_rels.append({"s": r["s"], "p": r["p"], "o": rng.choice(cand)})
    return new_types, new_rels


def _legacy_placebo(types, rels, seed=7):
    """L01 이전 위약(관계 안 목적어 치환·타입 집합 치환) — 결과 재현·감사용으로만 남긴다."""
    rng = random.Random(seed)
    tsets = [t for _, t in types]
    rng.shuffle(tsets)
    types = [(s, t) for (s, _), t in zip(types, tsets)]
    by_p = {}
    for r in rels:
        by_p.setdefault(r["p"], []).append(r)
    out = []
    for p, rs in by_p.items():
        objs = [r["o"] for r in rs]
        rng.shuffle(objs)
        out += [{"s": r["s"], "p": p, "o": o} for r, o in zip(rs, objs)]
    return types, out


def oracle_graph(facts, docs, placebo=False, seed=7, legacy_placebo=False):
    g = Graph()
    _tbox(g)
    ents = facts["entities"]
    for q, e in ents.items():
        node = U(S.ent_uri("wd:" + q))
        _add_names(g, node, e["names"])
        for d in e["docs"]:
            g.add((U(S.DOC + d), U(S.DESCRIBES), node))
    for c, rec in facts["classes"].items():
        cn = U(S.cls_uri("wd:" + c))
        _add_names(g, cn, [rec["label"]])
        for sup in rec["super"]:
            if sup in facts["classes"]:
                g.add((cn, RDFS.subClassOf, U(S.cls_uri("wd:" + sup))))
    types = list(facts["types"].items())
    rels = list(facts["relations"])
    if placebo:
        types, rels = (_legacy_placebo if legacy_placebo else placebo_facts)(types, rels, seed)
    for s, ts in types:
        for c in ts:
            g.add((U(S.ent_uri("wd:" + s)), RDF.type, U(S.cls_uri("wd:" + c))))
    for r in rels:
        g.add((U(S.ent_uri("wd:" + r["s"])), U(S.rel_uri(r["p"])), U(S.ent_uri("wd:" + r["o"]))))
    return g


def store(dataset="ontoharness"):
    """graphstore 로 백엔드 생성. 백엔드는 env GRAPHSTORE_BACKEND(기본 fuseki)."""
    sys.path.insert(0, os.getenv("GRAPHSTORE_SRC", os.path.join(
        os.path.dirname(__file__), "..", "..", "develop", "xgen-graphstore", "src")))
    from xgen_graphstore import create_store
    return create_store({"base_url": os.getenv("FUSEKI_URL", "http://localhost:3030"),
                         "dataset": dataset,
                         "password": os.getenv("FUSEKI_ADMIN_PASSWORD")})


def graph_name(run, arm):
    return f"{S.NS}g/{run}/{arm}"


async def load(g, run, arm):
    st = store()
    assert await st.ensure_dataset(), "데이터셋 생성 실패"
    gn = graph_name(run, arm)
    await st.clear_graph(gn)
    nt = g.serialize(format="nt")
    lines = nt.splitlines()
    for i in range(0, len(lines), 50000):
        r = await st.upload_ttl("\n".join(lines[i:i + 50000]) + "\n", graph_name=gn)
        if not r.get("success"):
            raise RuntimeError(f"업로드 실패: {r}")
    n = await st.get_triple_count(gn)
    await st.close()
    if n != len(g):  # 적재 손실을 조용히 넘기지 않는다
        raise RuntimeError(f"트리플 수 불일치: 로컬 {len(g)} vs 저장소 {n}")
    return n


def main():
    run, arm, src = sys.argv[2:5]
    docs = [json.loads(l) for l in open(sys.argv[5])] if len(sys.argv) > 5 else []
    data = json.load(open(src))
    stats = {}
    if arm in ("oracle", "placebo", "placebo2"):
        # placebo = HOLDOUT-1 까지 쓴 구 위약(감사용 보존), placebo2 = 모든 사실을 바꾸는 위약(L01 이후)
        g = oracle_graph(data, docs, placebo=(arm != "oracle"), legacy_placebo=(arm == "placebo"))
    else:
        g, stats = ontokit_graph(data, docs)
    n = asyncio.run(load(g, run, arm))
    print(f"{graph_name(run, arm)}: {n} triples {stats}")


if __name__ == "__main__":
    main()
