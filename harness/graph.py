"""그래프 팔 생성·적재 — 같은 스키마(harness.schema)로 오라클·위약·ontokit 그래프를 만들고
graphstore 를 통해 백엔드에 싣는다. 백엔드 교체는 graphstore 의 create_store 한 곳.

  python -m harness.graph load <run> <arm> <facts|ontokit_raw.json> [docs.jsonl]
    arm ∈ oracle | placebo | ontokit
"""
import asyncio
import json
import os
import random
import re
import sys

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from harness import schema as S

NKEY = URIRef(S.NS + "nkey")
U = URIRef


def _add_names(g, node, names):
    names = [n for n in dict.fromkeys(names) if n]
    if not names:
        return
    g.add((node, RDFS.label, Literal(names[0])))
    for n in names[1:]:
        g.add((node, SKOS.altLabel, Literal(n)))
    for n in names:
        k = S.norm(n)
        if k:
            g.add((node, NKEY, Literal(k)))


def _tbox(g):
    for key, (ko, desc, _wd, trans, inv) in S.RELATIONS.items():
        p = U(S.rel_uri(key))
        g.add((p, RDFS.label, Literal(ko)))
        g.add((p, RDFS.comment, Literal(desc)))


def oracle_graph(facts, docs, placebo=False, seed=7):
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
        # 음성 대조: 개수·차수는 그대로, 누가 무엇인지만 섞는다(전단사 치환)
        rng = random.Random(seed)
        tsets = [t for _, t in types]
        rng.shuffle(tsets)
        types = [(s, t) for (s, _), t in zip(types, tsets)]
        by_p = {}
        for r in rels:
            by_p.setdefault(r["p"], []).append(r)
        rels = []
        for p, rs in by_p.items():
            objs = [r["o"] for r in rs]
            rng.shuffle(objs)
            rels += [{"s": r["s"], "p": p, "o": o} for r, o in zip(rs, objs)]
    for s, ts in types:
        for c in ts:
            g.add((U(S.ent_uri("wd:" + s)), RDF.type, U(S.cls_uri("wd:" + c))))
    for r in rels:
        g.add((U(S.ent_uri("wd:" + r["s"])), U(S.rel_uri(r["p"])), U(S.ent_uri("wd:" + r["o"]))))
    return g


# KLUE-RE 라벨 → 하네스 스키마 정렬(어댑터). 주석이 근거다.
KLUE_MAP = {
    "per:place_of_birth": ("rel", "per:place_of_birth"),
    "per:place_of_death": ("rel", "per:place_of_death"),
    "per:schools_attended": ("rel", "per:schools_attended"),
    "per:employee_of": ("rel", "per:employee_of"),
    "per:origin": ("rel", "per:origin"),
    "per:spouse": ("rel", "per:spouse"),
    "per:parents": ("rel", "per:parents"),
    "per:children": ("rel", "per:children"),
    "org:founded_by": ("rel", "org:founded_by"),
    "org:place_of_headquarters": ("rel", "org:place_of_headquarters"),
    "org:top_members/employees": ("inv", "per:employee_of"),   # 조직→인물 = 인물 소속의 역
    "loc:located_in": ("rel", "loc:located_in"),   # ontokit 위치 서술 채널(4차, opt-in)
    "loc:country": ("rel", "loc:country"),
    "per:title": ("type", None),          # KLUE per:title = 직업·직위 → 타입으로
    "per:alternate_names": ("alias", None),
    "org:alternate_names": ("alias", None),
}


def ontokit_graph(raw, docs):
    g = Graph()
    _tbox(g)
    title_key = {}
    for d in docs:
        title_key.setdefault(S.norm(re.sub(r"\s*\([^)]*\)", "", d["title"])), []).append(d["doc_id"])

    def node(label):
        return U(S.ent_uri("ok:" + S.norm(label)))

    def cls(label):
        return U(S.cls_uri("ok:" + S.norm(label)))

    seen = set()

    def ensure(label):
        n = node(label)
        if n not in seen:
            seen.add(n)
            _add_names(g, n, [label])
            for d in title_key.get(S.norm(label), []):
                g.add((U(S.DOC + d), U(S.DESCRIBES), n))
        return n

    stats = {"entities": 0, "types": 0, "rels": 0, "rels_mapped": 0, "aliases": 0, "subclass": 0}
    for doc_id, ents in raw["ner_entities"].items():
        for e in ents:
            if not S.norm(e["entity"]):
                continue
            n = ensure(e["entity"])
            stats["entities"] += 1
            g.add((n, U(S.MENTIONED_IN), U(S.DOC + doc_id)))
            if e.get("class"):
                g.add((n, RDF.type, cls(e["class"])))
                _add_names(g, cls(e["class"]), [e["class"]])
                stats["types"] += 1
    for h in raw["concepts"].get("class_hierarchy", []):
        if S.norm(h["child"]) and S.norm(h["parent"]):
            g.add((cls(h["child"]), RDFS.subClassOf, cls(h["parent"])))
            _add_names(g, cls(h["child"]), [h["child"]])
            _add_names(g, cls(h["parent"]), [h["parent"]])
            stats["subclass"] += 1
    for r in raw["relations"]:
        stats["rels"] += 1
        m = KLUE_MAP.get(r.get("relation_label") or r["predicate"])
        if not m or not S.norm(r["subject"]) or not S.norm(r["object"]):
            continue
        kind, key = m
        sn, on = ensure(r["subject"]), ensure(r["object"])
        if kind == "rel":
            g.add((sn, U(S.rel_uri(key)), on))
        elif kind == "inv":
            g.add((on, U(S.rel_uri(key)), sn))
        elif kind == "type":
            g.add((sn, RDF.type, cls(r["object"])))
            _add_names(g, cls(r["object"]), [r["object"]])
        elif kind == "alias":
            g.add((sn, SKOS.altLabel, Literal(r["object"])))
            g.add((sn, NKEY, Literal(S.norm(r["object"]))))
            stats["aliases"] += 1
        stats["rels_mapped"] += 1
    return g, stats


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
    if arm in ("oracle", "placebo"):
        g = oracle_graph(data, docs, placebo=(arm == "placebo"))
    else:
        g, stats = ontokit_graph(data, docs)
    n = asyncio.run(load(g, run, arm))
    print(f"{graph_name(run, arm)}: {n} triples {stats}")


if __name__ == "__main__":
    main()
