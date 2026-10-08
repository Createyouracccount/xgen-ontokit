"""스키마 발견 모드 — 고정 스키마 없이 기존 XGEN 저장소(제품 빌드 그래프)에 하네스를 붙인다.

lotteimall-dev·develop 의 빌더는 술어를 자유롭게 만든다(xgen-domain#위치·#설립 …). 그래서
① 저장소에서 TBox 를 읽는다: 개체 사이에 실제로 쓰인 술어(로컬명=라벨)와 빈도, 타입 클래스
② 계획기 카탈로그를 그 술어로 동적 생성 — 질문을 저장소의 실제 술어로 계획
③ 컴파일러는 라벨 정규화 비교로 상수·클래스를 연결(저장소에 nkey 가 없다)
④ '수록된 항목' 한정은 표제 집합으로 사후 필터(문서 제목 = 개체 라벨)

  python -m harness.discover catalog <dataset> <graph>
"""
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from harness import schema as S

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
SKIP = {RDF_TYPE, "http://www.w3.org/2000/01/rdf-schema#label", "http://www.w3.org/2000/01/rdf-schema#domain",
        "http://www.w3.org/2000/01/rdf-schema#range", "http://www.w3.org/2000/01/rdf-schema#subClassOf"}


class Store:
    """읽기 전용 SPARQL 엔드포인트(Fuseki). graphstore FusekiBackend 와 같은 HTTP 계약."""

    def __init__(self, base="http://localhost:3030", dataset="xgen"):
        self.url = f"{base}/{dataset}/query"

    def q(self, sparql, timeout=600):
        req = urllib.request.Request(self.url + "?" + urllib.parse.urlencode({"query": sparql}),
                                     headers={"Accept": "application/sparql-results+json"})
        res = json.load(urllib.request.urlopen(req, timeout=timeout))
        if "results" not in res:
            raise RuntimeError("SPARQL 빈 응답")
        return res["results"]["bindings"]


def local(uri):
    return urllib.parse.unquote(re.split(r"[#/]", uri)[-1])


def catalog(st, graph, top=80):
    """개체-개체 술어 상위 N(빈도순). 반환 [(uri, 라벨, 빈도)]."""
    rows = st.q(f"""SELECT ?p (COUNT(*) AS ?n) WHERE {{ GRAPH <{graph}> {{
        ?s ?p ?o . FILTER(isIRI(?o)) ?o <{RDF_TYPE}> ?t . }} }} GROUP BY ?p ORDER BY DESC(?n) LIMIT {top * 2}""")
    out = []
    for b in rows:
        p = b["p"]["value"]
        if p in SKIP or "sourceChunk" in p or "sourceDocument" in p:
            continue
        out.append((p, local(p), int(b["n"]["value"])))
    return out[:top]


def _nf(var):
    """SPARQL 안의 라벨 정규화(공백·괄호 수식어 제거, 소문자) — harness.schema.norm 근사."""
    return (f'REPLACE(REPLACE(LCASE(STR({var})), "\\\\s*\\\\([^)]*\\\\)", ""), '
            f'"[\\\\s·\\\\-_.,]", "")')


def compile_discovered(plan, graph, prop_uri):
    """IR(관계 p 는 저장소 술어 라벨) → SPARQL. prop_uri: 라벨 → URI."""
    lines, n = [], [0]

    def fresh():
        n[0] += 1
        return f"?k{n[0]}"

    def term(t):
        if isinstance(t, str):
            return f"?{t}"
        if "var" in t:
            return f"?{t['var']}"
        v, lv = fresh(), fresh()
        lines.append(f"{v} <http://www.w3.org/2000/01/rdf-schema#label> {lv} . "
                     f"FILTER({_nf(lv)} = {json.dumps(S.norm(t['const']), ensure_ascii=False)})")
        return v
    for c in plan["where"]:
        if c["t"] == "isa":
            cv, lv = fresh(), fresh()
            lines.append(f"?{c['v']} <{RDF_TYPE}>/<http://www.w3.org/2000/01/rdf-schema#subClassOf>* {cv} . "
                         f"{cv} <http://www.w3.org/2000/01/rdf-schema#label> {lv} . "
                         f"FILTER({_nf(lv)} = {json.dumps(S.norm(c['class']), ensure_ascii=False)})")
        else:
            uri = prop_uri.get(c["p"])
            if uri is None:
                raise ValueError(f"저장소에 없는 술어: {c['p']}")
            lines.append(f"{term(c['s'])} <{uri}> {term(c['o'])} .")
    r = plan["return"]
    body = "\n    ".join(lines)
    return (f"SELECT DISTINCT ?{r} ?lab WHERE {{ GRAPH <{graph}> {{\n    {body}\n"
            f"    ?{r} <http://www.w3.org/2000/01/rdf-schema#label> ?lab .\n}} }} LIMIT 2000")


def planner_prompt(cat):
    lines = "\n".join(f"- {lab} (사용 {n}회)" for _, lab, n in cat)
    return f"""너는 질문을 그래프 질의계획(JSON)으로 바꾼다. 설명 없이 JSON 하나만 출력한다.

## 질의계획 형식
{{"op": "list" 또는 "count", "return": 반환할 변수, "where": [조건...]}}
- 타입: {{"t": "isa", "v": 변수, "class": "클래스 이름"}}
- 관계: {{"t": "rel", "s": 주어, "p": 관계라벨, "o": 목적어}} — 주어/목적어는 "x"·"y" 또는 {{"const": "이름"}} 또는 {{"var": "y"}}
"x" 는 문서 모음에 수록된 항목.

## 관계라벨 — 이 저장소에 실제로 있는 것만(서술어 형태). 질문 뜻에 가장 맞는 것을 고른다
{lines}

## 예시
질문: 이 문서 모음에서 '부산광역시'에 위치한 곳은 모두 몇인가?
{{"op":"count","return":"x","where":[{{"t":"rel","s":"x","p":"위치","o":{{"const":"부산광역시"}}}}]}}

질문: """


def plan_llm(prompt, question, model="qwen3:8b"):
    body = {"model": model, "prompt": prompt + question, "stream": False, "think": False,
            "format": "json", "options": {"temperature": 0, "seed": 0, "num_ctx": 4096}}
    req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    p = json.loads(json.load(urllib.request.urlopen(req, timeout=180))["response"])
    if "where" not in p or "op" not in p:
        raise ValueError("계획 형식 오류")
    p.setdefault("return", "x")
    return p


def run_bench(dataset, graph, bench, base_results, out, arm="G:xgen_product:llm"):
    """같은 벤치를 발견 모드로 돌려 base_results(r5.json 등)에 팔로 덧붙인다."""
    from harness.run import score_nodes
    st = Store(dataset=dataset)
    cat = catalog(st, graph)
    prompt = planner_prompt(cat)
    prop_uri = {lab: uri for uri, lab, _ in cat}
    titles = {S.norm(re.sub(r"\s*\([^)]*\)", "", json.loads(l)["title"]))
              for l in open("harness/data/wiki2/docs.jsonl")}
    qs = {q["id"]: q for q in json.load(open(bench))}
    res = json.load(open(base_results))
    for row in res["rows"]:
        q = qs[row["id"]]
        try:
            p = plan_llm(prompt, q["q"])
            sparql = compile_discovered(p, graph, prop_uri)
            nodes = {}
            for b in st.q(sparql):
                nodes.setdefault(b[p["return"]]["value"], []).append(b["lab"]["value"])
            ns = [{"uri": u, "names": ls} for u, ls in nodes.items()]
            if p["return"] == "x":   # '수록된 항목' 한정 = 표제 집합(사후 필터, ontokit 팔의 describes 와 같은 뜻)
                ns = [n for n in ns if any(S.norm(x) in titles for x in n["names"])]
            sc = score_nodes(q, ns)
            sc["plan"] = p
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            raise
        except Exception as e:
            sc = {"score": 0.0, "err": str(e)[:160]}
        row["arms"][arm] = sc
    res["arms"].append(arm)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    from harness.stats import summarize
    print(summarize(res))


if __name__ == "__main__":
    if sys.argv[1] == "catalog":
        for p, lab, n in catalog(Store(dataset=sys.argv[2]), sys.argv[3]):
            print(n, lab)
    elif sys.argv[1] == "bench":
        run_bench(*sys.argv[2:7])
