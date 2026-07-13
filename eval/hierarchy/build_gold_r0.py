"""Wikidata gold v2 — 정의문(schema:description) 보유 클래스 위주로 큰 셋 확보.
계층 유도는 '텍스트→추출'이라 정의문이 반드시 필요. 정의문 있는 child 위주로 회수."""
import urllib.request, urllib.parse, json, time

def sparql(q, retries=4):
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent":"ontokit-poc/1.0 (research)"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:
            if i==retries-1: raise
            time.sleep(4)

# 정의문 필수 + 여러 도메인. OFFSET 페이징으로 다양성 확보.
all_pairs, descs = [], {}
for offset in (0, 400, 800, 1200, 1600):
    q = f"""
    SELECT ?cL ?cD ?pL WHERE {{
      ?c wdt:P279 ?p .
      ?c rdfs:label ?cL . FILTER(lang(?cL)="ko")
      ?c schema:description ?cD . FILTER(lang(?cD)="ko")
      ?p rdfs:label ?pL . FILTER(lang(?pL)="ko")
    }} LIMIT 400 OFFSET {offset}
    """
    try:
        rows = sparql(q)
    except Exception as e:
        print(f"offset {offset} fail: {str(e)[:60]}"); continue
    for b in rows:
        cL, pL, cD = b["cL"]["value"], b["pL"]["value"], b["cD"]["value"]
        if cL != pL and len(cL)>=2 and len(pL)>=2 and "문서" not in cD:
            all_pairs.append((cL, pL)); descs[cL] = cD
    print(f"offset {offset}: 누적 {len(all_pairs)}쌍")
    time.sleep(1)

pairs = list(dict.fromkeys(all_pairs))
hetero = [(c,p) for c,p in pairs if not (c.endswith(p) and c!=p)]
print(f"\n최종 {len(pairs)}쌍, 이질 {len(hetero)}={len(hetero)/max(len(pairs),1)*100:.0f}%, 정의문 {len(descs)}")
json.dump({"pairs":pairs,"descs":descs}, open("wd_gold.json","w"), ensure_ascii=False, indent=1)
print("저장 wd_gold.json")
