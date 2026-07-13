"""Round-1 gold 구축 — 심판 지적 6개 반영:
 ①순환 차단: 추출 소스 = 한국어 위키피디아 lead(독립), Wikidata desc 아님.
 ②전이폐포: P279 다중홉 조상까지 gold 확장(층위 불일치 크레딧).
 ③substring baseline 데이터: 소스에 gold parent 가 substring 인 비율 기록.
 held-out: 클래스를 dev/test 분할(test 는 튜닝 중 미열람).
"""
import urllib.request, urllib.parse, json, time, re

UA = {"User-Agent": "ontokit-poc/1.0 (research; contact donguk)"}

def sparql(q, retries=4):
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": q, "format": "json"})
    req = urllib.request.Request(url, headers=UA)
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)["results"]["bindings"]
        except Exception:
            if i == retries-1: raise
            time.sleep(4)

def wiki_lead(title):
    t = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{t}"
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        if d.get("type") == "disambiguation":
            return ""
        return d.get("extract", "")
    except Exception:
        return ""

# 1) 기존 frozen gold 재사용(직접 P279 쌍) + QID 확보해 전이폐포 조상 추가
base = json.load(open("wd_gold_frozen.json"))
direct_pairs = [tuple(p) for p in base["pairs"]]
children = list(dict.fromkeys(c for c, p in direct_pairs))

# child 라벨 → QID + 전이폐포 조상(P279*) 라벨. 배치로.
def closure_for(labels_batch):
    """라벨들의 P279* 조상(한국어 라벨) 회수. label→set(ancestors)."""
    values = " ".join(f'"{l}"@ko' for l in labels_batch)
    q = f"""
    SELECT ?cL ?ancL WHERE {{
      VALUES ?cL {{ {values} }}
      ?c rdfs:label ?cL .
      ?c wdt:P279+ ?anc .
      ?anc rdfs:label ?ancL . FILTER(lang(?ancL)="ko")
    }} LIMIT 5000
    """
    out = {}
    try:
        for b in sparql(q):
            cL, aL = b["cL"]["value"], b["ancL"]["value"]
            if cL != aL:
                out.setdefault(cL, set()).add(aL)
    except Exception as e:
        print(f"  closure batch fail: {str(e)[:50]}")
    return out

print("전이폐포 조상 회수 중...")
closure = {}
B = 40
for i in range(0, len(children), B):
    batch = children[i:i+B]
    c = closure_for(batch)
    for k, v in c.items():
        closure.setdefault(k, set()).update(v)
    print(f"  {i+len(batch)}/{len(children)} (누적 조상보유 {len(closure)})")
    time.sleep(1)

# gold_closure: child -> set(모든 조상, 직접+전이)
gold_closure = {}
for c, p in direct_pairs:
    gold_closure.setdefault(c, set()).add(p)
for c, ancs in closure.items():
    gold_closure.setdefault(c, set()).update(ancs)

# 2) 독립 소스 = 한국어 위키피디아 lead (Wikidata desc 아님)
print("\n위키피디아 lead 회수 중...")
leads = {}
for i, c in enumerate(children):
    lead = wiki_lead(c)
    if lead:
        leads[c] = lead
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(children)} (lead 보유 {len(leads)})")
    time.sleep(0.05)

# 3) held-out split — 결정적(정렬 기반), test 25%
kids_with_lead = sorted(c for c in children if c in leads and c in gold_closure)
test = set(kids_with_lead[::4])       # 매 4번째 = test
dev = [c for c in kids_with_lead if c not in test]

out = {
    "direct_pairs": [list(p) for p in direct_pairs],
    "gold_closure": {k: sorted(v) for k, v in gold_closure.items()},
    "leads": leads,
    "dev": dev,
    "test": sorted(test),
}
json.dump(out, open("gold_r1.json", "w"), ensure_ascii=False, indent=1)
print(f"\n저장 gold_r1.json: 직접쌍 {len(direct_pairs)}, 폐포보유 {len(gold_closure)}, "
      f"lead {len(leads)}, dev {len(dev)}, test {len(test)}")
# substring baseline: lead 에 gold 조상이 substring 인 비율
sub = sum(1 for c in kids_with_lead if any(a in leads[c] for a in gold_closure[c]))
print(f"substring baseline: lead 에 gold조상 포함 = {sub}/{len(kids_with_lead)} "
      f"= {sub/len(kids_with_lead)*100:.1f}% (방법이 이걸 넘어야 함)")
