import urllib.request, urllib.parse, json, time, sys
UA={"User-Agent":"ontokit-research/1.0 (donguk0808@gmail.com)"}
def lead(title):
    t=urllib.parse.quote(title.replace(" ","_"))
    url=f"https://ko.wikipedia.org/api/rest_v1/page/summary/{t}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=10) as r:
            d=json.load(r)
        return "" if d.get("type")=="disambiguation" else d.get("extract","")
    except Exception:
        return ""
g=json.load(open("gold_r1.json"))
leads=g.get("leads",{})
todo=[c for c in g["gold_closure"] if not leads.get(c)]
print(f"남은 {len(todo)} (기존 {len([v for v in leads.values() if v])})", flush=True)
done=0
for i,c in enumerate(todo):
    l=lead(c)
    if l: leads[c]=l
    done+=1
    if done%50==0:
        g["leads"]=leads
        json.dump(g,open("gold_r1.json","w"),ensure_ascii=False)
        print(f"  {done}/{len(todo)} 저장 (총 lead {len([v for v in leads.values() if v])})", flush=True)
    time.sleep(0.2)
# 최종 저장 + split
kids=sorted(c for c in g["gold_closure"] if leads.get(c))
g["leads"]=leads; g["dev"]=[c for c in kids if kids.index(c)%4!=0]; g["test"]=[c for c in kids if kids.index(c)%4==0]
json.dump(g,open("gold_r1.json","w"),ensure_ascii=False,indent=1)
print(f"완료: lead {len(kids)}, dev {len(g['dev'])}, test {len(g['test'])}", flush=True)
