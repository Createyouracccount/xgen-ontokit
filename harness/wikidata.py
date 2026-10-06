"""Wikidata 수집 — 정답의 원천. 도구(ontokit) 출력과 무관한 외부 사실.

1) 코퍼스 문서 제목 → kowiki sitelink 로 QID·claims 를 받는다(wbgetentities, 50건/호출).
2) claims 가 가리키는 객체 QID 의 한국어 라벨·별칭을 받는다.
3) 객체의 P279(상위 클래스) 사슬을 받아 클래스 폐포를 계산할 수 있게 한다.

결과는 harness/data/wikidata/ 에 캐시한다. 재실행 시 네트워크를 다시 타지 않는다.

  python -m harness.wikidata harness/data/wiki2/docs.jsonl harness/data/wikidata
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

API = "https://www.wikidata.org/w/api.php"
UA = "xgen-ontokit-harness/0.1 (research; local evaluation)"

# 질문 축으로 쓰는 속성. 위키 리드 문장에 자주 서술되는 것만 고른다.
PROPS = ["P31", "P279", "P106", "P27", "P19", "P20", "P69", "P108", "P17", "P131",
         "P159", "P112", "P127", "P749", "P355", "P361", "P463", "P54", "P102",
         "P50", "P57", "P175", "P495", "P136", "P39", "P26", "P22", "P25", "P40"]


def _get(params, retries=8):
    q = urllib.parse.urlencode({**params, "format": "json"})
    for i in range(retries):
        try:
            req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
            r = json.load(urllib.request.urlopen(req, timeout=60))
            err = r.get("error")
            if err:  # 200 안의 오류(maxlag 등)를 '매핑 없음'으로 삼키지 않는다
                if err.get("code") == "maxlag" and i < retries - 1:
                    wait = max(5, int(float(err.get("lag", 5))) + 2)
                    print(f"  maxlag {err.get('lag')} — {wait}s 대기", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"wikidata API 오류: {err.get('code')} {err.get('info')}")
            return r
        except urllib.error.HTTPError as e:  # 429/5xx 만 재시도, Retry-After 를 지킨다
            if i == retries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * (i + 1)
            print(f"  HTTP {e.code} — {wait}s 대기", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))


def _claims(ent):
    out = {}
    for p in PROPS:
        vals = []
        for c in ent.get("claims", {}).get(p, []):
            dv = c.get("mainsnak", {}).get("datavalue", {})
            if dv.get("type") == "wikibase-entityid":
                vals.append(dv["value"]["id"])
        if vals:
            out[p] = vals
    return out


def _names(ent):
    lab = ent.get("labels", {}).get("ko", {}).get("value")
    als = [a["value"] for a in ent.get("aliases", {}).get("ko", [])]
    en = ent.get("labels", {}).get("en", {}).get("value")
    return {"label": lab, "aliases": als, "en": en}


def fetch_subjects(titles, cache):
    path = os.path.join(cache, "subjects.json")
    got = json.load(open(path)) if os.path.exists(path) else {}
    todo = [t for t in titles if t not in got]
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        r = _get({"action": "wbgetentities", "sites": "kowiki", "titles": "|".join(batch),
                  "props": "claims|labels|aliases|sitelinks", "languages": "ko|en",
                  "sitefilter": "kowiki"})
        by_title = {}
        for qid, ent in r.get("entities", {}).items():
            if qid.startswith("-") or "missing" in ent:
                continue
            t = ent.get("sitelinks", {}).get("kowiki", {}).get("title")
            if t:
                by_title[t] = {"qid": qid, **_names(ent), "claims": _claims(ent)}
        for t in batch:
            got[t] = by_title.get(t)  # None = 위키데이터 매핑 없음(명시적으로 기록)
        json.dump(got, open(path, "w"), ensure_ascii=False)
        time.sleep(1.0)
        print(f"  subjects {min(i + 50, len(todo))}/{len(todo)}", flush=True)
    return got


def fetch_entities(qids, cache, name, props="labels|aliases|claims"):
    path = os.path.join(cache, f"{name}.json")
    got = json.load(open(path)) if os.path.exists(path) else {}
    todo = sorted(set(q for q in qids if q not in got))
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        r = _get({"action": "wbgetentities", "ids": "|".join(batch), "props": props,
                  "languages": "ko|en"})
        for qid, ent in r.get("entities", {}).items():
            rec = _names(ent)
            if "claims" in props:
                rec["claims"] = {k: v for k, v in _claims(ent).items() if k in ("P31", "P279")}
            got[qid] = rec
        json.dump(got, open(path, "w"), ensure_ascii=False)
        time.sleep(1.0)
        if (i // 50) % 10 == 0:
            print(f"  {name} {min(i + 50, len(todo))}/{len(todo)}", flush=True)
    return got


def main():
    docs_path, cache = sys.argv[1], sys.argv[2]
    os.makedirs(cache, exist_ok=True)
    titles = [json.loads(l)["title"] for l in open(docs_path)]
    subj = fetch_subjects(titles, cache)
    mapped = {t: v for t, v in subj.items() if v}
    print(f"mapped {len(mapped)}/{len(titles)}")
    if len(mapped) < 0.5 * len(titles):  # 수집 실패를 정답 부재로 오독하지 않는다
        raise SystemExit("매핑률 50% 미만 — 수집 결함 의심, 중단")
    objs = {q for v in mapped.values() for vals in v["claims"].values() for q in vals}
    objs |= {v["qid"] for v in mapped.values()}
    ents = fetch_entities(objs, cache, "objects")
    # 클래스 폐포용: P31/P279 대상의 상위 사슬을 3단계까지 펼친다.
    frontier = {q for e in ents.values() for p in ("P31", "P279") for q in e.get("claims", {}).get(p, [])}
    for depth in range(3):
        frontier -= set(ents)
        if not frontier:
            break
        new = fetch_entities(frontier, cache, "objects")
        ents.update(new)
        frontier = {q for q2 in frontier for e in [ents.get(q2, {})]
                    for q in e.get("claims", {}).get("P279", [])}
    print(f"objects {len(ents)}")


if __name__ == "__main__":
    main()
