"""ER gold 빌드 — 한국어 위키피디아 redirect(넘겨주기) = 이명/동의어.

외부 신뢰 gold: redirect 표제어 → canonical 표제어 는 사람이 편집한 동의어 관계.
CC BY-SA 4.0(상업 OK). API 키 불필요. 조사로 접근 검증(HTTP 200, 40+/표제어).

누수 차단(계층 축 교훈): gold 소스=redirect 테이블. 빌더 입력 코퍼스와 분리.
평가는 held-out — seed 표제어 목록에서 수집, 빌더 학습에 미사용.

동의어 유형 태깅(난이도 분리):
  surface  = 형태소 baseline 이 이미 잡는 표면변이(공백·하이픈·대소문자만 차이)
  semantic = 표면 다른 의미변이(우한폐렴↔코로나19, COVID-19↔코로나19) ← ER 축의 핵심
하드 네거티브: 같은 canonical 아래 있지 않은, 표면 유사 쌍(형태소키 충돌 등).

사용: python3 build_gold.py  (API, ~수분). 출력 data/gold.json
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

API = "https://ko.wikipedia.org/w/api.php"
UA = "xgen-ontokit-eval/1.0 (research; ER gold build)"

# 다양한 도메인의 seed 표제어 — redirect 풍부한 개념(약어·외래어·이명 많은 것 위주).
# 특정 도메인 편향 방지 위해 IT·의학·기업·지명·기관·과학 혼합.
SEEDS = [
    "코로나바이러스감염증-19", "인공지능", "전자상거래", "블록체인", "사물인터넷",
    "삼성전자", "엘지전자", "현대자동차", "대한민국", "서울특별시",
    "미국", "중화인민공화국", "일본", "유럽 연합", "국제 연합",
    "세계보건기구", "국제통화기금", "북대서양 조약 기구", "경제협력개발기구",
    "머신러닝", "딥러닝", "자연어 처리", "컴퓨터 비전", "클라우드 컴퓨팅",
    "비트코인", "이더리움", "메타 (기업)", "구글", "마이크로소프트",
    "당뇨병", "고혈압", "심근경색", "뇌졸중", "결핵",
    "제2차 세계 대전", "냉전", "산업 혁명", "르네상스", "계몽주의",
    "상대성이론", "양자역학", "진화", "유전자", "광합성",
    "축구", "야구", "농구", "올림픽", "월드컵",
    "카카오 (기업)", "네이버 (기업)", "쿠팡", "배달의민족", "토스 (기업)",
]

_HANGUL_SPACE_HYPHEN = re.compile(r"[\s\-·・_()（）]")


def _get(params):
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                time.sleep(2 ** attempt)  # 1,2,4,8초 백오프
                continue
            raise


def redirects_of(title):
    """title 로 들어오는 모든 redirect 표제어(이명) 수집(페이지네이션)."""
    out, cont = [], {}
    while True:
        d = _get({"action": "query", "titles": title, "prop": "redirects",
                  "rdlimit": "500", **cont})
        for p in d.get("query", {}).get("pages", []):
            for r in p.get("redirects", []):
                out.append(r["title"])
        if "continue" in d:
            cont = {"rdcontinue": d["continue"]["rdcontinue"]}
            time.sleep(0.2)
        else:
            break
    return out


def _norm_surface(s):
    """공백·하이픈·괄호·대소문자 제거 정규화 — 표면변이 판별용(형태소 baseline 근사)."""
    return _HANGUL_SPACE_HYPHEN.sub("", s).lower()


def classify(alias, canonical):
    """surface(표면변이) vs semantic(의미변이). 형태소 baseline 이 잡는지 근사."""
    return "surface" if _norm_surface(alias) == _norm_surface(canonical) else "semantic"


def main():
    positives, seen = [], set()
    canon_aliases = {}  # canonical → [alias...]
    for i, seed in enumerate(SEEDS):
        try:
            reds = redirects_of(seed)
        except Exception as e:
            print(f"  ! {seed}: {e}", file=sys.stderr)
            continue
        # canonical = seed 자신(정식 표제어). alias 각각과 쌍.
        aliases = [a for a in reds if a and a != seed]
        canon_aliases[seed] = aliases
        for a in aliases:
            key = tuple(sorted((a, seed)))
            if key in seen:
                continue
            seen.add(key)
            positives.append({"a": a, "b": seed, "type": classify(a, seed)})
        print(f"[{i+1}/{len(SEEDS)}] {seed}: {len(aliases)} aliases", flush=True)
        time.sleep(1.0)

    # 하드 네거티브: 서로 다른 canonical 의 alias 끼리(같은 개체 아님) 쌍.
    # 표면 유사(정규화키 겹침)한 것 우선 — "전부 동의어" 허수아비 방어.
    negatives = []
    canons = list(canon_aliases)
    all_terms = [(a, c) for c, al in canon_aliases.items() for a in ([c] + al)]
    by_norm = {}
    for term, canon in all_terms:
        by_norm.setdefault(_norm_surface(term)[:2], []).append((term, canon))
    for bucket in by_norm.values():
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                (t1, c1), (t2, c2) = bucket[i], bucket[j]
                if c1 != c2 and t1 != t2:  # 다른 개체
                    negatives.append({"a": t1, "b": t2})
    # 네거티브를 positives 규모 근처로 캡(균형)
    negatives = negatives[:len(positives)]

    n_surface = sum(1 for p in positives if p["type"] == "surface")
    n_semantic = len(positives) - n_surface
    gold = {"positives": positives, "negatives": negatives,
            "meta": {"source": "ko.wikipedia redirect", "license": "CC BY-SA 4.0",
                     "seeds": len(SEEDS), "n_positive": len(positives),
                     "n_surface": n_surface, "n_semantic": n_semantic,
                     "n_negative": len(negatives)}}
    with open("data/gold.json", "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=1)
    print(f"\ngold: positives {len(positives)} (surface {n_surface} / "
          f"semantic {n_semantic}), negatives {len(negatives)} → data/gold.json")


if __name__ == "__main__":
    main()
