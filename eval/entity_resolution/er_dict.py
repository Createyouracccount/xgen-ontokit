"""ER 의미변이 채널 — 외부 사전(Wikidata) 조회로 표면 다른 동의어 판정. LLM 0회.

원리(계층 접미공유·관계 조사SVO의 ER 대응): 표면 문자열로는 "우한폐렴=코로나19"
불가. 두 표기가 **같은 Wikidata 개체(QID)로 링크되면 동의어**. 이는 위키피디아
redirect(gold)와 독립 소스(Wikidata entity linking)라 누수 없음.

무학습·결정적: Wikidata wbsearchentities(표기→QID) 조회 + QID 일치 판정. 캐시로
반복 조회 제거. LLM API 0회. 오프라인(사전 스냅샷) 모드도 지원.

⚠️ 평가 편의를 위해 온라인 조회(API)를 쓰나, 본체 배선 시엔 빌드타임에 사전
스냅샷을 만들어 오프라인 조회가 원칙(대용량서 API 왕복 불가). 여기선 채널의
'능력'을 gold 로 측정하는 게 목적.
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

WD_API = "https://www.wikidata.org/w/api.php"
UA = "xgen-ontokit-eval/1.0 (research; ER dict channel)"
_CACHE_PATH = "data/wd_qid_cache.json"
_NORM = re.compile(r"[\s]+")


def _get(params):
    url = WD_API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(1 + attempt)
                continue
            return {}  # 조회 실패는 빈 결과(hang 방지) — 해당 표기는 QID 없음 처리
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
                continue
            return {}


class WikidataERDict:
    """표기 → Wikidata QID 집합 조회 + QID 일치로 동의어 판정. 캐시 영속."""

    def __init__(self, cache_path=_CACHE_PATH, online=True, top_k=3):
        self._cache = {}
        self._path = cache_path
        self._online = online
        self._top_k = top_k
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self._cache = json.load(f)

    def _qids(self, term):
        key = _NORM.sub(" ", term.strip())
        if key in self._cache:
            return set(self._cache[key])
        if not self._online:
            return set()
        try:
            d = _get({"action": "wbsearchentities", "search": key,
                      "language": "ko", "uselang": "ko", "limit": self._top_k})
            qids = [r["id"] for r in d.get("search", [])]
        except Exception:
            qids = []
        self._cache[key] = qids
        time.sleep(0.15)
        return set(qids)

    def save(self):
        if self._path:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)

    def same_entity(self, a, b):
        """a·b 가 같은 Wikidata 개체(QID 교집합)면 동의어. top-1 우선 신뢰."""
        qa, qb = self._qids(a), self._qids(b)
        if not qa or not qb:
            return False
        return bool(qa & qb)
