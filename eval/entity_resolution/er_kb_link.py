"""ER W3C 정석 PoC — 개체를 외부 KB에 링크, 같은 개체면 sameAs(동의어). LLM 0회.

재프레이밍(사용자 지적 + W3C 표준): 동의어는 텍스트에서 '발견'하는 게 아니라
온톨로지가 owl:sameAs/skos:altLabel 로 '선언'하는 것. 우리 빌더 개체를 외부
KB 엔티티(QID)에 링크하면, 같은 QID = owl:sameAs = 동의어.

이 파일은 재설계 방향의 '가능성'을 gold 로 측정하는 PoC다. 핵심 관찰:
  현 er_dict(wbsearchentities 표기→QID)와 이 접근의 차이는 '링크 정확도'.
  entity linking 을 어떻게 하느냐가 순가치를 좌우한다.

누수 주의: gold=한국어 위키 redirect. Wikidata QID 링크는 위키 redirect 를
직접 참조하지 않으므로(엔티티 검색은 label 기반) 독립. 단 Wikidata 와 위키피디아가
같은 재단이라 완전 독립은 아님 — 심판 판정 대상.

⚠️ 이건 PoC. 실 배선은 빌드타임 KB 스냅샷 오프라인 링크가 원칙(API 왕복 불가).
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

WD_API = "https://www.wikidata.org/w/api.php"
UA = "xgen-ontokit-eval/1.0 (research; ER KB-link PoC)"
_CACHE = "data/kb_link_cache.json"


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
            return {}
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
                continue
            return {}


class KBEntityLinker:
    """표기 → Wikidata 엔티티(QID) 링크 + altLabel 확장. 같은 QID = sameAs.

    현 er_dict 와의 차이:
      - er_dict: 표기 검색 top-k QID 교집합(느슨)
      - 이것: top-1 링크 + 그 엔티티의 altLabel 도 같은 QID 로 등록(사전 확장).
        entity linking 을 '개체 단위'로 봄 — 표기가 아니라 개체가 1급.
    """

    def __init__(self, online=True):
        self._online = online
        self._term2qid = {}   # 표기 → top-1 QID
        self._qid2labels = {} # QID → {label, altLabels...}
        try:
            with open(_CACHE, encoding="utf-8") as f:
                c = json.load(f)
                self._term2qid = c.get("term2qid", {})
                self._qid2labels = c.get("qid2labels", {})
        except FileNotFoundError:
            pass

    def _link(self, term):
        """표기 → top-1 QID(entity linking). 캐시."""
        if term in self._term2qid:
            return self._term2qid[term]
        if not self._online:
            return None
        d = _get({"action": "wbsearchentities", "search": term,
                  "language": "ko", "uselang": "ko", "limit": 1})
        hits = d.get("search", [])
        qid = hits[0]["id"] if hits else None
        self._term2qid[term] = qid
        time.sleep(0.12)
        return qid

    def _labels(self, qid):
        """QID → 한국어 label + altLabels(사전 확장용)."""
        if qid in self._qid2labels:
            return self._qid2labels[qid]
        if not self._online or not qid:
            return []
        d = _get({"action": "wbgetentities", "ids": qid,
                  "props": "labels|aliases", "languages": "ko"})
        ent = d.get("entities", {}).get(qid, {})
        labels = []
        lab = ent.get("labels", {}).get("ko", {}).get("value")
        if lab:
            labels.append(lab)
        labels += [a["value"] for a in ent.get("aliases", {}).get("ko", [])]
        self._qid2labels[qid] = labels
        time.sleep(0.12)
        return labels

    def save(self):
        with open(_CACHE, "w", encoding="utf-8") as f:
            json.dump({"term2qid": self._term2qid,
                       "qid2labels": self._qid2labels}, f, ensure_ascii=False)

    def same_entity(self, a, b):
        """a·b 가 같은 KB 엔티티(QID)로 링크되면 동의어(owl:sameAs)."""
        qa, qb = self._link(a), self._link(b)
        return bool(qa and qb and qa == qb)
