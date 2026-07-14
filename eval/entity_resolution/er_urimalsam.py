"""ER 사전 채널 — 우리말샘 비슷한말(편찬 동의어)로 확정 동의어 판정. LLM 0회.

심판 3R 결론: 임베딩(KURE 포함)은 주제근접(삼성전자~LG전자 0.690)과 동의어
(전자상거래~이커머스 0.690)를 원리적으로 못 가름 → AUC 0.82 천장. 진짜 동의어는
**사람이 사전에 등재한 것**. 우리말샘 relation_info[type=비슷한말] = 편찬 동의어.

자원: 우리말샘(국립국어원, CC BY-SA 2.0 KR, 상업 OK). GitHub 미러
spellcheck-ko/korean-dict-nikl/opendict (API키 불필요, 110만 어휘).
누수 없음: gold=위키 redirect ⊥ 자원=우리말샘(독립 소스, 심판 반복 경고 회피).

방식: 비슷한말 쌍으로 union-find 동의어 집합 구성. 두 표기가 같은 집합 = 동의어.
결정적·무학습. LLM API 0회.
"""
from __future__ import annotations
import glob
import os
import re

_CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_HOMONYM = re.compile(r"\d+$")  # 표제어 끝 동형이의어 번호(백001)

# 심판 R4 적발: 비슷한말은 비이행적(구=구체/구절/구역 동형이의어)인데 union-find가
# 이행 병합 + 동형번호 제거로 sense 파괴 → 3만 표기 거대 블롭(미국=일본=한국 오병합).
# 수정: ①동형번호 보존(sense 키 분리) ②집합 크기 캡으로 블롭 폐기.
_MAX_SET = 30  # 동의어 집합 최대 멤버 — 초과 시 블롭(오병합)으로 간주해 무효


def _clean(w: str) -> str:
    """CDATA·붙임표 제거. ⚠️동형번호는 sense 구분 위해 보존(R4 버그수정)."""
    if not w:
        return ""
    m = _CDATA.search(w)
    if m:
        w = m.group(1)
    return w.replace("-", "").replace("^", "").strip()


def _surface(w: str) -> str:
    """동형번호 뗀 표면형 — 조회 시 사용(gold 표기엔 번호 없음)."""
    return _HOMONYM.sub("", _clean(w))


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


class UrimalsamER:
    """우리말샘 비슷한말 union-find. 같은 집합 = 동의어(확정)."""

    def __init__(self, xml_dir="data/urimalsam", cache="data/urimalsam_syn.txt"):
        self._uf = UnionFind()
        self._cache = cache
        if cache and os.path.exists(cache):
            self._load_cache()
        elif xml_dir and os.path.isdir(xml_dir):
            self._build(xml_dir)
            if cache:
                self._save_cache()

    def _build(self, xml_dir):
        # union-find 를 sense-keyed 표기(동형번호 보존)로 구성 — 비이행 병합 방지.
        items_re = re.compile(r"<item>.*?</item>", re.DOTALL)
        head_re = re.compile(r"<word>(.*?)</word>", re.DOTALL)
        rel_re = re.compile(r"<relation_info>.*?</relation_info>", re.DOTALL)
        type_re = re.compile(r"<type>(.*?)</type>", re.DOTALL)
        n = 0
        for path in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
            txt = open(path, encoding="utf-8").read()
            for it in items_re.findall(txt):
                hm = head_re.search(it)
                if not hm:
                    continue
                head = _clean(hm.group(1))  # sense 보존(백001)
                if not head:
                    continue
                for rel in rel_re.findall(it):
                    tm = type_re.search(rel)
                    if not tm or "비슷한말" not in tm.group(1):
                        continue
                    wm = head_re.search(rel)
                    if not wm:
                        continue
                    syn = _clean(wm.group(1))
                    if syn and syn != head:
                        self._uf.union(head, syn)
                        n += 1
        self._pairs_built = n
        self._finalize()

    def _finalize(self):
        """블롭 폐기(집합>_MAX_SET) + 표면형→유효 sense-대표 인덱스 구성.

        gold 표기엔 동형번호가 없으므로, 조회는 표면형으로. 한 표면형이 여러
        sense 에 걸리면 그 sense-대표들의 집합(surface2reps)으로 판정한다.
        """
        from collections import Counter
        root_size = Counter(self._uf.find(k) for k in self._uf.p)
        self._surface2reps = {}
        for k in self._uf.p:
            r = self._uf.find(k)
            if root_size[r] > _MAX_SET:      # 블롭 폐기(오병합)
                continue
            if root_size[r] < 2:             # 동의어 없는 단독은 무의미
                continue
            # k(sense-keyed, 백001) 와 그 표면형(백) 둘 다 조회 대상 — relation 의
            # syn 쪽 표기도 표면형으로 인덱싱돼야(gold 는 번호 없음). 동형번호 없는
            # 원표기도 union.p 에 있으면 함께 커버됨.
            self._surface2reps.setdefault(_surface(k), set()).add(r)
        self._blob_dropped = sum(1 for r, c in root_size.items() if c > _MAX_SET)

    def _save_cache(self):
        with open(self._cache, "w", encoding="utf-8") as f:
            for surf, reps in self._surface2reps.items():
                f.write(surf + "\t" + " ".join(sorted(reps)) + "\n")

    def _load_cache(self):
        self._surface2reps = {}
        with open(self._cache, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    self._surface2reps[parts[0]] = set(parts[1].split())

    def same_entity(self, a, b) -> bool:
        """두 표면형이 공유하는 sense-대표가 있으면 동의어(같은 비슷한말 집합)."""
        ra = self._surface2reps.get(_surface(a))
        rb = self._surface2reps.get(_surface(b))
        if not ra or not rb:
            return False
        return bool(ra & rb)

    def size(self):
        return len(getattr(self, "_surface2reps", {}))
