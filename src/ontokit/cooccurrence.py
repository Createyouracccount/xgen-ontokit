"""동시출현 약관계(coOccursWith) — 결정적·언어무관·목록 0. (0712 mixed20k 실증)

문제: LLM-free 관계추출(relation_ko)은 한국어 전용 + 조사규칙 정밀도 우선이라
관계 밀도가 낮다 — mixed20k 실측: 도메인 관계 31,662 = 전체 트리플의 1.1%,
영어 절반(1만 문서)은 관계 0. 검색 관계시드·답변 관계블록의 재료가 부족하다.

설계 (GraphRAG/LightRAG 계열 co-occurrence 관행의 결정적 구현):
- 같은 청크에 동시출현한 엔티티쌍 → 약한 관계 후보.
- 선별은 통계만(목록 0):
  ① pair df >= min_pair_df — 재사용 원칙(우연 1회 동시출현 배제, 클래스
     승격 필터의 df>=2 와 같은 termhood 논리).
  ② lift > lift_k — c·N > k·df_a·df_b (PMI > log k 와 동형). 허브 엔티티끼리
     우연히 자주 만나는 쌍(mixed20k 실측 '미국-선수' lift 1.45) 배제.
- 술어는 '관련'이 아니라 **함께언급(coOccursWith)**:
  ① 동시출현은 결정적으로 참 — 의미관계('X가 Y와 관련')를 단정하지 않는
     정직한 술어. ② 검색측 술어게이트(_seed_relational, 질문어-술어라벨
     매칭)가 한국어 상용어 '관련'에 오트리거하는 것을 회피.
- 라벨 자격은 콜백(label_ok) 주입 — 형태규칙(반복·기호심장·달력파편)은
  내장 기본값, 형태소 기반(단독 의존명사 등)은 호출측이 kiwi 로 조합 주입.

비용·스코프 공시:
- coarse 관계다(관계 종류 없음). SVO(relation_ko)와 술어를 분리해 소비측이
  정밀 관계를 우선(co-occ 는 recall 폴백)하도록 배선하는 것이 전제.
- **클래스 승격 필터의 has_rel 근거로 세면 필터가 무력화된다** — 호출측은
  has_rel 판정 쿼리에서 coOccursWith 를 반드시 제외할 것.
- 이미 SVO 로 연결된 쌍은 방출 제외(exclude_pairs) — 정밀 관계 희석 방지.
- 대칭 관계라 양방향 2트리플 방출(검색 단방향 쿼리 대칭성) — 볼륨 2배 공시.
- 쌍 카운터는 메모리 상주 — mixed20k(청크 2만·쌍 종수 388만) 실측 통과,
  수백만 청크 규모는 미검증(청크당 상한이 상계를 잡지만 실측 필요).
- 영어 서브워드 파편 라벨('ikos' 류 NER 아티팩트)은 어휘 없이 결정적 판별
  불가 — v1 미처리(pair df·lift 가 부분 억제).
- ⚠️ mixed-case 2글자 거부 규칙은 **일반 코퍼스 가정** — 화학기호(Fe·Cu·Li)·
  단위(Hz·eV)·경칭(Dr·St)을 죽인다. mixed20k 실측상 upstream 2글자 라벨은
  순 파편(Ur·Vm·Ad)이라 유효 손실 0 이었으나, 과학·화학 도메인 코퍼스 이식 시
  이 규칙을 끄거나 화이트리스트를 붙여야 한다(R2 적대검증 N2 공시).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable, Optional

PREDICATE_NAME = "coOccursWith"
PREDICATE_LABEL_KO = "함께언급"
PREDICATE_LABEL_EN = "co-occurs with"

# 닫힌 시간·달력 단위(문법 기능어 — 도메인 명사 아님). 숫자·기호를 걷어낸
# 잔여가 전부 이 집합이면 날짜 파편('0년', '월_1일', '1세기')으로 판정.
CALENDAR_UNITS = frozenset({
    "년", "월", "일", "시", "분", "초", "세기", "년대", "연도", "기원전", "기원후",
})

_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)  # 문자(숫자·기호·밑줄 제외)
_LATIN = re.compile(r"[A-Za-z]")
_HANGUL = re.compile(r"[가-힣]")
_CJK = re.compile(r"[가-힣一-鿿]")
_ALPHA_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)
# 괄호·고립 인용부호 — NER 이 원문 구절을 자른 파편의 흔적("Regional ) League",
# "' Union", "Margaret ("). 도메인 무관 형태 신호.
_PUNCT_FRAG = re.compile(r"[()\[\]{}]|^['\"‘’“”]|['\"‘’“”]$")
# 구두점 가장자리 파편 — 하이픈·쉼표·콜론 끝 or 하이픈·점·쉼표·콜론 시작
# ("North Rhine -", ". Michael"). 점 끝은 제외(약어 U.S./Inc./Jr. 보존).
_EDGE_PUNCT = re.compile(r"^[-.,:;]|[-,:;!]$")
# 한글 토큰 + 독립 숫자 토큰 — 한국어 명사에 숫자가 공백으로 붙으면 연도·조번호
# 잘림 파편("고려 892"=고려 892년, "제1조 2"). 한글 조건 없으면 영어 명명 패턴
# (Boeing 747·Apollo 11·iPhone 12·Area 51) 을 오살한다(R2 적대검증 N1). 라틴
# 모델명은 df·lift 로만 억제. 붙은 모델명(AH-64·B-1A)은 토큰이 안 쪼개져 보존.
_NUM_TOKEN = re.compile(r"^\d+$")


def default_label_ok(label: str) -> bool:
    """형태 기반 라벨 자격 — 언어무관 결정 규칙(클래스 승격 필터 ②③ 계열).

    거부: 빈 라벨 / 문자 2자 미만(기호·숫자 심장 '___', '0.5') /
    전체 이중반복('서울서울') / 달력 파편('0년', '년_1월', '월_1일', '1세기') /
    괄호·고립 인용부호 파편('North Rhine -', '. Michael') / 한글+독립숫자 잘림
    ('고려 892' — 영어 'Boeing 747'·'Apollo 11'은 정당 명명이라 보존) / 라틴 미소
    파편(소문자 시작 ∧ 최장 알파벳연쇄 ≤2: 'ho','ur','pi' — 0712 mixed20k 실측.
    all-caps 약어·모델명 SI/OH/PS2/An-26 은 대문자 시작이라 보존).
    """
    label = (label or "").strip()
    if not label:
        return False
    letters = _ALPHA.findall(label)
    if len(letters) < 2:
        return False
    n = len(label)
    if n >= 6 and n % 2 == 0 and label[: n // 2] == label[n // 2:]:
        return False
    # 달력 파편: 숫자·기호 제거 잔여를 달력 단위 집합의 조합으로 소진 가능한가
    residue = "".join(letters)
    if _consumes_all(residue, CALENDAR_UNITS):
        return False
    if _PUNCT_FRAG.search(label) or _EDGE_PUNCT.search(label):
        return False
    toks = label.split()
    if len(toks) >= 2 and _HANGUL.search(label) \
            and any(_NUM_TOKEN.match(t) for t in toks):
        return False
    # 라틴 미소 파편 — CJK·한글이 없는 라벨에만 적용(다국어 오손실 방지)
    if not _CJK.search(label):
        # 정확히 2글자 알파벳이면서 all-caps 아님('Ah'·'Re'·'Uk') → 이름·외국어
        # 조각. all-caps 2글자(US·UK·SI·OH)는 약어라 보존(0712 실측: mixed-case
        # 146건 전량 파편 vs all-caps 는 약어 다수).
        if len(label) == 2 and _LATIN.match(label) and label[1:2].isalpha() \
                and not label.isupper():
            return False
        m = _LATIN.search(label)
        if m and label[m.start()].islower():
            longest = max((len(x) for x in _ALPHA_RUN.findall(label)), default=0)
            if longest <= 2:
                return False
    return True


def _consumes_all(text: str, units: frozenset) -> bool:
    """text 가 units 원소들의 연접만으로 완전 소진되는지(greedy 최장일치)."""
    i, n = 0, len(text)
    while i < n:
        for u in sorted(units, key=len, reverse=True):
            if text.startswith(u, i):
                i += len(u)
                break
        else:
            return False
    return True


class CooccurrenceCollector:
    """청크 스트리밍으로 엔티티 동시출현을 집계, 통계 선별된 쌍만 방출.

    사용:
        col = CooccurrenceCollector()
        col.add_chunk(chunk_id, [(key, label), ...])   # key=URI 또는 이름
        for a, b, count in col.edges(exclude_pairs=svo_pairs): ...

    순수 집계(IO 없음). 그래프 방출(SPARQL INSERT)은 호출측 몫.
    """

    def __init__(self, *, min_pair_df: int = 3, lift_k: float = 2.0,
                 max_entities_per_chunk: int = 60,
                 label_ok: Optional[Callable[[str], bool]] = default_label_ok):
        self.min_pair_df = min_pair_df
        self.lift_k = lift_k
        self.max_entities_per_chunk = max_entities_per_chunk
        self.label_ok = label_ok
        self._pair: Counter = Counter()
        self._ent_df: Counter = Counter()
        self._chunks: set = set()
        self.stats = {"chunks": 0, "entities_rejected": 0, "chunks_truncated": 0}

    def add_chunk(self, chunk_id: str, entities: Iterable[tuple[str, str]]) -> None:
        """entities: (key, label) — 같은 청크 내 중복 key 는 1회로 집계."""
        if chunk_id in self._chunks:
            return
        self._chunks.add(chunk_id)
        self.stats["chunks"] += 1
        keys: dict = {}
        rejected = 0
        for key, label in entities:
            if key in keys:
                continue
            if self.label_ok is not None and not self.label_ok(label):
                rejected += 1
                continue
            keys[key] = True
        self.stats["entities_rejected"] += rejected
        es = sorted(keys)  # 결정적 순서
        if len(es) > self.max_entities_per_chunk:
            # 허브 청크 절단(폭발 방지) — 정렬 후 앞쪽 유지(결정적). 절단 발생은 통계로 공시.
            self.stats["chunks_truncated"] += 1
            es = es[: self.max_entities_per_chunk]
        for e in es:
            self._ent_df[e] += 1
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                self._pair[(es[i], es[j])] += 1

    def edges(self, exclude_pairs: Optional[set] = None) -> list[tuple[str, str, int]]:
        """선별 통과 쌍 [(a, b, count)] — a<b 정규순서, count 내림차순.

        exclude_pairs: 이미 정밀 관계(SVO)로 연결된 {(a, b)} (양방향 무관 —
        내부에서 정규화). 해당 쌍은 방출하지 않는다.
        """
        N = len(self._chunks)
        if N == 0:
            return []
        excl = set()
        for a, b in (exclude_pairs or ()):
            excl.add((a, b) if a <= b else (b, a))
        out = []
        for (a, b), c in self._pair.items():
            if c < self.min_pair_df:
                continue
            if c * N <= self.lift_k * self._ent_df[a] * self._ent_df[b]:
                continue
            if (a, b) in excl:
                continue
            out.append((a, b, c))
        out.sort(key=lambda t: (-t[2], t[0], t[1]))
        return out


# 단독 의존명사(NNB) — 닫힌 문법 집합. '마리'·'가지' 같은 수분류사가 NER
# 아티팩트로 엔티티가 된 경우를 거부한다(mixed20k 쌍 표본 실측 '마리-한국',
# '가지-경상남도'). 도메인 명사 목록이 아니라 품사 판정이므로 원칙 안.
def make_korean_label_ok(kiwi=None):
    """기본 형태규칙 + 한국어 단독 의존명사 거부 콜백 팩토리.

    kiwipiepy 미설치·미주입이면 기본 형태규칙만 동작(fail-open).
    반환 콜백은 CooccurrenceCollector(label_ok=...)에 주입.
    """
    holder = {"kiwi": kiwi, "tried": kiwi is not None}
    cache: dict = {}

    def _ok(label: str) -> bool:
        if not default_label_ok(label):
            return False
        if not holder["tried"]:
            holder["tried"] = True
            try:
                from kiwipiepy import Kiwi
                holder["kiwi"] = Kiwi()
            except ImportError:
                holder["kiwi"] = None
        k = holder["kiwi"]
        if k is None or not re.search(r"[가-힣]", label):
            return True
        if label not in cache:
            if len(cache) > 200_000:
                cache.clear()
            toks = k.analyze(label, top_n=1)[0][0]
            # ① 단독 의존명사(NNB) '마리'·'가지' 거부.
            # ② 조사 종결('조선엔'=조선+에+ㄴ, 'X는') 거부 — 종결 형태소가 조사(J*)
            #    면 원문에서 잘린 파편(NER 경계 오류). 닫힌 품사 판정, 목록 0.
            bad = (len(toks) == 1 and toks[0].tag == "NNB") or (
                bool(toks) and toks[-1].tag.startswith("J"))
            cache[label] = not bad
        return cache[label]

    return _ok
