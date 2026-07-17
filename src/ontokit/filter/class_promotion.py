"""클래스 승격 필터 — 통계·문법 기반 결정적 판정, LLM 0회. (0712 mixed20k 실증)

문제: LLM-free 추출은 클래스를 과생성한다 — mixed20k(위키 혼합 2만 문서) 실측
444,503 클래스 중 df=1(단일 청크 출처)이 84%, 그 표본 정크율 ~37%(파편·병합),
개체성 ~39%. 그래프 화면·클러스터·저장 전부를 노이즈가 지배.

원칙 (STOP_HEAD 하드코딩 금지 원칙과의 관계):
- 도메인 명사 블랙리스트는 두지 않는다. 판정은 ①코퍼스 내 통계(지지도 df,
  구조 참여)와 ②닫힌 문법 기능어 집합(지시·상대·문서구조어 — 도메인 무관)만 쓴다.
- 승격 기준: "온톨로지 클래스는 재사용되거나(df≥2) 구조에 참여할 때(관계·계층
  부모) 승격" — 1회 등장·무구조 고립 라벨은 클래스 부적격(termhood 표준 관행).
- 지지도 게이트는 재등장 기회가 있는 코퍼스에서만 의미 — 소형 코퍼스(기본
  chunk<5000)에선 자동 비활성(finreg 실측: 유효 금융개념 df=1 다수 → 게이트
  적용 시 오손실. 소형은 과생성 자체가 문제가 아님).

v1 비용·스코프 공시 (R1 적대검증 반영 — 헤드라인 왜곡 금지):
- "V-loss 0"은 **df≥2 정크규칙 스코프 한정**이다. 지지도 게이트는 고립 df=1
  유효 개념도 지운다 — GT 표본 기준 df1의 ~24%가 V(파트타임농부·최고혈압 등),
  게이트 탈락 ~37만 외삽 시 유효 개념 ~9만 삭제 추정. 이는 termhood 설계
  선택(1회 등장·무구조 라벨은 클래스 부적격)의 **의도된 비용**이며, 탈락분은
  사이드카(<graph>__filtered)에 기록되어 가역이다.
- 개체성 라벨(df=1 표본의 ~39%)도 고립이면 게이트로 삭제되며, 그중 상당수
  (표본 실측 59%, n=27)는 NER 인스턴스 채널에도 없어 그래프에서 완전 소실
  — "E는 보존"이 아니라 "E 는 정크 규칙의 표적이 아닐 뿐"이다. NER 신호
  기반 인스턴스 구제는 후속 과제.
- 라틴 전용 라벨의 정크(BC-AD 등)는 v1 미처리(표본 내 소수).
- 잔존 정크(동격 병합 '역사러시아'류): 수식 병합('고대일본어'=유효)과 통계로
  구분 불가해 v1 제외 — 승격셋 정크율 실측 20.3%→12.7%(GT139 기준, 단일
  라벨러 + 블라인드 재라벨 일치 80%로 ±노이즈 있음).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

_HANGUL = re.compile(r"[가-힣]")

# 명사성 종결 태그 — 이 밖(동사·부사·어미 등)으로 끝나는 라벨은 클래스 부적격.
NOUNISH_TAGS = {"NNG", "NNP", "NNB", "XSN", "SL", "SN", "XR"}

# 닫힌 지시·상대어 — 문법 기능어(도메인 명사 아님). head 위치면 파편 신호
# ("통합이전", "소년시절", "이탈리아전역"). 도메인 확장 금지.
# '목록'은 제외 — "재산목록"(상법) 등 문서유형 유효 클래스 오탐(finreg 실측).
REL_HEADS = frozenset({
    "이전", "이후", "당시", "오늘날", "시절", "무렵", "이래", "근처", "부근",
    "인근", "일대", "전역", "가운데", "부분", "경우", "때문", "일부",
})
# 닫힌 지시 관형어 — 첫 형태소면 지시 구성("해당국가", "당해연도").
# '본/각/우리'는 제외 — "본허가"(법령)·"각급법원"·"우리은행" 오탐 위험 > 이득
# (거짓 제거보다 잔존이 낫다는 원칙, finreg '본허가' 오탐 실측).
DEICTIC_FIRST = frozenset({"해당", "당해", "이번"})

# 지지도 게이트 활성 최소 코퍼스 크기(청크 수). 미만이면 게이트 비활성(정크 규칙만).
DEFAULT_GATE_MIN_CHUNKS = 5000


@dataclass
class PromotionDecision:
    keep: bool
    reason: str  # "" | gate | repeat | shape | relhead | deictic | overjoin


class ClassPromotionFilter:
    """클래스 승격 판정 — 순수 결정 함수(IO 없음). 삭제 실행은 호출측(인프라) 몫.

    사용:
        f = ClassPromotionFilter(corpus_chunks=20000)
        d = f.decide(label, df=3, has_rel=False, has_kid=False)
        d.keep, d.reason
    kiwi 미주입 시 lazy 생성(kiwipiepy 없으면 문법 규칙은 통과 처리 — 게이트만).
    """

    def __init__(self, *, corpus_chunks: Optional[int] = None,
                 gate_min_chunks: int = DEFAULT_GATE_MIN_CHUNKS,
                 min_df: int = 2, kiwi=None):
        # corpus_chunks=None(미상)이면 게이트 **비활성** — 미상 코퍼스가 소형이면
        # 유효 df=1 개념을 대량 오삭제한다. "거짓 제거보다 잔존이 낫다" 원칙의
        # fail-open(R1 적대검증 방향 반전). 대용량 정리는 호출측이 청크 수를
        # 실측해 넘겨서 활성화한다.
        self.gate_active = corpus_chunks is not None and corpus_chunks >= gate_min_chunks
        self.min_df = min_df
        self._kiwi = kiwi
        self._kiwi_tried = False
        self._cache: dict[str, list] = {}

    def _tokens(self, label: str):
        """명사성 독법 우선 선택 — 라벨은 원문 문맥에서 명사로 추출된 것인데, 단독
        재분석하면 Kiwi 가 동사·감탄사 독법을 1위로 뽑는 오분석이 흔하다(finreg
        실측: '예금'→IC+VA+EF, '주기'→VV+ETN, '실제명의'→MAG+NNB+JKG — 전부 2위
        독법이 정답 명사). top-3 중 명사성 종결 독법을 채택, 없으면 1위 반환."""
        if not self._kiwi_tried and self._kiwi is None:
            self._kiwi_tried = True
            try:
                from kiwipiepy import Kiwi
                self._kiwi = Kiwi()
            except ImportError:
                self._kiwi = None
        if self._kiwi is None:
            return None
        if label not in self._cache:
            if len(self._cache) > 200_000:  # 대용량 보호
                self._cache.clear()
            readings = [r[0] for r in self._kiwi.analyze(label, top_n=3)]
            best = readings[0]
            for toks in readings:
                if toks and toks[-1].tag in NOUNISH_TAGS and any(
                        t.tag in ("NNG", "NNP", "XR", "SL") for t in toks):
                    best = toks
                    break
            self._cache[label] = best
        return self._cache[label]

    def _en_pos(self, label: str):
        """라틴 라벨 nltk POS 태그열 — nltk/태거 미설치 시 None(게이트 생략)."""
        try:
            import nltk
            toks = label.split()
            if not toks:
                return []
            return [t for _, t in nltk.pos_tag(toks)]
        except Exception:
            return None

    def decide(self, label: str, *, df: int = 1, has_rel: bool = False,
               has_kid: bool = False, has_inst: bool = False) -> PromotionDecision:
        label = (label or "").strip()
        if not label:
            return PromotionDecision(False, "shape")

        # ① 지지도 게이트 — 고립(단일 출처 + 무구조) 라벨은 클래스 부적격.
        # 구조 = 관계 참여 ∨ 계층 부모 ∨ 인스턴스 보유(지우면 고아 인스턴스 발생).
        if self.gate_active and not (df >= self.min_df or has_rel or has_kid or has_inst):
            return PromotionDecision(False, "gate")

        # ② 전체 이중반복 병합("오에겐자부로오에겐자부로")
        n = len(label)
        if n >= 6 and n % 2 == 0 and label[: n // 2] == label[n // 2:]:
            return PromotionDecision(False, "repeat")

        if not _HANGUL.search(label):
            # R-en-2: 라틴 라벨 문법 게이트 — nltk POS 로 명사구 검증 (한국어 게이트와
            # 동일 원리: 클래스 후보는 명사 head 로 끝나는 명사구여야). nltk 미설치면
            # 기존대로 통과(v1 하위호환·폴백 공시).
            words = label.split()
            # 정서법 우선(폐형식): 문맥 없는 단독어 nltk 태깅은 불안정(Budapest→JJS,
            # King→VBG 실측) — TitleCase 종결어는 고유명사로 보고 POS 게이트 우회.
            if words and words[-1][:1].isupper() and not label.isupper():
                return PromotionDecision(True, "")
            en_tags = self._en_pos(label)
            if en_tags is None:
                return PromotionDecision(True, "")
            if not en_tags or not en_tags[-1].startswith("NN"):
                return PromotionDecision(False, "shape")  # 비명사 종결('provide'·'quickly')
            if len(label) <= 2 or (label.islower() and len(label.split()) == 1 and len(label) <= 3):
                return PromotionDecision(False, "shape")  # 약어 파편('AA'·'le')
            # 단독 소문자 1단어는 nltk 가 동사도 NN 오태깅('provide') — WordNet 명사
            # synset 존재로 판정(공개 어휘자원, 우리말샘 전례). wordnet 부재 시 통과.
            if label.islower() and len(label.split()) == 1:
                try:
                    from nltk.corpus import wordnet
                    if not wordnet.synsets(label, pos=wordnet.NOUN):
                        return PromotionDecision(False, "shape")
                except Exception:
                    pass
            return PromotionDecision(True, "")

        toks = self._tokens(label)
        if toks is None:
            return PromotionDecision(True, "")  # kiwi 없음 — 문법 규칙 생략

        tags = [t.tag for t in toks]
        # ③ 형태 게이트: 명사 전무 or 비명사 종결
        if not any(t in ("NNG", "NNP", "XR", "SL") for t in tags):
            return PromotionDecision(False, "shape")
        if tags[-1] not in NOUNISH_TAGS:
            return PromotionDecision(False, "shape")
        # ④ 지시·상대어 head("통합이전") / 지시 관형 시작("해당국가")
        if toks[-1].form in REL_HEADS or label in REL_HEADS:
            return PromotionDecision(False, "relhead")
        if len(toks) >= 2 and toks[0].form in DEICTIC_FIRST:
            return PromotionDecision(False, "deictic")
        # ⑤ 과결합(명사 4+ 연쇄·10자+인데 저지지 — "교통도로수도고속도로") —
        #    대형 코퍼스 전용. 법령류 소형 코퍼스는 긴 복합어가 정상 용어라
        #    ("한국채택국제회계기준", "기업구조개선기관전용사모집합투자기구")
        #    게이트와 함께 비활성(finreg 오탐 실측 반영).
        if self.gate_active:
            nn = sum(1 for t in tags if t in ("NNG", "NNP"))
            if nn >= 4 and df <= 2 and len(label) >= 10:
                return PromotionDecision(False, "overjoin")
        return PromotionDecision(True, "")

    def partition(self, items: Iterable[Mapping]) -> tuple[list, list]:
        """items: {name, df, has_rel, has_kid, has_inst} → (승격, 탈락 [(item, reason)])."""
        kept, dropped = [], []
        for it in items:
            d = self.decide(it["name"], df=int(it.get("df", 1)),
                            has_rel=bool(it.get("has_rel")), has_kid=bool(it.get("has_kid")),
                            has_inst=bool(it.get("has_inst")))
            (kept.append(it) if d.keep else dropped.append((it, d.reason)))
        return kept, dropped
