"""한국어 관계(objectProperty) 추출 — 조사 기반 SVO 트리플. LLM 0회.

한국어는 격조사가 문법역할을 명시(head-final SOV):
  주어=JKS(이/가)·JX(은/는), 목적어=JKO(을/를), 술어=동사성명사(NNG)+XSV(하/되).
  "금융위원회는 은행을 감독한다" → (금융위원회, 감독, 은행).

XGEN kg_builder 소비 스키마에 맞춰 {subject, predicate, object, predicate_type,
source_chunks} dict 를 반환. 하위(kg_builder)가 subject/object 를 인스턴스 노드로
자동 등록하므로 NER 등록과 독립.

정밀도 우선 설계: 한 절(clause)에 주어·목적어·술어가 모두 있을 때만 추출(불완전
문장은 버림). 자유 텍스트 오탐을 줄이되 정형 텍스트(법령/규정/정의문)에서 고신뢰.
"""
from __future__ import annotations
import re

from ..morphology.kiwi_nouns import STOP_HEAD

# 문장 분리 — 종결어미(SF) 또는 개행 기준. 절 경계를 넘는 오연결 방지.
_SENT_SPLIT = re.compile(r'(?<=[.!?。\n])\s+|(?<=다\.)\s*')

# 술어로 부적격한 동사성 명사(너무 일반적 → 관계로 무의미). 접미공유 STOP_HEAD 재사용 +
# 관계 특화 컷.
_STOP_PRED = STOP_HEAD | {
    "관하", "대하", "위하", "의하", "따르", "인하", "통하",  # 기능 동사
}

_HANGUL = re.compile(r"[가-힣]{2,}")


class KoreanRelationExtractor:
    """조사 기반 한국어 SVO 관계 추출. Kiwi 인스턴스 주입(없으면 생성)."""

    MAX_ARG_LEN = 20  # 주어/목적어 명사구 최대 글자수 (과결합 노이즈 컷)

    def __init__(self, kiwi=None):
        if kiwi is None:
            from kiwipiepy import Kiwi  # lazy — extras[korean]
            kiwi = Kiwi()
        self.kiwi = kiwi

    def _flush_noun(self, buf: list[str]) -> str:
        """연속 명사 버퍼 → 복합명사 표면형(2자 이상, 불용어/길이 컷)."""
        if not buf:
            return ""
        surf = "".join(buf)
        if len(surf) < 2 or len(surf) > self.MAX_ARG_LEN:
            return ""
        if surf in STOP_HEAD or not _HANGUL.fullmatch(surf):
            return ""
        return surf

    def _extract_sentence(self, sent: str) -> list[dict]:
        """한 문장에서 (subject, predicate, object) 트리플. 조사로 역할 판별."""
        toks = self.kiwi.tokenize(sent)
        subject = obj = predicate = ""
        noun_buf: list[str] = []
        out: list[dict] = []

        for t in toks:
            if t.tag in ("NNG", "NNP"):
                noun_buf.append(t.form)
                continue

            # 조사 도달 — 직전 명사구를 역할에 배정
            if t.tag in ("JKS", "JX"):          # 이/가/은/는 → 주어
                cand = self._flush_noun(noun_buf)
                if cand:
                    subject = cand
            elif t.tag == "JKO":                # 을/를 → 목적어
                cand = self._flush_noun(noun_buf)
                if cand:
                    obj = cand
            elif t.tag == "XSV":                # 하/되 → 직전 NNG 가 술어
                if noun_buf:
                    pred_cand = noun_buf[-1]
                    if (2 <= len(pred_cand) <= self.MAX_ARG_LEN
                            and pred_cand not in _STOP_PRED
                            and _HANGUL.fullmatch(pred_cand)):
                        predicate = pred_cand
                        # 술어 완성 시점에 S·P·O 다 있으면 트리플 확정
                        if subject and obj and predicate and subject != obj:
                            out.append({"subject": subject, "predicate": predicate,
                                        "object": obj})
                            # 다음 절 대비 목적어·술어 리셋(주어는 생략 대비 유지)
                            obj = predicate = ""
            noun_buf = []

        return out

    def extract(self, text: str, *, source_chunks: list[str]) -> list[dict]:
        """청크 텍스트 → 관계 dict 리스트(kg_builder 소비 스키마).

        반환: [{subject, predicate, object, predicate_type='ObjectProperty',
                source_chunks}, ...]  — 같은 (s,p,o) 중복 제거.
        """
        if not text or not text.strip():
            return []
        seen: set = set()
        out: list[dict] = []
        for sent in _SENT_SPLIT.split(text):
            if not sent.strip():
                continue
            for tri in self._extract_sentence(sent):
                key = (tri["subject"], tri["predicate"], tri["object"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({**tri, "predicate_type": "ObjectProperty",
                            "source_chunks": source_chunks})
        return out
