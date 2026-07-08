"""Kiwi 형태소 기반 한국어 복합명사 추출. extras[korean]=kiwipiepy.

finreg 489 실측: 연속 명사(NNG/NNP) 결합으로 복합명사 유지(여신전문금융업 등).
사용자사전으로 도메인 용어 정확 유지 가능.
"""
from __future__ import annotations
import re
from typing import Optional

_HANGUL_NOUN = re.compile(r"[가-힣]{2,}")

# subClassOf/클래스로 부적격한 일반명사·술어 (finreg 실측 노이즈 컷)
STOP_HEAD = {
    "경우", "때", "것", "바", "자", "수", "등", "및", "또는", "이하", "다음",
    "사항", "내용", "부분", "목적", "정의", "적용", "해당", "관련", "기타",
    "준용", "제출", "승계", "제한", "지급", "결의", "사유", "이익", "어느",
    "말한다", "한다", "각", "호", "목", "항",
    "수수", "개정", "신설", "삭제", "발생", "취득", "행위", "업무", "절차",
}


class KiwiNounExtractor:
    """Kiwi 래핑 — 복합명사 추출. kiwi 인스턴스와 도메인 사전을 주입받는다."""

    def __init__(self, kiwi=None, domain_words: Optional[list[str]] = None):
        if kiwi is None:
            from kiwipiepy import Kiwi  # lazy — extras[korean]
            kiwi = Kiwi()
        self.kiwi = kiwi
        if domain_words:
            for w in domain_words:
                try:
                    self.kiwi.add_user_word(w, "NNP")
                except Exception:
                    pass

    def compound_nouns(self, text: str) -> list[str]:
        """연속 NNG/NNP를 결합해 복합명사 후보. 2자+·불용어 제외·순수 한글·중복 제거."""
        toks = self.kiwi.tokenize(text)
        out, buf = [], []
        for t in toks:
            if t.tag in ("NNG", "NNP"):
                buf.append(t.form)
            else:
                if buf:
                    out.append("".join(buf) if len(buf) > 1 else buf[0])
                    buf = []
        if buf:
            out.append("".join(buf) if len(buf) > 1 else buf[0])
        seen, res = set(), []
        for n in out:
            if len(n) >= 2 and n not in STOP_HEAD and _HANGUL_NOUN.fullmatch(n) and n not in seen:
                seen.add(n)
                res.append(n)
        return res

    def last_noun(self, s: str) -> str:
        """구절의 마지막 복합명사 — 술어·조사 제거."""
        nouns = self.compound_nouns(s)
        return nouns[-1] if nouns else ""
