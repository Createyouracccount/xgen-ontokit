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

    # 클래스 후보 품질 상한 — 연속 명사가 과결합되면(예: "단편작품상중편작품상독립단체상")
    # 무의미한 긴 문자열이 클래스로 폭발. 형태소 수/길이 상한으로 컷.
    MAX_MORPHS = 4     # 결합 형태소 최대 개수 (초과 = 과결합 노이즈)
    MAX_LEN = 20       # 결합 결과 최대 글자수

    def compound_nouns(self, text: str, *, require_compound: bool = True) -> list[str]:
        """연속 NNG/NNP를 결합해 복합명사 후보.

        클래스 품질 필터:
        - 단일 일반명사(NNG 1개, 예: 형벌/비율/발급)는 클래스 부적격 → 제외
          (단, 단일 고유명사 NNP 는 유지 — 개체 타입일 수 있음).
        - 복합명사(2+ 형태소 결합)는 유지.
        - 과결합(MAX_MORPHS/MAX_LEN 초과)·불용어·비한글·중복 제외.
        require_compound=False 면 단일 NNG 도 허용(구버전 호환, last_noun 등 내부용).
        """
        toks = self.kiwi.tokenize(text)
        out: list[tuple[str, int, bool]] = []  # (표면형, 형태소수, 고유명사포함)
        buf: list[str] = []
        buf_has_nnp = False
        for t in toks:
            if t.tag in ("NNG", "NNP"):
                buf.append(t.form)
                if t.tag == "NNP":
                    buf_has_nnp = True
            else:
                if buf:
                    surf = "".join(buf) if len(buf) > 1 else buf[0]
                    out.append((surf, len(buf), buf_has_nnp))
                    buf, buf_has_nnp = [], False
        if buf:
            surf = "".join(buf) if len(buf) > 1 else buf[0]
            out.append((surf, len(buf), buf_has_nnp))
        seen, res = set(), []
        for surf, n_morph, has_nnp in out:
            if not (len(surf) >= 2 and _HANGUL_NOUN.fullmatch(surf)):
                continue
            if surf in STOP_HEAD or surf in seen:
                continue
            if n_morph > self.MAX_MORPHS or len(surf) > self.MAX_LEN:
                continue  # 과결합 노이즈
            # 단일 일반명사(NNG 1개)는 클래스 부적격 — 고유명사거나 복합만 통과
            if require_compound and n_morph == 1 and not has_nnp:
                continue
            seen.add(surf)
            res.append(surf)
        return res

    def last_noun(self, s: str) -> str:
        """구절의 마지막 복합명사 — 술어·조사 제거. 계층 접미사용이라 단일명사도 허용."""
        nouns = self.compound_nouns(s, require_compound=False)
        return nouns[-1] if nouns else ""
