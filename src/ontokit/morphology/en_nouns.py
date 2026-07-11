"""영어 명사구 추출 — nltk POS 기반. extras[english]=nltk.

Kiwi 복합명사(한국어)의 영어 대응물. 연속 명사(NN*)를 결합해 복합명사 후보.
클래스 품질 필터는 kiwi_nouns 와 동일 정책:
- 단일 일반명사(NN 1개, 예: company/system)는 클래스 부적격 → 제외
- 고유명사(NNP 포함) 또는 복합명사(2+ 결합)는 유지
- 과결합·불용어·중복 제외

nltk 데이터(punkt, averaged_perceptron_tagger)는 최초 사용 시 자동 다운로드.
"""
from __future__ import annotations
import re
from typing import Optional

_ALPHA = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
# 토큰 단위 검사용 — 전체가 라틴이어야 버퍼링(fullmatch). 시작만 검사(match)하면
# "Basel규제" 같은 라틴+한글 꼬리 토큰이 통과해 한글 혼입 클래스명이 방출된다(0711 실측).
# 마침표 허용은 "U.S." 류 약어 보존용.
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-.]*")

# 클래스로 부적격한 영어 일반명사(불용성 head). 위키/일반문서 노이즈 컷.
STOP_HEAD_EN = {
    "way", "time", "part", "kind", "type", "case", "example", "number",
    "form", "group", "member", "use", "fact", "point", "thing", "area",
    "name", "term", "list", "set", "value", "result", "system", "process",
    "state", "period", "range", "rate", "level", "order", "place", "side",
}


class EnglishNounExtractor:
    """nltk POS 태깅 기반 영어 복합명사 추출."""

    MAX_WORDS = 4      # 결합 단어 최대 개수 (초과 = 과결합)
    MAX_LEN = 40       # 결합 결과 최대 글자수

    def __init__(self, domain_words: Optional[list[str]] = None):
        self._ensured = False
        self.domain_words = {w.lower() for w in (domain_words or [])}

    def _ensure(self):
        if self._ensured:
            return
        import nltk  # lazy — extras[english]
        for pkg in ("punkt", "punkt_tab",
                    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        self._ensured = True

    def compound_nouns(self, text: str, *, require_compound: bool = True) -> list[str]:
        """연속 명사(NN*)를 결합한 복합명사 후보 리스트.

        require_compound=True(기본): 단일 일반명사(NN 1개, 고유명사 아님) 제외.
        """
        self._ensure()
        from nltk import word_tokenize, pos_tag
        try:
            tagged = pos_tag(word_tokenize(text))
        except Exception:
            return []

        out: list[tuple[str, int, bool]] = []  # (표면형, 단어수, 고유명사포함)
        buf: list[str] = []
        buf_has_nnp = False
        for w, t in tagged:
            # 전체가 라틴인 토큰만 버퍼링 — nltk 는 미지 토큰(한글 등)을 NN 으로 태깅하는
            # 경향이 있어, 혼합문에서 한글이 영어 복합명사에 붙으면("금융위원회는 Basel
            # Committee") 통째 소실되고, 시작만 검사하면 "Basel규제" 꼬리가 통과한다
            # (둘 다 0711 실측). 비라틴(부분 포함)은 경계로 취급.
            if t.startswith("NN") and _LATIN_TOKEN.fullmatch(w):
                buf.append(w)
                if t in ("NNP", "NNPS"):
                    buf_has_nnp = True
            else:
                if buf:
                    out.append((" ".join(buf), len(buf), buf_has_nnp))
                    buf, buf_has_nnp = [], False
        if buf:
            out.append((" ".join(buf), len(buf), buf_has_nnp))

        seen, res = set(), []
        for surf, n_word, has_nnp in out:
            s = surf.strip()
            if len(s) < 2 or not _ALPHA.match(s):
                continue
            key = s.lower()
            if key in seen:
                continue
            if n_word > self.MAX_WORDS or len(s) > self.MAX_LEN:
                continue  # 과결합 노이즈
            # 단일 일반명사(고유명사 아님)는 클래스 부적격 — 불용어면 항상 제외
            if require_compound and n_word == 1 and not has_nnp and key in STOP_HEAD_EN:
                continue
            if require_compound and n_word == 1 and not has_nnp and key not in self.domain_words:
                continue
            seen.add(key)
            res.append(s)
        return res
