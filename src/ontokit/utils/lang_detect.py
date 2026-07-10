"""경량 언어 감지 — 한글/영문 글자 비율. 외부 라이브러리 0.

혼합 코퍼스(한국어+영어)에서 청크 단위로 ko/en 을 판별해 언어별 추출
파이프라인(형태소·NER)을 라우팅하는 용도. 정밀 언어분류가 아니라
"한글 위주냐 영문 위주냐" 2분류면 충분하다.
"""
from __future__ import annotations
import re

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[a-zA-Z]")


def detect_lang(text: str) -> str:
    """텍스트의 주 언어 반환: 'ko' 또는 'en'.

    한글 글자 수 >= 영문 글자 수 이면 ko, 아니면 en.
    (한글이 조금이라도 우세하면 한국어 파이프라인 — Kiwi/KoELECTRA 가
    영문 토큰은 무시하고 한글만 처리하므로 혼합문에서도 안전.)
    """
    if not text:
        return "en"
    ko = len(_HANGUL.findall(text))
    en = len(_LATIN.findall(text))
    return "ko" if ko >= en else "en"
