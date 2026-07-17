"""R14b 보조 NER union 앙상블 — opt-in (env ONTOKIT_NER_AUX_MODEL).

현행 모델의 리콜 한계(미탐 GT 64%)를 보조 모델(예: soddokayo/koelectra-base-klue-ner,
Apache-2.0)의 **신규 스팬만** union 으로 보완. 최종심판(0717) 처분:
- 기본 off — 오탐 합의 10.0%는 n=60 에서 게이트(≤10%)를 배제 못 한 값일 뿐
  (Wilson CI 상단 20.1%). on 승격은 n≥150 블라인드 재판정 + 정탐 생존율 채점 조건.
- 게이트 전부 문법/구조 규칙(콘텐츠 하드코딩 0): 선후행 구두점 / 소문자 선도 라틴 /
  라틴 ≤2자 / 서수 파편 / 라틴 대문자+s 복수 약어 / '~어' 접미 언어명의 기관 클래스
  금지(형태소 접미 규칙 — 개체 리스트 아님) / 연도형 수량→날짜 재매핑 / label_ok 위생.
"""
from __future__ import annotations
import re


_YEAR = re.compile(r"(19|20)\d\d")


def aux_gate(e: dict, kiwi=None) -> dict | None:
    """보조 모델 방출 1건을 검열 — 통과 시 (재매핑된) dict, 기각 시 None."""
    w = (e.get("entity") or "").strip()
    if not w:
        return None
    if re.match(r"^[\W_]", w) or re.search(r"[\W_]$", w):
        return None  # 선·후행 구두점 파편
    t0 = w.split()[0]
    if re.fullmatch(r"[a-z][a-zA-Z]*", t0):
        return None  # 소문자 선도 라틴(고유명사 아님 — 'a De Mornay')
    if re.fullmatch(r"[A-Za-z]{1,2}", w):
        return None  # 라틴 2자 파편
    if re.match(r"^\d+\s*(th|st|nd|rd)\b", w) or re.match(r"^\d+\s+[a-z]", w):
        return None  # 서수/숫자+소문자 파편
    if re.fullmatch(r"[A-Z]{2,}s", w):
        return None  # 대문자 약어 복수형('MHKs') — 개체 아닌 총칭
    if kiwi is not None:
        from ontokit.instance_typing import label_ok
        if not label_ok(w, kiwi):
            return None
    out = dict(e, entity=w)
    if out.get("class") == "수량" and _YEAR.fullmatch(w):
        out["class"] = "날짜"  # KLUE QT 가 연도를 수량으로 태깅하는 계통 오류 교정
    if w.endswith("어") and out.get("class") == "기관":
        return None  # '~어' 접미 언어명이 기관으로 — 형태소 접미 규칙(키예어·수르지크어)
    return out


class EnsembleNER:
    """base NER + aux NER union — aux 는 base 미방출 스팬만, aux_gate 통과분만.

    base 와 동일 인터페이스(entities/entities_batch 일부)로 교체 주입 가능.
    """

    def __init__(self, base, aux, kiwi=None):
        self.base = base
        self.aux = aux
        self.kiwi = kiwi

    def entities(self, text: str, *, source_chunks: list[str]) -> list[dict]:
        out = self.base.entities(text, source_chunks=source_chunks)
        try:
            aux_ents = self.aux.entities(text, source_chunks=source_chunks)
        except Exception:
            return out  # 보조 실패는 비치명 — 현행 결과 그대로
        seen = {re.sub(r"\s+", "", e.get("entity", "")) for e in out}
        for e in aux_ents:
            k = re.sub(r"\s+", "", e.get("entity", ""))
            if not k or k in seen:
                continue
            g = aux_gate(e, self.kiwi)
            if g is not None:
                seen.add(k)
                out.append(g)
        return out
