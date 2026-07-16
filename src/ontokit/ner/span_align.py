"""R13-2 스팬-형태소 경계 정렬 + 인접 동클래스 인명 병합 (결정론, LLM 0콜).

실측 실패 모드(mixed20k 미탐 추적):
- 토큰 중간 절단: '찰스턴'→'찰스', '진양호'→'진양' — NER 서브워드 경계가 Kiwi
  토큰 중간에서 끊김 → **토큰 끝까지 확장**. '한국측/서울역' 오결합 없음(스팬이
  완전한 토큰 경계에서 끝나면 미확장 — 측/역은 별도 토큰).
- (기각) 인명 병합은 G-A 채점에서 인명 나열 오결합 실증 — 제거.
"""
from __future__ import annotations


def align_spans(text: str, ents: list[dict], kiwi=None) -> list[dict]:
    """NER 엔티티 스팬을 Kiwi 토큰 경계로 정렬 + 인접 인명 병합. in-place 아님."""
    if kiwi is None or not ents or not text:
        return ents
    try:
        toks = kiwi.tokenize(text)
    except Exception:
        return ents
    # 토큰 [start, end) 경계 목록
    bounds = [(t.start, t.start + len(t.form)) for t in toks]
    out: list[dict] = []
    for e in ents:
        st, en = e.get("start"), e.get("end")
        if isinstance(st, int) and isinstance(en, int):
            for ts, te in bounds:
                # 스팬 끝이 토큰 내부에서 끊김 → 토큰 끝까지 확장 (찰스턴)
                if ts < en < te and st >= ts - 0 and te - ts <= (en - st) + 8:
                    ext = text[st:te].strip()
                    if ext and len(ext) > len(e.get("entity", "")):
                        e = dict(e, entity=ext, end=te)
                    break
        out.append(e)
    # (제거됨, 0717 G-A) 인접 인명 병합은 나열된 서로 다른 인물을 오결합
    # ('R.A.디키'+'놀런', '데릭 로'+'마크' — 블라인드 2인 실증). 공백 인접만으로는
    # 단일 인명과 인명 나열을 구분 불가 — 판별 기제 없이 재도입 금지.
    return out
