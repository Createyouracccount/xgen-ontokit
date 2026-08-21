"""R7 어절 경계 정합 — 코퍼스 증거 기반 스팬 복원 (결정론, LLM 0콜).

## 왜 또 경계인가

정밀도 결함의 **57.5%가 절단**이다(독립 3인 감정, n=198). 그 하위 유형:

- **접미 조각** — `웍스`←CJ올리브네트웍스 · `날드`←맥도날드 · `레콤`←SK텔레콤
- **접두 조각** — `CJ`←CJ올리브네트웍스 · `카카오`←카카오게임즈 · `자원부`←산업통상자원부

R3(a)·R4·R5 는 이 둘을 **서로 다른 규칙**으로 따로 고치려다 전부 기각됐고,
R6 은 접미 조각을 **제거**만 해서 재현율을 잃었다.

**이 모듈은 하나의 규칙으로 둘 다 고친다.** 제거가 아니라 복원이므로 재현율이 오른다.

## 기제 — 어절 경계 + 코퍼스 증거

① **어절 핵심**: 스팬이 속한 공백·구두점 구분 단위에서 조사·어미를 뗀 나머지.
   한국어는 어절 내부에 공백이 없으므로 **붙여 쓴 것은 대개 한 단위**다.
   R4 가 실패한 이유는 어절 **내부**를 형태소 태그로 걸어 들어갔기 때문이다
   (`괴뢰정부`·`반미`). 여기서는 어절 **경계**만 본다 — 표층 사실이지 추측이 아니다.

② **코퍼스 증거 가드**: 원본 라벨이 코퍼스 어딘가에서 **독립 어절 핵심**으로 출현하면
   확장하지 않는다. 그 라벨은 스스로 설 수 있는 단위이므로 조각이 아니다.

   실측(ui_news100 538청크 · 확장 후보 195종):

       한국   → 한국경제      원본 독립 25회 → **차단** (한국 경제를 붙여 쓴 것)
       서울   → 서울추모공원   원본 독립 53회 → **차단**
       웍스   → CJ올리브네트웍스  원본 독립  0회 → 통과
       CJ    → CJ올리브네트웍스  원본 독립  0회 → 통과
       레콤   → SK텔레콤      원본 독립  0회 → 통과
       자원부 → 산업통상자원부   원본 독립  0회 → 통과
       휘소   → 이휘소        원본 독립  0회 → 통과

   가드 적용: 195종 중 **68종 차단 · 127종 통과**.

## ⚠️ 가드의 비용 (공시 — R5 가 스톱리스트 비용 미공시로 기각된 전례)

가드는 **정당한 수리도 차단한다.** 실측 예: `서울`→`서울추모공원` 은 옳은 복원인데
`서울` 이 독립 53회 출현해 차단된다. 68종 차단분에 이런 손실이 섞여 있으며,
그 수는 결과 문서에 공시한다.

## ⚠️ 잔존 오살 (사전 자백)

가드를 통과하고도 틀리는 경우가 실재한다. 실측: `SM` → `SM이성수`
(원문이 `SM 이성수` 를 붙여 씀 — 어절 판단으로는 구별 불가).
→ **`악화` 범주는 도달 가능하다.** R4·R5·R6 은 악화가 구조적으로 도달 불가여서
   임계가 무장 해제됐다. 이 처치는 그 함정에서 벗어나 있으며, 그 사실을 A/B 에서
   실측으로 보인다.

공시 `eval_runs/bench/demo_roster/r7_predeclare.md`.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# 기본 off — 심판 검증 전. 켜려면 ONTOKIT_WORD_BOUNDARY=on.
ENABLED = os.environ.get("ONTOKIT_WORD_BOUNDARY", "").lower() in ("on", "true", "1")

# 개체명 내부에 오지 않는 문자.
#
# ⛔ 초판은 가운뎃점을 **U+00B7 하나만** 넣고 "가운뎃점을 넣는 것이 필수"라고 선언했다.
#    공격 심판 실측: 코퍼스의 `현대차‧기아` 는 **U+2027**(HYPHENATION POINT)이라 통과했고,
#    `기아`(참)가 `현대차‧기아`(거짓)로 병합돼 소멸했다. `※CBS노컷뉴스` 도 같은 유형이다.
#    **확정 손실 3건 중 2건이 이 두 코드포인트 누락에서 나왔다.**
#    → 중간점 6종과 참조·원문자 기호를 전부 넣는다. 이 목록은 테스트가 코드포인트별로 건다.
TRIM = (
    "·‧・･•∙⋅⸱"                  # 중간점 6종+ — 전부 나열 구분자다
    "※ⓒ△▲◇◆□■○●☆★"          # 참조·불릿 기호 (기사 머리에 붙는다)
    "①②③④⑤⑥⑦⑧⑨⑩"              # 원문자 숫자 — Python isalnum() 이 True 라 안 잡힌다
    ",.!?;:'\"“”‘’()[]{}<>«»…~/\\|"
)

# 조사·어미·문장부호 태그 — 어절 끝에서 떼어낸다
_TAIL_TAGS = frozenset("""
JKS JKC JKG JKO JKB JKV JKQ JX JC
EF EC ETM ETN EP VCP XSV XSA
SF SP SE SS SO SW
""".split())

_SPLIT = re.compile(r"[\s" + re.escape(TRIM) + r"]+")


def _strip_tail(seg: str, kiwi) -> str:
    """어절에서 조사·어미·문장부호 꼬리를 뗀다."""
    if not seg:
        return ""
    try:
        toks = kiwi.tokenize(seg)
    except Exception:
        return seg.strip().strip(TRIM)
    cut = len(seg)
    for t in reversed(toks):
        if t.tag in _TAIL_TAGS:
            cut = min(cut, t.start)
        else:
            break
    return seg[:cut].strip().strip(TRIM)


def core_at(text: str, st: int, en: int, kiwi) -> str | None:
    """스팬 [st,en) 이 속한 어절의 핵심부. 못 구하면 None."""
    ws = st
    while ws > 0 and not text[ws - 1].isspace() and text[ws - 1] not in TRIM:
        ws -= 1
    we = en
    while we < len(text) and not text[we].isspace() and text[we] not in TRIM:
        we += 1
    return _strip_tail(text[ws:we], kiwi) or None


def corpus_cores(corpus: str, kiwi) -> dict:
    """코퍼스의 어절 핵심 빈도표 — 가드의 증거원."""
    out: dict[str, int] = {}
    for w in _SPLIT.split(corpus):
        if not w:
            continue
        c = _strip_tail(w, kiwi)
        if c:
            out[c] = out.get(c, 0) + 1
    return out


def repair(all_entities: dict, chunk_texts: dict, corpus: str, kiwi) -> dict:
    """엔티티 스팬을 어절 경계로 복원한다. in-place. 통계 dict 반환.

    `chunk_texts`: {chunk_id: text}. 스팬 검증에 쓴다 — `text[st:en] != entity` 면
    스팬이 신뢰할 수 없으므로 **건드리지 않는다**.
    """
    stats = {"검사": 0, "확장후보": 0, "가드차단": 0, "적용": 0}
    if not ENABLED or not all_entities or not corpus or kiwi is None:
        return stats
    cores = corpus_cores(corpus, kiwi)
    for ents in all_entities.values():
        for e in ents:
            lb = (e.get("entity") or "").strip()
            st, en = e.get("start"), e.get("end")
            sc = e.get("source_chunks") or []
            if not (lb and isinstance(st, int) and isinstance(en, int) and sc):
                continue
            text = chunk_texts.get(sc[0])
            if not text or text[st:en] != lb:      # 스팬 불일치 → 손대지 않는다
                continue
            stats["검사"] += 1
            c = core_at(text, st, en, kiwi)
            if not c or len(c) <= len(lb):
                continue
            stats["확장후보"] += 1
            if cores.get(lb, 0) > 0:               # ② 원본이 독립 단위면 조각 아님
                stats["가드차단"] += 1
                continue
            idx = text.find(c, max(0, st - len(c)))
            if idx < 0:
                continue
            e["entity"] = c
            e["start"], e["end"] = idx, idx + len(c)
            stats["적용"] += 1
    if stats["적용"]:
        logger.info("어절 경계 정합: 후보 %d · 가드차단 %d · 적용 %d",
                    stats["확장후보"], stats["가드차단"], stats["적용"])
    return stats
