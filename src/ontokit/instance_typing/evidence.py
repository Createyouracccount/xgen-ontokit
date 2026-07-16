"""T1 타이핑 증거 수집 — 정의문(첫문장/전문장 격리변형) + 동격·열거 패턴.

전부 폐집합 문법 판정(콘텐츠 어휘 0)·LLM 0콜. 독립 타이핑 패스와 정기 빌드가
이 함수를 공유한다(단일 진실원천 — R11 심판 요건).

채널별 게이트 (심판 3R 동결):
- def_first: 기존 definitional_pairs(제목-주어 정합·EF) 그대로 — 검증된 강채널.
- (제거됨, 0717 최종심판) def_mid(전문장)·appos(동격열거)는 블라인드 2인 G2 에서
  오탐 63~70%/37~43% — 승격클래스 하드게이트+시제컷으로도 계사문의 서술·역할·
  방향역전·상대지시어·부분관계 오탐을 못 막았다(4,133 트리플 revert). 4대 오탐
  유형은 eval_runs/typing/t1_reverted_midappos.json 참조. 판별 기제 없이 재도입 금지.
"""
from __future__ import annotations
import re
from typing import Iterable

from ontokit.hierarchy.hearst_ko import definitional_pairs, _subject_of, _noun_run_before

_SENT_SPLIT = re.compile(r"(?<=다)\.(?=\s|[가-힣A-Za-z])|[.!?]\s|\n")
_norm = lambda s: re.sub(r"\s+", "", s)


def _class_head_at(toks, i: int, promoted_classes: set[str]) -> str:
    """토큰 i 직전 명사 run 의 **형태소 경계** 접미 부분열 중 승격 클래스와 일치하는
    최장 접미 반환 (head-final: [보수,정당]→'정당'). 문자 접미가 아니라 형태소 단위 —
    '중국집'(단일 NNG)에서 '집' 오살 없음. 없으면 ""."""
    j = i - 1
    forms: list[str] = []
    while j >= 0 and (toks[j].tag.startswith("NN") or toks[j].tag in ("SL", "SN", "XR")):
        forms.insert(0, toks[j].form)
        j -= 1
    for k in range(len(forms)):
        cand = "".join(forms[k:])
        if cand in promoted_classes:
            return cand
    return ""


def collect_typing_evidence(text: str, kiwi,
                            entity_labels: set[str],
                            promoted_classes: set[str]) -> list[dict]:
    """청크 텍스트에서 (child=개체, parent=클래스) 타이핑 증거 수집.

    entity_labels/promoted_classes 는 공백 정규화된 집합. 반환 레코드:
    {"child", "parent", "kind": def_first|def_mid|appos}.
    """
    out: list[dict] = []
    if kiwi is None or not text:
        return out
    # ① def_first — 기존 채널 재사용(제목-주어 정합 게이트 포함)
    for hp in definitional_pairs(text, kiwi=kiwi):
        cn = _norm(hp["child"])
        if cn in entity_labels:
            out.append({"child": hp["child"], "parent": hp["parent"], "kind": "def_first"})

    return out


def select_types(evidences: Iterable[dict], entity_labels: set[str]) -> dict[str, tuple[str, int, str]]:
    """(child,parent) 증거 빈도 최다(동률 시 최장·사전순) 선택 — T0 규칙과 동일.
    parent 정합성: parent 가 개체명과 동일하면 서로 다른 child ≥2 일 때만 채택.
    반환: child_norm → (parent, 증거수, 대표 kind)."""
    freq: dict[tuple[str, str], list] = {}
    for ev in evidences:
        key = (_norm(ev["child"]), ev["parent"])
        freq.setdefault(key, []).append(ev["kind"])
    # parent 정합성 사전 집계
    children_of: dict[str, set] = {}
    for (cn, parent), kinds in freq.items():
        children_of.setdefault(_norm(parent), set()).add(cn)
    best: dict[str, tuple[str, int, str]] = {}
    for (cn, parent), kinds in sorted(freq.items()):
        pn = _norm(parent)
        if pn in entity_labels and len(children_of.get(pn, ())) < 2:
            continue  # 개체-as-클래스 (자식 ≥2 구제)
        cand = (len(kinds), len(parent), parent)
        cur = best.get(cn)
        if cur is None or cand > (cur[1], len(cur[0]), cur[0]):
            best[cn] = (parent, len(kinds), kinds[0])
    return best
