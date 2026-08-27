# -*- coding: utf-8 -*-
"""IMPROVEMENT_ROADMAP.md 의 기본값 계약이 코드와 일치하는지.

왜 (0824 실측): 로드맵이 v0.13.1 기준으로 **2개 버전 뒤처져** 있었다.
`enable_hearst=True` **기본 on** 이라고 적혀 있었으나 v0.14 에서 off 로 바뀌었고
(정의문 유래 의미 subClassOf 거짓률 88.5%), v0.15 의 `enable_qdt_gate` 는 표에 아예 없었다.
로드맵은 **다음 라운드가 "무엇이 켜져 있나"를 확인하는 문서**라, 틀리면 라운드 설계가 오도된다.

산문으로 "갱신하자"는 규범은 집행되지 않는다(판례 27-iii). 그래서 대조를 테스트로 만든다.
**코드가 정본이다** — 값이 갈리면 로드맵을 고쳐라(반대가 아니다).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from ontokit.extractors.deterministic_ko import DeterministicKoreanExtractor

ROADMAP = Path(__file__).resolve().parents[1] / "IMPROVEMENT_ROADMAP.md"
BLOCK = re.compile(
    r"<!-- ROADMAP_DEFAULTS_BEGIN.*?-->\s*```(.*?)```\s*<!-- ROADMAP_DEFAULTS_END -->",
    re.DOTALL,
)


def _declared() -> dict[str, bool]:
    m = BLOCK.search(ROADMAP.read_text(encoding="utf-8"))
    assert m, "로드맵에서 ROADMAP_DEFAULTS 블록을 못 찾았다 — 마커를 지우지 마라"
    out: dict[str, bool] = {}
    for line in m.group(1).strip().splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = {"True": True, "False": False}[v.strip()]
    return out


def _actual() -> dict[str, bool]:
    sig = inspect.signature(DeterministicKoreanExtractor.__init__)
    return {k: v.default for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty and isinstance(v.default, bool)}


def test_roadmap_declares_every_boolean_default():
    """새 플래그가 생기면 로드맵에도 적어야 한다 — 조용히 늘어나는 것을 막는다."""
    missing = sorted(set(_actual()) - set(_declared()))
    assert not missing, (
        f"코드에 있는데 로드맵 기본값 계약에 없다: {missing}. "
        "IMPROVEMENT_ROADMAP.md 의 ROADMAP_DEFAULTS 블록에 추가하라."
    )


def test_roadmap_has_no_phantom_flags():
    """반대로 로드맵에만 있는 유령 플래그도 막는다(제거된 기능의 잔재)."""
    phantom = sorted(set(_declared()) - set(_actual()))
    assert not phantom, f"로드맵에만 있고 코드엔 없다: {phantom}"


@pytest.mark.parametrize("flag", sorted(_actual()))
def test_roadmap_default_matches_code(flag):
    """값 일치. 갈리면 **로드맵을 고쳐라** — 코드가 정본이다."""
    dec, act = _declared(), _actual()
    assert dec[flag] == act[flag], (
        f"`{flag}`: 로드맵 {dec[flag]} vs 코드 {act[flag]}. "
        "코드가 정본이다 — 로드맵의 기본값 계약과 상태표를 함께 고쳐라."
    )
