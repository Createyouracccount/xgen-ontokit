# -*- coding: utf-8 -*-
"""P279 개념 게이트 (R13) — 보통명사를 개체 인스턴스에서 제외한다. 기본 OFF.

## 왜 이 처치인가

0822 **검색 계층** 실측이 지목했다. 추출 계층 분해는 절단 32.1% · 보통명사 62.3% 였으나,
실제 온톨로지 검색이 상위에 노출한 결함 35건은 **보통명사 100% · 절단 0%** 였다
(`eval_runs/search_layer/SEARCH_LAYER_2026_08_22.md`). 절단 라벨은 빈도가 낮아 랭킹에서
밀리고, 보통명사(`기자`·`회장`·`정부`·`반도체`)는 빈도가 높아 항상 상위에 뜬다.
**프록시(추출)가 아니라 최종 지표(검색)가 고른 첫 처치다**(판례 35).

## 판별력 (0822 실측)

| 집단 | 스냅샷 매칭 | P279>0 거부율 | Wilson95 |
|---|---|---|---|
| 검색 **결함** 35 | 23 | **73.9%** | [53.5, 87.5] |
| 검색 **정상** 120 | 81 | **13.6%** | [7.8, 22.7] |

판별력 **+60.3pp**, CI 비겹침.

## 설계 원칙

- **비대칭만 쓴다.** `P279 참여 ⇒ 개념`은 유효(Wikidata 공식 문서가 성문화).
  ⛔ `P31 있음 ⇒ instance`는 **무효** — 구조 오염이 도메인별 40~100%
  (arXiv:2411.15550 / 2511.04926). 이 모듈은 P31 을 개체 판정에 쓰지 않는다.
- **빌드 시점에 네트워크를 타지 않는다.** 동결 스냅샷 JSON 만 읽는다. 결정성 봉인 유지.
  스냅샷 수집은 `eval_runs/search_layer/harvest_p279.py`(빌드 경로 아님).
- **라벨 없음 = 판정불가.** 개념으로도 개체로도 넘기지 않고 **통과**시킨다(보수적).
  회색지대 기본값 금지 원칙 — "모르면 건드리지 않는다"를 명시적 통과로 고정.
- **taxon 별도 규칙.** Wikidata 는 생물 분류군을 `P31=Q16521` 로만 모델링해 P279=0 이다
  (`가리비`·`배추`·`갑오징어`가 게이트를 빠져나가는 것을 0822 에 실측). 별도로 거부한다.
- **단일 NNG 만 대상.** 복합명사·NNP 포함 표면형은 건드리지 않는다 — 오살을 구조적으로 줄인다.

## 알려진 오살 (0822 실측, 정상 노드 81 중 11)

`AI` · `FCEV` · `SUV` · `iOS` · `경기` · `메타버스` · `수도권` · `유튜버` · `중국` ·
`카카오` · `코로나` — **전부 약어·지명·기술용어**다. `CLASS_SCOPE` 로 적용 클래스를
좁히는 것이 처방이며, 전역 적용은 이 목록을 그대로 잃는다.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional

ENABLED = os.environ.get("ONTOKIT_CONCEPT_GATE", "").lower() in ("on", "true", "1")

# 적용 클래스. 빈 값(기본)이면 **전 클래스**. 0822 실측상 `용어`는 순손실이라
# 좁힐 때는 여기에 명시한다. 어휘가 아니라 **모델 자체 taxonomy 의 클래스명**이므로
# 층위 3(≤14항목, 증식하지 않음)이지 층위 0 어휘 목록이 아니다.
CLASS_SCOPE = tuple(
    c for c in os.environ.get("ONTOKIT_CONCEPT_GATE_CLASSES", "").split(",") if c.strip())

_SNAPSHOT: Optional[dict] = None
_SNAPSHOT_PATH = os.environ.get("ONTOKIT_P279_SNAPSHOT", "")


def load_snapshot(path: str = "") -> dict:
    """동결 스냅샷을 읽는다. 없으면 빈 dict — 게이트는 아무것도 거부하지 않는다.

    무증상 0 의심 원칙: 스냅샷이 비면 '거부 0'이 나오는데, 그것은 '개념이 없다'가
    아니라 '조회를 못 했다'이다. 호출자가 구별할 수 있도록 `stats["스냅샷"]`에 크기를 낸다.
    """
    global _SNAPSHOT
    p = path or _SNAPSHOT_PATH
    if not p or not os.path.exists(p):
        _SNAPSHOT = {}
        return _SNAPSHOT
    with open(p, encoding="utf-8") as f:
        _SNAPSHOT = json.load(f)
    return _SNAPSHOT


def is_concept(label: str, snapshot: Optional[dict] = None) -> Optional[bool]:
    """True=개념(거부) · False=개체 후보(통과) · None=판정불가(라벨 없음/미수집).

    None 과 False 를 뭉개지 않는다 — 판정불가율을 공시해야 하기 때문이다(판례 19).
    """
    snap = snapshot if snapshot is not None else (_SNAPSHOT if _SNAPSHOT is not None
                                                  else load_snapshot())
    row = snap.get(label)
    if row is None:
        return None
    p279, taxon = row.get("p279"), row.get("taxon")
    if p279 is None:
        return None                      # 한국어 라벨 부재 = 판정불가
    if taxon:
        return True                      # 생물 분류군 — P279=0 이어도 개념
    return p279 > 0


def single_common_noun(label: str, kiwi) -> bool:
    """단일 일반명사(NNG)인가. 게이트 적용 대상 판별 — 복합·고유명사는 건드리지 않는다."""
    toks = kiwi.tokenize(label)
    return len(toks) == 1 and toks[0].tag == "NNG"


def apply(all_entities: dict, kiwi, snapshot_path: str = "") -> dict:
    """all_entities 를 제자리 필터링. 반환은 통계(분모 포함 — 판례 16).

    all_entities: {doc_name: [entity_dict, ...]}
    """
    snap = load_snapshot(snapshot_path)
    stats = {"스냅샷": len(snap), "검사": 0, "대상아님": 0,
             "판정불가": 0, "거부": 0, "통과": 0, "거부목록": []}
    if not snap:
        stats["경고"] = ("스냅샷이 비어 있다 — 거부 0 은 '개념이 없다'가 아니라 "
                         "'조회를 못 했다'이다(무증상 0 의심 원칙).")
    for doc, ents in all_entities.items():
        keep = []
        for e in ents:
            lb = e.get("entity") or ""
            stats["검사"] += 1
            if CLASS_SCOPE and e.get("class") not in CLASS_SCOPE:
                stats["대상아님"] += 1
                keep.append(e)
                continue
            if not lb or not single_common_noun(lb, kiwi):
                stats["대상아님"] += 1
                keep.append(e)
                continue
            v = is_concept(lb, snap)
            if v is None:
                stats["판정불가"] += 1
                keep.append(e)            # 판정불가는 **명시적 통과**
            elif v:
                stats["거부"] += 1
                if len(stats["거부목록"]) < 60:
                    stats["거부목록"].append(lb)
            else:
                stats["통과"] += 1
                keep.append(e)
        all_entities[doc] = keep
    n = stats["검사"] or 1
    stats["거부율_전체"] = round(stats["거부"] / n * 100, 2)
    tgt = stats["판정불가"] + stats["거부"] + stats["통과"]
    stats["판정불가율_대상중"] = round(stats["판정불가"] / tgt * 100, 2) if tgt else 0.0
    return stats
