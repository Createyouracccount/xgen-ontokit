"""직업 타이핑 채널 (P106) — 외부지식 어휘집 기반 인스턴스 타이핑.

배경(0718~19 심판루프): 열거(enum) 천장의 병목은 "통로가 아니라 물" — 코퍼스
내 정의문·동격만으로는 직업류 타이핑이 희박해 천장 0.0185 에 갇혔고, SREDFM-ko
P106(전수 505 블라인드 2인 채점, either-오탐 2.2%) + Wikidata 역방향 전수(합의
정탐만 채택) 주입으로 0.0731(4.0배·0716 게이트 0.05 돌파)을 실측했다. 이 모듈은
그 사이드카 주입을 **빌드타임 채널**로 이식한다 — 임의 코퍼스에서 재현되도록.

어휘집(data/occupation_lexicon_ko.json.gz, 4,121쌍·검증 7개념):
- sredfm: SREDFM-ko P106 표면형 쌍(오프라인·CC-BY-SA), census either-오탐 블록 제거.
  채택 라운드의 오탐률 실측 2.2%(하한)가 이 생성 과정의 품질 추정치.
- wdqs_vetted: Wikidata 후보 중 블라인드 2인 **합의 정탐만**(348, 인간 검증).
비합의·오탐 503쌍은 증류 단계에서 제외됨(eval_runs/typing/p106_build_lexicon.py).

게이트(채택 라운드와 동일 — 변경 금지):
- 인물지배 컷: 라벨의 NER 클래스 분포에서 인물이 최대값이 아니면 제외(동음이의,
  동률은 통과). 채택 라운드는 그래프 전수 분포(r15_inst_dist=빌드 완료 후 상태)를
  썼고, 여기서도 정의문 타이핑 **이후** 분포로 동일 규칙 적용(r15 정합 — 정의문이
  세밀타입으로 이탈시킨 라벨은 지배 판정이 달라질 수 있음, 의도된 동작).
- 재타입 대상은 **인물 클래스 레코드만**(가산 의미론) — 동명 라벨의 비인물
  레코드(수지의 '지역')는 보존. 사이드카 가산 INSERT 와 정합(심판 D1 이행).
- 방출 형식: 정의문 타이핑 채널과 동일 — 대표(최고 conf) 직업은 entity dict
  "class" in-place 갱신, 복수 직업(갈릴레이=물리학자·수학자, 사이드카 3.8%)은
  동일 라벨 추가 레코드(kg_builder 는 레코드별 rdf:type add → 복수 타입 자연 지원).
- 상향 경로 보존: 직업클래스 ⊂ 인물 계층쌍 방출(정의문 채널의 TTA 패턴).
전부 결정론·LLM 0콜·빌드 네트워크 0(어휘집은 패키지 동봉 자산).
"""
from __future__ import annotations

import gzip
import json
import re
from importlib import resources
from typing import Optional

_norm = lambda s: re.sub(r"\s+", "", s or "")

PERSON_CLASS = "인물"  # koelectra TTA_LABEL_KO["PS"] — NER 대분류 라벨과 일치해야 함


def load_occupation_lexicon(path: Optional[str] = None) -> dict[str, list[dict]]:
    """어휘집 로드 → {정규화 라벨: [{concept, conf, src}, ...]} (conf 내림차순)."""
    if path:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    else:
        ref = resources.files("ontokit.instance_typing").joinpath(
            "data/occupation_lexicon_ko.json.gz")
        with ref.open("rb") as f:
            rows = json.loads(gzip.decompress(f.read()).decode("utf-8"))
    lex: dict[str, list[dict]] = {}
    for r in rows:
        lex.setdefault(_norm(r["label"]), []).append(r)
    for v in lex.values():
        v.sort(key=lambda r: (-float(r.get("conf") or 0), r["concept"]))
    return lex


def apply_occupation_typing(all_entities: dict[str, list],
                            lexicon: Optional[dict[str, list[dict]]] = None,
                            person_class: str = PERSON_CLASS,
                            ) -> tuple[list[tuple[str, str]], list[dict], int]:
    """어휘집 매칭 개체를 직업클래스로 재타입 (in-place).

    반환: (직업클래스⊂인물 계층쌍 리스트, 추가 방출할 복수직업 엔티티 레코드,
    재타입 개체 수). 추가 레코드는 호출측이 all_entities 의 해당 문서 리스트에
    append 한다(여기선 원본 순회 중이라 in-place append 금지).
    """
    if lexicon is None:
        lexicon = load_occupation_lexicon()

    # 인물지배 컷 — 빌드 NER 분포에서 라벨별 클래스 최다표 (채택 라운드와 동일 규칙)
    dist: dict[str, dict[str, int]] = {}
    for ents in all_entities.values():
        for e in ents:
            n = _norm(e.get("entity") or "")
            if n:
                d = dist.setdefault(n, {})
                c = e.get("class") or ""
                d[c] = d.get(c, 0) + 1
    # 인물이 **최대값 중 하나**면 통과 — 채택 라운드(census 505 채점)의 동률 처리와
    # 정합('사나' {인물:1,지역:1} 류 동률을 오살하지 않음. 동률 케이스도 census 에서
    # 정탐 판정됨). 인물이 단독 열세일 때만 컷.
    person_dom = {n for n, d in dist.items()
                  if d and d.get(person_class, 0) == max(d.values())}

    hier_pairs: set[tuple[str, str]] = set()
    extra_records: list[dict] = []
    n_typed = 0
    seen_extra: set[tuple[str, str]] = set()  # (norm, concept) — 추가 레코드 중복 방지
    for doc, ents in all_entities.items():
        for e in ents:
            n = _norm(e.get("entity") or "")
            rows = lexicon.get(n)
            if not rows or n not in person_dom:
                continue
            # 가산 의미론(심판 D1): **인물 클래스 레코드만** 재타입 — 동명 라벨의
            # 비인물 레코드(수지의 '지역' 등)는 보존한다. 채택 라운드의 사이드카가
            # 기존 타입을 지우지 않는 가산 INSERT 였던 것과 정합. 인물지배 컷을
            # 통과했으므로 인물 레코드는 최소 1개 존재(인물이 최대값 중 하나).
            if e.get("class") != person_class:
                continue
            primary = rows[0]["concept"]
            e["class"] = primary
            n_typed += 1
            hier_pairs.add((primary, person_class))
            for r in rows[1:]:  # 복수 직업 — 동일 라벨 추가 레코드로 병기
                key = (n, r["concept"])
                if key in seen_extra:
                    continue
                seen_extra.add(key)
                rec = dict(e)
                rec["class"] = r["concept"]
                extra_records.append({"doc": doc, "record": rec})
                hier_pairs.add((r["concept"], person_class))
    return sorted(hier_pairs), extra_records, n_typed
