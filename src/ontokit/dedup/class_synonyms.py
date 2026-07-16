"""클래스(TBox) 동의어 후보 생성 — 우리말샘 사전 lookup. LLM 0회, 자동 병합 없음.

R9(2026-07-16) 심판 판정의 코드화. 배경: 검색 시드가 질문어를 GT 와 다른
동의어 중복 클래스(국가 vs 나라)로 해소해 무효화되는 문제(R8). 인스턴스
ER(개방집합, 임베딩 AUC 0.81 천장)과 달리 클래스 라벨은 폐집합·소규모라
사전 lexical 매칭이 유효 — 단 사전 단독은 FP 지뢰밭임이 실측됐다
(동음이의 오만Oman↔종교, 약어 성우⊂유성우, 코퍼스-어의 불일치 '단체'
조직↔홑원소물질). 그래서 이 모듈은 **후보를 생성할 뿐 병합하지 않는다**:

  · 자동 병합 티어는 심판이 기각 — 코퍼스 신호(인스턴스 교집합)가 후보
    224쌍 중 1쌍에서만 평가 가능한 빈 게이트였다.
  · 병합(equivalentClass 링크)은 검수 통과 화이트리스트만. 검수는 오프라인
    (사람 또는 블라인드 채점 에이전트 합의) — 빌드 파이프라인 밖.

결정론 deny 게이트(전부 심판 승인, 새 어휘목록 없음):
  substr — 한쪽 라벨이 다른쪽의 부분문자열(약어/복합어 FP: 성우⊂유성우).
  ambig  — 라벨이 여러 synset 에 속함(다의어: 사상思想/死傷).
  hub    — 인스턴스 수가 임계 초과(NER 허브 '기관' 32K 오병합 방지).
  zero   — 양쪽 다 인스턴스 0(검색 이득 증명 불가 — 계층 트랙으로 보류).
"""
from __future__ import annotations

from .synonym_dict import SynonymDictDedup, _surface

HUB_INSTANCE_LIMIT = 5000


def class_synonym_candidates(
    labels_with_counts: dict[str, int],
    dict_dedup: SynonymDictDedup,
) -> list[dict]:
    """클래스 라벨→인스턴스 수 dict 에서 동의어 후보쌍을 생성한다.

    반환: [{"a", "b", "inst_a", "inst_b", "deny": [사유...]}] — deny 가 빈 쌍만
    검수 후보로 올릴 것. deny 사유도 반환해 검수 리포트에서 왜 제외됐는지 보인다.
    """
    surf2labels: dict[str, list[str]] = {}
    for lab in labels_with_counts:
        surf2labels.setdefault(_surface(lab), []).append(lab)

    # synset 대표 → 표면형들 (사전 그룹핑)
    rep2surfs: dict[str, set] = {}
    for surf in surf2labels:
        for rep in dict_dedup._s2reps.get(surf, ()):
            rep2surfs.setdefault(rep, set()).add(surf)

    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for surfs in rep2surfs.values():
        if len(surfs) < 2:
            continue
        ordered = sorted(surfs)
        for i, sa in enumerate(ordered):
            for sb in ordered[i + 1:]:
                key = (sa, sb)
                if key in seen:
                    continue
                seen.add(key)
                la, lb = surf2labels[sa][0], surf2labels[sb][0]
                ia, ib = labels_with_counts.get(la, 0), labels_with_counts.get(lb, 0)
                deny: list[str] = []
                if sa in sb or sb in sa:
                    deny.append("substr")
                if len(dict_dedup._s2reps.get(sa, ())) > 1 or \
                        len(dict_dedup._s2reps.get(sb, ())) > 1:
                    deny.append("ambig")
                if max(ia, ib) > HUB_INSTANCE_LIMIT:
                    deny.append("hub")
                if ia == 0 and ib == 0:
                    deny.append("zero")
                out.append({"a": la, "b": lb, "inst_a": ia, "inst_b": ib, "deny": deny})
    # 검수 우선순위: deny 없는 쌍 먼저, 인스턴스 합 내림차순(검색 영향 큰 순)
    out.sort(key=lambda x: (bool(x["deny"]), -(x["inst_a"] + x["inst_b"])))
    return out
