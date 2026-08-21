"""R6 접미 조각 절단 탐지 — 좌측 경계 폐쇄 검사 (결정론, LLM 0콜).

## 기제

개체명이라면 코퍼스 어딘가에서 **왼쪽 경계가 열린 채로** — 문두이거나 왼쪽 인접이
비단어 문자인 위치에서 — 최소 한 번은 등장해야 한다. 라벨의 **모든** 출현에서
왼쪽 인접이 단어 문자라면, 그 문자열은 코퍼스 안에서 **더 긴 문자열의 내부로만**
존재한다. 즉 잘려나온 조각이다.

실측(ui_news100 538청크 · 로스터 1,099 라벨):

    llerd  ← Sellerd          왼쪽 단어문자 2/2
    웍스   ← CJ올리브네트웍스   4/4
    베이션 ← 오픈이노베이션      20/20
    날드   ← 맥도날드           7/7
    운드   ← 사운드/파운드리     11/11
    레콤   ← SK텔레콤          5/5

대조군(경계가 열려 **잡히지 않는** 정상 라벨):
    주립대 0/3 · 한국부 0/3 · 신흥 0/5 · 아시아태평양본부 0/2

## 왜 제거만 하는가

R3(a)·R4·R5 는 전부 스팬을 **확장·수정**하는 처치였고 전부 기각됐다. 확장은
정답을 만들어야 하지만, 제거는 **쓰레기를 버리기만** 하면 된다. 판정 부담이 낮다.

## 한계 (일반화 금지)

이 검사는 **주어진 코퍼스 안에서만** 성립한다. 다른 코퍼스에서는 같은 라벨의
경계가 열릴 수 있다. 코퍼스가 작을수록 우연히 안 열릴 확률이 오르므로,
저빈도 라벨이 오살의 주 발생원이다 — `min_occurrences` 로 방어한다.

공시 `eval_runs/bench/demo_roster/r6_predeclare.md` · 결과 `r6_result.md`.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[가-힣A-Za-z0-9]")

# ⛔ **기본 off** — 공격 심판 REJECT(0821). R3(a)·R4·R5 와 동일 처분.
#
# 기각은 효과 방향이 아니라 **측정**에 대한 것이다. 확인된 치명 결함 3건:
#   D-1 McNemar 무장 해제 — imp/wor 가 둘 다 같은 봉인 기준선 판정에서 나와
#       `imp+wor` 가 고정된다. 처치는 판정을 안 바꾸고 행만 지우므로 이 검정은
#       "제거된 9건의 기준선 판정이 무엇이었나"의 **동어반복**이다(판례 14).
#       R4·R5 기각 사유의 **3회차 재발**. p=0.0039 주장은 철회했다.
#   D-2 최악 조작화 5.71%가 **산문에만 있었다**(판례 30·24). 봉인 채점기의
#       `osal_worst` 가 다수결 이후 카운터를 써 구조적으로 무력했다.
#   D-4 게이트 Δ +3.86pp 는 **분자 불변(154→154)의 분모 축소**다. 참 라벨을
#       하나도 더 찾지 못했다. 재현율 없이는 성능 지표가 아니다.
#
# 미이행 재회부 조건: ④재현율 동시 측정 ⑤처치 독립 표본 ⑥심판 A·B 변별력
# ⑦심판 C 반대 4건(`빈맥` 외 판정불가 3) 확정 감정.
#
# 실측된 사실(기각과 무관하게 유효): 제거 70건 중 3인 전원 거짓 66~70건,
# 명백한 오살은 `빈맥` 1건. 처치는 아마 옳다 — 그것을 증명한 방법이 무효다.
#
# 켜려면 ONTOKIT_SUFFIX_FRAGMENT=on.
ENABLED = os.environ.get("ONTOKIT_SUFFIX_FRAGMENT", "").lower() in (
    "on", "true", "1")

MIN_LEN = 2          # 1글자 라벨은 판단 불가 — 손대지 않는다
MIN_OCCUR = 1        # 출현 0회면 표적 아님(원문에 없는 라벨은 다른 결함이다)


def left_closed(label: str, corpus: str) -> bool:
    """라벨의 **모든** 출현에서 왼쪽 인접이 단어문자면 True(= 접미 조각 의심).

    출현 0회면 False — 원문에 없는 라벨은 이 처치의 표적이 아니다.
    """
    if len(label) < MIN_LEN:
        return False
    seen = 0
    for m in re.finditer(re.escape(label), corpus):
        seen += 1
        i = m.start()
        if i == 0 or not _WORD.match(corpus[i - 1]):
            return False        # 경계가 한 번이라도 열리면 조각이 아니다
    return seen >= MIN_OCCUR


def drop_suffix_fragments(all_entities: dict, corpus: str) -> int:
    """엔티티 맵에서 접미 조각 라벨을 제거한다. in-place. 제거 건수 반환.

    `all_entities` 는 `{doc_name: [entity_dict, ...]}`.
    라벨 단위로 한 번만 판정하고 전 문서에 적용한다 — 같은 라벨이 문서마다
    다르게 판정되면 결정성이 깨진다.
    """
    if not ENABLED or not corpus or not all_entities:
        return 0
    labels = {(e.get("entity") or "").strip()
              for ents in all_entities.values() for e in ents}
    labels.discard("")
    doomed = {lb for lb in labels if left_closed(lb, corpus)}
    if not doomed:
        return 0
    n = 0
    for ents in all_entities.values():
        kept = [e for e in ents if (e.get("entity") or "").strip() not in doomed]
        n += len(ents) - len(kept)
        ents[:] = kept
    logger.info("접미 조각 게이트: 라벨 %d종 · 레코드 %d건 드랍", len(doomed), n)
    return n
