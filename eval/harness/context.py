"""근거 컨텍스트 선택 — **독립 출현 우선**.

## 왜 필요한가

R7 공격 심판이 지적했다: *"심판이 받은 근거가 변경 지점이 아닌 경우가 소멸집합의 17.7%"*.
구 생성기는 `text.find(label)` 로 **첫 출현**을 잡는데, 짧은 라벨은 다른 단어 **내부**에
먼저 걸린다. 미이행으로 남았고 **R10 에서 재발했다**:

    '서부' → 심판이 본 것: `발굴에서부터` · `단계에서부터`   ← 라벨이 다른 단어 안에 있다
             실제 개체 자리: `남동·동서·남부·서부·중부발전`  ← 못 보여줬다

R10 실측: 코퍼스에 독립 출현이 있는데 컨텍스트가 못 보여준 항목 **14/152 = 9.2%**.
심판 B 가 그중 하나(`서부`)를 판정불가로 처리하며 스스로 적발했다.

## 규칙

라벨이 **독립 형태**(좌우 인접이 단어 문자가 아님)로 나오는 출현을 **우선** 뽑는다.
독립 출현이 하나도 없으면 그 사실 자체가 정보이므로(= 조각일 가능성) 내부 출현을
쓰되 **`[내부출현]` 표시**를 붙여 심판이 구분할 수 있게 한다.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[가-힣A-Za-z0-9]")
CTX = 90


def _standalone(text: str, i: int, j: int) -> bool:
    left = i > 0 and _WORD.match(text[i - 1])
    right = j < len(text) and _WORD.match(text[j])
    return not (left or right)


def contexts(labels, chunks, max_per_label=2, ctx=CTX):
    """{label: [context, ...]} — 독립 출현을 우선한다.

    chunks: [{'chunk_text': ...}, ...]
    """
    texts = [c["chunk_text"] for c in chunks]
    out = {}
    for lb in labels:
        if not lb:
            continue
        free, inside = [], []
        for t in texts:
            for m in re.finditer(re.escape(lb), t):
                i, j = m.start(), m.end()
                seg = t[max(0, i - ctx):j + ctx].replace("\n", " ")
                (free if _standalone(t, i, j) else inside).append(seg)
                if len(free) >= max_per_label:
                    break
            if len(free) >= max_per_label:
                break
        if free:
            out[lb] = free[:max_per_label]
        elif inside:
            # 독립 출현이 없다 — 그 자체가 정보다(조각일 가능성). 표시해서 넘긴다.
            out[lb] = ["[내부출현] " + s for s in inside[:max_per_label]]
        else:
            out[lb] = ["(원문 전체에서 이 문자열을 찾지 못함 — 판정불가 사유가 된다)"]
    return out
