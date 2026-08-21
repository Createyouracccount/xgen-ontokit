"""재현율 측정 — ① 표본 추출 + 맹검 열거 패킷 생성.

## 왜 이 도구가 필요한가

R6 까지 다섯 라운드가 **정밀도만** 재고 재현율을 못 쟀다. 그 결과:

- 제거 처치(R6)의 게이트 Δ 가 **분자 불변(154→154)의 분모 축소**였는데도
  "+3.86pp 개선"으로 읽혔다. 극단적으로 라벨을 전부 지우면 정밀도는 100% 로 간다.
- 공격 심판 재회부 조건 ④: *"재현율 동시 측정 — 없으면 Δ 주장 철회"*.
  **이 도구 없이는 어떤 제거 처치도 채택할 수 없다.**

## 기제

시스템 출력을 **보여주지 않고** 심판에게 청크 원문만 준다. 심판은 그 안의
인물·기관·지역을 **전수 열거**한다. 이것이 gold 다. 그 다음:

    재현율 = |gold ∩ 시스템| / |gold|

시스템 출력을 안 보여주는 것이 핵심이다 — 보여주면 확인 편향으로 gold 가
시스템 출력 쪽으로 끌려간다.

## 표층형 정책 (사전 고정)

심판에게 **원문에 나타난 그대로** 복사하라고 지시한다. 그러면 매칭이 다음 셋으로 갈린다:

- **정확일치** — 시스템 라벨 == gold
- **부분일치** — 한쪽이 다른 쪽을 포함 (`카카오` vs `카카오게임즈`)
  → 이것이 곧 **경계 결함**이며, 우리 결함 축의 57.5% 다. 별도 집계한다.
- **미탐** — 겹치는 것이 없음

부분일치를 재현율 분자에 넣을지는 `score.py` 가 **양쪽 다** 계산해 병기한다(판례 19).
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "..", "..", "..", "eval_runs", "bench",
                      "ui_news100_chunks_sealed.json")

SEED = 20260821_7      # 신선 표본 (판례 14 — 기존 표본은 전부 소진됨)
N_CHUNKS = 30


def build(seed: int = SEED, n: int = N_CHUNKS) -> dict:
    with open(CHUNKS, encoding="utf-8") as f:
        rows = json.load(f)
    rows = sorted(rows, key=lambda r: r["chunk_id"])     # 결정성
    picked = random.Random(seed).sample(rows, n)
    picked.sort(key=lambda r: r["chunk_id"])
    return {
        "seed": seed,
        "n_chunks": len(picked),
        "총문자수": sum(len(r["chunk_text"]) for r in picked),
        "지침": {
            "임무": "각 청크 원문을 읽고 그 안의 **인물·기관·지역 고유명사를 전수 열거**한다",
            "클래스": {
                "인물": "사람 이름. 기자·통신원 바이라인 포함",
                "기관": "회사·정부부처·단체·학교·언론사 등 조직",
                "지역": "국가·도시·행정구역·지명",
            },
            "표기": "⚠️ **원문에 나타난 문자열 그대로** 복사할 것. 정식명칭으로 고쳐 쓰지 말 것. "
                    "원문이 '한전'이면 '한전'으로, '카카오게임즈'면 '카카오게임즈'로 적는다.",
            "경계": "개체명의 **온전한 경계**로 적는다. 조사(이/가/을/를/은/는/의/에)는 빼고, "
                    "복합 고유명사는 통째로 적는다('카카오게임즈'를 '카카오'로 줄이지 말 것).",
            "제외": "제품명·차량모델·백신명·질병명·법령명은 위 3클래스가 아니므로 **열거하지 않는다**. "
                    "보통명사('정부'·'회사'·'대통령')도 열거하지 않는다.",
            "중복": "같은 개체가 한 청크에 여러 번 나오면 **한 번만** 적는다.",
            "완전성": "🔑 이 작업의 가치는 **빠뜨리지 않는 것**에 있다. 애매하면 넣고 "
                      "`불확실: true` 를 표시하라. 빠뜨리면 그 개체는 영원히 측정되지 않는다.",
        },
        "chunks": [{"chunk_id": r["chunk_id"], "text": r["chunk_text"]} for r in picked],
    }


if __name__ == "__main__":
    out = build()
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "recall_packet.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("chunks", "지침")},
                     ensure_ascii=False, indent=1))
