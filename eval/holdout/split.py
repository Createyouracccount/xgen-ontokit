# -*- coding: utf-8 -*-
"""분석셋 / 잠금 테스트셋 분리 — G3 조건 ②.

왜 필요한가. Dwork et al.(Science 2015)이 형식화한 **적응적 홀드아웃 소진**:
같은 홀드아웃을 반복해서 들여다보며 처치를 고르면 **홀드아웃 자체에 과적합**되고,
접근 횟수가 늘수록 타당성이 열화된다.

우리 실태(0821l 까지): 처치 후보를 **평가셋에서 뽑고** 같은 셋으로 채점했다.
`문화·제도 보통명사 필터` 후보가 R10 감정 65종을 **보고** 만들어질 뻔한 것이 그 예다.
내부 규칙 "처치를 고르게 한 지표는 그 처치의 검증 게이트가 될 수 없다"(판례 14)의
학술적 뒷받침이 바로 이 문헌이다.

실무 최소판:
  분석셋(analysis) — **무제한 열람**. 오류 감정·처치 후보 도출은 **여기서만**.
  잠금셋(locked)   — 접근 횟수를 **세고** 라운드당 1회로 제한. 라벨은 열지 않는다.

⚠️ 이 도구는 접근을 **물리적으로 막지 못한다**(파일은 읽을 수 있다). 하는 일은
   **접근을 기록하고 초과를 드러내는 것**이다. 규율의 대체물이 아니라 증거다.
"""
from __future__ import annotations
import hashlib
import json
import os

MAX_PER_ROUND = 1          # 라운드당 잠금셋 접근 허용 횟수


def _h(s: str, seed: str) -> int:
    return int(hashlib.sha256(f"{seed}|{s}".encode()).hexdigest()[:16], 16)


def split(items: list, seed: str, locked_frac: float = 0.5, key=str) -> dict:
    """해시 기반 결정적 분할 — 같은 seed·같은 항목이면 항상 같은 쪽으로 간다.

    난수 셔플이 아니라 **항목 해시**를 쓰는 이유: 나중에 항목이 추가돼도
    기존 항목의 소속이 **바뀌지 않는다**. 셔플이면 전체가 재배치돼 잠금셋이 오염된다.
    """
    if not 0 < locked_frac < 1:
        raise ValueError("locked_frac 는 (0,1)")
    cut = int(locked_frac * (1 << 64))
    analysis, locked = [], []
    for it in items:
        (locked if _h(key(it), seed) % (1 << 64) < cut else analysis).append(it)
    return {"seed": seed, "locked_frac": locked_frac,
            "analysis": analysis, "locked": locked,
            "n_analysis": len(analysis), "n_locked": len(locked)}


class Ledger:
    """잠금셋 접근 장부. 열 때마다 기록하고, 라운드당 상한 초과를 드러낸다."""

    def __init__(self, path: str):
        self.path = path
        self.log = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []

    def access(self, round_id: str, purpose: str, n_items: int) -> dict:
        prior = [e for e in self.log if e["round"] == round_id]
        entry = {"round": round_id, "purpose": purpose, "n_items": n_items,
                 "seq": len(prior) + 1}
        over = len(prior) + 1 > MAX_PER_ROUND
        entry["초과"] = over
        self.log.append(entry)
        json.dump(self.log, open(self.path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        if over:
            entry["경고"] = (
                f"⛔ 라운드 {round_id} 의 잠금셋 접근 {len(prior)+1}회 — 상한 {MAX_PER_ROUND}. "
                "적응적 홀드아웃 소진(Dwork et al. 2015). 이 라운드 결과에 접근 횟수를 공시하라.")
        return entry

    def summary(self) -> dict:
        by = {}
        for e in self.log:
            by.setdefault(e["round"], 0)
            by[e["round"]] += 1
        return {"총_접근": len(self.log), "라운드별": by,
                "초과_라운드": [r for r, n in by.items() if n > MAX_PER_ROUND]}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="분석셋/잠금셋 분리 (G3 조건 ②)")
    ap.add_argument("--items", required=True, help="JSON 배열 파일")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--field", default=None, help="항목이 dict 면 분할 키로 쓸 필드")
    ap.add_argument("--locked-frac", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = json.load(open(a.items, encoding="utf-8"))
    key = (lambda x: str(x[a.field])) if a.field else str
    r = split(items, a.seed, a.locked_frac, key)
    json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"분석셋 {r['n_analysis']} · 잠금셋 {r['n_locked']} (seed={a.seed})")
    print("⚠️ 잠금셋 라벨은 열지 않는다. 접근 시 Ledger.access() 로 기록하라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
