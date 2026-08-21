# -*- coding: utf-8 -*-
"""R12 게이트 채점기 — R11(first+트림)의 정식 판정.

⛔ 이 파일은 **심판 결과를 보기 전에** 커밋된다. 사후 조정 의혹을 구조로 차단한다.
사전 공시: eval_runs/bench/demo_roster/r12_predeclare.md

임계(사전 공시 §5 그대로, 여기서 바꾸지 않는다):
  채택 권고 : c/(b+c) >= 0.70  AND  McNemar p < 0.05
  기각      : b >= c
  보류      : 그 외
  판정불가 > 15% → 컨텍스트 결함 의심, 재감정 회부

⚠️ G3 조건 ③(심판 벤더 다양화) 미충족이므로 최대 산출은 **잠정** 채택 권고다.
   이 채점기는 그 사실을 판정 문자열에 **강제로 실어** 낸다 — 호출자가 뺄 수 없다.
"""
from __future__ import annotations
import collections
import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired import mcnemar_exact          # noqa: E402

ADOPT_RATIO = 0.70
UNDECIDED_MAX = 0.15
KAPPA_ALARM = 0.90                        # 판례 21 — 이 이상이면 독립성 의심
G3_UNMET = "③심판 벤더 다양화"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def cohen_kappa(a: list, b: list):
    """단일 범주 열이면 pe=1 이라 κ 가 정의되지 않는다 — 0 이나 1 로 위조하지 않는다."""
    if len(a) != len(b) or not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1 - pe) < 1e-12:
        return None                       # 미정의 — 판례 대로 명시
    return round((po - pe) / (1 - pe), 4)


def score(key_path: str, verdict_paths: dict) -> dict:
    key = {r["id"]: r for r in json.load(open(key_path, encoding="utf-8"))["key"]}
    votes, cols = collections.defaultdict(list), {}
    for j, p in verdict_paths.items():
        d = json.load(open(p, encoding="utf-8"))
        col = {}
        for v in d.get("verdicts", d):
            votes[v["id"]].append(v["판정"])
            col[v["id"]] = v["판정"]
        cols[j] = col

    ids = sorted(key)
    missing = {j: [i for i in ids if i not in c] for j, c in cols.items()}
    gold, undecided = {}, 0
    for i in ids:
        c = collections.Counter(votes.get(i, []))
        if not c:
            gold[i] = "판정불가"; undecided += 1; continue
        top, n = c.most_common(1)[0]
        if n >= 2 and top != "판정불가":
            gold[i] = top
        else:
            gold[i] = "판정불가"; undecided += 1

    # McNemar 표. group A = simple 전용, B = first+트림 전용.
    #   A 가 참  → 처치가 참을 잃음        = b(악화)
    #   A 가 거짓 → 처치가 거짓을 버림      = c(개선)
    #   B 가 참  → 처치가 참을 얻음        = c(개선)
    #   B 가 거짓 → 처치가 거짓을 만듦      = b(악화)
    b = c = 0
    detail = {"A_참": [], "A_거짓": [], "B_참": [], "B_거짓": []}
    for i in ids:
        g, grp = gold[i], key[i].get("group")
        if g == "판정불가":
            continue
        lb = key[i].get("label")
        if grp == "A":
            (detail["A_참"] if g == "참" else detail["A_거짓"]).append(lb)
            b += (g == "참"); c += (g == "거짓")
        else:
            (detail["B_참"] if g == "참" else detail["B_거짓"]).append(lb)
            c += (g == "참"); b += (g == "거짓")

    mc = mcnemar_exact(b, c)
    ratio = c / (b + c) if (b + c) else None
    lo, hi = wilson(c, b + c) if (b + c) else (0.0, 0.0)

    if ratio is None:
        verdict = "판정 불가 — 불일치 쌍 0(무증상 0 의심 원칙: 배선부터 의심)"
    elif b >= c:
        verdict = "⛔ 기각 — 악화가 개선 이상(b >= c)"
    elif ratio >= ADOPT_RATIO and mc["p"] is not None and mc["p"] < 0.05:
        verdict = f"✅ **잠정** 채택 권고 — G3 {G3_UNMET} 미충족이라 확정 아님"
    else:
        verdict = "보류 — 임계 미달"

    und_rate = undecided / len(ids)
    kap = {}
    js = sorted(cols)
    for x in range(len(js)):
        for y in range(x + 1, len(js)):
            a1 = [cols[js[x]].get(i, "미제출") for i in ids]
            a2 = [cols[js[y]].get(i, "미제출") for i in ids]
            kap[f"{js[x]}-{js[y]}"] = cohen_kappa(a1, a2)
    kv = [v for v in kap.values() if v is not None]

    return {
        "n": len(ids),
        "미제출": {j: len(m) for j, m in missing.items()},
        "다수결": dict(collections.Counter(gold.values())),
        "판정불가율": round(und_rate * 100, 2),
        "판정불가_경보": und_rate > UNDECIDED_MAX,
        "McNemar": mc,
        "b_악화": b, "c_개선": c,
        "이득률": None if ratio is None else round(ratio * 100, 2),
        "이득률_Wilson95": [round(lo * 100, 2), round(hi * 100, 2)],
        "임계": {"채택_이득률": ADOPT_RATIO * 100, "판정불가_상한": UNDECIDED_MAX * 100},
        "판정": verdict,
        "kappa": kap,
        "판례21_경보": bool(kv) and max(kv) >= KAPPA_ALARM,
        "G3_미충족": [G3_UNMET],
        "표본_한계": "불일치 항목만 추출 — 절대 정밀도 추정에 쓰지 말 것",
        "상세": detail,
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--verdicts", nargs="+", required=True, help="J=path 형식")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    vp = dict(s.split("=", 1) for s in a.verdicts)
    r = score(a.key, vp)
    json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in r.items() if k != "상세"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
