"""요약·통계 — 형태별 평균과 쌍 부트스트랩 95% CI(문항 재표집, 10,000회)."""
import json
import random
import sys

STRUCT = ("E1", "E2", "R", "A", "C", "M")


def boot_ci(diffs, n=10000, seed=0):
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(diffs)
    ms = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def summarize(res, base=None):
    rows, arms = res["rows"], res["arms"]
    forms = list(dict.fromkeys(r["form"] for r in rows))
    lines = [f"run={res['run']}  문항={len(rows)}",
             "| 팔 | " + " | ".join(forms) + " | 구조형 평균 | 조회(L) |",
             "|---|" + "---|" * (len(forms) + 2)]
    for arm in arms:
        cells = [f"{mean([r['arms'][arm]['score'] for r in rows if r['form'] == f]):.3f}" for f in forms]
        st = mean([r["arms"][arm]["score"] for r in rows if r["form"] in STRUCT])
        lk = mean([r["arms"][arm]["score"] for r in rows if r["form"] == "L"])
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | **{st:.3f}** | {lk:.3f} |")
    base = base or next((x for x in arms if x.startswith("V40")), None)
    if base:
        lines.append("")
        lines.append(f"쌍 차이(구조형, 기준 {base}) — 평균 Δ [95% CI]")
        for arm in arms:
            if arm == base:
                continue
            d = [r["arms"][arm]["score"] - r["arms"][base]["score"] for r in rows if r["form"] in STRUCT]
            lo, hi = boot_ci(d)
            lines.append(f"- {arm}: Δ={mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(d)})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(json.load(open(sys.argv[1])), sys.argv[2] if len(sys.argv) > 2 else None))
