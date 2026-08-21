# -*- coding: utf-8 -*-
"""대응(paired) 설계 통계 — G3 조건 ①.

왜 필요한가. 0822 기준 우리 재현율 골드는 n=74 · MDE 8.11pp · 검정력 8% 다.
Card et al.(EMNLP 2020)이 **은퇴를 권고한** WNLI(147건, MDE 5.26%)보다 나쁘다.
비대응(unpaired) 두 비율 검정으로 5pp 를 80% 검정력에 탐지하려면 팔당 ~1,450건이
필요해 사실상 불가능하다.

대응 설계로 바꾸면 필요량이 **불일치 쌍(discordant pairs) 수**에만 의존한다.
같은 골드 항목에 before/after 를 둘 다 돌리고, **판정이 갈린 항목만** 센다.
처치가 항목의 일부만 건드리므로 대부분은 일치(concordant)해서 검정에서 상쇄된다.

⛔ 이 모듈은 **정밀도를 계산하지 않는다.** 두 시스템의 짝지어진 판정 배열만 받는다.
   짝짓기 자체가 결함원이었으므로(R7 유령 짝 63건) 짝짓기는 호출자 책임이고,
   여기서는 **길이 일치와 값 도메인만 강제**한다.
"""
from __future__ import annotations
import math
from itertools import accumulate

# ── 이항 꼬리 (scipy 없이. 결정적 연산만) ────────────────────────────────────
def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_exact(b: int, c: int) -> dict:
    """McNemar 정확검정(양측). b = before만 맞음, c = after만 맞음.

    귀무가설: 불일치 쌍이 b/c 로 갈릴 확률이 각각 1/2.
    n = b + c 가 **0 이면 p-value 는 정의되지 않는다**(1.0 이 아니다) —
    "처치가 아무것도 안 바꿨다"와 "표본이 없다"를 구별하기 위해 None 을 낸다.
    """
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p": None,
                "판정": "불일치 쌍 0 — 검정 불가(처치 무효과와 무표본을 구별 못 함)"}
    lo = min(b, c)
    tail = sum(_binom_pmf(k, n) for k in range(lo + 1))
    p = min(1.0, 2 * tail)
    return {"b": b, "c": c, "n_discordant": n, "p": round(p, 6),
            "판정": ("유의(α=0.05)" if p < 0.05 else "유의하지 않음"),
            "방향": ("개선" if c > b else "악화" if b > c else "무방향")}


def paired_bootstrap(before: list[bool], after: list[bool],
                     n_boot: int = 10000, seed: int = 0) -> dict:
    """짝지어진 정확도 차이의 부트스트랩 CI. 결정적(seed 고정)."""
    if len(before) != len(after):
        raise ValueError(f"길이 불일치 {len(before)} != {len(after)} — 짝짓기 결함")
    n = len(before)
    if n == 0:
        raise ValueError("표본 0 — 부트스트랩 불가")
    d = [int(a) - int(b) for b, a in zip(before, after)]
    obs = sum(d) / n
    rng = _Lcg(seed)
    diffs = []
    for _ in range(n_boot):
        s = sum(d[rng.below(n)] for _ in range(n))
        diffs.append(s / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))]
    return {"n": n, "관측차": round(obs * 100, 2),
            "CI95": [round(lo * 100, 2), round(hi * 100, 2)],
            "판정": ("유의(0 미포함)" if lo > 0 or hi < 0 else "유의하지 않음(0 포함)")}


class _Lcg:
    """표준 난수 대신 고정 LCG — 결정성 봉인(같은 seed = 같은 결과, 플랫폼 무관)."""
    def __init__(self, seed: int): self.s = (seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
    def below(self, n: int) -> int:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self.s >> 33) % n


# ── 표본 크기 산정 ────────────────────────────────────────────────────────────
def required_discordant(delta_ratio: float, power: float = 0.80,
                        alpha: float = 0.05) -> int:
    """McNemar 에 필요한 **불일치 쌍** 수.

    delta_ratio = c/(b+c) — 불일치 중 개선 쪽 비율. 0.5 면 효과 없음.
    정규근사: n_disc >= (z_a/2 * 1 + z_b * 2*sqrt(r(1-r)))^2 / (2r-1)^2
    """
    if not 0 < delta_ratio < 1:
        raise ValueError("delta_ratio 는 (0,1)")
    if abs(delta_ratio - 0.5) < 1e-9:
        return -1                      # 효과 0 — 어떤 표본으로도 탐지 불가
    za = 1.959963985                   # α=0.05 양측
    zb = {0.80: 0.8416212, 0.90: 1.2815516, 0.95: 1.6448536}.get(power)
    if zb is None:
        raise ValueError("power 는 0.80/0.90/0.95")
    r = delta_ratio
    num = za + zb * 2 * math.sqrt(r * (1 - r))
    return math.ceil(num * num / ((2 * r - 1) ** 2))


def required_gold(delta_ratio: float, touch_rate: float,
                  power: float = 0.80, alpha: float = 0.05) -> dict:
    """골드 전체 크기. touch_rate = 처치가 판정을 바꾸는 항목 비율(= 불일치율).

    ⚠️ touch_rate 는 **추정치**다. 라운드마다 실측해서 갱신하라 —
       과대 추정하면 표본이 모자라 검정력이 없고, 과소 추정하면 낭비다.
    """
    nd = required_discordant(delta_ratio, power, alpha)
    if nd < 0:
        return {"필요_불일치쌍": None, "필요_골드": None,
                "판정": "delta_ratio=0.5 — 효과 0 이라 어떤 표본으로도 탐지 불가"}
    if not 0 < touch_rate <= 1:
        raise ValueError("touch_rate 는 (0,1]")
    return {"필요_불일치쌍": nd, "필요_골드": math.ceil(nd / touch_rate),
            "delta_ratio": delta_ratio, "touch_rate": touch_rate,
            "power": power, "alpha": alpha}


def main(argv=None):
    import argparse, json
    ap = argparse.ArgumentParser(description="대응 설계 통계 (G3 조건 ①)")
    ap.add_argument("--plan", action="store_true", help="필요 골드 크기 표")
    ap.add_argument("--mcnemar", nargs=2, type=int, metavar=("B", "C"))
    a = ap.parse_args(argv)
    if a.mcnemar:
        print(json.dumps(mcnemar_exact(*a.mcnemar), ensure_ascii=False, indent=2))
    if a.plan:
        print(f"{'고침:깨짐':>10}{'delta_ratio':>13}{'필요 불일치쌍':>15}"
              f"{'  필요 골드(불일치율별)'}")
        print(f"{'':>38}" + "".join(f"{t:>10.0%}" for t in (0.05, 0.10, 0.20, 0.40)))
        print("-" * 80)
        for fix, brk in ((2, 1), (3, 1), (4, 1), (5, 1), (9, 1)):
            r = fix / (fix + brk)
            nd = required_discordant(r)
            row = f"{f'{fix}:{brk}':>10}{r:>13.3f}{nd:>15}"
            for t in (0.05, 0.10, 0.20, 0.40):
                row += f"{math.ceil(nd / t):>10}"
            print(row)
        print("\n비교: 비대응 설계로 5pp 탐지 시 팔당 ~1,450건 (p≈0.63 부근)")
    if not (a.plan or a.mcnemar):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
