"""집합 감정 게이트 — 하네스 5조건을 **전부 코드로** 강제한다.

R4·R5·R6·R7 네 라운드가 감정 설계에서 죽었다. 그 사유를 하나씩 코드로 막는다:

| 사유 | 라운드 | 이 모듈의 처리 |
|---|---|---|
| 정답 누설 | R7 | `packet.py` 가 배포본에 넣을 방법을 없앰 |
| 심판 1인이 결과를 좌우 | R7 | **LOO 를 기본 출력** · 뒤집히면 자동 기각 |
| 여유 작은 임계에 최악값 미적용 | R6·R7 | **모든 임계에 최악 조작화 의무** |
| 도달 불가 임계를 통과로 셈 | R4·R5·R6·R7 | **도달 가능성 실측** · 불가면 통과 무효 |
| 사전 등록 자를 산문으로 처리 | R5·R7 | `ruler.py` 를 **채점기가 강제 적용** |
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ruler import apply as ruler_apply    # noqa: E402

JUDGES = ("A", "B", "C")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    if not n:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def majority(votes):
    c = Counter(votes)
    for cand in ("참", "거짓"):
        if c[cand] >= 2:
            return cand
    return "판정불가"


def score(pack, key, verdicts, gain_group, loss_group,
          th_net=10, th_loss_rate=0.20, th_gain_rate=0.50):
    """집합 감정 채점. 임계는 호출자가 **실행 전에** 고정해 넘긴다."""
    items = {i["id"]: i for i in pack["items"]}
    grp = {k["id"]: k["group"] for k in key["key"]}
    lab = {i: items[i]["label"] for i in items}
    complete = [i for i in items if all(i in verdicts[j] for j in JUDGES)]

    # ① 다수결 → ② 사전 등록 자 강제 적용 (K-1: 심판 일치는 자를 이기지 못한다)
    raw_maj = {i: majority([verdicts[j][i]["판정"] for j in JUDGES]) for i in complete}
    ruled = ruler_apply(raw_maj, lab)
    maj = ruled["판정"]

    def tally(ids):
        c = Counter(maj[i] for i in ids)
        judged = c["참"] + c["거짓"]
        p, lo, hi = wilson(c["참"], judged) if judged else (0, 0, 0)
        unan = [i for i in ids
                if all(verdicts[j][i]["판정"] == "참" for j in JUDGES)
                and maj[i] == "참"]
        doubt = [i for i in ids
                 if any(verdicts[j][i]["판정"] in ("참", "판정불가") for j in JUDGES)]
        return {"n": len(ids), "분해": dict(c), "참": c["참"],
                "비율": round(p, 4), "CI95": [round(lo, 4), round(hi, 4)],
                "만장일치참": len(unan), "만장일치비율": round(len(unan) / len(ids), 4)
                if ids else 0.0,
                "최악(원표참+판정불가)": len(doubt)}

    G = tally([i for i in complete if grp[i] == gain_group])
    L = tally([i for i in complete if grp[i] == loss_group])
    net = G["참"] - L["참"]

    # ③ LOO — 1인 제외로 임계가 뒤집히면 자동 기각
    loo = {}
    gain_ids = [i for i in complete if grp[i] == gain_group]
    for out_j in JUDGES:
        keep = [j for j in JUDGES if j != out_j]
        k = sum(1 for i in gain_ids
                if all(verdicts[j][i]["판정"] == "참" for j in keep)
                and maj[i] == "참")
        loo[f"{out_j} 제외"] = {"이득": k, "이득률": round(k / len(gain_ids), 4),
                                 "통과": k / len(gain_ids) >= th_gain_rate}
    loo_ok = all(v["통과"] for v in loo.values())

    # ④ 도달 가능성 — 손실 분자가 0 이 아닐 수 있는가
    loss_dissent = sum(1 for i in complete if grp[i] == loss_group
                       and any(verdicts[j][i]["판정"] != "거짓" for j in JUDGES))
    reach = {"손실 분자 도달": {"관측 손실": L["참"],
                                "반대의견 있는 손실후보": loss_dissent,
                                "도달가능": loss_dissent > 0}}

    # ⑤-a **누설 검증** — id 가 집합을 누설하는가. 이것이 0 이어야 ⑤-b 가 무해하다.
    #    실측: R7(하네스 이전) r = +0.861 → R8(하네스) r = +0.026.
    ids_sorted = sorted(grp)
    yv = [1 if grp[i] == gain_group else 0 for i in ids_sorted]
    mx = (len(ids_sorted) - 1) / 2
    my = sum(yv) / len(yv)
    num = sum((k - mx) * (yv[k] - my) for k in range(len(ids_sorted)))
    den = (sum((k - mx) ** 2 for k in range(len(ids_sorted)))
           * sum((v - my) ** 2 for v in yv)) ** 0.5
    leak_r = num / den if den else 0.0

    # ⑤-b 심판별 **판정 순서**와 id 의 상관.
    #    ⚠️ 이 값이 ±1 이어도 ⑤-a 가 0 이면 **무해**하다 — 팩 배열 순서대로 작업한
    #    것일 뿐 정답에 도달할 수 없다. R7 재회부 조건 #2(|ρ|>0.3 무효)는 id 가
    #    집합을 누설하던 조건에서 나온 것이며, 하네스가 그 전제를 제거했다.
    #    규칙을 없애지 않고 **두 값을 함께** 내어 판단 근거를 남긴다.
    order = {}
    for j in JUDGES:
        seq = [v["id"] for v in verdicts[j].values()]
        m = (len(seq) - 1) / 2
        d2 = sum((k - m) ** 2 for k in range(len(seq)))
        order[j] = round(sum((k - m) * (seq[k] - m)
                             for k in range(len(seq))) / d2, 4) if d2 else None

    pair = {}
    for a, b in combinations(JUDGES, 2):
        va = [verdicts[a][i]["판정"] for i in complete]
        vb = [verdicts[b][i]["판정"] for i in complete]
        kp = kappa(va, vb)
        pair[f"{a}-{b}"] = {
            "일치율": round(sum(1 for x, y in zip(va, vb) if x == y) / len(complete), 4),
            "kappa": round(kp, 4) if kp is not None else "미정의(pe=1)"}

    gate = {
        f"순효과>=+{th_net}": {"값": net, "통과": net >= th_net},
        f"손실률<={th_loss_rate:.0%}": {"값": L["비율"],
                                        "통과": L["비율"] <= th_loss_rate},
        f"이득률(다수결)>={th_gain_rate:.0%}": {"값": G["비율"],
                                                "통과": G["비율"] >= th_gain_rate},
        f"이득률(만장일치)>={th_gain_rate:.0%}": {"값": G["만장일치비율"],
                                                  "통과": G["만장일치비율"] >= th_gain_rate},
        f"이득률 CI하한>={th_gain_rate:.0%}": {"값": G["CI95"][0],
                                               "통과": G["CI95"][0] >= th_gain_rate},
        "최악순효과>0": {"값": G["만장일치참"] - L["최악(원표참+판정불가)"],
                         "통과": (G["만장일치참"] - L["최악(원표참+판정불가)"]) > 0,
                         "주": "이득=만장일치참 · 손실=원표최악 — 진짜 불리한 조합"},
    }
    return {
        "표본": {"이득후보": G["n"], "손실후보": L["n"], "3인전원판정": len(complete)},
        "사전등록자_덮어씀": {"수": ruled["덮어쓴수"], "내역": ruled["자_덮어씀"]},
        "이득": G, "손실": L, "순효과": net,
        "leave_one_out": loo,
        "임계_도달가능성": reach,
        "판례21_독립성": pair,
        "누설검증": {"id↔집합 상관": round(leak_r, 4),
                     "판정순서↔id 상관": order,
                     "판정": ("누설 없음 — 순서 상관은 무해"
                              if abs(leak_r) < 0.15 else
                              "⛔ id 가 집합을 누설한다. 순서 상관이 유해하다"),
                     "참고": "R7(하네스 이전) id↔집합 = +0.861"},
        "임계판정": gate,
        "최종": ("채택 후보" if all(g["통과"] for g in gate.values()) and loo_ok
                 and all(r["도달가능"] for r in reach.values()) else "채택 안 함"),
    }


def main(work, pack_path, key_path, gain_group, loss_group, out):
    pack = json.load(open(pack_path, encoding="utf-8"))
    key = json.load(open(key_path, encoding="utf-8"))
    V = {}
    for j in JUDGES:
        d = json.load(open(os.path.join(work, f"r8_verdicts_{j}.json"), encoding="utf-8"))
        V[j] = {v["id"]: v for v in d["verdicts"]}
    res = score(pack, key, V, gain_group, loss_group)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
