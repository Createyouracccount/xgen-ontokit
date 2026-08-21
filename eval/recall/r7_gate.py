"""R7 어절 경계 정합 — 게이트 채점.

⚠️ **심판 스폰 전에 커밋된다.** git 해시를 사전 공시에 박는다(R6 교훈).

## 설계 — 짝짓기 없는 집합 감정

처치는 라벨을 **바꾼다**. before→after 짝을 만들려 했으나 두 결함이 드러났다:
① 문자열 포함 폴백이 가짜 짝 63건 생성 ② 확장이 start 를 앞으로 옮겨 다른 라벨의
자리로 이동하면서 **오귀속**(`모공원`→`서울추모공원` 을 `서울`→로 보고).

→ 짝 대신 **소멸 집합 / 신설 집합을 각각 맹검 감정**한다. 오귀속이 원천 불가능하다.

    이득 = 신설 중 참    (없던 옳은 라벨이 생김)
    손실 = 소멸 중 참    (있던 옳은 라벨이 사라짐)
    순효과 = 이득 - 손실

## 임계 (공시 `r7_predeclare.md` §3 — 여기 하드코딩, 인자로 받지 않는다)
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from itertools import combinations

JUDGES = ("A", "B", "C")

TH_NET = 10          # 순효과 ≥ +10 라벨. 미달이면 채택 안 함
TH_LOSS_RATE = 0.20  # 손실률(소멸 중 참) > 20% → 롤백
TH_GAIN_RATE = 0.50  # 이득률(신설 중 참) < 50% → 채택 안 함


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


def _tally(ids, V, items):
    """다수결 + 원표 기반 최악/보수 조작화를 **한 번에** 낸다.

    ⛔ R6 에서 최악값을 다수결 **이후** 카운터로 계산해 구조적으로 무력했다
       (`majority()` 가 판정불가를 삼켜 항상 0 이 나왔다). 여기서는 원표를 쓴다.
    """
    maj = {i: majority([V[j][i]["판정"] for j in JUDGES]) for i in ids}
    dm = Counter(maj.values())
    judged = dm["참"] + dm["거짓"]
    raw_true = [i for i in ids if any(V[j][i]["판정"] == "참" for j in JUDGES)]
    raw_doubt = [i for i in ids
                 if any(V[j][i]["판정"] in ("참", "판정불가") for j in JUDGES)]
    # 가장 불리한 조작화 — 3인 **만장일치 참**만 참으로 센다.
    # 여유가 작은 임계(이득률은 여유 2건)에는 이것이 필수다(공격 심판 S2).
    unan = [i for i in ids if all(V[j][i]["판정"] == "참" for j in JUDGES)]
    p, lo, hi = wilson(dm["참"], judged)
    return {
        "n": len(ids), "분해": dict(dm), "판단선것": judged,
        "참(다수결)": dm["참"], "비율": round(p, 4),
        "CI95": [round(lo, 4), round(hi, 4)],
        "보수(원표 참)": {"k": len(raw_true), "비율": round(len(raw_true) / len(ids), 4)},
        "만장일치 참": {"k": len(unan), "비율": round(len(unan) / len(ids), 4)},
        "최악(원표 참+판정불가)": {"k": len(raw_doubt),
                                   "비율": round(len(raw_doubt) / len(ids), 4)},
        "_maj": maj,
    }


def main(work):
    pack = json.load(open(os.path.join(work, "r7_adjudicator_pack.json"), encoding="utf-8"))
    items = {i["id"]: i for i in pack["items"]}
    V = {}
    for j in JUDGES:
        d = json.load(open(os.path.join(work, f"r7_verdicts_{j}.json"), encoding="utf-8"))
        V[j] = {v["id"]: v for v in d["verdicts"]}
    complete = [i for i in items if all(i in V[j] for j in JUDGES)]

    gone_ids = [i for i in complete if items[i]["_set"] == "소멸"]
    new_ids = [i for i in complete if items[i]["_set"] == "신설"]
    G, N = _tally(gone_ids, V, items), _tally(new_ids, V, items)

    gain, loss = N["참(다수결)"], G["참(다수결)"]
    net = gain - loss

    # ⛔ 초판은 `gain_w` 에 **이득의 최대값**(원표 참+판정불가 = 57)을 썼다.
    #    그 결과 "최악값"(51)이 점추정(48)보다 **커서** 최악값이 아니었고,
    #    임계4 가 무장 해제됐다(공격 심판 S3). 공시 §3 문면은
    #    `이득 = 다수결 참 · 손실 = 원표 최악` 이다. 문면대로 고친다(K-1).
    #    현행·공시·최불리 세 값을 **전부** 출력해 반사실을 병기한다(판례 29).
    gain_w = gain                                    # 공시 문면: 이득은 다수결 참
    loss_w = G["최악(원표 참+판정불가)"]["k"]         # 손실은 원표 최악
    gain_unan = N["만장일치 참"]["k"]                 # 가장 불리한 이득

    pair = {}
    for a, b in combinations(JUDGES, 2):
        va = [V[a][i]["판정"] for i in complete]
        vb = [V[b][i]["판정"] for i in complete]
        k = kappa(va, vb)
        pair[f"{a}-{b}"] = {
            "일치율": round(sum(1 for x, y in zip(va, vb) if x == y) / len(complete), 4),
            "kappa": round(k, 4) if k is not None else "미정의(pe=1)"}

    gate = {
        f"순효과>=+{TH_NET}": {"값": net, "임계": TH_NET, "통과": net >= TH_NET},
        f"손실률<={TH_LOSS_RATE:.0%}": {"값": G["비율"], "임계": TH_LOSS_RATE,
                                        "통과": G["비율"] <= TH_LOSS_RATE},
        f"이득률(다수결)>={TH_GAIN_RATE:.0%}": {"값": N["비율"], "임계": TH_GAIN_RATE,
                                                 "통과": N["비율"] >= TH_GAIN_RATE},
        # ⛔ 초판은 여유가 **2건뿐인** 이득률에 최악 조작화를 안 달았다(판례 19 우회).
        f"이득률(만장일치)>={TH_GAIN_RATE:.0%}": {
            "값": N["만장일치 참"]["비율"], "임계": TH_GAIN_RATE,
            "통과": N["만장일치 참"]["비율"] >= TH_GAIN_RATE},
        f"이득률 CI하한>={TH_GAIN_RATE:.0%}": {
            "값": N["CI95"][0], "임계": TH_GAIN_RATE,
            "통과": N["CI95"][0] >= TH_GAIN_RATE},
        "최악순효과>0": {"값": gain_w - loss_w, "임계": 0,
                         "통과": (gain_w - loss_w) > 0,
                         "반사실": {"공시문면": gain_w - loss_w,
                                    "초판구현(무장해제)": N["최악(원표 참+판정불가)"]["k"] - loss_w,
                                    "최불리(만장일치이득−원표최악손실)": gain_unan - loss_w,
                                    "점추정": net}},
    }
    # ── 재회부 조건 #3: leave-one-out — 1인 제외로 임계가 뒤집히면 자동 기각 ──
    loo = {}
    for out_j in JUDGES:
        keep = [j for j in JUDGES if j != out_j]
        k = sum(1 for i in new_ids if all(V[j][i]["판정"] == "참" for j in keep))
        loo[f"{out_j} 제외"] = {"이득": k, "이득률": round(k / len(new_ids), 4),
                                 "임계통과": k / len(new_ids) >= TH_GAIN_RATE}
    loo_ok = all(v["임계통과"] for v in loo.values())

    # ── 재회부 조건 #6: 각 임계의 기각 영역 **도달 가능성** 실측 ──
    #    도달 불가 임계는 통과로 세지 않는다. 무장 해제된 임계는 임계가 아니다.
    dissent_gone = sum(1 for i in gone_ids
                       if any(V[j][i]["판정"] != "거짓" for j in JUDGES))
    reach = {
        "최악순효과>0": {
            "필요": f"손실 원표최악 > {gain_w}",
            "관측 소멸 반대의견": f"{dissent_gone}/{len(gone_ids)}"
                                   f" = {dissent_gone/len(gone_ids):.2%}",
            "필요 반대의견율": f"{gain_w/len(gone_ids):.2%}",
            "도달가능": dissent_gone > gain_w},
        "순효과>=+10": {
            "필요": "이득−손실 < 10",
            "도달가능": True,
            "주": "이득이 10 미만이거나 손실이 이득−10 초과면 실패. 원리적 도달 가능"},
    }
    res = {
        "표본": {"소멸": G["n"], "신설": N["n"], "3인전원판정": len(complete)},
        "leave_one_out": loo,
        "임계_도달가능성": reach,
        "손실(소멸 중 참)": G, "이득(신설 중 참)": N,
        "순효과": {"다수결": net, "이득": gain, "손실": loss,
                   "최악": gain_w - loss_w},
        "판례21_독립성": pair,
        "임계판정": gate,
        # 도달 불가 임계는 통과로 세지 않는다(#6) · LOO 로 뒤집히면 자동 기각(#3)
        "최종": ("채택 후보"
                 if all(g["통과"] for g in gate.values())
                 and loo_ok
                 and all(r.get("도달가능", True) for r in reach.values())
                 else "채택 안 함"),
    }
    # ⚠️ detail 을 **먼저** 만든다 — `_maj` 를 지운 뒤 참조하면 KeyError 다(실측).
    #    이 수정은 **출력 순서만** 바꾼다. 임계·집계 규칙은 한 줄도 건드리지 않았다.
    detail = [{"id": i, "label": items[i]["label"], "집합": items[i]["_set"],
               "표": [V[j][i]["판정"] for j in JUDGES],
               "다수결": (G["_maj"] if items[i]["_set"] == "소멸" else N["_maj"]).get(i)}
              for i in complete]
    for d in (res["손실(소멸 중 참)"], res["이득(신설 중 참)"]):
        d.pop("_maj", None)
    json.dump({"요약": res, "항목별": detail},
              open(os.path.join(work, "r7_gate_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
