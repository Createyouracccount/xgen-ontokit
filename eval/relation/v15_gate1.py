"""v15 게이트1 — C4 무결성 + G1a + G1b(3분기) + G1c(생산 유효성).

설계 v15_design.md v2 봉인 조건:
  C4: sanity(v15 환경 base 재학습) vs v14 base — holdout F1 Δ≤0.001.
  G1a: 전체 F1 ≥ base ∧ colleagues F1 ≥0.35 ∧ watchlist 5종 −3pt 이내.
  G1b-채택: no_rel PER-PER R ≥0.856 ∧ alt_names·친족 P/R −1.5pt 이내.
  G1b-가설: no_rel PER-PER R ≥0.78 (T1 0.708 대비 회복). 반증 = 0.708±2pt.
  G1c: conf≥0.5 조건부 colleagues F1 ≥0.35 ∧ TP conf≥0.5 커버리지 ≥70%.
"""
import json
import pathlib
import sys

import torch

sys.path.insert(0, ".")
from eval_re import micro_f1, per_class            # noqa: E402
from labels import LABEL2ID                        # noqa: E402
from v14_gate1 import predict, is_per_per, block_pr, family_pr, WATCH  # noqa: E402

COL = LABEL2ID["per:colleagues"]
NOREL = LABEL2ID["no_relation"]


def main():
    rows = json.load(open("data/holdout.json", encoding="utf-8"))
    gold = [r["label"] for r in rows]
    pp_idx = [i for i, r in enumerate(rows) if is_per_per(r)]
    out = {}
    for name, md in [("base", "model_re_v14_base"), ("sanity", "model_re_v15_sanity"),
                     ("T1", "model_re_v14_T1"), ("cw", "model_re_v15_cw")]:
        preds, confs = predict(md, rows)
        m = micro_f1(gold, preds)
        pc = {r[0]: r for r in per_class(gold, preds, top=99)}
        g_pp = [gold[i] for i in pp_idx]
        p_pp = [preds[i] for i in pp_idx]
        g1b = {"alternate_names": block_pr(g_pp, p_pp, LABEL2ID["per:alternate_names"]),
               "family_block": family_pr(g_pp, p_pp),
               "no_rel_perper": block_pr(g_pp, p_pp, NOREL)}
        # G1c — conf≥0.5 조건부
        tp_confs = [c for g, p, c in zip(gold, preds, confs) if g == COL and p == COL]
        cov = (sum(1 for c in tp_confs if c >= 0.5) / len(tp_confs)) if tp_confs else 0.0
        tp5 = sum(1 for g, p, c in zip(gold, preds, confs)
                  if g == COL and p == COL and c >= 0.5)
        fp5 = sum(1 for g, p, c in zip(gold, preds, confs)
                  if g != COL and p == COL and c >= 0.5)
        sup = sum(1 for g in gold if g == COL)
        fn5 = sup - tp5
        p5 = tp5 / (tp5 + fp5) if tp5 + fp5 else 0.0
        r5 = tp5 / (tp5 + fn5) if tp5 + fn5 else 0.0
        f15 = 2 * p5 * r5 / (p5 + r5) if p5 + r5 else 0.0
        out[name] = {"micro_f1": m, "colleagues": pc.get("per:colleagues"),
                     "watch": {w: pc.get(w) for w in WATCH}, "g1b_perper": g1b,
                     "g1c": {"tp_cov_ge05": round(cov, 4), "cond_f1": round(f15, 4),
                             "cond_P": round(p5, 4), "cond_R": round(r5, 4),
                             "cond_tp": tp5, "cond_fp": fp5, "uncond_sup": sup}}
        print(f"[{name}] F1={m['micro_f1']:.4f} col={pc.get('per:colleagues')}")
        print(f"    G1b={g1b}")
        print(f"    G1c={out[name]['g1c']}")

    base, cw = out["base"], out["cw"]
    v = {}
    v["C4_sanity"] = abs(out["sanity"]["micro_f1"]["micro_f1"]
                         - base["micro_f1"]["micro_f1"]) <= 0.001
    col_f1 = cw["colleagues"][6] if cw["colleagues"] else 0.0
    v["G1a"] = (cw["micro_f1"]["micro_f1"] >= base["micro_f1"]["micro_f1"]
                and col_f1 >= 0.35
                and all((cw["watch"][w][6] - base["watch"][w][6]) >= -0.03
                        for w in WATCH if base["watch"].get(w) and cw["watch"].get(w)))
    nr = cw["g1b_perper"]["no_rel_perper"]["R"]
    ok_blocks = all(
        cw["g1b_perper"][b][k] - base["g1b_perper"][b][k] >= -0.015
        for b in ("alternate_names", "family_block") for k in ("P", "R"))
    v["G1b_adopt"] = nr >= 0.856 and ok_blocks
    v["G1b_hypothesis"] = nr >= 0.78
    v["G1b_refute_zone"] = abs(nr - 0.708) <= 0.02
    v["G1c"] = (cw["g1c"]["cond_f1"] >= 0.35 and cw["g1c"]["tp_cov_ge05"] >= 0.70)
    out["verdicts"] = v
    result = (pathlib.Path(__file__).resolve().parents[3]
              / "eval_runs/relations/v15_gate1_result.json")
    json.dump(out, open(result, "w"), ensure_ascii=False, indent=1)
    print("VERDICTS:", v)


if __name__ == "__main__":
    main()
