"""v14 게이트1 — KLUE holdout: G1a(전체·colleagues·watchlist) + G1b(PER-PER 블록).

설계 v14_design.md D4:
  G1a: 전체 F1 ≥0.6169 ∧ colleagues F1 ≥0.35 ∧ watchlist(member_of·top_members·
       founded_by·employee_of·title) 각 v13c 대비 −3pt 이내.
  G1b: PER-PER 블록(alternate_names / 친족 묶음 / no_relation PER-PER 서브셋)
       각 리콜·정밀도 −3pt 이내 — base+T1 arm 의무 측정(T2 분리 귀속) + final.
  conf 연동 봉인 공시: colleagues 정탐 conf 히스토그램(0.05 bin)·중앙값.

사용: python3 v14_gate1.py   (모델: v13c 기준 + T1 + final)
"""
import collections
import json
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, ".")
from eval_re import micro_f1, per_class            # noqa: E402
from labels import LABELS, LABEL2ID                # noqa: E402
from train_encoder import MAX_LEN, mark            # noqa: E402

WATCH = ["org:member_of", "org:top_members/employees", "org:founded_by",
         "per:employee_of", "per:title"]
FAMILY = ["per:spouse", "per:siblings", "per:children", "per:parents",
          "per:other_family"]
COL = LABEL2ID["per:colleagues"]
NOREL = LABEL2ID["no_relation"]


@torch.no_grad()
def predict(model_dir, rows, batch=64):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()
    preds, confs = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([mark(r) for r in chunk], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        prob = model(**enc).logits.softmax(-1)
        preds += prob.argmax(-1).tolist()
        confs += prob.max(-1).values.tolist()
    return preds, confs


def is_per_per(r):
    return (r["subject_entity"].get("type") == "PER"
            and r["object_entity"].get("type") == "PER")


def block_pr(gold, preds, label_id):
    tp = sum(1 for g, p in zip(gold, preds) if g == label_id and p == label_id)
    fp = sum(1 for g, p in zip(gold, preds) if g != label_id and p == label_id)
    fn = sum(1 for g, p in zip(gold, preds) if g == label_id and p != label_id)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": round(p, 4), "R": round(r, 4)}


def family_pr(gold, preds):
    fam = {LABEL2ID[x] for x in FAMILY}
    tp = sum(1 for g, p in zip(gold, preds) if g in fam and p in fam)
    fp = sum(1 for g, p in zip(gold, preds) if g not in fam and p in fam)
    fn = sum(1 for g, p in zip(gold, preds) if g in fam and p not in fam)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": round(p, 4), "R": round(r, 4)}


def main():
    rows = json.load(open("data/holdout.json", encoding="utf-8"))
    gold = [r["label"] for r in rows]
    pp_idx = [i for i, r in enumerate(rows) if is_per_per(r)]
    out = {}
    # R1b 재게이트(v14_gate1_repair_verdict.md): 델타 기준 = base 재현.
    # v13c 행은 공시 전용(판정 사용 금지).
    for name, md in [("v13c", "model_re_v13c"), ("base", "model_re_v14_base"),
                     ("T1", "model_re_v14_T1"), ("T2", "model_re_v14_T2"),
                     ("final", "model_re_v14_final")]:
        preds, confs = predict(md, rows)
        m = micro_f1(gold, preds)
        pc = {r[0]: r for r in per_class(gold, preds, top=99)}
        col = pc.get("per:colleagues")
        # colleagues 정탐 conf 히스토그램(0.05 bin)·중앙값 — 봉인 공시
        tp_confs = [c for g, p, c in zip(gold, preds, confs)
                    if g == COL and p == COL]
        hist = collections.Counter(int(c / 0.05) * 0.05 for c in tp_confs)
        # G1b — PER-PER 서브셋
        g_pp = [gold[i] for i in pp_idx]
        p_pp = [preds[i] for i in pp_idx]
        g1b = {"alternate_names": block_pr(g_pp, p_pp,
                                           LABEL2ID["per:alternate_names"]),
               "family_block": family_pr(g_pp, p_pp),
               "no_rel_perper": block_pr(g_pp, p_pp, NOREL)}
        out[name] = {
            "micro_f1": m,
            "colleagues": col,
            "colleagues_tp_conf": {
                "median": round(statistics.median(tp_confs), 4) if tp_confs else None,
                "hist_0.05": {f"{k:.2f}": v for k, v in sorted(hist.items())}},
            "watch": {w: pc.get(w) for w in WATCH},
            "g1b_perper": g1b}
        print(f"[{name}] F1={m['micro_f1']:.4f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} | colleagues={col}")
        for w in WATCH:
            print(f"    {w}: {pc.get(w)}")
        print(f"    G1b: {g1b}")

    # 판정 — 봉인 조건(v14_gate1_repair_verdict.md §4): 모든 델타 = base 재현 대비
    base = out["base"]
    base_f1 = base["micro_f1"]["micro_f1"]
    verdicts = {"integrity_base": abs(base_f1 - 0.6169) <= 0.005}
    for arm in ("T1", "T2", "final"):
        fin = out[arm]
        col_f1 = fin["colleagues"][6] if fin["colleagues"] else 0.0
        ok_watch = all(
            (fin["watch"][w][6] - base["watch"][w][6]) >= -0.03
            for w in WATCH if base["watch"].get(w) and fin["watch"].get(w))
        ok_g1b = True
        for blk in ("alternate_names", "family_block", "no_rel_perper"):
            b, f = base["g1b_perper"][blk], out[arm]["g1b_perper"][blk]
            if f["P"] - b["P"] < -0.03 or f["R"] - b["R"] < -0.03:
                ok_g1b = False
                print(f"  G1b 위반[{arm}/{blk}]: P {b['P']}→{f['P']} "
                      f"R {b['R']}→{f['R']}")
        verdicts[f"G1a_{arm}"] = (fin["micro_f1"]["micro_f1"] >= base_f1
                                  and col_f1 >= 0.35 and ok_watch)
        verdicts[f"G1b_{arm}"] = ok_g1b
    out["verdicts"] = verdicts
    result_path = (pathlib.Path(__file__).resolve().parents[3]
                   / "eval_runs/relations/v14_gate1_result_regate.json")
    json.dump(out, open(result_path, "w"), ensure_ascii=False, indent=1)
    print("VERDICTS:", verdicts)
    passing = [a for a in ("T1", "T2", "final")
               if verdicts[f"G1a_{a}"] and verdicts[f"G1b_{a}"]]
    print("통과 arm:", passing if passing else "없음 → 라운드 자동 기각(봉인 조건 5)")


if __name__ == "__main__":
    main()
