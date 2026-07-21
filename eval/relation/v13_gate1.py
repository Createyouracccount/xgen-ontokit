"""v13 게이트1 — KLUE holdout: v12 vs v13 F1·watchlist 클래스·conf 분포 (설계 v13_design_v2.md 게이트1+조건3)."""
import json
import statistics
import sys

import torch

sys.path.insert(0, ".")
from eval_re import micro_f1, per_class
from labels import LABELS, LABEL2ID
from train_encoder import MAX_LEN, mark

WATCH = ["org:member_of", "per:colleagues"]


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


def main():
    rows = json.load(open("data/holdout.json", encoding="utf-8"))
    gold = [r["label"] for r in rows]
    out = {}
    import os
    cand = os.getenv("V13_GATE_MODEL", "model_re_v13")
    for name, md in [("v12", "model_re_aug_v12"), ("v13", cand)]:
        preds, confs = predict(md, rows)
        m = micro_f1(gold, preds)
        true_confs = [c for g, c in zip(gold, confs) if g != 0]
        med = statistics.median(true_confs)
        pc = {LABELS[0]: None}
        pc = {r[0]: r for r in per_class(gold, preds, top=30)}
        out[name] = {"micro_f1": m, "true_conf_median": med,
                     "watch": {w: pc.get(w) for w in WATCH}}
        print(f"[{name}] F1={m['micro_f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"| 진관계 conf 중앙값={med:.4f}")
        for w in WATCH:
            print(f"    {w}: {pc.get(w)}")
    json.dump(out, open(
        "/Users/kimdu/company/xgen-levelup/eval_runs/relations/v13_gate1_result.json", "w"),
        ensure_ascii=False, indent=1)
    d_f1 = out["v13"]["micro_f1"]["micro_f1"] - out["v12"]["micro_f1"]["micro_f1"]
    d_conf = out["v13"]["true_conf_median"] - out["v12"]["true_conf_median"]
    print(f"ΔF1={d_f1:+.4f}  Δconf중앙값={d_conf:+.4f}")
    print("GATE1:", "PASS" if out["v13"]["micro_f1"]["micro_f1"] >= 0.62 else "FAIL",
          "| conf 재론 트리거:", "YES" if d_conf < -0.10 else "no")


if __name__ == "__main__":
    main()
