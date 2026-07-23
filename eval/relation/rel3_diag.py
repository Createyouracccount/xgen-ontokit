#!/usr/bin/env python3
"""rel3 진단 — colleagues 추론단 보정의 헤드룸 실측 (tune 전용, 홀드아웃 미접촉).

질문: v13c가 colleagues를 놓칠 때(gold=colleagues ∧ pred≠colleagues),
P(colleagues)가 얼마나 아깝게 지는가? 그리고 "argmax가 no_rel인데
P(colleagues)≥τ면 colleagues 방출" 룰의 τ별 회수/오발화 트레이드오프는?

출력: τ 스윕 표(회수 TP↑ vs 오발화 FP↑, PER-PER no_rel 전환 수) — 설계 입력용.
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from labels import LABEL2ID                     # noqa: E402
from train_encoder import MAX_LEN, mark         # noqa: E402
from m2_eval_aug import load                    # noqa: E402

COL = LABEL2ID["per:colleagues"]
NOREL = LABEL2ID["no_relation"]


@torch.no_grad()
def probs(model_dir, rows, batch=64):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([mark(r) for r in chunk], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        out.append(model(**enc).logits.softmax(-1).cpu())
    return torch.cat(out)


def main():
    rows = load("tune")
    gold = torch.tensor([r["label"] for r in rows])
    per = torch.tensor([1 if (r["subject_entity"].get("type") == "PER"
                              and r["object_entity"].get("type") == "PER") else 0
                        for r in rows], dtype=torch.bool)
    p = probs("model_re_v13c", rows)
    pred = p.argmax(-1)
    pcol = p[:, COL]

    n_col = int((gold == COL).sum())
    miss = (gold == COL) & (pred != COL)
    print(f"tune colleagues sup={n_col}, v13c TP={int(((gold==COL)&(pred==COL)).sum())}, "
          f"miss={int(miss.sum())} (miss 중 pred=no_rel {int((miss&(pred==NOREL)).sum())})")
    # miss 케이스의 P(colleagues) 분포
    q = torch.quantile(pcol[miss], torch.tensor([0.5, 0.75, 0.9]))
    print(f"miss P(col) 중앙값 {q[0]:.3f} / p75 {q[1]:.3f} / p90 {q[2]:.3f}")

    # 룰: pred==no_rel ∧ P(col)≥τ → colleagues 로 전환
    print(f"\n{'τ':>5} {'회수TP':>6} {'오발화FP':>8} {'전환P':>7} {'신P':>6} {'신R':>6} {'신F1':>6} {'PP노렐전환':>9}")
    base_tp = int(((gold == COL) & (pred == COL)).sum())
    base_fp = int(((gold != COL) & (pred == COL)).sum())
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        fire = (pred == NOREL) & (pcol >= tau)
        rec_tp = int((fire & (gold == COL)).sum())
        rec_fp = int((fire & (gold != COL)).sum())
        # 오발화 중 실제 no_rel(PER-PER) — G1b 손상 상당분
        pp_norel = int((fire & (gold == NOREL) & per).sum())
        tp, fp = base_tp + rec_tp, base_fp + rec_fp
        fn = n_col - tp
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn)
        F1 = 2 * P * R / (P + R) if P + R else 0
        conv_p = rec_tp / (rec_tp + rec_fp) if rec_tp + rec_fp else 0
        print(f"{tau:5.2f} {rec_tp:6d} {rec_fp:8d} {conv_p:7.3f} {P:6.3f} {R:6.3f} {F1:6.3f} {pp_norel:9d}")


if __name__ == "__main__":
    main()
