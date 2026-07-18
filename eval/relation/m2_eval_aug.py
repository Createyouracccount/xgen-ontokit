"""M2 — 증강 모델 vs 기존 모델 비교 평가.

사용: python3 m2_eval_aug.py tune            (반복 비교용)
      python3 m2_eval_aug.py holdout         (심판 단계 전용 — JUDGE_PROTOCOL 준수)
두 모델(model_re, model_re_aug)을 같은 split 에서 micro-F1(no_relation 제외 표준)로 비교.
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from eval_re import report
from train_encoder import MAX_LEN, mark


def load(name):
    with open(f"data/{name}.json", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def predict(model_dir, rows, batch=64):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()
    preds = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([mark(r) for r in chunk], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        logits = model(**enc).logits
        preds += logits.argmax(-1).tolist()
    return preds


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "tune"
    rows = load(split)
    gold = [r["label"] for r in rows]
    for md in ["model_re", "model_re_aug"]:
        preds = predict(md, rows)
        report(f"{md} @ {split}", gold, preds)


if __name__ == "__main__":
    main()
