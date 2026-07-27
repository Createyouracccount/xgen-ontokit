"""스케일링 라운드 평가 — 지정 모델 디렉토리의 KLUE F1/P/R + per:colleagues.

사용: python3 scaling_eval.py <model_dir> <holdout|tune>
eval_encoder.py 의 E1(raw 인코더) 경로와 동일 지표. 게이트·앙상블 미적용
(v13c 발효본도 raw + 파이프라인 게이트이므로 raw 비교가 공정).
"""
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from eval_re import report, per_class
from train_encoder import MAX_LEN, mark


@torch.no_grad()
def predict(model_dir, rows, batch=64):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev).eval()
    preds = []
    t0 = time.time()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([mark(r) for r in chunk], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    elapsed = time.time() - t0
    print(f"inference: {len(rows)} rows in {elapsed:.1f}s = {len(rows)/elapsed:.1f} rows/s")
    return preds


def main():
    model_dir, which = sys.argv[1], sys.argv[2]
    rows = json.load(open(f"data/{which}.json", encoding="utf-8"))
    golds = [r["label"] for r in rows]
    preds = predict(model_dir, rows)
    report(f"{model_dir} @{which}", golds, preds)
    print("\nper-class:")
    for row in per_class(golds, preds, top=30):
        print("  ", row)
    print("SCALE-EVAL-DONE")


if __name__ == "__main__":
    main()
