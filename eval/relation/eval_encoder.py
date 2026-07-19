"""로컬 RE 인코더 평가 + 규칙 앙상블 ablation.

  E1 encoder            : 인코더 단독
  E1g encoder+type-gate  : 인코더 출력에 라벨 타입제약(결정적 후처리)
  E2 rules→encoder      : 규칙 발화 우선(고정밀), 미발화 시 인코더
  E3 encoder→rules      : 인코더 우선, no_relation 일 때만 규칙
사용: python3 eval_encoder.py tune | python3 eval_encoder.py holdout(심판 전용)

⚠️주의(B3 심판 D1): 이 스크립트의 E2/E3 는 **게이트 적용 인코더** 기반. 0719 처분
재판의 보고 수치(tune 0.8325/0.8301/0.8096)는 **raw 인코더** 기반 앙상블 — 게이트는
실측 유해(holdout 0.6274→0.5996)라 프로덕션 미사용이며, 재현하려면 gate 를 끄고
앙상블할 것. 처분 결론(0719 B3 94/100): 앙상블 양방향 순손실 → relation_ko 는
가용성 폴백 전용으로 확정, E2/E3 승격 영구 기각(a1_judge_verdict.md B3).
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from eval_re import report, per_class
from extractor_rules import predict as rule_predict
from labels import LABELS, type_ok
from train_encoder import MAX_LEN, mark


def load(name):
    with open(f"data/{name}.json", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def encoder_predict(rows, batch=64):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("model_re")
    model = AutoModelForSequenceClassification.from_pretrained("model_re")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev).eval()
    preds, logits_all = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([mark(r) for r in chunk], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        logits = model(**enc).logits.cpu()
        preds.extend(logits.argmax(-1).tolist())
        logits_all.append(logits)
    return preds, torch.cat(logits_all)


def gate(preds, logits, rows):
    """타입제약 위반 예측을 제약 만족 차순위로 교정(결정적)."""
    out = []
    for p, lg, r in zip(preds, logits, rows):
        st, ot = r["subject_entity"]["type"], r["object_entity"]["type"]
        if type_ok(LABELS[p], st, ot):
            out.append(p)
            continue
        for cand in lg.argsort(descending=True).tolist():
            if type_ok(LABELS[cand], st, ot):
                out.append(cand)
                break
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "tune"
    rows = load(which)
    golds = [r["label"] for r in rows]

    enc, logits = encoder_predict(rows)
    encg = gate(enc, logits, rows)
    rules = [rule_predict(r) for r in rows]
    e2 = [ru if ru != 0 else e for ru, e in zip(rules, encg)]
    e3 = [e if e != 0 else ru for ru, e in zip(rules, encg)]

    report(f"E1 encoder        @{which}", golds, enc)
    report(f"E1g encoder+gate  @{which}", golds, encg)
    report(f"rules(참고)       @{which}", golds, rules)
    report(f"E2 rules→encoder  @{which}", golds, e2)
    report(f"E3 encoder→rules  @{which}", golds, e3)
    print("\nper-class (E1g):")
    for row in per_class(golds, encg):
        print("  ", row)


if __name__ == "__main__":
    main()
