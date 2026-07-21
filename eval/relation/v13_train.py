"""v13 — 하드 네거티브/포지티브 파인튜닝 (설계 eval_runs/relations/v13_design_v2.md, 패널 재심 통과).

v12 레시피(KLUE−tune + SREDFM aug v12_fixed) + 하드셋(채점 확정 ~750행) ×UPSAMPLE.
- no_relation 전체 비율 35% 상한 — 초과 시 배율 자동 하향(설계 D3).
- 하드셋 dev 15% 학습 제외(과적합 감시), 에폭별 dev-loss 상승 시 조기종료(주심 재심 b).
재현: /opt/miniconda3/bin/python3 v13_train.py  (MPS)
  env: V13_HARD_PATH(기본 eval_runs/relations/v13_hardset.jsonl), V13_UPSAMPLE(16), V13_OUT(model_re_v13)
"""
import collections
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from labels import LABEL2ID
from train_encoder import BATCH, LR, MODEL, SEED, SPECIALS, REDataset

AUG_PATH = os.getenv("M2_AUG_PATH",
                     "/Users/kimdu/company/xgen-levelup/eval_runs/relations/m2_aug_v12_fixed.jsonl")
AUG_CAP = int(os.getenv("M2_AUG_CAP", "8000"))
HARD_PATH = os.getenv("V13_HARD_PATH",
                      "/Users/kimdu/company/xgen-levelup/eval_runs/relations/v13_hardset.jsonl")
UPSAMPLE = int(os.getenv("V13_UPSAMPLE", "16"))
NO_REL_MAX = 0.35   # 설계 D3 상한
MAX_EPOCHS = int(os.getenv("V13_EPOCHS", "4"))
OUT_DIR = os.getenv("V13_OUT", "model_re_v13")


def load_aug():
    rows, cnt = [], collections.Counter()
    with open(AUG_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if cnt[r["label"]] >= AUG_CAP:
                continue
            cnt[r["label"]] += 1
            rows.append({"sentence": r["sentence"], "subject_entity": r["subject_entity"],
                         "object_entity": r["object_entity"], "label": LABEL2ID[r["label"]]})
    print("aug:", sum(cnt.values()))
    return rows


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import pyarrow.parquet as pq

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t = pq.read_table("data/klue_re_train.parquet")
    train_all = [{c: t.column(c)[i].as_py() for c in t.column_names} for i in range(t.num_rows)]
    tune_guids = {r["guid"] for r in json.load(open("data/tune.json", encoding="utf-8"))}
    base = [r for r in train_all if r["guid"] not in tune_guids] + load_aug()

    hard = [json.loads(l) for l in open(HARD_PATH, encoding="utf-8")]
    random.shuffle(hard)
    n_dev = max(1, int(len(hard) * 0.15))
    hard_dev, hard_tr = hard[:n_dev], hard[n_dev:]
    json.dump(hard_dev, open(os.path.join(os.path.dirname(HARD_PATH), "v13_hard_dev.json"), "w"),
              ensure_ascii=False)

    def enc(r):
        return {"sentence": r["sentence"], "subject_entity": r["subject_entity"],
                "object_entity": r["object_entity"], "label": LABEL2ID[r["label"]]}

    # no_relation 35% 상한 — 업샘플 배율 자동 하향(설계 D3)
    ups = UPSAMPLE
    while True:
        total = len(base) + len(hard_tr) * ups
        no_rel = (sum(1 for r in base if r["label"] == LABEL2ID["no_relation"])
                  + sum(1 for r in hard_tr if r["label"] == "no_relation") * ups)
        if no_rel / total <= NO_REL_MAX or ups <= 1:
            break
        ups -= 1
    print(f"upsample x{ups} (no_rel 비율 {no_rel/total:.3f})")
    rows = base + [enc(r) for r in hard_tr] * ups
    random.shuffle(rows)
    print(f"base {len(base)} + hard {len(hard_tr)}x{ups} = {len(rows)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=30)
    model.resize_token_embeddings(len(tok))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)

    dl = DataLoader(REDataset(rows, tok), batch_size=BATCH, shuffle=True)
    dev_dl = DataLoader(REDataset([enc(r) for r in hard_dev], tok), batch_size=BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = len(dl) * MAX_EPOCHS
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, steps)

    def dev_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in dev_dl:
                b = {k: v.to(dev) for k, v in b.items()}
                tot += model(**b).loss.item() * b["labels"].size(0)
                n += b["labels"].size(0)
        model.train()
        return tot / max(1, n)

    best, best_ep = float("inf"), -1
    model.train()
    step = 0
    for ep in range(MAX_EPOCHS):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"ep{ep} step {step}/{steps} loss {out.loss.item():.4f}", flush=True)
        dl_ = dev_loss()
        print(f"[dev] ep{ep} hard_dev_loss {dl_:.4f}", flush=True)
        if dl_ < best:
            best, best_ep = dl_, ep
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)
            print(f"[dev] saved (best ep{ep})", flush=True)
        elif ep - best_ep >= 1:
            print(f"[dev] early stop at ep{ep} (best ep{best_ep})", flush=True)
            break

    print(f"saved → {OUT_DIR}/ (best ep{best_ep}, dev {best:.4f})")
    print("V13-TRAIN-DONE")


if __name__ == "__main__":
    main()
