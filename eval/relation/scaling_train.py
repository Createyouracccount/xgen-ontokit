"""베이스 모델 스케일링 라운드 — v13c 레시피 고정, 베이스/LR만 env 로 교체.

v13_train.py 와 동일 레시피(aug cap 8000, 하드셋 ×1, SEED 20260714, 마커,
MAX_LEN 180, BATCH 32, MAX_EPOCHS 4, hard_dev 조기종료). 움직이는 변수는
SCALE_MODEL(베이스)과 SCALE_LR(large LR 스윕 전용)뿐이다.

재현: SCALE_MODEL=klue/roberta-base SCALE_OUT=model_re_scale_base \
      /opt/miniconda3/bin/python3 scaling_train.py
  env: SCALE_MODEL(필수), SCALE_LR(기본 train_encoder.LR=3e-5), SCALE_OUT(필수)
"""
import json
import os
import pathlib
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from labels import LABEL2ID
from train_encoder import BATCH, LR, SEED, SPECIALS, REDataset

MODEL = os.environ["SCALE_MODEL"]
SCALE_LR = float(os.getenv("SCALE_LR", str(LR)))
SCALE_WARMUP = float(os.getenv("SCALE_WARMUP", "0"))   # warmup 비율(예: 0.1)
OUT_DIR = os.environ["SCALE_OUT"]

_EVAL_RUNS = pathlib.Path(__file__).resolve().parents[3] / "eval_runs"
AUG_PATH = os.getenv("M2_AUG_PATH", str(_EVAL_RUNS / "relations/m2_aug_v12_fixed.jsonl"))
AUG_CAP = int(os.getenv("M2_AUG_CAP", "8000"))
HARD_PATH = os.getenv("V13_HARD_PATH", str(_EVAL_RUNS / "relations/v13_hardset.jsonl"))
UPSAMPLE = 1        # v13c 확정값
NO_REL_MAX = 0.35
MAX_EPOCHS = 4


def load_aug():
    import collections
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

    t0 = time.time()
    t = pq.read_table("data/klue_re_train.parquet")
    train_all = [{c: t.column(c)[i].as_py() for c in t.column_names} for i in range(t.num_rows)]
    tune_guids = {r["guid"] for r in json.load(open("data/tune.json", encoding="utf-8"))}
    base = [r for r in train_all if r["guid"] not in tune_guids] + load_aug()

    hard = [json.loads(l) for l in open(HARD_PATH, encoding="utf-8")]
    random.shuffle(hard)
    n_dev = max(1, int(len(hard) * 0.15))
    hard_dev, hard_tr = hard[:n_dev], hard[n_dev:]

    def enc(r):
        return {"sentence": r["sentence"], "subject_entity": r["subject_entity"],
                "object_entity": r["object_entity"], "label": LABEL2ID[r["label"]]}

    rows = base + [enc(r) for r in hard_tr] * UPSAMPLE
    random.shuffle(rows)
    print(f"model {MODEL} lr {SCALE_LR} | base {len(base)} + hard {len(hard_tr)}x{UPSAMPLE} = {len(rows)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=30)
    model.resize_token_embeddings(len(tok))
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)

    dl = DataLoader(REDataset(rows, tok), batch_size=BATCH, shuffle=True)
    dev_dl = DataLoader(REDataset([enc(r) for r in hard_dev], tok), batch_size=BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=SCALE_LR)
    steps = len(dl) * MAX_EPOCHS
    if SCALE_WARMUP > 0:
        w = int(steps * SCALE_WARMUP)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / max(1, w) if s < w else max(0.0, (steps - s) / max(1, steps - w)))
        print(f"warmup {SCALE_WARMUP} ({w} steps)")
    else:
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
                print(f"ep{ep} step {step}/{steps} loss {out.loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
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

    print(f"saved → {OUT_DIR}/ (best ep{best_ep}, dev {best:.4f}, "
          f"elapsed {time.time()-t0:.0f}s)")
    print("SCALE-TRAIN-DONE")


if __name__ == "__main__":
    main()
