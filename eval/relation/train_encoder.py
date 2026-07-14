"""KLUE-RE 로컬 인코더 파인튜닝 — klue/roberta-small + typed entity marker.

불변식: LLM 호출 0회. 로컬 인코더는 허용(NER modu-ner 전례).
라이선스: KLUE 모델·데이터 CC BY-SA 4.0 (github.com/KLUE-benchmark/KLUE) — 상용 OK.

누수 차단:
  - 학습 = train − tune (26,470) : tune 6,000 이 train 표본이므로 제외해 공정 평가
  - tune  = 반복 비교용, holdout = 심판 전용(미접촉)

입력 형식: 문장에 typed marker 삽입(KLUE-RE 표준 관행)
  "[S:ORG] 금호고속 [/S] [O:PER] 이덕연 [/O] 사장 …" → [CLS] 분류(30 클래스)

재현: /opt/miniconda3/bin/python3 train_encoder.py  (MPS, ~20분)
가중치는 model_re/ 에 저장(≈270MB — git 미커밋, 스크립트로 재현).
"""
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MODEL = "klue/roberta-small"
MAX_LEN = 180
BATCH = 32
EPOCHS = 3
LR = 3e-5
SEED = 20260714

SUBJ_TYPES = ["PER", "ORG"]
OBJ_TYPES = ["PER", "ORG", "LOC", "DAT", "POH", "NOH"]
SPECIALS = ([f"[S:{t}]" for t in SUBJ_TYPES] + ["[/S]"] +
            [f"[O:{t}]" for t in OBJ_TYPES] + ["[/O]"])


def mark(row) -> str:
    sent = row["sentence"]
    s, o = row["subject_entity"], row["object_entity"]
    spans = sorted([
        (s["start_idx"], s["end_idx"], f"[S:{s['type']}]", "[/S]"),
        (o["start_idx"], o["end_idx"], f"[O:{o['type']}]", "[/O]"),
    ], key=lambda x: -x[0])
    for st, en, opn, cls in spans:
        sent = sent[:st] + f" {opn} " + sent[st:en + 1] + f" {cls} " + sent[en + 1:]
    return sent


class REDataset(Dataset):
    def __init__(self, rows, tok):
        self.enc = tok([mark(r) for r in rows], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        self.y = torch.tensor([r["label"] for r in rows])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.y[i]}


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    import pyarrow.parquet as pq
    t = pq.read_table("data/klue_re_train.parquet")
    train_all = [{c: t.column(c)[i].as_py() for c in t.column_names} for i in range(t.num_rows)]
    tune_guids = {r["guid"] for r in json.load(open("data/tune.json", encoding="utf-8"))}
    train = [r for r in train_all if r["guid"] not in tune_guids]
    print(f"train {len(train)} (tune {len(tune_guids)} 제외)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=30)
    model.resize_token_embeddings(len(tok))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)

    dl = DataLoader(REDataset(train, tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = len(dl) * EPOCHS
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, steps)

    model.train()
    step = 0
    for ep in range(EPOCHS):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"ep{ep} step {step}/{steps} loss {out.loss.item():.4f}", flush=True)

    model.save_pretrained("model_re")
    tok.save_pretrained("model_re")
    print("saved → model_re/")


if __name__ == "__main__":
    main()
