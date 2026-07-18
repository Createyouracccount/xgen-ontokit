"""M2 — SREDFM-ko 증강 파인튜닝 (train_encoder.py 조건 동일 + 증강 블렌드).

증강원: eval_runs/relations/m2_sredfm_klue_aug.jsonl (오라벨 필터 4종 검증판 —
블라인드 2인 합의 오라벨 2.1%, Wilson 상단 5.9%; m2_sredfm_filter_verdict.json).
무모호 P-id 22종 → KLUE 18라벨 매핑분 179,260건 중 라벨당 AUG_CAP 상한 표집
(편중 억제: 출생일 61k 가 KLUE 분포를 익사시키지 않게). 매핑 22종 중 P22/P25(parents)·
P1056(product) 등 2라벨은 실산출 0건(사문 — 0718 심판 정정 반영).

심판 판정(0718, 95/100 채택): holdout 0.5924→0.6259(+0.0335, paired bootstrap
2000/2000 유의), SREDFM↔holdout 교차오염 0.79% 무해(제거해도 개선폭 불변).
감시 목록: org:founded_by −0.148(sup 11)·org:top_members −0.009 — 차기 라운드 추적.

산출: model_re_aug/ (기존 model_re/ 는 불변 — 비교 기준 보존).
평가: eval_encoder.py 의 mark/holdout 재사용은 m2_eval_aug.py 참조.
재현: /opt/miniconda3/bin/python3 m2_train_aug.py  (MPS)
"""
import collections
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from labels import LABEL2ID
from train_encoder import (BATCH, EPOCHS, LR, MODEL, SEED, SPECIALS, REDataset)

import os
# 0718b 심판 반려 조치: 캡 재표집 교란(파일 행수→셔플 순열→학습셋 26% 교체) 차단 —
# 사전 고정셋(M2_AUG_PATH, 캡 기적용)을 주면 캡 로직이 그대로 통과(선착순 캡 무해)
AUG_PATH = os.getenv("M2_AUG_PATH",
                     "/Users/kimdu/company/xgen-levelup/eval_runs/relations/m2_sredfm_klue_aug.jsonl")
AUG_CAP = int(os.getenv("M2_AUG_CAP", "8000"))  # 라벨당 증강 상한
OUT_DIR = os.getenv("M2_OUT", "model_re_aug_v11")


def load_aug():
    rows, cnt = [], collections.Counter()
    with open(AUG_PATH, encoding="utf-8") as f:
        for line in f:  # 파일은 이미 셔플됨(seed 20260718) — 선착순 cap = 무작위 표집
            r = json.loads(line)
            lab = LABEL2ID[r["label"]]
            if cnt[r["label"]] >= AUG_CAP:
                continue
            cnt[r["label"]] += 1
            rows.append({"sentence": r["sentence"], "subject_entity": r["subject_entity"],
                         "object_entity": r["object_entity"], "label": lab})
    print("증강 사용:", sum(cnt.values()), dict(cnt.most_common(8)))
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
    train = [r for r in train_all if r["guid"] not in tune_guids]
    aug = load_aug()
    rows = train + aug
    random.shuffle(rows)
    print(f"train {len(train)} + aug {len(aug)} = {len(rows)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=30)
    model.resize_token_embeddings(len(tok))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)

    dl = DataLoader(REDataset(rows, tok), batch_size=BATCH, shuffle=True)
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
            if step % 200 == 0:
                print(f"ep{ep} step {step}/{steps} loss {out.loss.item():.4f}", flush=True)

    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"saved → {OUT_DIR}/")
    print("M2-TRAIN-DONE")


if __name__ == "__main__":
    main()
