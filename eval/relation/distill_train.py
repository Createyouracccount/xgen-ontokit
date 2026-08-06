"""증류 2단계 — student(base) KD 학습. teacher 메모리 0(캐시 logits 사용).

L = α·KL(student/T ‖ teacher/T)·T²  +  (1-α)·CE(student, hard_label)

1단계(distill_cache_logits.py)가 만든 logits 캐시를 행별로 정합해 학습.
학습셋 구성은 build_distill_rows() 단일 소스 → 캐시와 순서 100% 일치.
scaling_train.py 와 동일 레시피(마커·MAX_LEN 180·BATCH 32·hard_dev 조기종료),
움직이는 변수는 student 베이스 + KD 하이퍼(T·α)뿐.

재현: DISTILL_STUDENT=klue/roberta-base DISTILL_CACHE=distill_logits.pt \
      DISTILL_T=4 DISTILL_ALPHA=0.7 DISTILL_OUT=model_re_distill_base_T4a07 \
      /path/venv/bin/python distill_train.py
  env: DISTILL_STUDENT(필수), DISTILL_CACHE(필수), DISTILL_OUT(필수),
       DISTILL_T(기본 4), DISTILL_ALPHA(기본 0.7), DISTILL_LR(기본 3e-5)
"""
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_encoder import BATCH, LR, MAX_LEN, SEED, SPECIALS, mark
from distill_cache_logits import build_distill_rows

MAX_EPOCHS = int(os.getenv("DISTILL_EPOCHS") or "4")


class KDDataset(Dataset):
    """행별 (인코딩, hard label, teacher logits) — 캐시 logits 를 인덱스로 정합."""
    def __init__(self, rows, teacher_logits, tok):
        self.enc = tok([mark(r) for r in rows], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        self.y = torch.tensor([r["label"] for r in rows])
        self.tl = teacher_logits
        assert len(self.y) == self.tl.size(0), "row/logits 정합 실패"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return ({k: v[i] for k, v in self.enc.items()}
                | {"labels": self.y[i], "teacher_logits": self.tl[i]})


def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    """KD: soft KL(T² 스케일) + hard CE."""
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard, soft.item(), hard.item()


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    STUDENT = os.environ["DISTILL_STUDENT"]
    CACHE = os.environ["DISTILL_CACHE"]
    OUT_DIR = os.environ["DISTILL_OUT"]
    T = float(os.getenv("DISTILL_T", "4"))
    ALPHA = float(os.getenv("DISTILL_ALPHA", "0.7"))
    D_LR = float(os.getenv("DISTILL_LR", str(LR)))

    torch.manual_seed(SEED)
    t0 = time.time()

    rows, dev_rows = build_distill_rows()
    cache = torch.load(CACHE)
    tl = cache["logits"]
    assert tl.size(0) == len(rows), f"캐시 {tl.size(0)} != rows {len(rows)} — 규약 불일치"
    print(f"student {STUDENT} T {T} α {ALPHA} lr {D_LR} | rows {len(rows)} "
          f"(teacher={cache['teacher']})", flush=True)

    tok = AutoTokenizer.from_pretrained(STUDENT)
    tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=30)
    model.resize_token_embeddings(len(tok))
    force = os.getenv("DISTILL_DEVICE", "").lower()
    if force in ("cpu", "cuda", "mps"):
        dev = force
    else:
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {dev}", flush=True)
    model.to(dev)

    dl = DataLoader(KDDataset(rows, tl, tok), batch_size=BATCH, shuffle=True)
    # dev 는 hard label CE 만(teacher logits 없이) — 조기종료 기준은 v13c 와 동일 축.
    dev_ds = KDDataset(dev_rows, torch.zeros(len(dev_rows), 30), tok)
    dev_dl = DataLoader(dev_ds, batch_size=BATCH)

    opt = torch.optim.AdamW(model.parameters(), lr=D_LR)
    steps = len(dl) * MAX_EPOCHS
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, steps)

    def dev_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in dev_dl:
                labels = b["labels"].to(dev)
                inp = {k: v.to(dev) for k, v in b.items()
                       if k not in ("labels", "teacher_logits")}
                loss = F.cross_entropy(model(**inp).logits, labels)
                tot += loss.item() * labels.size(0)
                n += labels.size(0)
        model.train()
        return tot / max(1, n)

    best, best_ep = float("inf"), -1
    model.train()
    step = 0
    for ep in range(MAX_EPOCHS):
        for b in dl:
            labels = b["labels"].to(dev)
            teacher_logits = b["teacher_logits"].to(dev)
            inp = {k: v.to(dev) for k, v in b.items()
                   if k not in ("labels", "teacher_logits")}
            logits = model(**inp).logits
            loss, soft, hard = kd_loss(logits, teacher_logits, labels, T, ALPHA)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} "
                      f"(soft {soft:.4f} hard {hard:.4f}) ({time.time()-t0:.0f}s)", flush=True)
        dl_ = dev_loss()
        print(f"[dev] ep{ep} hard_dev_loss {dl_:.4f}", flush=True)
        if dl_ < best:
            best, best_ep = dl_, ep
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)
            print(f"[dev] saved (best ep{ep})", flush=True)
        elif ep - best_ep >= int(os.getenv("DISTILL_PATIENCE") or "1"):
            print(f"[dev] early stop at ep{ep} (best ep{best_ep})", flush=True)
            break

    print(f"saved → {OUT_DIR}/ (best ep{best_ep}, dev {best:.4f}, "
          f"elapsed {time.time()-t0:.0f}s)", flush=True)
    print("DISTILL-TRAIN-DONE", flush=True)


if __name__ == "__main__":
    main()
