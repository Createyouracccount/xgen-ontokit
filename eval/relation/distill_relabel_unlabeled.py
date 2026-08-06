"""레이블 없는 증류 — 경로1: 기존 태깅셋(SREDFM 등)을 teacher 로 재태깅.

사용자 의도("레이블 없는 증류"): 원천의 기존 라벨은 **버리고**, large teacher 가
30-way soft label 을 새로 생성한다. 문장+엔티티쌍만 취한다(NER 불필요 — 이미 태깅됨).

고확신 필터: teacher softmax max > CONF_MIN 이고 argmax != no_relation 인 행만 채택.
  → teacher 오류(≈33%)가 student 로 새는 것을 차단. 회색지대(저확신) 폐기.

산출: {sentence, subject_entity, object_entity, teacher_label(int), teacher_conf} jsonl.
  이후 distill 학습셋에 합쳐 재학습(별도 캐시 단계에서 이 파일도 mark→logits).

재현:
  DISTILL_TEACHER=/path/model_re_large \
  RELABEL_SRC=../../../eval_runs/relations/m2_sredfm_klue_aug_v1.jsonl \
  RELABEL_OUT=distill_relabel_sredfm.jsonl \
  DISTILL_DEVICE=cuda /path/venv/bin/python distill_relabel_unlabeled.py
env: DISTILL_TEACHER(필수), RELABEL_SRC(필수), RELABEL_OUT(기본 distill_relabel.jsonl),
     CONF_MIN(기본 0.9), RELABEL_CAP(기본 0=전량), DISTILL_TEACHER_FP16(기본 1)
"""
import ast
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from labels import LABEL2ID, LABELS
from train_encoder import BATCH, MAX_LEN, SEED, mark

NO_REL = LABEL2ID["no_relation"]


def _parse_entity(e):
    """SREDFM 은 subject_entity 를 문자열 dict 로 저장 — ast 로 복원."""
    return ast.literal_eval(e) if isinstance(e, str) else e


def load_src(path, cap):
    """원천 로드 — 기존 라벨은 버리고 문장+엔티티쌍만. mark() 규약 검증."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s, o = _parse_entity(r["subject_entity"]), _parse_entity(r["object_entity"])
            # mark() 가 요구하는 필수 키 — 하나라도 없으면 스킵(조용한 오염 방지)
            if not all(k in s for k in ("start_idx", "end_idx", "type")):
                continue
            if not all(k in o for k in ("start_idx", "end_idx", "type")):
                continue
            rows.append({"sentence": r["sentence"], "subject_entity": s,
                         "object_entity": o, "src": r.get("source", "?")})
            if cap and len(rows) >= cap:
                break
    return rows


class MarkDataset(Dataset):
    """라벨 없이 mark() 만 — teacher 추론 전용."""
    def __init__(self, rows, tok):
        self.enc = tok([mark(r) for r in rows], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TEACHER = os.environ["DISTILL_TEACHER"]
    SRC = os.environ["RELABEL_SRC"]
    OUT = os.getenv("RELABEL_OUT", "distill_relabel.jsonl")
    CONF_MIN = float(os.getenv("CONF_MIN") or "0.9")
    CAP = int(os.getenv("RELABEL_CAP") or "0")
    FP16 = os.getenv("DISTILL_TEACHER_FP16", "1") == "1"

    torch.manual_seed(SEED)
    t0 = time.time()

    rows = load_src(SRC, CAP)
    print(f"teacher {TEACHER} | src {SRC} rows {len(rows)} CONF_MIN {CONF_MIN}", flush=True)

    tok = AutoTokenizer.from_pretrained(TEACHER)
    force = os.getenv("DISTILL_DEVICE", "").lower()
    if force in ("cpu", "cuda", "mps"):
        dev = force
    else:
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if dev == "cpu" else (torch.float16 if FP16 else torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=30, dtype=dtype)
    print(f"device {dev} dtype {dtype}", flush=True)
    model.to(dev).eval()

    dl = DataLoader(MarkDataset(rows, tok), batch_size=BATCH, shuffle=False)  # 순서 보존
    preds, confs = [], []
    with torch.no_grad():
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            probs = torch.softmax(model(**b).logits.float(), dim=-1).cpu()
            conf, pred = probs.max(dim=-1)
            preds.append(pred)
            confs.append(conf)
            if (i + 1) % 50 == 0:
                print(f"  batch {i+1}/{len(dl)} ({time.time()-t0:.0f}s)", flush=True)

    preds = torch.cat(preds).tolist()
    confs = torch.cat(confs).tolist()

    # 고확신 필터: conf > CONF_MIN 이고 no_relation 아님
    kept, drop_lowconf, drop_norel = [], 0, 0
    for r, p, c in zip(rows, preds, confs):
        if p == NO_REL:
            drop_norel += 1
            continue
        if c < CONF_MIN:
            drop_lowconf += 1
            continue
        kept.append({"sentence": r["sentence"], "subject_entity": r["subject_entity"],
                     "object_entity": r["object_entity"], "label": p,
                     "teacher_conf": round(c, 4), "src": r["src"]})

    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 라벨 분포 — colleagues 등 어려운 라벨이 실제로 잡혔나
    from collections import Counter
    labdist = Counter(LABELS[r["label"]] for r in kept)
    print(f"채택 {len(kept):,} / 전체 {len(rows):,} "
          f"(no_rel 폐기 {drop_norel:,}, 저확신 폐기 {drop_lowconf:,})", flush=True)
    print("상위 라벨:", labdist.most_common(12), flush=True)
    print(f"saved → {OUT} ({time.time()-t0:.0f}s)", flush=True)
    print("DISTILL-RELABEL-DONE", flush=True)


if __name__ == "__main__":
    main()
