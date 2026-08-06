"""경로2 STEP4~5 — 추출된 엔티티쌍(distill_extract_pairs 산출)을 teacher 태깅 + 필터.

입력: distill_extract_pairs.py 산출 jsonl — 각 행에 'marked'(teacher 입력 문자열,
  [S:type]..[/S] [O:type]..[/O] 삽입 완료)가 있음. 그대로 토크나이즈해 teacher 추론.

이중선별 v1(고정 임계, 1차 실증):
  - teacher 고확신: max softmax > CONF_MIN, no_relation 제외
  - (student 불확실 축은 v2에서 — 먼저 teacher 필터 실효부터 실측)

산출: 채택된 쌍 {sentence, subject_entity, object_entity, label(int), teacher_conf, src}.
  + 필터 통계(no_rel 폐기·저확신 폐기·라벨분포) 출력 — NER 노이즈를 실제로 걸러내는지 실측.

재현:
  DISTILL_TEACHER=model_re_scale_large_fp16 TAG_SRC=distill_pairs_wiki100.jsonl \
  TAG_OUT=distill_tagged_wiki100.jsonl CONF_MIN=0.9 DISTILL_DEVICE=cuda \
  /path/venv/bin/python distill_tag_pairs.py
"""
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from labels import LABEL2ID, LABELS
from train_encoder import BATCH, MAX_LEN, SEED

NO_REL = LABEL2ID["no_relation"]


class MarkedDataset(Dataset):
    """추출물의 'marked' 문자열을 그대로 토크나이즈(teacher 입력)."""
    def __init__(self, marked_texts, tok):
        self.enc = tok(marked_texts, truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TEACHER = os.environ["DISTILL_TEACHER"]
    SRC = os.environ["TAG_SRC"]
    OUT = os.getenv("TAG_OUT", "distill_tagged.jsonl")
    CONF_MIN = float(os.getenv("CONF_MIN") or "0.9")
    FP16 = os.getenv("DISTILL_TEACHER_FP16", "1") == "1"

    torch.manual_seed(SEED)
    t0 = time.time()

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    marked = [r["marked"] for r in rows]
    print(f"teacher {TEACHER} | src {SRC} rows {len(rows)} CONF_MIN {CONF_MIN}", flush=True)

    tok = AutoTokenizer.from_pretrained(TEACHER)
    force = os.getenv("DISTILL_DEVICE", "").lower()
    if force in ("cpu", "cuda", "mps"):
        dev = force
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if dev == "cpu" else (torch.float16 if FP16 else torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=30, dtype=dtype)
    print(f"device {dev} dtype {dtype}", flush=True)
    model.to(dev).eval()

    dl = DataLoader(MarkedDataset(marked, tok), batch_size=BATCH, shuffle=False)
    preds, confs = [], []
    with torch.no_grad():
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            probs = torch.softmax(model(**b).logits.float(), dim=-1).cpu()
            c, p = probs.max(dim=-1)
            preds.append(p)
            confs.append(c)
            if (i + 1) % 50 == 0:
                print(f"  batch {i+1}/{len(dl)} ({time.time()-t0:.0f}s)", flush=True)

    preds = torch.cat(preds).tolist()
    confs = torch.cat(confs).tolist()

    def with_idx(sent, ent):
        """재학습 mark()가 start_idx/end_idx 를 요구 — sentence.find 로 채움.
        미출현 시 None(해당 쌍 폐기 — 마킹 불가)."""
        w = ent["word"]
        i = sent.find(w)
        if i < 0:
            return None
        return {"word": w, "type": ent["type"], "start_idx": i, "end_idx": i + len(w) - 1}

    kept, drop_norel, drop_lowconf, drop_noidx = [], 0, 0, 0
    for r, p, c in zip(rows, preds, confs):
        if p == NO_REL:
            drop_norel += 1
            continue
        if c < CONF_MIN:
            drop_lowconf += 1
            continue
        se = with_idx(r["sentence"], r["subject_entity"])
        oe = with_idx(r["sentence"], r["object_entity"])
        if se is None or oe is None:   # 마킹 불가(재학습서 mark 실패) → 폐기
            drop_noidx += 1
            continue
        kept.append({"sentence": r["sentence"], "subject_entity": se,
                     "object_entity": oe, "label": p,
                     "teacher_conf": round(c, 4), "src": r.get("src", "?")})

    with open(OUT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    labdist = Counter(LABELS[r["label"]] for r in kept)
    print(f"채택 {len(kept):,} / 전체 {len(rows):,} "
          f"(no_rel 폐기 {drop_norel:,} = {drop_norel/len(rows)*100:.0f}%, "
          f"저확신 폐기 {drop_lowconf:,} = {drop_lowconf/len(rows)*100:.0f}%, "
          f"위치불가 폐기 {drop_noidx:,})", flush=True)
    print("채택 라벨 상위:", labdist.most_common(15), flush=True)
    print(f"saved → {OUT} ({time.time()-t0:.0f}s)", flush=True)
    print("DISTILL-TAG-DONE", flush=True)


if __name__ == "__main__":
    main()
