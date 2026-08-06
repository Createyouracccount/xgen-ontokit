"""증류 1단계 — teacher(large) logits 오프라인 캐시.

메모리 제약(DGX 가용 ~11GB, 스왑 12GB 사용 중) 우회: teacher 와 student 를 동시에
올리지 않는다. 여기서 teacher 를 일시 로딩해 학습셋 전체의 logits 를 추출·저장하고
언로드한다. 2단계(distill_train.py)는 이 캐시만 읽어 teacher 메모리 0 으로 학습.

⚠️ 규약 동결 필수: 학습셋 구성·마킹·MAX_LEN·라벨 순서가 distill_train.py 와 100%
동일해야 logits 가 행별로 정합한다. → scaling_train.py 와 동일 로딩 경로를 공유하도록
build_distill_rows() 한 곳에서만 행을 만든다(양쪽이 import).

재현: DISTILL_TEACHER=/path/to/model_re_large \
      /path/venv/bin/python distill_cache_logits.py
  env: DISTILL_TEACHER(필수, teacher 경로), DISTILL_CACHE(기본 distill_logits.pt),
       DISTILL_TEACHER_FP16(기본 1 — 메모리 절감)
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
from train_encoder import BATCH, SEED, SPECIALS, REDataset

_EVAL_RUNS = pathlib.Path(__file__).resolve().parents[3] / "eval_runs"
AUG_PATH = os.getenv("M2_AUG_PATH", str(_EVAL_RUNS / "relations/m2_aug_v12_fixed.jsonl"))
AUG_CAP = int(os.getenv("M2_AUG_CAP", "8000"))
HARD_PATH = os.getenv("V13_HARD_PATH", str(_EVAL_RUNS / "relations/v13_hardset.jsonl"))
UPSAMPLE = 1


def build_distill_rows():
    """v13c/scaling 과 동일 학습셋 구성 — 단일 소스(양쪽 스크립트가 이걸 씀).

    SEED 고정 셔플까지 포함해 행 순서를 결정론적으로 만든다 → logits 캐시가
    학습 시 DataLoader(shuffle=True) 와 무관하게 행별로 정합(캐시는 인덱스 키).
    """
    import collections
    import pyarrow.parquet as pq

    random.seed(SEED)
    np.random.seed(SEED)

    t = pq.read_table("data/klue_re_train.parquet")
    train_all = [{c: t.column(c)[i].as_py() for c in t.column_names} for i in range(t.num_rows)]
    tune_guids = {r["guid"] for r in json.load(open("data/tune.json", encoding="utf-8"))}

    aug, cnt = [], collections.Counter()
    with open(AUG_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if cnt[r["label"]] >= AUG_CAP:
                continue
            cnt[r["label"]] += 1
            aug.append({"sentence": r["sentence"], "subject_entity": r["subject_entity"],
                        "object_entity": r["object_entity"], "label": LABEL2ID[r["label"]]})

    base = [r for r in train_all if r["guid"] not in tune_guids] + aug

    hard = [json.loads(l) for l in open(HARD_PATH, encoding="utf-8")]
    random.shuffle(hard)
    n_dev = max(1, int(len(hard) * 0.15))
    hard_dev, hard_tr = hard[:n_dev], hard[n_dev:]

    def enc(r):
        return {"sentence": r["sentence"], "subject_entity": r["subject_entity"],
                "object_entity": r["object_entity"], "label": LABEL2ID[r["label"]]}

    rows = base + [enc(r) for r in hard_tr] * UPSAMPLE
    random.shuffle(rows)   # SEED 고정 → 결정론적. 캐시와 학습이 같은 순서.
    dev_rows = [enc(r) for r in hard_dev]
    return rows, dev_rows


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TEACHER = os.environ["DISTILL_TEACHER"]
    CACHE = os.getenv("DISTILL_CACHE", "distill_logits.pt")
    FP16 = os.getenv("DISTILL_TEACHER_FP16", "1") == "1"

    torch.manual_seed(SEED)
    t0 = time.time()

    rows, dev_rows = build_distill_rows()
    print(f"teacher {TEACHER} fp16={FP16} | rows {len(rows)} dev {len(dev_rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TEACHER)
    # teacher 는 이미 SPECIALS 로 학습된 체크포인트 → resize 불필요(vocab 일치 가정).
    # DGX GPU 는 vLLM 3엔진이 통짜 예약(96GB) → CUDA OOM. logits 추출은 1회성 forward라
    # DISTILL_DEVICE=cpu 로 강제 가능(통합메모리 CPU 경로, 느리지만 캐시는 한 번만).
    force = os.getenv("DISTILL_DEVICE", "").lower()
    if force in ("cpu", "cuda", "mps"):
        dev = force
    else:
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    # CPU 는 fp16 행렬곱 미지원 → fp32 강제.
    dtype = torch.float32 if dev == "cpu" else (torch.float16 if FP16 else torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=30, dtype=dtype)
    print(f"device {dev} dtype {dtype}", flush=True)
    model.to(dev).eval()

    dl = DataLoader(REDataset(rows, tok), batch_size=BATCH, shuffle=False)  # 순서 보존 필수
    logits_all = []
    with torch.no_grad():
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
            out = model(**b).logits.float().cpu()   # fp32 로 저장(수치 안정)
            logits_all.append(out)
            if (i + 1) % 50 == 0:
                print(f"  batch {i+1}/{len(dl)} ({time.time()-t0:.0f}s)", flush=True)

    logits = torch.cat(logits_all, dim=0)
    assert logits.size(0) == len(rows), f"logits {logits.size(0)} != rows {len(rows)}"
    torch.save({"logits": logits, "n_rows": len(rows), "teacher": TEACHER}, CACHE)
    print(f"saved → {CACHE} shape {tuple(logits.shape)} ({time.time()-t0:.0f}s)", flush=True)
    print("DISTILL-CACHE-DONE", flush=True)


if __name__ == "__main__":
    main()
