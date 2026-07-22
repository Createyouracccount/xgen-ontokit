#!/usr/bin/env python3
"""v14 arm 하드셋 결정적 생성 (설계 v2 D2, 주심 C1·C4 이행).

Base = v13_hardset.jsonl(880, 불변 — v13c 재현은 V13_HARD_PATH 로 그대로).
생성 arm 파일(각각 V13_HARD_PATH 로 주입, aug/캡/시드 불변 = 순수처치):
  v14_hard_T1.jsonl    = v13_hardset + KLUE colleagues 425 ×K 복제
  v14_hard_T1rm.jsonl  = v13_hardset − no_relation ∧ PER-PER 쌍 (억압원 제거)
  v14_hard_T2.jsonl    = v13_hardset + T2 합의 하드네거(문체 상한 강제)
  v14_hard_final.jsonl = (T1계 승자, 기본 T1) + T2
K 는 env V14_K(기본 2 — 선택 규율은 설계 D2, tune 기준 사전 규칙).

C4 문체 편중 상한 강제: T2 최종셋에서 트리거 어휘별 비율 ≤40% ∧ 문서당 ≤3문장.
위반 시 결정론 드롭 — 문서 분산 우선(문서당 초과분부터), 그다음 트리거 초과분을
conf 내림차순 유지·오름차순 드롭. 어서션으로 재검증.
"""
import collections
import json
import os
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
EVAL_RUNS = HERE.parents[2] / "eval_runs" / "relations"
K = int(os.getenv("V14_K", "2"))

sys.path.insert(0, str(HERE))
from labels import LABEL2ID  # noqa: E402
ID2 = {v: k for k, v in LABEL2ID.items()}


def load_hardset():
    rows = []
    for line in open(EVAL_RUNS / "v13_hardset.jsonl", encoding="utf-8"):
        rows.append(json.loads(line))
    assert len(rows) == 880, len(rows)
    return rows


def klue_colleagues():
    df = pd.read_parquet(HERE / "data" / "klue_re_train.parquet")
    tune = {r["guid"] for r in json.load(open(HERE / "data" / "tune.json"))}
    col_id = LABEL2ID["per:colleagues"]
    sel = df[(df["label"] == col_id) & (~df["guid"].isin(tune))]
    assert len(sel) == 425, len(sel)
    rows = []
    for _, r in sel.iterrows():
        rows.append({"sentence": r["sentence"],
                     "subject_entity": dict(r["subject_entity"]),
                     "object_entity": dict(r["object_entity"]),
                     "label": "per:colleagues", "source": "v14_t1_colleagues"})
    return rows


def t2_negatives():
    path = EVAL_RUNS / "v14_t2_consensus.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    # C4 상한 강제 — ① 문서당 ≤3 (conf 내림차순 유지)
    rows.sort(key=lambda r: -r.get("score", 0))
    by_doc = collections.Counter()
    kept = []
    for r in rows:
        if by_doc[r["doc"]] < 3:
            by_doc[r["doc"]] += 1
            kept.append(r)
    # ② 트리거 어휘별 ≤40% — 초과 트리거는 conf 오름차순 드롭
    total = len(kept)
    trig = collections.Counter(r["trigger"] for r in kept)
    while trig and max(trig.values()) > 0.40 * total:
        worst = max(trig, key=trig.get)
        # 해당 트리거 중 최저 conf 1건 드롭
        cand = min((r for r in kept if r["trigger"] == worst),
                   key=lambda r: r.get("score", 0))
        kept.remove(cand)
        total = len(kept)
        trig = collections.Counter(r["trigger"] for r in kept)
    assert all(v <= 0.40 * total + 1e-9 for v in trig.values()), trig
    assert max(collections.Counter(r["doc"] for r in kept).values()) <= 3
    print(f"T2: 합의 {len(rows)} → 상한 후 {total} (트리거 {dict(trig)})")
    out = []
    for r in kept:
        out.append({"sentence": r["sentence"],
                    "subject_entity": r["subject_entity"],
                    "object_entity": r["object_entity"],
                    "label": "no_relation", "source": "v14_t2_visit_neg"})
    return out


def is_per_per(row):
    st = (row.get("subject_entity") or {}).get("type")
    ot = (row.get("object_entity") or {}).get("type")
    return st == "PER" and ot == "PER"


def dump(name, rows):
    p = EVAL_RUNS / name
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")


def main():
    base = load_hardset()
    col = klue_colleagues()
    dump("v14_hard_T1.jsonl", base + col * K)
    t1rm = [r for r in base if not (r["label"] == "no_relation" and is_per_per(r))]
    print(f"T1rm: PER-PER no_rel 제거 {len(base) - len(t1rm)}건")
    dump("v14_hard_T1rm.jsonl", t1rm)
    if (EVAL_RUNS / "v14_t2_consensus.jsonl").exists():
        t2 = t2_negatives()
        dump("v14_hard_T2.jsonl", base + t2)
        winner = os.getenv("V14_T1_WINNER", "T1")
        t1part = (base + col * K) if winner == "T1" else t1rm + col * 0
        if winner == "T1rm":
            t1part = t1rm
        dump("v14_hard_final.jsonl", (t1part if winner == "T1rm"
                                      else base + col * K) + t2)
    else:
        print("T2 합의 파일 없음 — T1/T1rm 만 생성")


if __name__ == "__main__":
    main()
