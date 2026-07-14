"""KLUE-RE parquet → JSON 분할 준비.

누수 차단 프로토콜(I2):
  - tune  = train 에서 시드 고정 표본 6,000 — 규칙 반복·패턴 튜닝은 여기서만.
  - holdout = 공식 validation 7,765 전량 — 반복 중 미접촉, 심판 채점 전용.
데이터 출처: https://huggingface.co/datasets/klue/klue (re config, CC BY-SA 4.0).
재현: curl -L -o data/klue_re_{train,validation}.parquet <HF resolve URL> 후 본 스크립트.
"""
import json
import random

import pyarrow.parquet as pq


def rows(path):
    t = pq.read_table(path)
    cols = t.column_names
    return [{c: t.column(c)[i].as_py() for c in cols} for i in range(t.num_rows)]


def main():
    train = rows("data/klue_re_train.parquet")
    dev = rows("data/klue_re_validation.parquet")
    rng = random.Random(20260714)
    tune = rng.sample(train, 6000)
    with open("data/tune.json", "w", encoding="utf-8") as f:
        json.dump(tune, f, ensure_ascii=False)
    with open("data/holdout.json", "w", encoding="utf-8") as f:
        json.dump(dev, f, ensure_ascii=False)
    print(f"tune {len(tune)} / holdout {len(dev)}")


if __name__ == "__main__":
    main()
