#!/usr/bin/env python3
"""v14 R1b(a) — v13c 시점 hard_dev 분할 결정론 복원·동결.

v13c 학습과 동일 절차: v13_hardset.jsonl(880, 파일 순서) → random.seed(SEED)
→ shuffle → 앞 15%(132행) = dev. random.shuffle 은 위치에만 작용하므로
range(880) 셔플로 동일 순열 재현 → dev = 원 파일 인덱스 집합으로 동결
(하드셋에 완전 중복 4행이 있어 키 매칭 대신 인덱스 매칭).
v13c 로그 'hard 748x1' 과 일치 검증. 출력: eval_runs/relations/v14_frozen_dev.json
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, ".")
from train_encoder import SEED  # noqa: E402

EVAL_RUNS = pathlib.Path(__file__).resolve().parents[3] / "eval_runs" / "relations"


def main():
    hard = [json.loads(l) for l in open(EVAL_RUNS / "v13_hardset.jsonl",
                                        encoding="utf-8")]
    assert len(hard) == 880, len(hard)
    idx = list(range(880))
    random.seed(SEED)
    random.shuffle(idx)
    n_dev = max(1, int(880 * 0.15))
    dev_idx = idx[:n_dev]
    assert 880 - n_dev == 748, "v13c 로그(hard 748x1)와 불일치"
    out = {"indices": dev_idx, "rows": [hard[i] for i in dev_idx]}
    json.dump(out, open(EVAL_RUNS / "v14_frozen_dev.json", "w"),
              ensure_ascii=False)
    print(f"frozen dev {n_dev}행(인덱스 동결) 저장, train 잔여 748 = v13c 일치")


if __name__ == "__main__":
    main()
