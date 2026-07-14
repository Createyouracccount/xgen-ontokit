"""정직 baseline 3종 (허수아비 금지 — 계층 R1 교훈).

B0 always-no_relation : micro-F1 정의상 0 (floor 의 floor, 참고용)
B1 type-pair prior    : train 에서 (subj_type, obj_type)별 최빈 관계라벨.
                        결정적·무학습·1분 구현 — 이것이 정직한 floor.
B2 ontokit SVO 연결력  : 현 조사 SVO 가 gold 관계쌍의 subj-obj 를 애초에
                        '연결'이나 하는가(타입 분류 이전의 재현 상한).
"""
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from eval_re import report


def load(name):
    with open(f"data/{name}.json", encoding="utf-8") as f:
        return json.load(f)


def b1_prior(train_rows, eval_rows):
    prior = defaultdict(Counter)
    for r in train_rows:
        prior[(r["subject_entity"]["type"], r["object_entity"]["type"])][r["label"]] += 1
    preds = []
    for r in eval_rows:
        c = prior.get((r["subject_entity"]["type"], r["object_entity"]["type"]))
        preds.append(c.most_common(1)[0][0] if c else 0)
    return preds


def b2_svo_connectivity(eval_rows):
    """ontokit 조사 SVO(relation_ko)가 gold 관계문장에서 (subj,obj) 쌍을 연결하는 비율.

    타입 분류가 아니라 '연결' 자체의 재현 상한 — 현 본체 로직의 정직한 커버리지.
    """
    sys.path.insert(0, "../../src")
    from kiwipiepy import Kiwi
    from ontokit.extractors.relation_ko import KoreanRelationExtractor  # 현 본체 관계 채널

    ex = KoreanRelationExtractor(Kiwi(), enable_carry=False)
    total = hit = 0
    for r in eval_rows:
        if r["label"] == 0:
            continue
        total += 1
        sw, ow = r["subject_entity"]["word"], r["object_entity"]["word"]
        try:
            triples = ex.extract(r["sentence"], source_chunks=["x"])
        except Exception:
            triples = []
        for t in triples:
            s, o = t.get("subject", ""), t.get("object", "")
            if ((sw in s or s in sw) and (ow in o or o in ow)) or \
               ((sw in o or o in sw) and (ow in s or s in ow)):
                hit += 1
                break
    return hit, total


if __name__ == "__main__":
    import pyarrow.parquet as pq
    t = pq.read_table("data/klue_re_train.parquet")
    train = [{c: t.column(c)[i].as_py() for c in t.column_names} for i in range(t.num_rows)]
    which = sys.argv[1] if len(sys.argv) > 1 else "tune"
    rows = load(which)
    golds = [r["label"] for r in rows]

    report(f"B0 always-no_relation @{which}", golds, [0] * len(golds))
    report(f"B1 type-pair prior @{which}", golds, b1_prior(train, rows))
    if "--svo" in sys.argv:
        hit, total = b2_svo_connectivity(rows)
        print(f"[B2 SVO 연결력 @{which}] gold관계쌍 {total} 중 연결 {hit} = {hit/total:.1%}")
