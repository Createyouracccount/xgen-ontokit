#!/usr/bin/env python3
"""ONTOKIT_RELATION_CONF_MIN 배선 검증 (conf 스윕 라운드 0722) — 스크립트 판정, LLM 무관.

픽스처 = gate2 실기사 news_1843 청크(colleagues 쌍 conf ~0.638 — 0.6/0.7 사이라 판별력).
검증 4종(전부 별도 프로세스 — env 는 ctor 에서 읽으므로):
  1. env unset ×2 → 산출 바이트 동일 (결정성 + 기본값 불변)
  2. env=0.99   → 산출이 실제로 줄어야 함. 동일하면 배선 미연결 → FAIL
  3. env=0.65   → unset 대비 부분집합이며 conf<0.65 항목 전멸
  4. env="nan"  → 거부·기본값 폴백 = unset 과 바이트 동일 (주심 D5: nan 이
     통과하면 score<nan 항상 False 로 conf 게이트 조용한 무력화)
사용: cd xgen-ontokit/eval/relation && PYTHONPATH=../../src python3 conf_wiring_check.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model_re_v13c")
FIXTURE = os.path.join(HERE, "conf_wiring_fixture.txt")

_WORKER = r"""
import json, sys
from ontokit.ner.koelectra import KoElectraNER
from ontokit.extractors.relation_encoder_ko import KoreanRelationEncoder
text = open(sys.argv[2], encoding="utf-8").read()
enc = KoreanRelationEncoder(model=sys.argv[1], ner=KoElectraNER())
rels = enc.extract(text, source_chunks=["fixture"])
rows = sorted((r["subject"], r["predicate"], r["object"], round(r["score"], 4)) for r in rels)
print(json.dumps(rows, ensure_ascii=False))
"""


def run(env_val):
    env = dict(os.environ)
    env.pop("ONTOKIT_RELATION_CONF_MIN", None)
    if env_val is not None:
        env["ONTOKIT_RELATION_CONF_MIN"] = env_val
    env["PYTHONPATH"] = os.path.join(HERE, "..", "..", "src")
    p = subprocess.run([sys.executable, "-c", _WORKER, MODEL, FIXTURE],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"worker 실패(env={env_val}):\n{p.stderr[-2000:]}")
    return p.stdout.strip()


def main():
    a1, a2 = run(None), run(None)
    assert a1 == a2, "FAIL: env unset 2회 산출 불일치 — 결정성 깨짐"
    base = json.loads(a1)
    print(f"[1/4] PASS unset ×2 바이트 동일 — 방출 {len(base)}건")
    assert base, "FAIL: 픽스처에서 방출 0건 — 픽스처 부적합"

    hi = json.loads(run("0.99"))
    assert len(hi) < len(base), (
        f"FAIL: env=0.99 인데 방출 불변({len(base)}→{len(hi)}) — 배선 미연결. 중단.")
    print(f"[2/4] PASS env=0.99 → 방출 {len(base)}→{len(hi)}건 (실제로 움직임)")

    mid = json.loads(run("0.65"))
    mid_set, base_set = set(map(tuple, mid)), set(map(tuple, base))
    assert mid_set <= base_set, "FAIL: env=0.65 산출이 unset 의 부분집합이 아님"
    assert all(r[3] >= 0.65 for r in mid), "FAIL: conf<0.65 항목이 컷을 통과"
    cut = [r for r in base if r[3] < 0.65]
    assert cut, "FAIL: 픽스처에 0.5~0.65 구간 항목이 없어 판별 불능"
    print(f"[3/4] PASS env=0.65 → 부분집합·컷 정확 (컷 {len(cut)}건: "
          + "; ".join(f"{s} {p} {o} {c}" for s, p, o, c in cut) + ")")

    nan_out = run("nan")
    assert nan_out == a1, (
        "FAIL: env=nan 산출이 unset 과 다름 — nan 거부·기본값 폴백이 안 됨"
        " (D5: 게이트 무력화 위험)")
    print("[4/4] PASS env=nan → 거부·기본값 폴백 (unset 과 바이트 동일)")
    print("배선 검증 전체 PASS")


if __name__ == "__main__":
    main()
