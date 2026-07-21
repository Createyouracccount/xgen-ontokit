#!/usr/bin/env python3
"""sg1 스팬 게이트 배선 검증 — 스크립트 판정, LLM 무관 (conf 라운드 wiring_check 동형).

픽스처 = gate3 개발 코퍼스(sg1_dev_malformed79.json) 청크 8건 — 개발용 허용 자료.
⚠️ 채택 게이트 표본(sg1_fresh_sample_ids.json)은 여기서 사용 금지.

검증 4종(전부 별도 프로세스 — env 는 실행 시 읽힘):
  1. OFF ×2 → 산출 바이트 동일 (결정성 + 기본값 무영향)
  2. OFF 산출에 경계절단 스팬이 실재해야(픽스처 적격성) — boundary_ok 로 판정
  3. ON → OFF 의 부분집합 ∧ 경계절단 스팬 전멸 ∧ 제거분 == OFF 의 절단 집합
     (게이트가 정확히 절단만 제거)
  4. 다시 OFF → 1번과 바이트 동일 (규칙을 끄면 MALFORMED 가 실제로 되돌아옴 —
     안 돌아오면 배선 미연결/상태 오염 → 중단)
사용: cd xgen-ontokit && PYTHONPATH=src /opt/miniconda3/bin/python eval/relation/sg1_wiring_check.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
DEV = os.path.join(ROOT, "..", "eval_runs", "typing", "sg1_dev_malformed79.json")

_WORKER = r"""
import json, sys
from ontokit.ner.koelectra import KoElectraNER
from ontokit.morphology.kiwi_nouns import KiwiNounExtractor
from ontokit.extractors.deterministic_ko import DeterministicKoreanExtractor
chunks = json.load(open(sys.argv[1], encoding="utf-8"))
ner = KoElectraNER()
kiwi = KiwiNounExtractor().kiwi
all_entities = {}
buf = [(f"doc{i}", c, [f"c{i}"]) for i, c in enumerate(chunks)]
DeterministicKoreanExtractor._run_ner_batched(ner, buf, all_entities, kiwi=kiwi)
rows = []
for doc, ents in sorted(all_entities.items()):
    for e in ents:
        rows.append([doc, e["entity"], e.get("start"), e.get("end")])
print(json.dumps(sorted(rows, key=str), ensure_ascii=False))
"""


def run(gate_on: bool, fixture: str) -> str:
    env = dict(os.environ)
    env["ONTOKIT_NER_SPAN_GATE"] = "1" if gate_on else "0"
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    p = subprocess.run([sys.executable, "-c", _WORKER, fixture],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"worker 실패(gate={gate_on}):\n{p.stderr[-2000:]}")
    return p.stdout.strip()


def main():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from ontokit.ner.span_gate import boundary_ok

    mal = json.load(open(DEV, encoding="utf-8"))
    chunks, seen = [], set()
    for m in mal:
        if m["file"] not in seen:
            seen.add(m["file"])
            chunks.append(m["chunk"])
        if len(chunks) >= 8:
            break
    fixture = os.path.join(HERE, "sg1_wiring_fixture.json")
    json.dump(chunks, open(fixture, "w", encoding="utf-8"), ensure_ascii=False)

    off1, off2 = run(False, fixture), run(False, fixture)
    assert off1 == off2, "FAIL: OFF 2회 산출 불일치 — 결정성 깨짐"
    off = json.loads(off1)
    print(f"[1/4] PASS OFF ×2 바이트 동일 — 방출 {len(off)}건")

    # 픽스처 적격성: OFF 방출에 경계절단 스팬 실재
    idx = {f"doc{i}": c[:1200] for i, c in enumerate(chunks)}
    cut = [r for r in off if isinstance(r[2], int)
           and not boundary_ok(idx[r[0]], r[2], r[3])]
    assert cut, "FAIL: OFF 방출에 경계절단 스팬 0건 — 픽스처 부적격"
    print(f"[2/4] PASS OFF 방출에 절단 스팬 {len(cut)}건 실재 — 예: "
          + "; ".join(f"{r[0]}:{r[1]}" for r in cut[:5]))

    on = json.loads(run(True, fixture))
    on_set, off_set = {tuple(map(str, r)) for r in on}, {tuple(map(str, r)) for r in off}
    assert on_set <= off_set, "FAIL: ON 산출이 OFF 의 부분집합이 아님"
    on_cut = [r for r in on if isinstance(r[2], int)
              and not boundary_ok(idx[r[0]], r[2], r[3])]
    assert not on_cut, f"FAIL: ON 인데 절단 스팬 {len(on_cut)}건 통과 — 배선 미연결"
    removed = off_set - on_set
    cut_set = {tuple(map(str, r)) for r in cut}
    assert removed == cut_set, (
        f"FAIL: 제거분({len(removed)}) != 절단 집합({len(cut_set)}) — 과잉/과소 제거")
    print(f"[3/4] PASS ON → 절단 {len(removed)}건 정확 제거(과잉 제거 0)")

    off3 = run(False, fixture)
    assert off3 == off1, "FAIL: 재-OFF 산출이 최초 OFF 와 다름 — MALFORMED 미복원"
    print("[4/4] PASS 재-OFF → MALFORMED 복원 (바이트 동일)")
    print("배선 검증 전체 PASS")


if __name__ == "__main__":
    main()
