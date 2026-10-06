"""dev 소표본에서 LLM 추출 단독·ontokit 합집합의 관계/타입 재현율(모델 선택용, dev 전용)."""
import json
import sys

from harness.extract_llm import merge
from harness.relrecall import recall
from harness.typerecall import type_recall

docs = {json.loads(l)["doc_id"] for l in open(sys.argv[1])}
F = json.load(open("harness/bench/wiki2_oracle_facts.json"))
empty = {"ner_entities": {}, "relations": [], "concepts": {"class_hierarchy": []}}
json.dump(empty, open("/tmp/claude_oh_empty.json", "w"))
for llm in sys.argv[2:]:
    merge("/tmp/claude_oh_empty.json", llm, "/tmp/claude_oh_m.json")
    m = json.load(open("/tmp/claude_oh_m.json"))
    r, t = recall(m, F, docs), type_recall(m, F, docs)
    print(f"{llm}: 관계 {r['hit']}/{r['facts']}={r['recall']:.3f} 타입 {t['hit']}/{t['facts']}={t['recall']:.3f} 방출관계 {r['relations_emitted']}")
