"""ontokit 빌드 — 코퍼스 → ontokit 원출력(JSON). LLM 0회, 로컬 모델만.

  ONTOKIT_RELATION_ENCODER_MODEL=eval/relation/model_re_v13c \
  python -m harness.build_ontokit harness/data/wiki2/docs.jsonl harness/data/r1/ontokit_raw.json
"""
import asyncio
import json
import os
import sys
import time


def main():
    docs_path, out = sys.argv[1], sys.argv[2]
    import torch
    torch.set_num_threads(int(os.getenv("HARNESS_TORCH_THREADS", "4")))  # 0904 실측: 4스레드 최적
    from ontokit.builder.ontology_builder import OntologyBuilder
    from ontokit.ner.koelectra import KoElectraNER

    docs = [json.loads(l) for l in open(docs_path)]
    # 문서 단위로 넘긴다 — 키는 doc_id(제목 중복 방지), 청크 id 는 doc_id#index
    payload = {}
    for d in docs:
        payload.setdefault(d["doc_id"], []).append(
            {"chunk_id": f'{d["doc_id"]}#{d["chunk_index"]}', "chunk_text": d["text"],
             "chunk_index": d["chunk_index"]})
    # 제품 어댑터(feature/ontology-extractor-axis-0903 ontokit_extractor.py)와 같은 구성.
    # 코퍼스가 한국어라 영어 NER 은 생략(로드 비용만 큼).
    builder = OntologyBuilder(ner=KoElectraNER())
    t0 = time.time()
    concepts, entities, relations, data = asyncio.run(builder.build(payload))
    sec = time.time() - t0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"concepts": concepts, "ner_entities": entities, "relations": relations,
               "data_properties": data, "seconds": sec,
               "env": {k: v for k, v in os.environ.items() if k.startswith("ONTOKIT_")}},
              open(out, "w"), ensure_ascii=False)
    n_ent = sum(len(v) for v in entities.values()) if isinstance(entities, dict) else len(entities)
    print(f"{sec:.0f}s classes={len(concepts.get('classes', []))} entities={n_ent} "
          f"relations={len(relations)} -> {out}")


if __name__ == "__main__":
    main()
