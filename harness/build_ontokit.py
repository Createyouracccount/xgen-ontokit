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
    from ontokit.pipeline import extract   # ontokit build CLI 와 같은 경로(harness/docs/S01)

    docs = [json.loads(l) for l in open(docs_path)]
    # HARNESS_HEARST=1 → ontokit 정의문 채널(enable_hearst) opt-in(3차). 백과체 전용 권장 채널.
    t0 = time.time()
    raw = asyncio.run(extract(docs, hearst=os.getenv("HARNESS_HEARST") == "1"))
    concepts, entities, relations, data = (raw["concepts"], raw["ner_entities"], raw["relations"],
                                           raw["data_properties"])
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
