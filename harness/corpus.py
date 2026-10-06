"""코퍼스 덤프 — 로컬 Qdrant 컬렉션의 청크 본문을 jsonl 로 고정한다.

wiki2(한국어 위키 리드 3,000건)는 이전 LLM-free 실험에서 0벡터로 적재돼 있어
본문만 가져오고 벡터는 vector.py 가 제품 임베딩으로 다시 만든다.

  python -m harness.corpus wiki2_idfix3 harness/data/wiki2/docs.jsonl
"""
import json
import os
import sys
import urllib.request

QDRANT = os.getenv("QDRANT_URL", "http://localhost:6333")


def _post(path, body):
    req = urllib.request.Request(QDRANT + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def dump(collection: str):
    offset, docs = None, []
    while True:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = _post(f"/collections/{collection}/points/scroll", body)["result"]
        for p in r["points"]:
            pl = p["payload"]
            if pl.get("type") == "collection_metadata":
                continue
            docs.append({"doc_id": pl.get("wiki_doc_id") or pl.get("document_id"),
                         "title": pl.get("title") or pl.get("file_name"),
                         "chunk_index": int(pl.get("chunk_index") or 0),
                         "text": pl.get("chunk_text") or ""})
        offset = r.get("next_page_offset")
        if offset is None:
            break
    docs.sort(key=lambda d: (d["doc_id"], d["chunk_index"]))
    return docs


def main():
    collection, out = sys.argv[1], sys.argv[2]
    docs = dump(collection)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    empty = sum(1 for d in docs if not d["text"].strip())
    print(f"{len(docs)} chunks, {len({d['doc_id'] for d in docs})} docs, empty={empty} -> {out}")


if __name__ == "__main__":
    main()
