"""벡터 기준선 — 제품과 같은 임베딩·같은 검색 API 를 쓴다.

색인(index): documents 컨테이너 안에서 제품의 EmbeddingFactory(=제품 설정 임베딩)로
코퍼스를 임베딩해 새 Qdrant 컬렉션에 넣는다. 키는 컨테이너 밖으로 나오지 않는다.
검색(search): 제품 API `/api/retrieval/documents/search` 를 그대로 호출한다.

  python -m harness.vector index harness/data/wiki2/docs.jsonl oh_wiki2_vec
  python -m harness.vector search oh_wiki2_vec "질문" 40
"""
import json
import os
import subprocess
import sys
import urllib.request

DOCS_API = os.getenv("DOCS_API", "http://localhost:8003")
CONTAINER = os.getenv("DOCS_CONTAINER", "full-stack-xgen-documents-1")

# 컨테이너 안에서 실행되는 색인 스크립트. stdin 으로 jsonl 을 받는다.
_INDEX_SRC = r'''
import asyncio, json, sys, uuid
sys.path.insert(0, "/app")
from xgen_sdk.config import ConfigClient
from service.embedding.embedding_factory import EmbeddingFactory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

coll = sys.argv[1]
docs = [json.loads(l) for l in sys.stdin if l.strip()]
emb = EmbeddingFactory.create_embedding_client(ConfigClient())
print("provider", emb.get_provider_info(), flush=True)
qc = QdrantClient(url="http://qdrant:6333")

async def run():
    vecs = []
    for i in range(0, len(docs), 64):
        vecs += await emb.embed_documents([d["text"] for d in docs[i:i + 64]])
        print("embedded", len(vecs), flush=True)
    return vecs

vecs = asyncio.run(run())
assert len(vecs) == len(docs), (len(vecs), len(docs))
dim = len(vecs[0])
zero = sum(1 for v in vecs if not any(v))
assert zero == 0, f"0벡터 {zero}건 — 임베딩 실패를 색인하지 않는다"
if qc.collection_exists(coll):
    qc.delete_collection(coll)
qc.create_collection(coll, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
pts = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, d["doc_id"] + "#" + str(d["chunk_index"]))),
                   vector=v,
                   payload={"collection_name": coll, "document_id": d["doc_id"], "chunk_index": d["chunk_index"],
                            "chunk_text": d["text"], "file_name": d["title"], "title": d["title"]})
       for d, v in zip(docs, vecs)]
for i in range(0, len(pts), 256):
    qc.upsert(coll, pts[i:i + 256])
print("indexed", len(pts), "dim", dim, flush=True)
'''


def index(docs_path, coll):
    with open(docs_path) as f:
        data = f.read()
    r = subprocess.run(["docker", "exec", "-i", CONTAINER, "python3", "-c", _INDEX_SRC, coll],
                       input=data, capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        raise SystemExit(r.stderr[-3000:])


def search(coll, query, k=40):
    """제품 검색 API. 결과가 0건이면 조용히 넘기지 않고 예외 — 0벡터 사고 재발 방지."""
    body = {"collection_name": coll, "query_text": query, "limit": k, "score_threshold": 0.0}
    req = urllib.request.Request(f"{DOCS_API}/api/retrieval/documents/search",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json", "X-User-ID": "1", "X-User-Name": "admin",
                                          "X-User-Superuser": "true"})
    res = json.load(urllib.request.urlopen(req, timeout=120)).get("results", [])
    if not res:
        raise RuntimeError(f"벡터 검색 0건: {coll} / {query[:40]}")
    return [{"doc_id": r["document_id"], "title": r.get("file_name"), "score": r["score"],
             "text": r.get("chunk_text") or ""} for r in res]


if __name__ == "__main__":
    if sys.argv[1] == "index":
        index(sys.argv[2], sys.argv[3])
    else:
        for h in search(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 10):
            print(round(h["score"], 3), h["title"])
