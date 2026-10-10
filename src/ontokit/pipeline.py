"""문서 → 추출 결과(raw). `ontokit build` 와 측정 하네스(harness/build_ontokit.py)가 같은 경로를 쓴다.

LLM 0회. extras[korean,ner] 필요(Kiwi 형태소 + KoELECTRA NER). 관계는 env ONTOKIT_RELATION_ENCODER_MODEL
(관계 인코더 가중치 경로)이 있어야 나온다 — 없으면 관계 0건(타입·클래스만).

docs = [{"doc_id", "title", "text", "chunk_index"}] — 한 항목이 한 청크다.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

CHUNK_CHARS = 1200   # = ner.koelectra.MAX_NER_CHARS — NER 이 한 번에 보는 길이. 넘는 부분은 개체를 못 얻는다
TEXT_SUFFIXES = (".txt", ".md")


def chunk_text(text, limit=CHUNK_CHARS):
    """문단(빈 줄) 경계로 limit 이하 청크로 묶는다. limit 을 넘는 문단은 문장 끝에서, 그래도 길면 limit 에서 자른다."""
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        while len(para) > limit:
            cut = max(para.rfind(e, 0, limit) for e in (". ", "다.", "\n", "? ", "! "))
            cut = cut + 2 if cut > 0 else limit
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            pieces.append(para)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 2 + len(p) > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def read_docs(path):
    """입력 → 청크 목록.

    - .jsonl: 한 줄 = 한 청크 {"doc_id", "text", "title"?, "chunk_index"?} (그대로 — 측정 하네스와 같은 형식)
    - 폴더: 안의 .txt·.md 파일(하위 폴더 포함). doc_id = 상대 경로, title = 파일 이름(확장자 제외), chunk_text 로 분할
    """
    p = Path(path)
    if p.is_dir():
        docs = []
        for f in sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in TEXT_SUFFIXES):
            rel = f.relative_to(p).as_posix()
            for i, c in enumerate(chunk_text(f.read_text(encoding="utf-8", errors="replace"))):
                docs.append({"doc_id": rel, "title": f.stem, "text": c, "chunk_index": i})
        return docs
    docs = []
    with open(p, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            if "doc_id" not in d or "text" not in d:
                raise ValueError(f"{path}:{n} — doc_id 와 text 가 필요합니다")
            d.setdefault("title", d["doc_id"])
            d.setdefault("chunk_index", 0)
            docs.append(d)
    return docs


async def extract(docs, *, hearst=False):
    """청크 목록 → {"concepts", "ner_entities", "relations", "data_properties"}.

    hearst=True: 정의문 채널(백과체 전용 권장, 3차 측정). 기본 off.
    """
    from .builder.ontology_builder import OntologyBuilder
    from .ner.koelectra import KoElectraNER

    # 문서 단위로 넘긴다 — 키는 doc_id(제목 중복 방지), 청크 id 는 doc_id#index
    payload = {}
    for d in docs:
        payload.setdefault(d["doc_id"], []).append(
            {"chunk_id": f'{d["doc_id"]}#{d["chunk_index"]}', "chunk_text": d["text"],
             "chunk_index": d["chunk_index"]})
    # 제품 어댑터와 같은 구성. 영어 NER 은 생략(한국어 코퍼스 기준, 로드 비용만 큼).
    ner = KoElectraNER()
    extractor = None
    if hearst:
        from . import DeterministicKoreanExtractor
        extractor = DeterministicKoreanExtractor(ner=ner, enable_hearst=True)
    builder = OntologyBuilder(extractor=extractor, ner=ner)
    concepts, entities, relations, data = await builder.build(payload)
    return {"concepts": concepts, "ner_entities": entities, "relations": relations, "data_properties": data}


def relation_channel_on():
    return bool(os.getenv("ONTOKIT_RELATION_ENCODER_MODEL"))
