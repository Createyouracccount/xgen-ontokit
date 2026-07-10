"""영어 NER — dslim/bert-base-NER. extras[ner]=transformers+torch.

KoElectraNER 와 동일 인터페이스(.entities()). 영어 문서 인스턴스 추출용.
dslim/bert-base-NER: MIT, CoNLL-2003 F1~0.91, ~110M. PER/ORG/LOC/MISC.
언어 감지(detect_lang)로 영어 청크에만 라우팅해 사용.
"""
from __future__ import annotations
from typing import Optional


class EnglishNER:
    """HF NER 파이프라인 래핑(영어). 지연 로드."""

    DEFAULT_MODEL = "dslim/bert-base-NER"

    def __init__(self, model: Optional[str] = None, pipeline=None):
        self._pipe = pipeline
        self._model = model or self.DEFAULT_MODEL

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline as hf_pipeline  # lazy — extras[ner]
            self._pipe = hf_pipeline("ner", model=self._model,
                                     aggregation_strategy="simple")

    def entities(self, text: str, *, source_chunks: list[str], max_len: int = 512) -> list[dict]:
        self._ensure()
        out = []
        try:
            for e in self._pipe(text[:max_len]):
                w = (e.get("word", "") or "").replace("##", "").strip()
                if len(w) >= 2:
                    out.append({"entity": w, "class": e.get("entity_group", "ENTITY"),
                                "type": "INSTANCE", "source_chunks": source_chunks})
        except Exception:
            pass
        return out

    def entities_batch(self, texts: list[str], *, source_chunks_list: list[list[str]],
                       max_len: int = 512, batch_size: int = 32) -> list[list[dict]]:
        """배치 forward 추론 — KoElectraNER.entities_batch 와 동일 계약(대용량 완주용)."""
        self._ensure()
        results: list[list[dict]] = [[] for _ in texts]
        if not texts:
            return results
        try:
            batched = self._pipe([t[:max_len] for t in texts], batch_size=batch_size)
        except Exception:
            return results
        for i, ents in enumerate(batched):
            sc = source_chunks_list[i]
            for e in ents or []:
                w = (e.get("word", "") or "").replace("##", "").strip()
                if len(w) >= 2:
                    results[i].append({"entity": w, "class": e.get("entity_group", "ENTITY"),
                                       "type": "INSTANCE", "source_chunks": sc})
        return results
