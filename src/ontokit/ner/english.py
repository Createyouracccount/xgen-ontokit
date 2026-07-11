"""영어 NER — dslim/bert-base-NER. extras[ner]=transformers+torch.

KoElectraNER 와 동일 인터페이스(.entities()/.entities_batch()). 영어 문서 인스턴스용.
dslim/bert-base-NER: MIT, CoNLL-2003 F1~0.91, ~110M. PER/ORG/LOC/MISC.
언어 감지(detect_lang)로 영어 청크에만 라우팅해 사용.
"""
from __future__ import annotations
import logging
import threading
from typing import Optional

from .koelectra import MAX_NER_CHARS

logger = logging.getLogger(__name__)

# CoNLL 라벨 → 한국어 클래스명 — 한국어 NER(TTA→인물/기관)와 클래스명 통일.
# 통일 안 하면 혼합 코퍼스에서 "인물"과 "PER"가 별개 owl:Class 로 공존해
# 인물 질의 시 영어 문서 인스턴스가 통째 누락된다(0711 적대리뷰 MED —
# kg_builder 가 entity "class" 문자열 그대로 클래스 URI 생성). 닫힌 집합(4개).
CONLL_LABEL_KO = {"PER": "인물", "ORG": "기관", "LOC": "지역", "MISC": "기타"}


class EnglishNER:
    """HF NER 파이프라인 래핑(영어). 지연 로드."""

    DEFAULT_MODEL = "dslim/bert-base-NER"

    def __init__(self, model: Optional[str] = None, pipeline=None):
        self._pipe = pipeline
        self._model = model or self.DEFAULT_MODEL
        # 동시 빌드의 tokenizer 동시 호출 방지 — koelectra 와 동일(0711).
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._pipe is None:
                from transformers import pipeline as hf_pipeline  # lazy — extras[ner]
                self._pipe = hf_pipeline("ner", model=self._model,
                                         aggregation_strategy="simple")

    def _to_dicts(self, ents, source_chunks: list[str]) -> list[dict]:
        out = []
        for e in ents or []:
            w = (e.get("word", "") or "").replace("##", "").strip()
            if len(w) >= 2:
                g = e.get("entity_group", "ENTITY")
                out.append({"entity": w, "class": CONLL_LABEL_KO.get(g, g),
                            "type": "INSTANCE", "source_chunks": source_chunks})
        return out

    def entities(self, text: str, *, source_chunks: list[str],
                 max_len: int = MAX_NER_CHARS) -> list[dict]:
        self._ensure()
        try:
            with self._lock:
                ents = self._pipe(text[:max_len])
        except Exception:
            logger.warning("영어 NER 단건 추론 실패 — 해당 청크 엔티티 생략", exc_info=True)
            return []
        return self._to_dicts(ents, source_chunks)

    def entities_batch(self, texts: list[str], *, source_chunks_list: list[list[str]],
                       max_len: int = MAX_NER_CHARS, batch_size: int = 32) -> list[list[dict]]:
        """배치 forward — KoElectraNER.entities_batch 와 동일 계약(서브배치 격리+단건 폴백)."""
        self._ensure()
        results: list[list[dict]] = [[] for _ in texts]
        if not texts:
            return results
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            try:
                with self._lock:
                    batched = self._pipe([t[:max_len] for t in chunk],
                                         batch_size=batch_size)
                if len(batched) != len(chunk):
                    raise RuntimeError(
                        f"배치 출력 {len(batched)} != 입력 {len(chunk)}")
                for j, ents in enumerate(batched):
                    results[start + j] = self._to_dicts(
                        ents, source_chunks_list[start + j])
            except Exception:
                logger.warning(
                    "영어 NER 배치 실패(%d~%d) — 단건 폴백", start, start + len(chunk),
                    exc_info=True)
                for j, t in enumerate(chunk):
                    results[start + j] = self.entities(
                        t, source_chunks=source_chunks_list[start + j],
                        max_len=max_len)
        return results
