"""KoELECTRA NER — 한국어 엔티티(인스턴스) 추출. extras[ner]=transformers+torch.

finreg 실측: monologg/koelectra-base-v3-naver-ner, 로드~100s(1회 캐시)·추론 0.44s/문장.
금융 도메인은 KF-DeBERTa(MIT, 금융 FN-NER 91.80)가 더 강함 — model 인자로 교체 가능.
"""
from __future__ import annotations
from typing import Optional


class KoElectraNER:
    """HF NER 파이프라인 래핑. 지연 로드(사용 안 하면 안 깔림)."""

    DEFAULT_MODEL = "monologg/koelectra-base-v3-naver-ner"

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
                w = e.get("word", "").replace("##", "").strip()
                if len(w) >= 2:
                    out.append({"entity": w, "class": e.get("entity_group", "ENTITY"),
                                "type": "INSTANCE", "source_chunks": source_chunks})
        except Exception:
            pass
        return out
