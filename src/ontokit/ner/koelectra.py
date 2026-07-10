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

    def entities_batch(self, texts: list[str], *, source_chunks_list: list[list[str]],
                       max_len: int = 512, batch_size: int = 32) -> list[list[dict]]:
        """여러 청크를 배치 forward 로 추론 — CPU 실측 891ms→430ms/청크(2배).

        청크별 entities() 반복은 forward 를 청크 수만큼 개별 실행해 대용량(2만 청크)
        에서 완주 불가(0710 mixed20k 실측: 단건 297분 vs 배치32 143분 추정).
        반환: texts 와 같은 순서의 리스트-of-리스트(i번째 = texts[i]의 엔티티).
        배치 전체 실패 시 빈 결과(단건 entities 와 동일한 실패 격리)."""
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
