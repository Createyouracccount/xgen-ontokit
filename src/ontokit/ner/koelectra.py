"""KoELECTRA NER — 한국어 엔티티(인스턴스) 추출. extras[ner]=transformers+torch.

기본 모델 = Leo97/KoELECTRA-small-v3-modu-ner (모두의말뭉치 2021, TTA 15분류).
0711 실전 5텍스트 맞대결로 구모델(monologg naver-ner 2020) 교체:
- 구모델은 라벨이 `ORG-B` 역순 형식이라 HF aggregation(`B-ORG` 기대)이 깨져
  entity_group 이 전부 'B'/'I' 로 잘리고 단어 파편화(1글자 파편 6 vs 1)가 심각.
- modu-ner: 라벨 표준 B-XX(정상 결합), CPU 배치 98ms vs 217ms(2.2배),
  55MB vs 450MB. 리콜은 다소 낮으나 온톨로지 인스턴스는 정밀도 우선
  (노이즈 인스턴스는 그래프 오염, 검색 리콜은 벡터 leg 가 캐리).
금융 도메인은 KF-DeBERTa(MIT, 금융 FN-NER 91.80)가 더 강함 — model 인자로 교체 가능.
"""
from __future__ import annotations
from typing import Optional

# TTA 15 분류(모두의말뭉치) 코드 → 온톨로지 클래스명. 모델 라벨셋 = 닫힌 집합이라
# 수동 목록 금지 원칙(무한증식) 대상 아님. 미등록 코드는 원문 코드 유지(안전).
TTA_LABEL_KO = {
    "PS": "인물", "FD": "분야", "TR": "이론", "AF": "인공물", "OG": "기관",
    "LC": "지역", "CV": "문명", "DT": "날짜", "TI": "시간", "QT": "수량",
    "EV": "사건", "AM": "동물", "PT": "식물", "MT": "물질", "TM": "용어",
}


class KoElectraNER:
    """HF NER 파이프라인 래핑. 지연 로드(사용 안 하면 안 깔림)."""

    DEFAULT_MODEL = "Leo97/KoELECTRA-small-v3-modu-ner"

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
                    g = e.get("entity_group", "ENTITY")
                    out.append({"entity": w, "class": TTA_LABEL_KO.get(g, g),
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
                    g = e.get("entity_group", "ENTITY")
                    results[i].append({"entity": w, "class": TTA_LABEL_KO.get(g, g),
                                       "type": "INSTANCE", "source_chunks": sc})
        return results
