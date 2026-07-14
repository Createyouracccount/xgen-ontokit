"""영어 NER — dslim/bert-base-NER. extras[ner]=transformers+torch.

KoElectraNER 와 동일 인터페이스(.entities()/.entities_batch()). 영어 문서 인스턴스용.
dslim/bert-base-NER: MIT, CoNLL-2003 F1~0.91, ~110M. PER/ORG/LOC/MISC.
언어 감지(detect_lang)로 영어 청크에만 라우팅해 사용.
"""
from __future__ import annotations
import logging
import os
import re
import threading
from typing import Optional

from .koelectra import MAX_NER_CHARS

logger = logging.getLogger(__name__)

# CoNLL 라벨 → 한국어 클래스명 — 한국어 NER(TTA→인물/기관)와 클래스명 통일.
# 통일 안 하면 혼합 코퍼스에서 "인물"과 "PER"가 별개 owl:Class 로 공존해
# 인물 질의 시 영어 문서 인스턴스가 통째 누락된다(0711 적대리뷰 MED —
# kg_builder 가 entity "class" 문자열 그대로 클래스 URI 생성). 닫힌 집합(4개).
CONLL_LABEL_KO = {"PER": "인물", "ORG": "기관", "LOC": "지역", "MISC": "기타"}

# MISC 방출 차단(기본) — CoNLL MISC 는 의미 클래스가 아니라 "나머지" 쓰레기통이라
# '기타' 클래스로 직행하면 혼합 코퍼스에서 검색 불가 고아 인스턴스가 수만 건
# 쌓인다(mixed20k 실측 20,797건 = 인스턴스 4위 ~14%, 전량 라틴 토큰, SVO·계층
# 참여 구조적 0). PER/LOC/ORG 는 인물/지역/기관 공유 클래스로 유지 — 영어
# 집계·열거 질의는 생존(0714 적대심판 조건부 채택; "ko 청크 혼입 개체 회귀"
# 우려는 en-지배 청크만 이 경로라 허수 판정). dict 에서 키만 빼면 raw "MISC"
# 클래스로 역방출되는 함정(심판 적발)이 있어 명시 필터로 구현.
# env ONTOKIT_NER_EMIT_MISC=1 로 구동작 복원 가능.
_EMIT_MISC = os.environ.get("ONTOKIT_NER_EMIT_MISC", "") == "1"

# 영어 NER 최소 신뢰도 — koelectra 의 ONTOKIT_NER_MIN_SCORE(0.40)는 한국어 실측
# 보정값이라 복붙 금지(심판 조건: dslim/bert-base-NER 스코어 분포는 별개). env 를
# ONTOKIT_NER_MIN_SCORE_EN 으로 분리, 기본 0.0(=off) — en 표본 히스토그램으로
# 보정하기 전까지 정직하게 무보정.
DEFAULT_MIN_SCORE_EN = float(os.environ.get("ONTOKIT_NER_MIN_SCORE_EN", "0") or 0)

# 문자(letter) 없는 표면 컷 — '12'·'2004'·'65' 류 아티팩트가 인스턴스로 방출되던
# 무게이트 구멍(mixed20k 실측, koelectra 에만 있던 게이트의 en 대칭 일부).
# 닫힌 문자클래스 규칙 — 임계 보정이 필요 없어 즉시 안전.
_HAS_LETTER = re.compile(r"[^\W\d_]")


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
            if len(w) < 2:
                continue
            g = e.get("entity_group", "ENTITY")
            if g == "MISC" and not _EMIT_MISC:      # 쓰레기통 라벨 미방출(파일 상단 주석)
                continue
            if not _HAS_LETTER.search(w):           # 숫자·기호뿐인 표면 컷
                continue
            score = float(e.get("score", 1.0) or 1.0)
            if score < DEFAULT_MIN_SCORE_EN:        # 기본 0.0=off, 보정 후 env 로 활성
                continue
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
