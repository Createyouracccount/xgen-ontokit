"""ER 임베딩 채널 — 로컬 문장 임베딩으로 의미변이 동의어 판정. LLM 0회.

심판 판정(2026-07-14): 텍스트 기반 ER 폐기 아님, 채널 교체. 날것 klue-roberta 도
residual 62%@P0.90 회수 — 동의어 신호는 분포에 실재. 형태소 정규화가 원리적 불가한
의미변이(전자상거래~이커머스 0.902, 인공지능~AI 0.989)를 코사인 유사도로 잡는다.

임베딩=로컬 추론(API 0회). 사용자 결정(2026-07-14): LLM-free 불변식과 양립 허용
(관계 인코더·NER 과 동일 계열). 정본 docs/ontokit_ER_심판_텍스트접근_채널교체_2026_07_14.md.

blocking+matching(ER 문헌 표준): 후보쌍을 임베딩 유사도로 매칭. threshold 로
정밀도 통제(높을수록 정밀, 낮을수록 재현). 모델 교체 가능(기본 klue-roberta,
KURE-v1 등 SentenceTransformer 호환 지정 가능).
"""
from __future__ import annotations
import threading

import torch


class EmbeddingER:
    """로컬 임베딩 코사인 유사도로 동의어 판정. 모델 주입/교체 가능.

    model_kind:
      "mean"  = AutoModel + mean-pool(klue/roberta-small 등 일반 인코더)
      "st"    = SentenceTransformer(KURE-v1 등 문장임베딩 전용, 있으면 우선)
    """

    def __init__(self, model="klue/roberta-small", threshold=0.90, model_kind="mean",
                 max_len=32):
        self._name = model
        self._threshold = threshold
        self._kind = model_kind
        self._max_len = max_len
        self._tok = None
        self._model = None
        self._st = None
        self._cache = {}
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._kind == "st" and self._st is None:
                from sentence_transformers import SentenceTransformer  # extras
                self._st = SentenceTransformer(self._name)
            elif self._kind == "mean" and self._model is None:
                from transformers import AutoModel, AutoTokenizer  # extras
                self._tok = AutoTokenizer.from_pretrained(self._name)
                self._model = AutoModel.from_pretrained(self._name).eval()

    @torch.no_grad()
    def _embed(self, texts):
        self._ensure()
        if self._kind == "st":
            import numpy as np
            v = self._st.encode(texts, normalize_embeddings=True)
            return torch.tensor(np.asarray(v))
        enc = self._tok(texts, padding=True, truncation=True,
                        max_length=self._max_len, return_tensors="pt")
        out = self._model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(v, dim=-1)

    def _vec(self, term):
        if term not in self._cache:
            self._cache[term] = self._embed([term])[0]
        return self._cache[term]

    def similarity(self, a, b) -> float:
        return float((self._vec(a) * self._vec(b)).sum())

    def same_entity(self, a, b) -> bool:
        return self.similarity(a, b) >= self._threshold

    def embed_all(self, terms):
        """배치 임베딩(캐시 워밍) — 평가 시 전체 표기 미리."""
        todo = [t for t in terms if t not in self._cache]
        for i in range(0, len(todo), 128):
            chunk = todo[i:i + 128]
            vs = self._embed(chunk)
            for t, v in zip(chunk, vs):
                self._cache[t] = v
