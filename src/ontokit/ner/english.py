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

# 어절 경계 복원(R1b, 0816e) — WordPiece 연속 조각이 그대로 개체 표층형으로 방출되던
# 결함. `word` 의 `##` 를 지우면 '이건 단어 중간 조각'이라는 모델 자신의 신호가 사라져
# 'Eusebius'→'bius'·'Smash Mouth'→'mash Mouth'·'Sasanian'→'Sasan' 이 인스턴스가 됐다
# (mixed20k 확정 파편 62건 중 영어 49건, end-to-end 재현 43/49=87.8%).
# 파이프라인은 문자 오프셋을 이미 주고 있었고(실측), 코드가 그것을 버리고 있었다.
# → 오프셋으로 어절 중간 절단을 판정해 **원문에서 어절 경계까지 되돌린다**.
# 확장은 아래 닫힌 문자클래스 안에서만 — 공백·하이픈·아포스트로피·마침표에서 멈춘다
# (과확장은 없던 오류를 만들고 과소복원은 기존 오류를 덜 고칠 뿐이라 보수 쪽 고정).
# ⚠️ gate_spans(sg1) 의 영어 적용은 하지 않는다 — 별도 기각 이력이 있고 영어 오살률
# 미측정. 오프셋은 이 복원의 입력으로만 쓴다. env ONTOKIT_EN_SPAN_REPAIR=off 로 비활성.
_WORDCHAR_EN = re.compile(r"[0-9A-Za-zÀ-ÿĀ-ſ]")
# 확장 슬라이스의 공백 정규화 — 원문 개행·다중공백·NBSP 가 라벨로 새는 것을 막는다
# (0816e 심판 F-11 실측 0.91%: 'Mato Grosso  Demographics'·'Basil\xa0II').
_WS_COLLAPSE = re.compile(r"\s+")
_SPAN_REPAIR_EN = os.environ.get("ONTOKIT_EN_SPAN_REPAIR", "").lower() not in ("off", "false", "0")


def _expand_to_word(text: str, start: int, end: int) -> tuple[int, int]:
    """스팬이 어절 중간에서 끊겼으면 어절 경계까지 확장. 아니면 그대로."""
    while start > 0 and _WORDCHAR_EN.match(text[start - 1]):
        start -= 1
    while end < len(text) and _WORDCHAR_EN.match(text[end]):
        end += 1
    return start, end


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

    def _to_dicts(self, ents, source_chunks: list[str],
                  text: str | None = None) -> list[dict]:
        out = []
        seen = set()
        for e in ents or []:
            w = (e.get("word", "") or "").replace("##", "").strip()
            if len(w) < 2:
                # 표면 컷은 **확장 전** 표층형에 적용한다. 확장 뒤로 밀면 처치 전 차단되던
                # 1글자 스팬이 어절 전체로 늘어나 **신설 방출 채널**이 생긴다(0816e 심판
                # F-08 실측: 개체 +6.35%·중복 265건). 그 채널의 정밀도는 미측정이고
                # 사전 공시에도 없었으므로, 게이트 순서를 원복해 채널을 열지 않는다.
                continue
            st, en = e.get("start"), e.get("end")
            # 어절 경계 복원(R1b) — **스팬이 실제로 확장될 때만** 표층형을 교체한다.
            # ⚠️0816e 심판 F-07: 초판은 정합 스팬에도 무조건 `text[st:en]` 로 덮어써서
            # 파이프라인의 정규화된 표기가 원문 표기로 강등됐다('St. Louis'→'St.Louis'
            # = 같은 도시가 두 라벨로 분열, 개행·NBSP 유입 0.91%). 확장이 일어나지
            # 않으면 손대지 않는다 — 사전 공시 §1의 트리거 문면 그대로.
            if (_SPAN_REPAIR_EN and text is not None
                    and isinstance(st, int) and isinstance(en, int)
                    and 0 <= st < en <= len(text)):
                ns, ne = _expand_to_word(text, st, en)
                if (ns, ne) != (st, en):
                    st, en = ns, ne
                    w = _WS_COLLAPSE.sub(" ", text[st:en]).strip()
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
            # start/end 방출 — koelectra 와 대칭(그쪽은 R13에 이미 보존). 없던 정보를
            # 만드는 게 아니라 파이프라인이 주던 것을 버리지 않는 것.
            key = (w, st, en)
            if key in seen:                 # 서로 다른 조각이 같은 어절로 확장되면 중복 방출
                continue                    # (0816e F-08 실측 265건). 'He'+'ik' → 'Heikki' ×2
            seen.add(key)
            out.append({"entity": w, "class": CONLL_LABEL_KO.get(g, g),
                        "type": "INSTANCE", "source_chunks": source_chunks,
                        "start": st, "end": en})
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
        return self._to_dicts(ents, source_chunks, text[:max_len])

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
                        ents, source_chunks_list[start + j], chunk[j][:max_len])
            except Exception:
                logger.warning(
                    "영어 NER 배치 실패(%d~%d) — 단건 폴백", start, start + len(chunk),
                    exc_info=True)
                for j, t in enumerate(chunk):
                    results[start + j] = self.entities(
                        t, source_chunks=source_chunks_list[start + j],
                        max_len=max_len)
        return results
