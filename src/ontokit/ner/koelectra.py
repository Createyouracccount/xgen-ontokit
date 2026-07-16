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
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# TTA 15 분류(모두의말뭉치) 코드 → 온톨로지 클래스명. 모델 라벨셋 = 닫힌 집합이라
# 수동 목록 금지 원칙(무한증식) 대상 아님. 미등록 코드는 원문 코드 유지(안전).
# CV 는 TTA 표준명이 '문명(CIVILIZATION)'이지만 실제 커버가 법률·직위·음식·스포츠
# 등 제도·문화 전반(실측: 보험업법·대통령·김치·태권도가 전부 CV)이라, 온톨로지
# 클래스명으로 '문명'은 오독을 부름 → 커버 범위를 그대로 읽는 '문화·제도'로 표기.
TTA_LABEL_KO = {
    "PS": "인물", "FD": "분야", "TR": "이론", "AF": "인공물", "OG": "기관",
    "LC": "지역", "CV": "문화·제도", "DT": "날짜", "TI": "시간", "QT": "수량",
    "EV": "사건", "AM": "동물", "PT": "식물", "MT": "물질", "TM": "용어",
}

# NER 입력 글자수 상한. 모델 한계는 512 "토큰"이고 transformers 가 자동 truncate
# 하므로 crash 방어용이 아니라 연산량 상한이다. 한국어 ~2.2자/토큰이라 1200자
# ≈ 512토큰 근방 — 이전 512"자" 절단은 과잉 보수로 청크 후반부 엔티티를 통째
# 버렸다(0711 적대리뷰 MED). 1200자 초과분은 여전히 커버 밖(정직한 한계).
MAX_NER_CHARS = 1200

# 엔티티 최소 신뢰도. aggregation_strategy="simple" 은 저확신 오탐도 그대로 방출한다
# (실측: "…대한민국의 전자 회사이다"의 수식어 '전자'가 OG score 0.28 로 통과 →
#  관계 그래프에서 '삼성전자'와 별개 '전자' 노드로 파편화, e2e 빌드 검증서 적발).
# 0.40 은 오탐만 컷하는 보수값 — 코퍼스 실측상 진짜 엔티티 최저가 0.50(메신저)이라
# 0.40 미만은 사실상 오탐뿐. 정밀도 우선 원칙(파일 상단 주석)의 신뢰도 축 구현이다.
# env ONTOKIT_NER_MIN_SCORE 로 조정(도메인별 재보정 가능).
DEFAULT_MIN_SCORE = 0.40


class KoElectraNER:
    """HF NER 파이프라인 래핑. 지연 로드(사용 안 하면 안 깔림)."""

    DEFAULT_MODEL = "Leo97/KoELECTRA-small-v3-modu-ner"

    def __init__(self, model: Optional[str] = None, pipeline=None):
        self._pipe = pipeline
        self._model = model or self.DEFAULT_MODEL
        import os
        try:
            self._min_score = float(os.getenv("ONTOKIT_NER_MIN_SCORE", DEFAULT_MIN_SCORE))
        except ValueError:
            self._min_score = DEFAULT_MIN_SCORE
        # 동시 빌드 2개가 같은 인스턴스를 서로 다른 to_thread 워커에서 쓸 때
        # HF fast tokenizer(Rust)의 "Already borrowed" 충돌·모델 이중 로드를 방지
        # (0711 적대리뷰 HIGH — factory 가 단일 인스턴스를 전 빌드에 공유).
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._pipe is None:
                from transformers import pipeline as hf_pipeline  # lazy — extras[ner]
                self._pipe = hf_pipeline("ner", model=self._model,
                                         aggregation_strategy="simple")

    def _to_dicts(self, ents, source_chunks: list[str]) -> list[dict]:
        """HF 파이프라인 원시 출력 → 엔티티 dict (2자 미만 파편 컷 + 한글 클래스명)."""
        out = []
        for e in ents or []:
            w = (e.get("word", "") or "").replace("##", "").strip()
            if len(w) >= 2 and float(e.get("score", 1.0)) >= self._min_score:
                g = e.get("entity_group", "ENTITY")
                # start/end 오프셋 보존 (R13) — 스팬-형태소 경계 정렬·윈도우 dedup 의
                # 선결 재료. HF 파이프라인이 주는 원문 문자 오프셋 그대로.
                out.append({"entity": w, "class": TTA_LABEL_KO.get(g, g),
                            "type": "INSTANCE", "source_chunks": source_chunks,
                            "start": e.get("start"), "end": e.get("end")})
        return out

    def entities(self, text: str, *, source_chunks: list[str],
                 max_len: int = MAX_NER_CHARS) -> list[dict]:
        self._ensure()
        try:
            with self._lock:
                ents = self._pipe(text[:max_len])
        except Exception:
            logger.warning("NER 단건 추론 실패 — 해당 청크 엔티티 생략", exc_info=True)
            return []
        out = self._to_dicts(ents, source_chunks)
        # R13-1 경량 2패스 — max_len 초과 청크의 후반부(절단 사각) 1회 추가 추론.
        # 실측: 미탐 GT 112표본에서 2패스가 +17건 회수(전반부만은 +8). 문자열 기준
        # union(오프셋은 전반부 기준만 유효 — 후반부 방출분은 오프셋 무효화).
        # 기본 off (0717 G-A): 후반부는 표·참고문헌 잔재 구간이라 파편 다발
        # ('년 4월 27일'·'urse' — 블라인드 2인 오탐 40~61% 실증). GT 회수 +17건의
        # 가치는 실재하므로 위생 게이트 결합·재채점 후 재판정 — 그 전까지 opt-in.
        import os
        if len(text) > max_len and os.getenv("ONTOKIT_NER_TWO_PASS", "0") == "1":
            try:
                with self._lock:
                    tail = self._pipe(text[max_len:max_len * 2])
                seen = {e["entity"] for e in out}
                for e in self._to_dicts(tail, source_chunks):
                    if e["entity"] not in seen:
                        e["start"] = e["end"] = None  # 후반부 오프셋은 전문 기준 아님
                        out.append(e)
            except Exception:
                pass  # 후반부 실패는 비치명 — 전반부 결과 유지
        return out

    def entities_batch(self, texts: list[str], *, source_chunks_list: list[list[str]],
                       max_len: int = MAX_NER_CHARS, batch_size: int = 32) -> list[list[dict]]:
        """여러 청크를 배치 forward 로 추론 — CPU 실측 단건 891ms→배치 430ms/청크.

        청크별 entities() 반복은 forward 를 청크 수만큼 개별 실행해 대용량(2만 청크)
        에서 완주 불가(0710 mixed20k 실측: 단건 297분 vs 배치 143분 추정).
        반환: texts 와 같은 순서의 리스트-of-리스트(i번째 = texts[i]의 엔티티).

        실패 격리 = 서브배치 단위: batch_size 조각으로 나눠 추론하고, 조각이
        실패하면 그 조각만 단건 폴백으로 재시도한다. 이전 구현은 그룹 전체(수백
        청크)를 단일 try 에 넣어 텍스트 1개의 예외가 그룹 전량을 무성 소실시켰다
        (0711 적대리뷰 HIGH — "문서 500 조용한 누락"과 동형 함정). 폴백의 단건
        실패는 entities() 가 로그 남기고 빈 리스트 반환(청크 단위 격리 복원)."""
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
                    "NER 배치 실패(%d~%d) — 단건 폴백", start, start + len(chunk),
                    exc_info=True)
                for j, t in enumerate(chunk):
                    results[start + j] = self.entities(
                        t, source_chunks=source_chunks_list[start + j],
                        max_len=max_len)
        return results
