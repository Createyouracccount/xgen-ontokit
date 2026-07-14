"""한국어 관계추출 인코더 — KLUE-RE 파인튜닝 로컬 인코더. extras[relation-encoder].

외부 gold(KLUE-RE, CC BY-SA 4.0) 심판루프 90/100 검증(2026-07-14). holdout micro-F1
0.5924 = KLUE 공식 RoBERTa-small baseline 60.89 정합. 규칙 조사SVO(KLUE 연결력 0.8%)
대비 근본 개선. 정본 docs/ontokit_관계_KLUE-RE_인코더_심판루프_90_2026_07_14.md,
학습 재현 eval/relation/train_encoder.py.

불변식(빌더 철학): **LLM API 호출 0회.** 이 인코더는 로컬 추론만(NER modu-ner 전례와
동일 계열). transformers+torch 는 extras 라 미설치 시 import 실패 → 상위(deterministic_ko)
가 규칙 채널로 폴백. 즉 "설치 안 하면 안 켜진다"가 불변.

입력: (문장, subject, object) 개체쌍 → 30개 KLUE 관계 라벨. NER(KoElectra)이 인스턴스를
주므로, 이 채널은 한 청크 내 개체쌍을 조합해 관계 타입을 분류한다. typed entity marker
방식(학습과 동일): "[S:ORG] 금호고속 [/S] [O:PER] 이덕연 [/O] 사장 …".

모델 교체(나중에 다른 모델로): 이 클래스는 본체 관계 채널 계약
(.extract(text, *, source_chunks) -> list[dict])을 구현하는 드롭인 채널 중 하나다.
  ① 같은 KLUE 형식 다른 가중치 → env ONTOKIT_RELATION_ENCODER_MODEL 만 변경(로컬경로/HF id).
     조건: 30 KLUE 라벨 출력 + 위 typed marker 입력 이해.
  ② 더 큰 모델 → eval/relation/train_encoder.py MODEL 한 줄 바꿔 재학습.
  ③ 코드 직접 주입 → DeterministicKoreanExtractor(relation_extractor=enc, ner=...)(env보다 우선).
  ④ 다른 라벨 체계·방식 → .extract() 계약으로 감싼 새 클래스를 ③처럼 주입.
정본: eval/relation/README.md '모델 교체' 절.
"""
from __future__ import annotations
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# KLUE-RE 30 관계(공식 순서). id→라벨. no_relation(0)은 미방출.
KLUE_RE_LABELS = [
    "no_relation", "org:dissolved", "org:founded", "org:place_of_headquarters",
    "org:alternate_names", "org:member_of", "org:members",
    "org:political/religious_affiliation", "org:product", "org:founded_by",
    "org:top_members/employees", "org:number_of_employees/members",
    "per:date_of_birth", "per:date_of_death", "per:place_of_birth",
    "per:place_of_death", "per:place_of_residence", "per:origin",
    "per:employee_of", "per:schools_attended", "per:alternate_names",
    "per:parents", "per:children", "per:siblings", "per:spouse",
    "per:other_family", "per:colleagues", "per:product", "per:religion",
    "per:title",
]

# modu-ner(TTA 15) 온톨로지 클래스명 → KLUE-RE 개체 타입(subj∈{PER,ORG}, obj 6종).
# 닫힌 매핑(모델 라벨셋=유한). KoElectra TTA_LABEL_KO 한글 클래스명 기준.
#   PS 인물→PER, OG 기관→ORG, LC 지역→LOC, DT 날짜→DAT, QT 수량→NOH.
#   그 외(인공물·용어·사건 등 고유명)→POH(KLUE 기타 고유명 슬롯).
_CLASS_TO_KLUE_TYPE = {
    "인물": "PER", "기관": "ORG", "지역": "LOC", "날짜": "DAT", "수량": "NOH",
    "시간": "DAT",
    # 나머지 고유명은 POH(KLUE 'POH'=기타 고유명). 미등록도 POH 폴백.
}
# 관계 subject 는 KLUE 정의상 PER 또는 ORG 만. object 는 6종.
_SUBJ_TYPES = frozenset({"PER", "ORG"})

# 개체쌍 폭발 방지 — 한 청크에서 만들 최대 쌍 수. 대용량 빌드 보호.
# ⚠️상한 삭감은 관계 손실 직결(mixed20k 밀집청크 실측: 60→40 시 관계 -26%, 잘리는
# 쌍에서도 실관계 다수 — "먼 쌍은 희박" 가정 반증). 속도는 상한이 아니라 배치추론으로
# 사야 한다(extract_batch). 60 유지.
MAX_PAIRS_PER_CHUNK = 60
# 인코더 forward 배치 크기 — self._pipe 에 미지정 시 HF 는 1건씩 순차 추론(CPU 손해).
# NER(entities_batch=32)과 동일하게 배치화. mixed20k 실측: 밀집청크 batch=1 82.6초→
# batch=32 40.0초(2.07배), 관계 결과 불변. 상한 삭감(관계 손실) 대신 이걸로 속도 확보.
ENC_BATCH_SIZE = 32
# 인코더 입력 토큰 상한(학습 MAX_LEN=180과 일치).
MAX_LEN = 180


def _klue_type(cls: str) -> str:
    return _CLASS_TO_KLUE_TYPE.get(cls, "POH")


def _mark(sentence: str, s_word: str, s_type: str, o_word: str, o_type: str) -> Optional[str]:
    """typed entity marker 삽입(학습 train_encoder.mark 와 동일 규약).

    두 개체의 최초 출현 위치에 마커 삽입. 미출현·중첩 시 None(해당 쌍 스킵)."""
    si = sentence.find(s_word)
    oi = sentence.find(o_word)
    if si < 0 or oi < 0:
        return None
    s_span = (si, si + len(s_word) - 1)
    o_span = (oi, oi + len(o_word) - 1)
    # 중첩 개체쌍은 마커가 깨짐 — 스킵
    if not (s_span[1] < o_span[0] or o_span[1] < s_span[0]):
        return None
    spans = sorted([
        (s_span[0], s_span[1], f"[S:{s_type}]", "[/S]"),
        (o_span[0], o_span[1], f"[O:{o_type}]", "[/O]"),
    ], key=lambda x: -x[0])
    out = sentence
    for st, en, opn, cls in spans:
        out = out[:st] + f" {opn} " + out[st:en + 1] + f" {cls} " + out[en + 1:]
    return out


class KoreanRelationEncoder:
    """KLUE-RE 파인튜닝 인코더 관계 채널. 지연 로드(사용 안 하면 안 깔림).

    deterministic_ko 의 relation_extractor 인터페이스(.extract(text, source_chunks))를
    구현 — 규칙 KoreanRelationExtractor 와 드롭인 호환. NER 결과(인스턴스)를 개체쌍으로
    조합해 관계를 분류한다. NER 미주입 시 개체쌍을 만들 수 없어 빈 결과(안전).
    """

    #: 학습 가중치 경로 — 기본 없음. env ONTOKIT_RELATION_ENCODER_MODEL 로 지정.
    #: 미지정 시 이 채널을 만들지 않는다(상위가 규칙 폴백). HF hub id 또는 로컬 경로.
    ENV_MODEL = "ONTOKIT_RELATION_ENCODER_MODEL"

    def __init__(self, model: Optional[str] = None, ner=None, *,
                 pipeline=None, min_score: float = 0.0):
        self._model = model or os.getenv(self.ENV_MODEL)
        if not self._model:
            raise ValueError(
                f"관계 인코더 모델 경로 미지정 — {self.ENV_MODEL} env 또는 model 인자 필요. "
                "학습: eval/relation/train_encoder.py → model_re/")
        self._ner = ner
        self._pipe = pipeline
        self._min_score = min_score
        self._lock = threading.Lock()

    def warmup(self):
        """모델을 즉시 로드(지연 로드 강제 트리거). 경로 오류·extras 미설치를
        생성 직후 적발하려는 상위(deterministic_ko)가 호출 — 실패 시 예외 전파."""
        self._ensure()

    def _ensure(self):
        with self._lock:
            if self._pipe is None:
                # lazy — extras[relation-encoder]=transformers+torch. 미설치 시 여기서
                # ImportError → 상위(deterministic_ko)가 규칙 폴백(불변식).
                from transformers import pipeline as hf_pipeline
                self._pipe = hf_pipeline(
                    "text-classification", model=self._model,
                    tokenizer=self._model, top_k=1)

    def _pairs(self, entities: list[dict]) -> list[tuple]:
        """청크 인스턴스 → 관계 후보 개체쌍. subj∈{PER,ORG}, 근접 우선 상한."""
        typed = [(e["entity"], _klue_type(e.get("class", ""))) for e in entities
                 if e.get("entity")]
        # dedup(같은 표면형+타입)
        seen, uniq = set(), []
        for w, t in typed:
            if (w, t) not in seen:
                seen.add((w, t))
                uniq.append((w, t))
        pairs = []
        for i, (sw, stype) in enumerate(uniq):
            if stype not in _SUBJ_TYPES:
                continue
            for j, (ow, otype) in enumerate(uniq):
                if i == j or sw == ow:
                    continue
                pairs.append((sw, stype, ow, otype))
        return pairs[:MAX_PAIRS_PER_CHUNK]

    def extract(self, text: str, *, source_chunks: list[str]) -> list[dict]:
        """청크 → 관계 dict 리스트(kg_builder 소비 스키마, 규칙 채널과 동일).

        반환: [{subject, predicate, object, predicate_type='ObjectProperty',
                source_chunks, relation_label, score}, ...]
        NER(self._ner) 없으면 빈 리스트(개체쌍 불가). no_relation 은 미방출."""
        if not text or not text.strip() or self._ner is None:
            return []
        ents = self._ner.entities(text, source_chunks=source_chunks)
        pairs = self._pairs(ents)
        if not pairs:
            return []
        self._ensure()
        marked, meta = [], []
        for sw, st, ow, ot in pairs:
            m = _mark(text, sw, st, ow, ot)
            if m is not None:
                marked.append(m[:MAX_LEN * 4])  # 넉넉히 자른 뒤 토크나이저가 토큰 절단
                meta.append((sw, ow))
        if not marked:
            return []
        try:
            with self._lock:
                # batch_size 지정 — 미지정 시 HF 가 쌍을 1건씩 순차 forward(CPU 실측
                # 2배 손해). 청크 내 쌍을 배치 forward. 청크 간 배치는 이득 미미(실측
                # 1.01배) — HF 스케줄러가 이미 근사 처리, 추가 복잡도 없이 이걸로 충분.
                preds = self._pipe(marked, truncation=True, max_length=MAX_LEN,
                                   batch_size=ENC_BATCH_SIZE)
        except Exception:
            logger.warning("관계 인코더 추론 실패 — 청크 관계 생략", exc_info=True)
            return []
        out, seen = [], set()
        for (sw, ow), pred in zip(meta, preds):
            top = pred[0] if isinstance(pred, list) else pred
            label = top.get("label", "")
            score = float(top.get("score", 0.0))
            # HF 라벨이 'LABEL_10' 형식이면 인덱스 파싱
            if label.startswith("LABEL_"):
                label = KLUE_RE_LABELS[int(label.split("_")[1])]
            if label == "no_relation" or score < self._min_score:
                continue
            key = (sw, label, ow)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "subject": sw, "predicate": label, "object": ow,
                "predicate_type": "ObjectProperty", "source_chunks": source_chunks,
                "relation_label": label, "score": round(score, 4),
            })
        return out
