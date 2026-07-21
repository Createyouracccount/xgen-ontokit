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
import re
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

# 개체쌍 폭발 방지 — 한 청크에서 만들 최대 쌍 수(문장 스코프 합산, 청크당). 대용량 빌드 보호.
# rel2(0721): 쌍 생성이 문장 스코프로 바뀌어 "먼 쌍" 자체가 후보에서 빠진다 — 과거
# "상한 삭감=관계 -26%" 실측은 청크 전조합 시절 수치로, 문장 스코프에서는 상한 도달
# 자체가 드묾. 상한 의미는 이제 병리적 밀집 문장 보호용 안전핀. 60 유지(청크당).
MAX_PAIRS_PER_CHUNK = 60
# 인코더 forward 배치 크기 — self._pipe 에 미지정 시 HF 는 1건씩 순차 추론(CPU 손해).
# NER(entities_batch=32)과 동일하게 배치화. mixed20k 실측: 밀집청크 batch=1 82.6초→
# batch=32 40.0초(2.07배), 관계 결과 불변. 상한 삭감(관계 손실) 대신 이걸로 속도 확보.
ENC_BATCH_SIZE = 32
# 인코더 입력 토큰 상한(학습 MAX_LEN=180과 일치).
MAX_LEN = 180

# ── rel2 방출 게이트 (설계 eval_runs/typing/rel2_design_v2.md, 패널 채택 86/100) ──
# 실측 근거: ui_news100 잔여표본 96 정밀도 8.6% → G4(문장)+G5(트리거) 후 41%(CORRECT 손실 0),
# 차단분 층화 30 오살 0/30. 전부 결정론·LLM 0콜.
# 기본 conf 하한(G3) — 분포 중첩(FABRICATED 중앙값 0.878)으로 판별력 낮음이 공시된 보조
# 게이트. 저확신 꼬리(방향/술어 오류 0.66~0.71)만 제거. CORRECT min 0.498 대비 0.5.
DEFAULT_MIN_SCORE = 0.5

# NER(TTA/modu-ner)이 실제로 방출하는 확정 클래스 집합 — 여기 없는 클래스(또는 무클래스)는
# '미상'으로 간주해 게이트를 **통과**시킨다(렌즈B R1: 게이트는 확증된 모순만 차단.
# 약타이핑 코퍼스에서 미상을 죽이면 정탐 학살).
_KNOWN_CLASSES = frozenset({
    "인물", "기관", "지역", "날짜", "시간", "수량", "문화·제도", "인공물", "용어",
    "사건", "동물", "식물", "이론", "물질",
})
# 술어군별 목적어 허용 클래스(확정 클래스가 이 집합 밖이면 차단; 미상은 통과).
_OBJ_ALLOW: dict[str, frozenset] = {}
for _p in ("org:top_members/employees", "org:founded_by", "per:parents", "per:children",
           "per:siblings", "per:spouse", "per:other_family", "per:colleagues"):
    _OBJ_ALLOW[_p] = frozenset({"인물", "기관"})
for _p in ("org:dissolved", "org:founded", "per:date_of_birth", "per:date_of_death"):
    _OBJ_ALLOW[_p] = frozenset({"날짜", "시간"})
for _p in ("org:place_of_headquarters", "per:place_of_birth", "per:place_of_death",
           "per:place_of_residence"):
    # 기관 허용 — 병원 출생지·사옥 등 관용(렌즈C R6)
    _OBJ_ALLOW[_p] = frozenset({"지역", "기관"})
for _p in ("per:origin",):
    _OBJ_ALLOW[_p] = frozenset({"지역", "문화·제도"})
for _p in ("per:employee_of", "per:schools_attended", "org:member_of", "org:members"):
    _OBJ_ALLOW[_p] = frozenset({"기관", "지역"})
for _p in ("org:number_of_employees/members",):
    _OBJ_ALLOW[_p] = frozenset({"수량"})
for _p in ("per:title",):
    _OBJ_ALLOW[_p] = frozenset({"문화·제도", "용어"})

# G2: 상대 시간표현 목적어 — 문서 날짜 앵커 없인 사실 성립 불능(앵커불능). 절대
# 일자(1969년, 5월 4일, 2026년 1분기)는 보존 — 단독 "N분기"만 차단.
_REL_TIME = re.compile(
    r"^(이달|이번 ?[주달해]|지난 ?[주달해]|다음 ?[주달해]|올해|지난해|작년|내년"
    r"|올 ?상반기|올 ?하반기|오전|오후|이날|당시|이후|이전|최근|당일|어제|오늘|내일"
    r"|하반기|상반기|분기|[1-4] ?분기|지난|.*개월 ?동안?|두 ?달.*|.*[년달] ?만)$")

# G5: 술어 트리거 어휘(어간 매칭 — 활용형 흡수). 해당 문장에 어간이 하나도 없으면 방출
# 거부. 미정의 술어는 통과(보수 — 순차단 전용이라 최악이 '개선 없음').
# ⚠️검증 라이더(주심 재심): 이 목록은 코드 동결 — 검증 라운드 중 수정 금지, 단위테스트
# 는 이 상수 기준. env 오버라이드 없음(배포 확장은 차기, 재현성 우선).
_TRIGGER: dict[str, tuple] = {
    "org:top_members/employees": ("대표", "회장", "사장", "부회장", "이사", "선임", "임명",
                                  "취임", "기자", "총재", "원장", "위원장", "장관", "교수",
                                  "CEO", "연구원", "근무", "재직", "일하", "합류"),
    "per:employee_of": ("대표", "회장", "소속", "재직", "근무", "일하", "합류", "기자",
                        "장관", "교수", "원장", "위원장", "연구원", "임명", "선임", "취임"),
    "per:title": ("대표", "회장", "사장", "부회장", "장관", "총재", "기자", "교수",
                  "위원장", "원장", "후보", "대통령", "의원", "이사", "CEO"),
    "org:member_of": ("소속", "산하", "계열", "자회사", "가입", "일원", "그룹", "편입"),
    "org:members": ("소속", "산하", "계열", "자회사", "가입", "일원", "그룹", "편입"),
    "org:founded": ("설립", "창립", "창업", "출범", "개원", "발족", "세우", "세웠",
                    "차리", "차렸", "만들", "일으"),
    "org:founded_by": ("설립", "창립", "창업", "출범", "세우", "세웠", "차리", "차렸",
                       "만들", "일으"),
    "org:dissolved": ("해산", "해체", "폐업", "청산", "파산", "폐지", "문닫", "문을 닫"),
    "per:schools_attended": ("졸업", "입학", "수학", "출신", "학사", "석사", "박사",
                             "다니", "다녔", "나오", "나왔"),
    "org:place_of_headquarters": ("본사", "본점", "사옥", "소재", "위치"),
    "per:origin": ("출신", "국적", "태생"),
    "per:place_of_birth": ("출생", "태어"),
    "org:product": ("출시", "제품", "서비스", "생산", "판매", "개발", "선보"),
    "per:children": ("아들", "딸", "자녀", "장남", "장녀", "차남", "차녀"),
    "per:parents": ("아버지", "어머니", "부친", "모친", "아들", "딸"),
    "per:siblings": ("동생", "형제", "자매", "누나", "언니"),
    "per:spouse": ("아내", "남편", "부인", "배우자", "결혼"),
}

# 폴백 문장분할 예외(렌즈B R3): ⑴숫자. 숫자("1969. 5. 4.") ⑵영문 약어("Inc.", "U.S.")
# — Kiwi 가용 시 Kiwi 가 정본, 이 정규식은 폴백 전용.
_SENT_SPLIT_FALLBACK = re.compile(
    r"(?<![0-9])(?<![A-Za-z])(?<=다)\.(?=\s)|(?<![0-9A-Za-z])(?<=[!?])\s")
_KIWI_HOLDER: dict = {}


def _split_sentences(text: str) -> list[str]:
    """문장 분할 — Kiwi 우선(빌더 기존 의존성), 실패 시 보수적 정규식 폴백."""
    if "kiwi" not in _KIWI_HOLDER:
        try:
            from kiwipiepy import Kiwi  # lazy — extras[korean]
            _KIWI_HOLDER["kiwi"] = Kiwi()
        except Exception:
            _KIWI_HOLDER["kiwi"] = None
    kiwi = _KIWI_HOLDER["kiwi"]
    if kiwi is not None:
        try:
            return [s.text for s in kiwi.split_into_sents(text) if s.text.strip()]
        except Exception:
            pass
    return [s for s in _SENT_SPLIT_FALLBACK.split(text) if s and s.strip()]


def _gate(label: str, s_cls: str, o_cls: str, obj_surface: str, sentence: str) -> Optional[str]:
    """rel2 방출 게이트 — 차단 사유명 반환(통과 시 None). 확증된 모순만 차단(미상 통과).

    S: 술어 접두-주어 타입 교차 오류(PER 주어에 org:*, ORG/LOC 주어에 per:*).
       국가류(지역) 주어는 org:* 허용(렌즈B R2; 현 쌍생성은 PER/ORG만이라 실발화 없음 — 계약 명시).
    T: 상대 시간표현 목적어(G2). O: 술어별 목적어 서명(G1'). G5: 술어 트리거 어휘.
    """
    if label.startswith("per:") and s_cls in _KNOWN_CLASSES and s_cls != "인물":
        return "S"
    if label.startswith("org:") and s_cls in _KNOWN_CLASSES and s_cls not in ("기관", "지역"):
        return "S"
    if _REL_TIME.match(obj_surface.strip()):
        return "T"
    allow = _OBJ_ALLOW.get(label)
    if allow and o_cls in _KNOWN_CLASSES and o_cls not in allow:
        return "O"
    cues = _TRIGGER.get(label)
    if cues and not any(c in sentence for c in cues):
        return "G5"
    return None


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
                 pipeline=None, min_score: float = DEFAULT_MIN_SCORE):
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

    def _pairs(self, entities: list[dict], sentences: list[str]) -> list[tuple]:
        """청크 인스턴스 → 관계 후보 개체쌍 — **문장 스코프**(rel2 G4).

        KLUE-RE 는 문장 단위 태스크. 청크(다문장) 전조합은 학습 분포 밖(OOD) 쌍을
        인코더에 던져 고확신 날조를 양산했다(ui_news100 실측: FABRICATED 66%가
        문장 경계 밖 쌍). 같은 문장에 동시 출현하는 쌍만 후보로 생성하고, 마킹
        입력도 그 문장으로 한다. subj∈{PER,ORG}(학습 마커 분포 유지 — 지역 주어
        허용은 게이트 계약에만 명시, 쌍 미생성이라 실발화 없음). 상한은 청크당.
        반환: (sw, stype, ow, otype, s_cls, o_cls, sentence)"""
        typed = []
        seen_e = set()
        for e in entities:
            w = e.get("entity")
            if not w or w in seen_e:
                continue
            seen_e.add(w)
            typed.append((w, e.get("class", "") or ""))
        pairs = []
        for sent in sentences:
            within = [(w, c) for w, c in typed if w in sent]
            for i, (sw, s_cls) in enumerate(within):
                stype = _klue_type(s_cls)
                if stype not in _SUBJ_TYPES:
                    continue
                for j, (ow, o_cls) in enumerate(within):
                    if i == j or sw == ow:
                        continue
                    pairs.append((sw, stype, ow, _klue_type(o_cls), s_cls, o_cls, sent))
                    if len(pairs) >= MAX_PAIRS_PER_CHUNK:
                        return pairs
        return pairs

    def extract(self, text: str, *, source_chunks: list[str]) -> list[dict]:
        """청크 → 관계 dict 리스트(kg_builder 소비 스키마, 규칙 채널과 동일).

        반환: [{subject, predicate, object, predicate_type='ObjectProperty',
                source_chunks, relation_label, score}, ...]
        NER(self._ner) 없으면 빈 리스트(개체쌍 불가). no_relation 은 미방출."""
        if not text or not text.strip() or self._ner is None:
            return []
        ents = self._ner.entities(text, source_chunks=source_chunks)
        sentences = _split_sentences(text)
        pairs = self._pairs(ents, sentences)
        if not pairs:
            return []
        self._ensure()
        marked, meta = [], []
        for sw, st, ow, ot, s_cls, o_cls, sent in pairs:
            # G4: 마킹 입력도 청크 전체가 아니라 해당 문장 — 학습(문장) 분포 정합.
            m = _mark(sent, sw, st, ow, ot)
            if m is not None:
                marked.append(m[:MAX_LEN * 4])  # 넉넉히 자른 뒤 토크나이저가 토큰 절단
                meta.append((sw, ow, s_cls, o_cls, sent))
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
        cuts = {"S": 0, "T": 0, "O": 0, "G5": 0, "conf": 0}
        for (sw, ow, s_cls, o_cls, sent), pred in zip(meta, preds):
            top = pred[0] if isinstance(pred, list) else pred
            label = top.get("label", "")
            score = float(top.get("score", 0.0))
            # HF 라벨이 'LABEL_10' 형식이면 인덱스 파싱
            if label.startswith("LABEL_"):
                label = KLUE_RE_LABELS[int(label.split("_")[1])]
            if label == "no_relation":
                continue
            if score < self._min_score:
                cuts["conf"] += 1
                continue
            # rel2 방출 게이트(G1'/G2/G5) — 확증된 모순만 차단, 미상 통과.
            g = _gate(label, s_cls, o_cls, ow, sent)
            if g:
                cuts[g] += 1
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
        # 관측성(설계 검증 게이트) — 채널별 컷 카운트. 재빌드 감사에서 게이트 발화 재현용.
        if any(cuts.values()):
            logger.info("[rel-gate] cuts=%s kept=%d", cuts, len(out))
        return out
