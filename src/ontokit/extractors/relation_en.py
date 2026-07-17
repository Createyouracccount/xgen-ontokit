"""영어 관계 추출 — spaCy 의존 파싱 SVO (R-en-1, LLM 0콜).

한국어 조사 SVO 와 동형의 계약(extract(text) → [{"subject","predicate","object"}]).
채널: ①nsubj–VERB–dobj (능동 SVO) ②nsubjpass–VERB–agent/pobj (수동 복원)
③nsubj–VERB–prep–pobj (전치사 목적어 — 술어에 전치사 표기).
게이트(전부 문법/구조): 논항은 개체성 명사구(PROPN 포함 or 2단어+ NP)만,
대명사 주어 컷, be/have 조동사 컷(계사문은 정의문 채널 소관).
extras[english-relations] = spacy + en_core_web_sm (미설치 시 생성자에서 ImportError).
"""
from __future__ import annotations
import threading

# 계사·경조동사 — 관계 술어로 무가치(폐집합)
_STOP_VERBS = {"be", "have", "do", "make", "take", "get", "become", "seem"}


class EnglishRelationExtractor:
    def __init__(self, nlp=None, model: str = "en_core_web_sm"):
        if nlp is None:
            import spacy
            nlp = spacy.load(model, disable=["lemmatizer"] if False else [])
        self.nlp = nlp
        self._lock = threading.Lock()

    @staticmethod
    def _np(tok) -> str:
        """토큰의 명사구(subtree 중 핵심부) — 관사·한정사 제외한 연속 구."""
        words = [t for t in tok.subtree if t.dep_ not in ("det", "punct", "prep", "pobj", "relcl", "appos", "acl")
                 and abs(t.i - tok.i) <= 4]
        words = sorted(words, key=lambda t: t.i)
        # 연속 구간만(관계절 등으로 끊기면 head 쪽)
        run, cur = [], []
        for t in words:
            if cur and t.i != cur[-1].i + 1:
                if tok in cur:
                    break
                cur = []
            cur.append(t)
        run = cur if tok in cur else [tok]
        return " ".join(t.text for t in run).strip()

    @staticmethod
    def _arg_ok(tok, phrase: str) -> bool:
        if tok.pos_ == "PRON" or not phrase:
            return False  # 대명사·공백 논항 컷
        # 절단 파편 컷(합의 채점 실증): 'dangers such'(such-as 절단), "'s central …"
        # (소유 클리틱 선도), '… that'(보문소 꼬리) — 전부 폐형식 문법 표지.
        if phrase.endswith((" such", " that")) or phrase.startswith(("'s ", "'s")):
            return False
        # 개체성: 고유명사 포함 or 2단어 이상 명사구 (단어 하나짜리 보통명사는 잡음)
        has_propn = any(t.pos_ == "PROPN" for t in tok.subtree if abs(t.i - tok.i) <= 4)
        return has_propn or len(phrase.split()) >= 2

    def extract(self, text: str, source_chunks=None) -> list[dict]:
        if not text:
            return []
        with self._lock:
            doc = self.nlp(text[:2000])
        out, seen = [], set()
        for tok in doc:
            if tok.pos_ != "VERB" or tok.lemma_.lower() in _STOP_VERBS:
                continue
            subs = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
            if not subs:
                continue
            sub = subs[0]
            objs = [(c, tok.lemma_) for c in tok.children if c.dep_ == "dobj"]
            # 수동: nsubjpass + by-agent → (agent, verb, subject) 복원
            if sub.dep_ == "nsubjpass":
                for prep in (c for c in tok.children if c.dep_ == "agent"):
                    for pobj in (c for c in prep.children if c.dep_ == "pobj"):
                        sp, op = self._np(pobj), self._np(sub)
                        if self._arg_ok(pobj, sp) and self._arg_ok(sub, op):
                            key = (sp, tok.lemma_, op)
                            if key not in seen:
                                seen.add(key)
                                out.append({"subject": sp, "predicate": tok.lemma_, "object": op})
                continue
            # 전치사 목적어: verb+prep 를 술어로. 시간·날짜 pobj 는 부가어(adjunct)라
            # 관계 목적어가 아님 — 'invade in June' 류 유령(합의 채점 실증) 컷.
            for prep in (c for c in tok.children if c.dep_ == "prep"):
                for pobj in (c for c in prep.children if c.dep_ == "pobj"):
                    if pobj.ent_type_ in ("DATE", "TIME"):
                        continue
                    objs.append((pobj, f"{tok.lemma_} {prep.text.lower()}"))
            sp = self._np(sub)
            if not self._arg_ok(sub, sp):
                continue
            for obj, pred in objs:
                op = self._np(obj)
                if not self._arg_ok(obj, op) or op == sp:
                    continue
                key = (sp, pred, op)
                if key not in seen:
                    seen.add(key)
                    out.append({"subject": sp, "predicate": pred, "object": op})
        return out
