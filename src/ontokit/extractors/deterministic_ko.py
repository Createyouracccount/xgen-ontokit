"""한국어·영어 LLM-free 온톨로지 추출기 — Extractor 프로토콜 구현.

Kiwi 복합명사(ko)+nltk 명사구(en) 클래스 + 언어별 NER 엔티티 + 접미공유 subClassOf
계층(ko=문자, en=단어 단위). 관계(조사 SVO)는 한국어만. LLM 0회.
finreg 489 실측: 4.5초/$0, 클래스 3156·subClassOf 1710. 검색 A/B에서 gpt-4o와 동일(0.947).

XGEN pipeline은 이것을 gpt-4o DocumentOntologyExtractor 대신 주입 가능(같은 4-tuple 계약).
"""
from __future__ import annotations
import asyncio
import re
import threading
from typing import Optional

from ..morphology.kiwi_nouns import KiwiNounExtractor
from ..hierarchy.suffix_share import induce_suffix_hierarchy
from ..utils.lang_detect import detect_lang

_HANGUL = re.compile(r"[가-힣]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")  # 라틴 2자+ 연속 (단독 기호·항번호 제외)


class DeterministicKoreanExtractor:
    def __init__(self, kiwi=None, ner=None, domain_words: Optional[list[str]] = None,
                 en_nouns=None, en_ner=None, relation_extractor=None,
                 enable_relations: bool = True, auto_english: bool = True):
        """kiwi: Kiwi 인스턴스(없으면 생성, extras[korean]).
        ner: KoElectraNER 인스턴스(None이면 한국어 엔티티 추출 생략, extras[ner]).
        domain_words: 사용자사전 도메인 용어(한국어 Kiwi 사용자사전 + 영어 단일명사 허용목록).
        en_nouns: EnglishNounExtractor(None이면 auto_english 에 따라 자동 배선, extras[english]).
        en_ner: EnglishNER(None이면 영어 엔티티 추출 생략, extras[ner]).
        auto_english: True(기본)면 nltk 설치 시 en_nouns 자동 생성 — 영어 클래스가
          별도 주입 없이 나온다. en_ner 는 torch 모델 로드가 무거워 자동화하지 않음(명시 주입만).

        혼합 코퍼스(한국어+영어): 명사(클래스)는 한·영 추출기를 **둘 다** 실행해
        혼합 청크의 소수언어 용어("Basel III 규제"의 Basel)를 보존한다(v0.6, 자연 직교 —
        Kiwi 는 라틴 무시, en_nouns 는 한글 무시). NER 는 비용(모델 forward) 때문에
        지배언어 라우팅 유지. 관계는 조사 기반이라 한국어 청크만."""
        self.nouns = KiwiNounExtractor(kiwi, domain_words)
        self.ner = ner
        self.en_nouns = en_nouns
        if self.en_nouns is None and auto_english:
            # extras[english](nltk) 설치돼 있으면 영어 명사추출 자동 배선(#4 auto-wire).
            import importlib.util
            if importlib.util.find_spec("nltk") is not None:
                from ..morphology.en_nouns import EnglishNounExtractor
                self.en_nouns = EnglishNounExtractor(domain_words)
        self.en_ner = en_ner
        # 한국어 관계(objectProperty) 추출 — 조사 기반 SVO. Kiwi 인스턴스 공유.
        self.relations = None
        if enable_relations:
            if relation_extractor is not None:
                self.relations = relation_extractor
            else:
                from .relation_ko import KoreanRelationExtractor
                self.relations = KoreanRelationExtractor(kiwi=self.nouns.kiwi)
        # 동시 빌드 2개가 같은 extractor 인스턴스(factory 공유)를 서로 다른
        # to_thread 워커에서 돌릴 때 Kiwi 동시 호출을 직렬화(스레드 안전성 미보장
        # 방어, 0711 적대리뷰 HIGH). NER 는 자체 락 보유 — 이 락은 형태소·관계 커버.
        self._lock = threading.Lock()

    @staticmethod
    def _run_ner_batched(ner, buf: list[tuple[str, str, list]],
                         all_entities: dict[str, list]) -> None:
        """수집된 (doc_name, text, sc) 버퍼를 배치 NER 로 추론해 all_entities 에 병합.

        entities_batch 미구현 NER(커스텀 주입)은 청크별 entities() 폴백 — 결과 동일,
        속도만 단건. 내장 KoElectraNER/EnglishNER 는 배치 경로."""
        if ner is None or not buf:
            return
        texts = [t for _, t, _ in buf]
        scs = [sc for _, _, sc in buf]
        batch_fn = getattr(ner, "entities_batch", None)
        if batch_fn is not None:
            results = batch_fn(texts, source_chunks_list=scs)
            # 커스텀 주입 NER 의 배치 구현이 길이를 어길 수 있음 — zip 이 꼬리를
            # 무성 절단하지 않도록 검증 후 단건 폴백(0711 리뷰 PLAUSIBLE 방어).
            if len(results) != len(buf):
                results = [ner.entities(t, source_chunks=sc)
                           for t, sc in zip(texts, scs)]
        else:
            results = [ner.entities(t, source_chunks=sc) for t, sc in zip(texts, scs)]
        for (doc_name, _, _), ents in zip(buf, results):
            if ents:
                all_entities.setdefault(doc_name, []).extend(ents)

    async def extract(
        self,
        documents: dict[str, list[dict]],
        *,
        domain: str = "",
        existing: Optional[dict] = None,
    ) -> tuple[dict, dict, list, list]:
        """비동기 진입점 — CPU 본문을 워커 스레드로 격리.

        본문(Kiwi 형태소·NER 추론)은 100% 동기 CPU 라, 이벤트 루프에서 직접 돌면
        그룹(100청크)당 수십 초씩 루프가 통째로 멈춘다(0710 mixed20k 실증: 폴링·
        헬스체크·로그 전부 마비 → "NER CPU 추론 무한정지"로 job failed). to_thread
        격리로 서버는 살아있고, 호출측(pipeline)의 그룹 사이 취소 체크도 동작한다.
        Kiwi·torch 는 C 확장이라 추론 중 GIL 을 놓아 메인 루프 응답성이 유지된다."""
        return await asyncio.to_thread(
            self._extract_sync, documents, domain=domain, existing=existing)

    def _extract_sync(
        self,
        documents: dict[str, list[dict]],
        *,
        domain: str = "",
        existing: Optional[dict] = None,
    ) -> tuple[dict, dict, list, list]:
        # 인스턴스 락 — 동시 빌드가 공유 extractor 를 쓸 때 전체 추출을 직렬화.
        # (병렬성 손실보다 Kiwi/tokenizer 동시 호출 미정의 동작이 훨씬 위험)
        with self._lock:
            return self._extract_impl(documents, domain=domain, existing=existing)

    def _extract_impl(
        self,
        documents: dict[str, list[dict]],
        *,
        domain: str = "",
        existing: Optional[dict] = None,
    ) -> tuple[dict, dict, list, list]:
        all_entities: dict[str, list] = {}
        all_relations: list = []
        all_data_props: list = []

        # 클래스 이름 → source_chunks(set) 딕셔너리 누적 — 매 청크 merge_concepts(O(T·C),
        # 내부 리스트 선형탐색까지 겹쳐 사실상 제곱)를 폐기. 청크당 O(1) dict 갱신 후
        # 루프 밖에서 1회 리스트화. existing(이어서 빌드) 클래스도 이 dict 로 흡수한다.
        class_chunks: dict[str, set] = {}
        if existing:
            for c in existing.get("classes", []):
                nm = c.get("name")
                if nm:
                    class_chunks.setdefault(nm, set()).update(c.get("source_chunks", []))

        # en 라우팅 침묵 방지(#5) — 영어 청크인데 영어 도구 미주입이면 조용히 스킵되던 것을
        # stats 로 노출. "문서 500 조용한 누락" 류 함정 재발 차단.
        skipped_en_chunks = 0

        # NER 배치 수집 버퍼 — 청크별 단건 forward(CPU 891ms/청크, 2만 청크=297분)를
        # 언어별로 모아 배치 forward(430ms/청크)로 바꾼다. (doc_name, text, sc) 튜플.
        ko_ner_buf: list[tuple[str, str, list]] = []
        en_ner_buf: list[tuple[str, str, list]] = []

        for doc_name, chunks in documents.items():
            for ch in chunks:
                cid = ch.get("chunk_id")
                text = ch.get("chunk_text", "")
                if not text.strip():
                    continue
                sc = [cid] if cid else []
                lang = detect_lang(text)
                # ① 명사→클래스: 한·영 이중 추출 (v0.6) — 혼합 청크의 소수언어 용어 보존.
                #   Kiwi 는 라틴 무시, en_nouns 는 한글 무시라 자연 직교(간섭·중복 없음).
                #   v0.5 까지는 ko/en 택1 라우팅이라 "Basel III 규제" 류 혼합청크에서
                #   소수언어 용어가 통째 소실됐다(0710 실측).
                nouns: list[str] = []
                if _HANGUL.search(text):
                    nouns += self.nouns.compound_nouns(text)
                if self.en_nouns is not None:
                    if _LATIN_WORD.search(text):
                        nouns += self.en_nouns.compound_nouns(text)
                elif lang == "en":
                    # en-지배 청크인데 영어 도구 없음 — 영어 "명사추출" 스킵 집계.
                    # (한글이 섞여 있으면 한국어 leg 는 동작했을 수 있음 — "통째 스킵" 아님)
                    skipped_en_chunks += 1
                for n in nouns:
                    class_chunks.setdefault(n, set()).update(sc)
                # ② NER 수집: 지배언어 라우팅 유지 — 모델 forward 가 비용 지배적이라
                #   이중 실행하지 않음(혼합청크의 소수언어 엔티티는 커버 밖, README 명시).
                if lang == "en":
                    if self.en_ner is not None:
                        en_ner_buf.append((doc_name, text, sc))
                elif self.ner is not None:
                    ko_ner_buf.append((doc_name, text, sc))
                # ③ 관계(objectProperty) — 조사 기반 SVO. 한글이 있으면 실행(지배언어
                #   무관) — en-지배 혼합청크의 한국어 문장 관계가 통째 소실되던 것 수정
                #   (0711 리뷰: 클래스는 이중추출로 살리면서 관계만 버리는 비일관).
                #   조사(JKS/JKO)는 한글 문장에만 나타나므로 영어 문장엔 원리적으로 무해.
                if self.relations is not None and _HANGUL.search(text):
                    rels = self.relations.extract(text, source_chunks=sc)
                    if rels:
                        all_relations.extend(rels)

        # ② NER → 인스턴스 엔티티 — 언어별 배치 forward 1회.
        self._run_ner_batched(self.ner, ko_ner_buf, all_entities)
        self._run_ner_batched(self.en_ner, en_ner_buf, all_entities)

        # 루프 밖 1회 리스트화 — merged 스키마 구성.
        merged = {
            "classes": [{"name": nm, "description": "", "parent": None,
                         "source_chunks": list(chunks)}
                        for nm, chunks in class_chunks.items()],
            "object_properties": list(existing.get("object_properties", [])) if existing else [],
            "datatype_properties": list(existing.get("datatype_properties", [])) if existing else [],
            "class_hierarchy": list(existing.get("class_hierarchy", [])) if existing else [],
        }
        if skipped_en_chunks:
            merged["skipped_en_chunks"] = skipped_en_chunks

        # ④ 계층: 전체 클래스에 접미공유 1회 (청크 경계 무관). 인덱스화+허브필터(O(N·L²)).
        #   한국어 head-final 특성으로 복합명사 접미가 상위 개념(생명보험업⊂보험업).
        #   정의문(Hearst) 계층은 실측상 노이즈가 이득을 상쇄해 미채택.
        merged["class_hierarchy"].extend(induce_suffix_hierarchy(set(class_chunks.keys())))
        # ⚠️pair 단위 dedup — existing 이어빌드 시 기존 hierarchy + 재유도 결과가 겹쳐
        # 패스마다 동일 pair 가 선형 증식(2→4→6, 0711 리뷰 실측)했고, 중복 pair 는
        # OWL disjoint 의 형제 리스트에 중복 URI 를 넣어 자기-disjoint(unsatisfiable)
        # 트리플까지 유발했다. 순서 보존 dedup.
        seen_pairs: set = set()
        uniq_hier = []
        for h in merged["class_hierarchy"]:
            k = (h.get("parent"), h.get("child"))
            if k[0] and k[1] and k[0] != k[1] and k not in seen_pairs:
                seen_pairs.add(k)
                uniq_hier.append(h)
        merged["class_hierarchy"] = uniq_hier

        return merged, all_entities, all_relations, all_data_props
