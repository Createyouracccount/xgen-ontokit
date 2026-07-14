"""한국어·영어 LLM-free 온톨로지 추출기 — Extractor 프로토콜 구현.

Kiwi 복합명사(ko)+nltk 명사구(en) 클래스 + 언어별 NER 엔티티 + 접미공유 subClassOf
계층(ko=문자, en=단어 단위). 관계(조사 SVO)는 한국어만. LLM 0회.
finreg 489 실측: 4.5초/$0, 클래스 3156·subClassOf 1710. 검색 A/B에서 gpt-4o와 동일(0.947).

XGEN pipeline은 이것을 gpt-4o DocumentOntologyExtractor 대신 주입 가능(같은 4-tuple 계약).
"""
from __future__ import annotations
import asyncio
import logging
import re
import threading
from typing import Optional

from ..morphology.kiwi_nouns import KiwiNounExtractor
from ..hierarchy.suffix_share import induce_suffix_hierarchy
from ..hierarchy.hearst_ko import definitional_pairs
from ..utils.lang_detect import detect_lang

logger = logging.getLogger(__name__)

_HANGUL = re.compile(r"[가-힣]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")  # 라틴 2자+ 연속 (단독 기호·항번호 제외)

# 클래스 승격 게이트의 df 조건이 활성화되는 최소 코퍼스 청크 수. 이보다 작은
# 코퍼스(도메인 문서 수십~수백 청크)는 정당 용어도 대부분 df=1 이라 df 컷이
# TBox 절멸이 됨(심판 A-2) — 계층참여·NER동일명 조건만 적용.
CLASS_DF_GATE_MIN_CHUNKS = 500


class DeterministicKoreanExtractor:
    def __init__(self, kiwi=None, ner=None, domain_words: Optional[list[str]] = None,
                 en_nouns=None, en_ner=None, relation_extractor=None,
                 enable_relations: bool = True, auto_english: bool = True,
                 enable_hearst: bool = True):
        """kiwi: Kiwi 인스턴스(없으면 생성, extras[korean]).
        ner: KoElectraNER 인스턴스(None이면 한국어 엔티티 추출 생략, extras[ner]).
        domain_words: 사용자사전 도메인 용어(한국어 Kiwi 사용자사전 + 영어 단일명사 허용목록).
        en_nouns: EnglishNounExtractor(None이면 auto_english 에 따라 자동 배선, extras[english]).
        en_ner: EnglishNER(None이면 영어 엔티티 추출 생략, extras[ner]).
        auto_english: True(기본)면 nltk 설치 시 en_nouns 자동 생성 — 영어 클래스가
          별도 주입 없이 나온다. en_ner 는 torch 모델 로드가 무거워 자동화하지 않음(명시 주입만).
        enable_hearst: True(기본, v0.12~)면 정의문 계층(hearst_ko) 배선 — 접미공유가
          원리적 불가한 이질계층(강아지⊂동물, 신용공여⊂거래)을 종결 패턴(계사/genus/서술/
          속하는)으로 유도. 외부 gold(Wikidata P279) 심판루프 89/100 검증. 비정의문
          문장엔 발화 안 함(오탐 낮음, 법령체 스팟체크 0). ⚠️v0.11 대비 동작 변화:
          정의문("X는…Y이다")이 있는 한국어 코퍼스에 이질계층 subClassOf 가 추가된다
          (순수 접미공유 출력과 다름). 기존 동작이 필요하면 enable_hearst=False.

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
                self.relations = self._default_relation_extractor()
        # 코퍼스레벨 관계추출기(extract_corpus, 예: hybrid) 판별 — 청크별 extract 대신
        # 루프 뒤 1회 호출(예산 가드레일이 코퍼스 전체 기준으로 작동).
        self._rel_is_corpus = (self.relations is not None
                               and hasattr(self.relations, "extract_corpus")
                               and not hasattr(self.relations, "extract"))
        # 동시 빌드 2개가 같은 extractor 인스턴스(factory 공유)를 서로 다른
        # to_thread 워커에서 돌릴 때 Kiwi 동시 호출을 직렬화(스레드 안전성 미보장
        # 방어, 0711 적대리뷰 HIGH). NER 는 자체 락 보유 — 이 락은 형태소·관계 커버.
        # Hearst 정의문 계층 — 기본 off(과거 실측 '노이즈가 이득 상쇄' 판단 존중).
        # 법령체 따옴표 정의문('"X"이란 … 말한다')만 커버하는 보수 패턴이라
        # 정의문 밀도 높은 코퍼스(법령·규정)에서 opt-in 으로 켠다. A/B 순도 실측 후 결정.
        self.enable_hearst = enable_hearst
        self._lock = threading.Lock()

    def _default_relation_extractor(self):
        """관계 채널 선택 — env ONTOKIT_RELATION_ENCODER_MODEL 지정 + import 가능 시
        KLUE-RE 인코더(심판 90/100), 아니면 규칙 조사SVO 폴백.

        불변식: env 미지정이거나 extras[relation-encoder] 미설치면 규칙 채널.
        "설치·설정 안 하면 인코더는 안 켜진다"(사용자 요건). 인코더는 NER(self.ner)로
        개체쌍을 만들므로 NER 도 있어야 실효 — NER 없으면 인코더는 빈 결과라 규칙 사용.
        정본: docs/ontokit_관계_KLUE-RE_인코더_심판루프_90_2026_07_14.md."""
        import os
        from .relation_ko import KoreanRelationExtractor

        model = os.getenv("ONTOKIT_RELATION_ENCODER_MODEL")
        if model and self.ner is not None:
            try:
                from .relation_encoder_ko import KoreanRelationEncoder
                enc = KoreanRelationEncoder(model=model, ner=self.ner)
                enc.warmup()  # 모델 즉시 로드 — 실패 시 여기서 규칙 폴백(지연 로드로
                              # 매 청크 실패하는 것 방지). extras 미설치·경로 오류 적발.
                logger.info("관계 채널 = KLUE-RE 인코더(%s)", model)
                return enc
            except Exception:
                # extras[relation-encoder] 미설치·모델 로드 실패 → 규칙 폴백(불변식)
                logger.warning("관계 인코더 로드 실패 — 규칙 조사SVO 폴백", exc_info=True)
        return KoreanRelationExtractor(kiwi=self.nouns.kiwi)

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
        hearst_pairs: list[dict] = []
        n_chunks = 0  # 처리 청크 수 — 승격 게이트의 소규모 코퍼스 판정용

        # NER 배치 수집 버퍼 — 청크별 단건 forward(CPU 891ms/청크, 2만 청크=297분)를
        # 언어별로 모아 배치 forward(430ms/청크)로 바꾼다. (doc_name, text, sc) 튜플.
        ko_ner_buf: list[tuple[str, str, list]] = []
        en_ner_buf: list[tuple[str, str, list]] = []
        rel_chunk_buf: list[dict] = []  # 코퍼스레벨 관계추출(hybrid)용 청크 수집

        for doc_name, chunks in documents.items():
            for ch in chunks:
                cid = ch.get("chunk_id")
                text = ch.get("chunk_text", "")
                if not text.strip():
                    continue
                n_chunks += 1
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
                # 관계 추출기가 코퍼스레벨(extract_corpus, 예: hybrid top-up)이면
                # 청크를 모아 루프 뒤 1회 호출(예산 가드레일이 코퍼스 전체 기준으로
                # 작동해야 LLM-free 가치 보존). 청크별 extract 면 여기서 즉시(인코더는
                # extract 내부에서 청크 내 쌍을 batch_size 로 배치 forward — CPU 2배).
                if self.relations is not None and _HANGUL.search(text):
                    if self._rel_is_corpus:
                        rel_chunk_buf.append({"chunk_id": cid, "chunk_text": text})
                    else:
                        rels = self.relations.extract(text, source_chunks=sc)
                        if rels:
                            all_relations.extend(rels)
                # ④' 정의문 계층(opt-in) — 접미공유가 원리적 불가한 이질 상위어
                #   (강아지⊂동물, 계란빵⊂음식). 종결 패턴(계사/genus/서술/속하는) 정밀
                #   targeting. 외부 gold(Wikidata P279) 심판루프 89/100 검증(순증 +0.30,
                #   이질계층 77%). kiwi 주입 → 강화 채널, 미주입 → 따옴표 폴백.
                if self.enable_hearst and _HANGUL.search(text):
                    for hp in definitional_pairs(text, self.nouns.last_noun,
                                                 kiwi=self.nouns.kiwi):
                        hearst_pairs.append(hp)
                        # 정의쌍의 클래스가 명사추출에 안 잡혔어도 존재 보장(+출처 청크)
                        class_chunks.setdefault(hp["child"], set()).update(sc)
                        class_chunks.setdefault(hp["parent"], set()).update(sc)

        # ② NER → 인스턴스 엔티티 — 언어별 배치 forward 1회.
        self._run_ner_batched(self.ner, ko_ner_buf, all_entities)
        self._run_ner_batched(self.en_ner, en_ner_buf, all_entities)

        # ③' 코퍼스레벨 관계추출(hybrid top-up) — 수집한 한국어 청크 1회 처리.
        #   예산 가드레일(청크% ∨ 달러)이 코퍼스 전체 기준으로 작동해 LLM-free 가치
        #   보존. extract_corpus 는 async 라 이 sync 본문에서 새 이벤트루프로 실행
        #   (이 본문 자체가 pipeline 의 to_thread 워커라 실행 루프 없음 — asyncio.run 안전).
        if self._rel_is_corpus and rel_chunk_buf:
            try:
                import asyncio as _asyncio
                rels, _rep = _asyncio.run(
                    self.relations.extract_corpus(rel_chunk_buf))
                if rels:
                    all_relations.extend(rels)
            except Exception:
                pass  # hybrid 실패는 비치명 — 규칙 결과(있으면) 유지, 관계 없이 진행

        # ④ 계층: 전체 클래스에 접미공유 1회 (청크 경계 무관). 인덱스화+허브필터(O(N·L²)).
        #   한국어 head-final 특성으로 복합명사 접미가 상위 개념(생명보험업⊂보험업, 동종계층).
        #   접미공유(동종)와 정의문(이질, hearst_ko 종결패턴)은 상보 — merge 시 superset.
        #   정의문 계층은 enable_hearst(기본 on, v0.12~, 외부gold 심판루프 89/100 검증).
        #   ⚠️클래스 승격 게이트(④')보다 먼저 — 계층 참여가 게이트의 보존 조건이므로
        #   전체 클래스 후보 위에서 유도해야 df=1 자식(안성농업전문학교 류)이 살아남는다.
        all_hier = list(existing.get("class_hierarchy", [])) if existing else []
        all_hier.extend(
            induce_suffix_hierarchy(set(class_chunks.keys()), kiwi=self.nouns.kiwi))
        if hearst_pairs:
            all_hier.extend(hearst_pairs)
        # ⚠️pair 단위 dedup — existing 이어빌드 시 기존 hierarchy + 재유도 결과가 겹쳐
        # 패스마다 동일 pair 가 선형 증식(2→4→6, 0711 리뷰 실측)했고, 중복 pair 는
        # OWL disjoint 의 형제 리스트에 중복 URI 를 넣어 자기-disjoint(unsatisfiable)
        # 트리플까지 유발했다. 순서 보존 dedup.
        seen_pairs: set = set()
        uniq_hier = []
        for h in all_hier:
            k = (h.get("parent"), h.get("child"))
            if k[0] and k[1] and k[0] != k[1] and k not in seen_pairs:
                seen_pairs.add(k)
                uniq_hier.append(h)

        # ④' 클래스 승격 게이트 (심판 OR-게이트) — 보존 = 계층 참여 OR df≥2.
        #   mixed20k 실측: 클래스 95%가 df=1(단일 청크 출현) 고아 → completeness 5%,
        #   text-index 노이즈. 순수 df≥2 는 계층 자식·소코퍼스 정당용어를 학살(심판
        #   A-1/A-2 기각) → 계층 참여 클래스는 df 무관 보존 + 소규모 코퍼스(<500청크)
        #   는 df 조건 비활성(76~309청크 도메인 코퍼스의 df=1 정당용어 보호).
        #   NER 동일명 강등 — 개체명이 클래스로 이중 존재(TBox/ABox 중복, 실측 17.3%:
        #   PeterCushing·베트남·광주광역시)하면 **고아에 한해** 강등(연결된 동명은
        #   punning 합법 — NER 오탐 1건이 정당 TBox 를 지우는 비용 구조 차단, 심판 B).
        #   탈락은 침묵하지 않고 stats 로 방출(no silent caps).
        hier_names = {h["parent"] for h in uniq_hier} | {h["child"] for h in uniq_hier}
        inst_norms = {e.get("entity", "").replace(" ", "")
                      for ents in all_entities.values() for e in ents}
        inst_norms.discard("")
        df_gate_active = n_chunks >= CLASS_DF_GATE_MIN_CHUNKS
        kept_classes: dict[str, set] = {}
        n_drop_df1 = n_drop_dup = 0
        for nm, chunks in class_chunks.items():
            if nm not in hier_names:
                if nm.replace(" ", "") in inst_norms:
                    n_drop_dup += 1      # 동일명 인스턴스 존재 — ABox 가 담당
                    continue
                if df_gate_active and len(chunks) < 2:
                    n_drop_df1 += 1      # df=1 ∧ 고아 — 어느 검색 leg 에도 기여 없음
                    continue
            kept_classes[nm] = chunks

        merged = {
            "classes": [{"name": nm, "description": "", "parent": None,
                         "source_chunks": list(chunks)}
                        for nm, chunks in kept_classes.items()],
            "object_properties": list(existing.get("object_properties", [])) if existing else [],
            "datatype_properties": list(existing.get("datatype_properties", [])) if existing else [],
            "class_hierarchy": uniq_hier,
        }
        if skipped_en_chunks:
            merged["skipped_en_chunks"] = skipped_en_chunks
        if n_drop_df1 or n_drop_dup:
            merged["class_gate_stats"] = {
                "kept": len(kept_classes), "dropped_df1_orphan": n_drop_df1,
                "dropped_ner_dup": n_drop_dup, "df_gate_active": df_gate_active,
            }
            logger.info("클래스 승격 게이트: %d 후보 → %d 보존 (df1고아 %d·NER동일명 %d 탈락, df게이트=%s)",
                        len(class_chunks), len(kept_classes), n_drop_df1, n_drop_dup,
                        df_gate_active)

        return merged, all_entities, all_relations, all_data_props
