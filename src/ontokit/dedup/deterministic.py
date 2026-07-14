"""결정적 dedup — 형태소 정규화 키 기반 클래스·인스턴스 병합. LLM 0회.

XGEN ClassDeduplicator 의 1·3단계(규칙/형태소 기반)를 라이브러리로 이식.
2·4단계(LLM 동의어)는 대용량서 컨텍스트 초과(3097배@800만)라 제외.

원리: 같은 개념이 조사/어미/특수문자만 다르면 형태소 명사키가 같아 병합.
in-memory concepts 4-tuple 변환만 — 인프라(Fuseki/DB) 결합 0.
"""
from __future__ import annotations
import re
import threading

_CLEAN = re.compile(r'[\s_\-·•/\\()（）「」『』【】\[\]]+')
_NOISE_EN = re.compile(
    r'(to|from|of|by|with|for|the|and|or|is|are|has|have|belongs|connection'
    r'|link|relation|mapping|reference)')


class DeterministicDedup:
    """Kiwi 형태소 정규화 키로 동의어 병합. kiwi 인스턴스 주입(없으면 생성)."""

    def __init__(self, kiwi=None, synonym_dict=None):
        """kiwi: Kiwi 인스턴스(없으면 생성, extras[korean]).
        synonym_dict: SynonymDictDedup 인스턴스(선택). None 이면 env
          ONTOKIT_SYNONYM_DICT 지정 시 자동 로드 — 우리말샘 비슷한말로 의미변이
          (전자상거래=이커머스) 추가 병합. 미지정·미설치면 형태소 정규화만(불변).
          ⚠️ 사전은 고정밀 확정 동의어 병합(P 1.000)이지 광범위 recall 아님
          (ER 심판 5R: 임베딩 주제근접 미분리·전통사전 현대어 미수록). 신중한 opt-in."""
        if kiwi is None:
            from kiwipiepy import Kiwi  # extras[korean]
            kiwi = Kiwi()
        self._kiwi = kiwi
        self._syn = synonym_dict or self._auto_synonym_dict()
        # 동시 빌드 2개가 같은 dedup 인스턴스를 서로 다른 to_thread 워커에서 쓸 때
        # Kiwi.analyze 동시 호출을 직렬화(스레드 안전성 미보장 방어). 0711 적대리뷰 지적.
        self._lock = threading.Lock()

    @staticmethod
    def _auto_synonym_dict():
        """env ONTOKIT_SYNONYM_DICT 지정 시 사전 채널 자동 로드, 아니면 None(형태소만).
        '설정 안 하면 안 켜진다' 불변식 — 로드 실패 시에도 형태소 폴백."""
        import logging
        import os
        if not os.getenv("ONTOKIT_SYNONYM_DICT"):
            return None
        try:
            from .synonym_dict import SynonymDictDedup
            return SynonymDictDedup()
        except Exception:
            logging.getLogger(__name__).warning(
                "동의어 사전 로드 실패 — 형태소 정규화만 사용", exc_info=True)
            return None

    def _noun_key(self, name: str, *, strip_en_noise: bool = False) -> str:
        cleaned = _CLEAN.sub('', (name or '').strip())
        if not cleaned:
            return ''
        try:
            toks = self._kiwi.analyze(cleaned)[0][0]
            nouns = [t.form for t in toks if t.tag.startswith('N') or t.tag in ('SL', 'SH', 'SW')]
            if nouns:
                key = ''.join(nouns).lower()
                if strip_en_noise:
                    key = _NOISE_EN.sub('', key)
                return key
        except Exception:
            pass
        return cleaned.lower()

    def _rename_by_key(self, names: list[str], *, strip_en_noise: bool = False) -> dict[str, str]:
        """같은 정규화키 → 첫 등장 이름을 canonical 로. {중복: canonical}."""
        seen, rename = {}, {}
        for name in names:
            if not name:
                continue
            key = self._noun_key(name, strip_en_noise=strip_en_noise)
            if not key:
                continue
            if key in seen:
                if name != seen[key]:
                    rename[name] = seen[key]
            else:
                seen[key] = name
        return rename

    def compute_rename_map(self, concepts: dict, ner_entities: dict) -> dict[str, str]:
        """클래스·인스턴스·속성 결정적 rename map. concepts/entities 는 읽기만."""
        with self._lock:  # Kiwi.analyze 동시 호출 직렬화 (동시 빌드 방어)
            rename: dict[str, str] = {}
            # 클래스 (우선) — 아래 인스턴스/속성 맵과 키가 충돌하면 클래스 canonical 이 승자.
            cls_names = [c.get("name", "") for c in concepts.get("classes", []) if c.get("name")]
            rename.update(self._rename_by_key(cls_names))
            # 인스턴스 — 클래스 맵과 모순되는 항목(같은 k 또는 역방향 k↔v)은 버린다.
            #   3개 독립 맵을 무조건 update 로 합치면 교차 사이클(클래스 맵 A→B +
            #   인스턴스 맵 B→A)이 가능하고, 사이클은 아래 평탄화 while 을 무한루프로
            #   만들어 취소 불능 to_thread 안에서 좀비 빌드가 된다(0711 적대리뷰 HIGH).
            inst_names = [e.get("entity", "") for ents in ner_entities.values() for e in ents]
            for k, v in self._rename_by_key(inst_names).items():
                if k not in rename and rename.get(v) != k:
                    rename[k] = v
            # ObjectProperty (영문 노이즈 제거) — 동일한 모순 필터.
            prop_names = [p.get("name", "") for p in concepts.get("object_properties", []) if p.get("name")]
            for k, v in self._rename_by_key(prop_names, strip_en_noise=True).items():
                if k not in rename and rename.get(v) != k:
                    rename[k] = v
            # 사전 채널(선택): 형태소키가 달라도 우리말샘 동의어면 추가 병합
            #   (전자상거래=이커머스). 형태소 맵과 모순되는 항목은 버린다(위와 동형).
            if self._syn is not None:
                syn_targets = cls_names + inst_names
                for k, v in self._syn.rename_by_synonym(syn_targets).items():
                    if k not in rename and rename.get(v) != k:
                        rename[k] = v
        # 체인 평탄화 + self-map 제거. visited 가드 — 위 필터가 놓친 어떤 형태의
        # 사이클(3항 이상 순환 등)도 종료를 보장(재방문 시 그 지점에서 멈춤).
        flat = {}
        for k, v in rename.items():
            if not k or not v:
                continue
            visited = {k}
            while v in rename and rename[v] != v and v not in visited:
                visited.add(v)
                v = rename[v]
            if k != v:
                flat[k] = v
        return flat

    def apply(self, rename: dict[str, str], concepts: dict, ner_entities: dict,
              relations: list, data_properties: list) -> tuple[dict, dict, list, list]:
        """rename map 을 4-tuple 전체에 적용(병합)."""
        if not rename:
            return concepts, ner_entities, relations, data_properties

        def r(n): return rename.get(n, n)

        # 클래스 병합(이름 기준 dedup)
        seen_cls, new_cls = set(), []
        for c in concepts.get("classes", []):
            nm = r(c.get("name"))
            if nm and nm not in seen_cls:
                seen_cls.add(nm)
                new_cls.append({**c, "name": nm})
        concepts = {**concepts, "classes": new_cls}
        # 계층 rename + pair 중복 제거 — rename 으로 서로 다른 pair 가 같은 pair 로
        # 수렴하면((A,C)+(B,C), B→A) 중복이 재생성되므로 여기서 걸러야 한다(0711).
        seen_pairs: set = set()
        new_hier = []
        for h in concepts.get("class_hierarchy", []):
            p, c = r(h.get("parent")), r(h.get("child"))
            if p != c and (p, c) not in seen_pairs:
                seen_pairs.add((p, c))
                new_hier.append({"parent": p, "child": c})
        concepts["class_hierarchy"] = new_hier
        # 엔티티 이름+class rename — 이전 구현은 class 만 바꾸고 entity 이름은 방치해
        # "인스턴스 병합"이 사실상 no-op 이었다(0711 적대리뷰: 삼성전자/삼성 전자가
        # Fuseki 에 URI 2개로 남음). 이름까지 canonical 로 치환.
        ner_entities = {
            doc: [{**e, "entity": r(e.get("entity")), "class": r(e.get("class"))}
                  for e in ents]
            for doc, ents in ner_entities.items()
        }
        # 관계 S·P·O 전부 rename — predicate 만 바꾸면 subject/object 가 병합 전
        # 이름으로 남아 인스턴스 노드와 어긋난다. rename 후 self-loop 는 버림.
        relations = [
            {**rel, "subject": r(rel.get("subject")),
             "predicate": r(rel.get("predicate")), "object": r(rel.get("object"))}
            for rel in relations
            if r(rel.get("subject")) != r(rel.get("object"))
        ]
        return concepts, ner_entities, relations, data_properties
