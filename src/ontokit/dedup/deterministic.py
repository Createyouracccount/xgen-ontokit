"""결정적 dedup — 형태소 정규화 키 기반 클래스·인스턴스 병합. LLM 0회.

XGEN ClassDeduplicator 의 1·3단계(규칙/형태소 기반)를 라이브러리로 이식.
2·4단계(LLM 동의어)는 대용량서 컨텍스트 초과(3097배@800만)라 제외.

원리: 같은 개념이 조사/어미/특수문자만 다르면 형태소 명사키가 같아 병합.
in-memory concepts 4-tuple 변환만 — 인프라(Fuseki/DB) 결합 0.
"""
from __future__ import annotations
import re

_CLEAN = re.compile(r'[\s_\-·•/\\()（）「」『』【】\[\]]+')
_NOISE_EN = re.compile(
    r'(to|from|of|by|with|for|the|and|or|is|are|has|have|belongs|connection'
    r'|link|relation|mapping|reference)')


class DeterministicDedup:
    """Kiwi 형태소 정규화 키로 동의어 병합. kiwi 인스턴스 주입(없으면 생성)."""

    def __init__(self, kiwi=None):
        if kiwi is None:
            from kiwipiepy import Kiwi  # extras[korean]
            kiwi = Kiwi()
        self._kiwi = kiwi

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
        rename: dict[str, str] = {}
        # 클래스
        cls_names = [c.get("name", "") for c in concepts.get("classes", []) if c.get("name")]
        rename.update(self._rename_by_key(cls_names))
        # 인스턴스
        inst_names = [e.get("entity", "") for ents in ner_entities.values() for e in ents]
        rename.update(self._rename_by_key(inst_names))
        # ObjectProperty (영문 노이즈 제거)
        prop_names = [p.get("name", "") for p in concepts.get("object_properties", []) if p.get("name")]
        rename.update(self._rename_by_key(prop_names, strip_en_noise=True))
        # 체인 평탄화 + self-map 제거
        flat = {}
        for k, v in rename.items():
            if not k or not v:
                continue
            while v in rename and rename[v] != v:
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
        # 계층 rename
        concepts["class_hierarchy"] = [
            {"parent": r(h.get("parent")), "child": r(h.get("child"))}
            for h in concepts.get("class_hierarchy", [])
            if r(h.get("parent")) != r(h.get("child"))
        ]
        # 엔티티 class rename
        ner_entities = {
            doc: [{**e, "class": r(e.get("class"))} for e in ents]
            for doc, ents in ner_entities.items()
        }
        # 관계 predicate rename
        relations = [{**rel, "predicate": r(rel.get("predicate"))} for rel in relations]
        return concepts, ner_entities, relations, data_properties
