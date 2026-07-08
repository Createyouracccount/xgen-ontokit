"""추출기 공통 — concepts 병합(중복 제거, source_chunks 합산).

XGEN DocumentOntologyExtractor._merge_concepts 를 라이브러리 자체에 포함(외부 주입 불필요).
"""
from __future__ import annotations


def merge_concepts(existing: dict, new: dict) -> dict:
    """스키마 병합. 클래스는 이름 기준 중복 제거하되 source_chunks는 합산."""
    merged = {
        "classes": list(existing.get("classes", [])),
        "object_properties": list(existing.get("object_properties", [])),
        "datatype_properties": list(existing.get("datatype_properties", [])),
        "class_hierarchy": list(existing.get("class_hierarchy", [])),
    }
    names = {c.get("name") for c in merged["classes"]}
    obj = {p.get("name") for p in merged["object_properties"]}
    dat = {p.get("name") for p in merged["datatype_properties"]}
    hier = {(h.get("parent"), h.get("child")) for h in merged["class_hierarchy"]}

    for cls in new.get("classes", []):
        nm = cls.get("name")
        if not nm:
            continue
        if nm not in names:
            merged["classes"].append(cls)
            names.add(nm)
        else:
            for ex in merged["classes"]:
                if ex.get("name") == nm:
                    s = set(ex.get("source_chunks", []))
                    s.update(cls.get("source_chunks", []))
                    ex["source_chunks"] = list(s)
                    break
    for p in new.get("object_properties", []):
        if p.get("name") and p["name"] not in obj:
            merged["object_properties"].append(p); obj.add(p["name"])
    for p in new.get("datatype_properties", []):
        if p.get("name") and p["name"] not in dat:
            merged["datatype_properties"].append(p); dat.add(p["name"])
    for h in new.get("class_hierarchy", []):
        k = (h.get("parent"), h.get("child"))
        if k not in hier:
            merged["class_hierarchy"].append(h); hier.add(k)
    return merged
