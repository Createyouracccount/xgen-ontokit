"""정답셋·문항 생성 — Wikidata 사실 중 **해당 문서 본문에 실제로 적힌 것만** 정답으로 쓴다.

도구(ontokit) 출력과 무관하게 만든다(§6 표본 편향·GT 오염 방지). 문항은 그래프를 보기 전에
고정하고 해시를 남긴다(사전 공시).

형태(구조형 6 + 대조 1, 어느 형태도 30% 를 넘지 않음):
  E1 타입 열거          — "직업이 가수인 인물 모두"
  E2 계층 폐포 열거     — 상위 클래스로 물음, 본문엔 하위 클래스만 적힘("음악가" ← 가수·작곡가)
  R  역관계 열거        — "서울에서 태어난 인물 모두"
  A  집계(개수)         — "국적이 일본인 인물은 몇 명"
  C  교차 조건          — "국적이 미국이고 직업이 배우인 인물"
  M  다중 홉            — "서울에서 태어난 인물들이 다닌 학교"
  L  단순 조회(대조)    — "X 의 출생지는" — 벡터에 유리한 형태, 회귀 감시용

  python -m harness.gold harness/data/wiki2/docs.jsonl harness/data/wikidata harness/bench
"""
import collections
import hashlib
import json
import os
import random
import re
import sys

from harness.schema import RELATIONS, TYPE_PROPS, WD_TO_REL, norm

_HANGUL = re.compile(r"[가-힣]")
# 2글자 라벨은 뒤에 이어지는 한글이 조사·서술어일 때만 인정("배우" ≠ "배우자")
_JOSA = ("이", "가", "은", "는", "을", "를", "의", "와", "과", "로", "으로", "에", "에서", "이다",
         "였", "이며", "이자", "로서", "출신", "겸", "인", "이고", "도", "만", "들")


def mentions(text, name):
    """본문이 name 을 서술하는가 — 2글자는 조사 경계, 3글자 이상은 부분일치."""
    if not name or len(name) < 2:
        return False
    if len(name) >= 3:
        return name in text
    for m in re.finditer(re.escape(name), text):
        nxt = text[m.end():m.end() + 3]
        if not nxt or not _HANGUL.match(nxt[0]) or nxt.startswith(_JOSA):
            return True
    return False


def names_of(rec):
    out = [rec.get("label")] + list(rec.get("aliases") or [])
    return [n for n in dict.fromkeys(out) if n]


class Facts:
    """본문 근거가 확인된 사실 집합(오라클 그래프의 원천)."""

    def __init__(self, docs, subjects, objects):
        self.objects = objects
        self.ent = {}          # qid → {"names": [...], "docs": [...], "subject": bool}
        self.types = collections.defaultdict(dict)   # s → {class_qid: 본문 표면형}
        self.occ_classes = set()                     # P106(직업)에서 온 타입
        self.rels = collections.defaultdict(dict)    # (s, rel) → {o: 표면형}
        self.subject_of_doc = {}
        for d in docs:
            sj = subjects.get(d["title"])
            if not sj:
                continue
            s, text = sj["qid"], d["text"]
            title = re.sub(r"\s*\([^)]*\)", "", d["title"]).strip()
            nm = list(dict.fromkeys([title] + names_of(sj)))
            self.ent.setdefault(s, {"names": nm, "docs": [], "subject": True})
            self.ent[s]["subject"] = True
            self.ent[s]["docs"].append(d["doc_id"])
            self.subject_of_doc[d["doc_id"]] = s
            for p, vals in sj["claims"].items():
                for o in vals:
                    orec = objects.get(o) or {}
                    hit = next((n for n in names_of(orec) if mentions(text, n)), None)
                    if not hit:
                        continue
                    if o == s:   # 자기 참조(예: 오스만 제국 P17 오스만 제국)는 사실로 쓰지 않는다
                        continue
                    if p in TYPE_PROPS:
                        self.types[s][o] = hit
                        if p == "P106":
                            self.occ_classes.add(o)
                    elif p in WD_TO_REL:
                        self.rels[(s, WD_TO_REL[p])][o] = hit
                        if o not in self.ent:
                            self.ent[o] = {"names": names_of(orec), "docs": [], "subject": False}

    # ── TBox: 클래스 상위 사슬(위키데이터 P279) ──
    def ancestors(self, c, depth=4):
        seen, frontier = {c}, {c}
        for _ in range(depth):
            nxt = set()
            for x in frontier:
                nxt |= set((self.objects.get(x) or {}).get("claims", {}).get("P279", []))
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        return seen

    def label(self, q):
        rec = self.objects.get(q) or {}
        return rec.get("label") or (self.ent.get(q) or {}).get("names", [None])[0]

    def members(self, c, occ_only=False):
        """c 의 인스턴스(폐포) — 직접 타입이 c 이거나 c 의 하위 클래스."""
        out = {}
        for s, ts in self.types.items():
            if occ_only:
                ts = {t: v for t, v in ts.items() if t in self.occ_classes}
            direct = [t for t in ts if t == c]
            via = [t for t in ts if t != c and c in self.ancestors(t)]
            if direct or via:
                out[s] = "direct" if direct else "sub"
        return out


def item(f, q, docs=None):
    """정답 항목 — 매칭용 이름들 + 근거 문서(벡터 상한 계산용: 이 문서가 근거에 들어와야 답할 수 있다)."""
    return {"key": q, "names": f.ent.get(q, {}).get("names") or names_of(f.objects.get(q) or {}),
            "docs": docs if docs is not None else f.ent.get(q, {}).get("docs", [])}


REL_Q = {
    "per:place_of_birth": "'{o}'에서 태어난 인물",
    "per:place_of_death": "'{o}'에서 사망한 인물",
    "per:schools_attended": "'{o}'을(를) 다닌 인물",
    "per:employee_of": "'{o}'에 소속된 적이 있는 인물",
    "per:origin": "국적이 '{o}'인 인물",
    "org:founded_by": "'{o}'이(가) 설립한 조직",
    "org:place_of_headquarters": "본부(본사)가 '{o}'에 있는 조직",
    "loc:located_in": "'{o}'에 속한(위치한) 곳",
    "loc:country": "'{o}'에 있는 곳",
}
HOP_Q = {  # 다중 홉의 두 번째 관계 → 물을 대상
    "per:schools_attended": "다닌 학교",
    "per:employee_of": "소속 조직",
    "per:place_of_birth": "출생지",
}
LOOKUP_Q = {
    "per:place_of_birth": "'{s}'의 출생지는 어디인가?",
    "per:schools_attended": "'{s}'이(가) 다닌 학교는?",
    "per:origin": "'{s}'의 국적은?",
    "org:founded_by": "'{s}'의 설립자는?",
    "org:place_of_headquarters": "'{s}'의 본부(본사)는 어디에 있나?",
    "loc:country": "'{s}'은(는) 어느 나라에 있나?",
}


def surface(f, o, pairs):
    """질문에 쓸 이름 — 본문에서 가장 많이 쓰인 표면형."""
    c = collections.Counter(pairs)
    return c.most_common(1)[0][0] if c else f.label(o)


def build(f, seed=20261006):
    rng = random.Random(seed)
    used = set()
    qs = []

    def add(form, q, plan, gold, meta):
        qs.append({"id": f"{form}{sum(1 for x in qs if x['form'] == form) + 1:03d}", "form": form,
                   "q": q, "plan": plan, "gold": gold, "meta": meta})

    # 클래스 후보 = 직접 타입 ∪ '직업으로 분류된' 직업 상위 클래스(2단계 이내).
    # 위키데이터 P279 의 일반 사슬은 '생물학적 과정'·'단체' 같은 비자연 상위를 만든다(0차 점검에서 확인).
    classes = {t for ts in f.types.values() for t in ts}
    OCC_KIND = {"Q28640", "Q12737077", "Q4164871", "Q88789639", "Q66715801"}  # 직업·전문직·직위·예술직·직업군
    occ_super = set()
    for t in f.occ_classes:
        for a in f.ancestors(t, depth=2):
            kinds = set((f.objects.get(a) or {}).get("claims", {}).get("P31", []))
            if a != t and (kinds & OCC_KIND or a in f.occ_classes):
                occ_super.add(a)
    closure_classes = classes | occ_super
    cls_rows = []
    for c in sorted(closure_classes):
        if not (f.objects.get(c) or {}).get("label"):
            continue
        m = f.members(c, occ_only=(c in occ_super and c not in classes))
        if 3 <= len(m) <= 80:
            sub = sum(1 for v in m.values() if v == "sub")
            cls_rows.append((c, m, sub / len(m)))
    rng.shuffle(cls_rows)
    is_occ = {t for s, ts in f.types.items() for t in ts
              if t in {o for s2 in [s] for o in (f.objects.get(t) or {}).get("claims", {}).get("P31", [])}}

    def cls_phrase(c):
        return f"'{f.label(c)}'에 해당하는 인물·항목"

    def isa(c):
        return {"t": "isa", "v": "x", "class": f.label(c)}

    e1 = [r for r in cls_rows if r[2] == 0][:30]
    # E2 는 직업 계층에서만 — 위키데이터 P279 의 일반 사슬은 '제어'·'행정중심' 같은 비자연 상위를 만든다
    e2 = [r for r in cls_rows if r[2] >= 0.4 and r[0] in occ_super][:30]
    for c, m, _ in e1:
        used.add(("cls", c))
        add("E1", f"이 문서 모음에 수록된 것 중 {cls_phrase(c)}을 모두 나열해줘.",
            {"op": "list", "return": "x", "where": [isa(c)]},
            {"items": [item(f, s) for s in sorted(m)]}, {"class": c, "n": len(m)})
    for c, m, ratio in e2:
        used.add(("cls", c))
        add("E2", f"이 문서 모음에 수록된 것 중 {cls_phrase(c)}(하위 분류 포함)을 모두 나열해줘.",
            {"op": "list", "return": "x", "where": [isa(c)]},
            {"items": [item(f, s) for s in sorted(m)]}, {"class": c, "n": len(m), "via_sub": round(ratio, 2)})

    # 역관계: (rel, o) → 주어 집합
    inv = collections.defaultdict(dict)
    for (s, rel), objs in f.rels.items():
        for o, hit in objs.items():
            inv[(rel, o)][s] = hit
    rel_rows = [(k, v) for k, v in inv.items() if k[0] in REL_Q and 3 <= len(v) <= 60]
    rng.shuffle(rel_rows)
    # 관계 종류가 한쪽에 몰리지 않게 관계별 상한
    per_rel = collections.Counter()
    r_rows, a_rows = [], []
    for k, v in rel_rows:
        if per_rel[k[0]] >= 8:
            continue
        per_rel[k[0]] += 1
        (r_rows if len(r_rows) < 40 else a_rows).append((k, v))
    for (rel, o), subs in r_rows:
        used.add(("rel", rel, o))
        name = surface(f, o, subs.values())
        add("R", f"이 문서 모음에서 {REL_Q[rel].format(o=name)}을(를) 모두 나열해줘.",
            {"op": "list", "return": "x", "where": [{"t": "rel", "s": "x", "p": rel, "o": {"const": name}}]},
            {"items": [item(f, s) for s in sorted(subs)]}, {"rel": rel, "obj": o, "n": len(subs)})
    # 집계: 역관계 잔여 + 클래스 잔여
    agg = [("rel", k, v) for k, v in a_rows[:15]]
    agg += [("cls", c, m) for c, m, _ in cls_rows if ("cls", c) not in used][:30 - len(agg)]
    for kind, k, v in agg:
        if kind == "rel":
            rel, o = k
            name = surface(f, o, v.values())
            q = f"이 문서 모음에서 {REL_Q[rel].format(o=name)}은(는) 모두 몇인가?"
            plan = {"op": "count", "return": "x", "where": [{"t": "rel", "s": "x", "p": rel, "o": {"const": name}}]}
        else:
            q = f"이 문서 모음에 수록된 것 중 {cls_phrase(k)}은(는) 모두 몇 개인가?"
            plan = {"op": "count", "return": "x", "where": [isa(k)]}
        add("A", q, plan, {"count": len(v), "items": [item(f, s) for s in sorted(v)]},
            {"kind": kind, "n": len(v)})

    # 교차: 국적/출생지 × 클래스
    conj = []
    for (rel, o), subs in inv.items():
        if rel not in ("per:origin", "per:place_of_birth", "per:schools_attended"):
            continue
        for c, m, _ in cls_rows:
            both = set(subs) & set(m)
            if 2 <= len(both) <= 40 and len(both) < min(len(subs), len(m)):
                conj.append((rel, o, c, both, subs))
    rng.shuffle(conj)
    seen_pair = set()
    for rel, o, c, both, subs in conj:
        if len([x for x in qs if x["form"] == "C"]) >= 25 or (rel, o) in seen_pair:
            continue
        seen_pair.add((rel, o))
        name = surface(f, o, subs.values())
        add("C", f"이 문서 모음에서 {REL_Q[rel].format(o=name)} 중 {cls_phrase(c)}을 모두 나열해줘.",
            {"op": "list", "return": "x",
             "where": [{"t": "rel", "s": "x", "p": rel, "o": {"const": name}}, isa(c)]},
            {"items": [item(f, s) for s in sorted(both)]}, {"rel": rel, "obj": o, "class": c, "n": len(both)})

    # 다중 홉: (rel1=o) 인 x 들의 rel2 대상
    hops = []
    for (rel1, o), subs in inv.items():
        if rel1 not in REL_Q:
            continue
        for rel2 in HOP_Q:
            if rel2 == rel1:
                continue
            ys, sup = {}, collections.defaultdict(list)
            for s in subs:
                for y, hit in f.rels.get((s, rel2), {}).items():
                    ys[y] = hit
                    sup[y] += f.ent[s]["docs"]
            if 3 <= len(ys) <= 40 and len(subs) >= 2:
                hops.append((rel1, o, rel2, ys, subs, sup))
    rng.shuffle(hops)
    for rel1, o, rel2, ys, subs, sup in hops[:25]:
        name = surface(f, o, subs.values())
        add("M", f"이 문서 모음에서 {REL_Q[rel1].format(o=name)}들의 {HOP_Q[rel2]}을(를) 모두 나열해줘.",
            {"op": "list", "return": "y",
             "where": [{"t": "rel", "s": "x", "p": rel1, "o": {"const": name}},
                       {"t": "rel", "s": "x", "p": rel2, "o": {"var": "y"}}]},
            {"items": [item(f, y, sorted(set(sup[y]))) for y in sorted(ys)]}, {"rel1": rel1, "obj": o, "rel2": rel2, "n": len(ys)})

    # 단순 조회(대조군)
    look = [(s, rel, objs) for (s, rel), objs in f.rels.items() if rel in LOOKUP_Q and 1 <= len(objs) <= 3
            and f.ent.get(s, {}).get("subject")]
    rng.shuffle(look)
    for s, rel, objs in look[:30]:
        add("L", LOOKUP_Q[rel].format(s=f.ent[s]["names"][0]),
            {"op": "list", "return": "y",
             "where": [{"t": "rel", "s": {"const": f.ent[s]["names"][0]}, "p": rel, "o": {"var": "y"}}]},
            {"items": [item(f, o, f.ent[s]["docs"]) for o in sorted(objs)]}, {"rel": rel, "subj": s, "n": len(objs)})

    # 정답 = 계획의 의미를 사실 집합 위에서 계산한 값(질의 의미와 일치).
    # 1차 오라클 대조에서 '독일'(독일·서독…) 같은 이름 중의성과 클래스 폐포 범위가
    # 정답 생성과 질의 의미 사이에서 어긋났음을 확인 → 이름 합집합·전체 타입 폐포로 통일.
    ev = Evaluator(f)
    for q in qs:
        sol = ev.solve(q["plan"])
        q["gold"]["items"] = [item(f, k, docs) for k, docs in sorted(sol.items())]
        if q["form"] == "A":
            q["gold"]["count"] = len(sol)
        q["meta"]["n"] = len(sol)
    return qs


class Evaluator:
    """IR 계획을 사실 집합 위에서 직접 푼다 — 컴파일러(SPARQL)와 독립된 두 번째 구현."""

    def __init__(self, f):
        self.f = f
        self.by_name = collections.defaultdict(set)
        for q, e in f.ent.items():
            for n in e["names"]:
                self.by_name[norm(n)].add(q)
        for q, rec in f.objects.items():
            for n in names_of(rec):
                self.by_name[norm(n)].add(q)
        self.subjects = {q for q, e in f.ent.items() if e["subject"] and e["docs"]}
        self.edges = collections.defaultdict(set)   # (rel) → {(s, o)}
        for (s, rel), objs in f.rels.items():
            for o in objs:
                self.edges[rel].add((s, o))

    def pairs(self, rel):
        out = set(self.edges[rel])
        inv = RELATIONS[rel][4]
        if inv:
            out |= {(o, s) for s, o in self.edges[inv]}
        return out

    def isa(self, s, cname):
        cs = self.by_name.get(norm(cname), set())
        return any(t in cs or cs & self.f.ancestors(t) for t in self.f.types.get(s, {}))

    def solve(self, plan):
        binds = [{}]
        for c in plan["where"]:
            nxt = []
            for b in binds:
                if c["t"] == "isa":
                    cands = [b[c["v"]]] if c["v"] in b else list(self.subjects if c["v"] == "x" else self.f.types)
                    nxt += [{**b, c["v"]: s} for s in cands if self.isa(s, c["class"])]
                    continue
                for s, o in self.pairs(c["p"]):
                    nb, ok = dict(b), True
                    for t, val in ((c["s"], s), (c["o"], o)):
                        if isinstance(t, dict) and "const" in t:
                            ok &= val in self.by_name.get(norm(t["const"]), set())
                        else:
                            v = t if isinstance(t, str) else t["var"]
                            if v in nb and nb[v] != val:
                                ok = False
                            nb[v] = val
                    if ok:
                        nxt.append(nb)
            binds = nxt
        binds = [b for b in binds if "x" not in b or b["x"] in self.subjects]
        r = plan["return"]
        sol = collections.defaultdict(set)
        for b in binds:
            src = b.get("x", b[r])
            sol[b[r]].update(self.f.ent.get(src, {}).get("docs", []))
            if r != "x" and isinstance(plan["where"][0].get("s"), dict):   # 상수 주어(L)
                sol[b[r]].update(d for q in self.by_name.get(norm(plan["where"][0]["s"]["const"]), set())
                                 for d in self.f.ent.get(q, {}).get("docs", []))
        return {k: sorted(v) for k, v in sol.items()}


def export_facts(f):
    """오라클 그래프 원천 — 본문 근거가 있는 사실만."""
    classes = {t for ts in f.types.values() for t in ts}
    tbox = {}
    for c in classes:
        for a in f.ancestors(c):
            sup = (f.objects.get(a) or {}).get("claims", {}).get("P279", [])
            tbox[a] = {"label": f.label(a), "super": sup}
    return {
        "entities": f.ent,
        "types": {s: list(ts) for s, ts in f.types.items()},
        "relations": [{"s": s, "p": rel, "o": o} for (s, rel), objs in f.rels.items() for o in objs],
        "classes": tbox,
    }


def main():
    docs_path, wd, out = sys.argv[1], sys.argv[2], sys.argv[3]
    docs = [json.loads(l) for l in open(docs_path)]
    subjects = json.load(open(os.path.join(wd, "subjects.json")))
    objects = json.load(open(os.path.join(wd, "objects.json")))
    f = Facts(docs, subjects, objects)
    qs = build(f)
    os.makedirs(out, exist_ok=True)
    blob = json.dumps(qs, ensure_ascii=False, indent=1)
    open(os.path.join(out, "wiki2_bench.json"), "w").write(blob)
    json.dump(export_facts(f), open(os.path.join(out, "wiki2_oracle_facts.json"), "w"), ensure_ascii=False)
    sha = hashlib.sha256(blob.encode()).hexdigest()[:16]
    forms = collections.Counter(q["form"] for q in qs)
    print(f"subjects={sum(1 for e in f.ent.values() if e['subject'])} rel_facts="
          f"{sum(len(v) for v in f.rels.values())} type_facts={sum(len(v) for v in f.types.values())}")
    print(f"questions={len(qs)} forms={dict(forms)} max_share={max(forms.values()) / len(qs):.0%} sha={sha}")


if __name__ == "__main__":
    main()
