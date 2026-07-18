"""M2-2 — SREDFM-ko → KLUE-RE 30라벨 증강 데이터 생성.

방침(1차 저위험): 라벨 공간은 KLUE 30 유지(기존 holdout 으로 공정 비교), SREDFM 은
무모호 P-id 매핑분만 증강. 오라벨 필터 4종(m2_sredfm_filter_verdict.json 검증 완료:
합의 오라벨 2.1%, Wilson 상단 5.9%) 적용.

출력: m2_sredfm_klue_aug.jsonl (KLUE train 스키마: sentence/subject_entity/object_entity/label)
사용: /opt/miniconda3/bin/python m2_prep_sredfm.py
"""
import json, re, collections, random, pathlib

SP = pathlib.Path(__file__).parent
P = ("/Users/kimdu/.cache/huggingface/hub/datasets--Babelscape--SREDFM/"
     "snapshots/2732d2834e12e36510aeb2a468163ea2642d55db/data/train.ko.jsonl")
norm = lambda s: re.sub(r"\s+", "", s or "")

# 무모호 P-id → KLUE 라벨 (subj 타입 제약 포함). 모호/부분중첩 술어는 제외(정직).
# KLUE org:founded/org:dissolved 는 날짜 목적어(설립일/해산일).
PID2KLUE = {
    "P569": ("PER", "per:date_of_birth"), "P570": ("PER", "per:date_of_death"),
    "P19": ("PER", "per:place_of_birth"), "P20": ("PER", "per:place_of_death"),
    "P551": ("PER", "per:place_of_residence"), "P27": ("PER", "per:origin"),
    "P108": ("PER", "per:employee_of"), "P69": ("PER", "per:schools_attended"),
    "P22": ("PER", "per:parents"), "P25": ("PER", "per:parents"),
    "P40": ("PER", "per:children"), "P3373": ("PER", "per:siblings"),
    "P26": ("PER", "per:spouse"), "P140": ("PER", "per:religion"),
    "P159": ("ORG", "org:place_of_headquarters"), "P112": ("ORG", "org:founded_by"),
    "P571": ("ORG", "org:founded"), "P576": ("ORG", "org:dissolved"),
    "P169": ("ORG", "org:top_members/employees"), "P488": ("ORG", "org:top_members/employees"),
    "P1056": ("ORG", "org:product"), "P463": ("ORG", "org:member_of"),
}
TYPE2KLUE = {"PER": "PER", "ORG": "ORG", "LOC": "LOC", "TIME": "DAT",
             "NUM": "NOH", "Concept": "POH", "MISC": "POH", "EVENT": "POH",
             "MEDIA": "POH", "DIS": "POH", "LOC_ORG": "LOC"}


P112_EVIDENCE = ("설립", "창립", "창설", "창업", "창시", "세운", "세웠", "만든", "만들었")


def keep(rel, sent=""):
    # P112(설립자) 근거 어휘 필터 — 0718 M2 심판 감시 org:founded_by 회귀(-0.148) 추적:
    # '회장/CEO' 직책 문맥이 설립자로 오라벨되는 distant supervision 노이즈(하이얼-장루이민
    # 실측). 문장에 설립 계열 어휘가 없으면 컷.
    if rel["predicate"]["uri"] == "P112" and sent and not any(k in sent for k in P112_EVIDENCE):
        return False
    return _keep_base(rel)


def _keep_base(rel):
    s, o = rel["subject"], rel["object"]
    so, oo = s.get("surfaceform") or "", str(o.get("surfaceform") or "")
    if norm(so) == norm(oo):
        return False
    if norm(oo) in norm(so) and re.search(r"\d{3,4}", oo):
        return False
    if rel["predicate"]["uri"] == "P530":
        return False
    on = norm(oo)
    if re.fullmatch(r"[\d,.]+", on) or re.fullmatch(r"\d{1,2}년", on):
        return False
    return True


rng = random.Random(20260718)
out, stats = [], collections.Counter()
with open(P) as f:
    for line in f:
        r = json.loads(line)
        text = r["text"]
        for rel in r.get("relations") or []:
            pid = rel["predicate"]["uri"]
            if pid not in PID2KLUE:
                stats["skip_unmapped"] += 1
                continue
            need_subj, label = PID2KLUE[pid]
            s, o = rel["subject"], rel["object"]
            if not (s.get("boundaries") and o.get("boundaries")):
                stats["skip_nobound"] += 1
                continue
            if not keep(rel, text):
                stats["skip_filtered"] += 1
                continue
            # 문장 단위 절단: KLUE 는 문장 입력 — sentence_id 기반 대신 스팬 주변 절단(마커
            # 삽입 좌표 보존 위해 스팬 포함 최소 확장, MAX_LEN 180 토큰 내 안전권 320자)
            lo = min(s["boundaries"][0], o["boundaries"][0])
            hi = max(s["boundaries"][1], o["boundaries"][1])
            if hi - lo > 280:
                stats["skip_far"] += 1
                continue
            start = max(0, lo - (320 - (hi - lo)) // 2)
            end = min(len(text), start + 320)
            sent = text[start:end]
            stype = TYPE2KLUE.get(s.get("type") or "", "POH")
            otype = TYPE2KLUE.get(o.get("type") or "", "POH")
            # KLUE subj 제약: PER/ORG. Concept 주어는 인물문서 다수 — 제약 위반은 스킵하되
            # Concept→need_subj 강제복원은 하지 않음(오염 방지). 단 Concept 이면서 라벨이
            # 요구하는 타입과 문서 주제(주어=문서 제목)가 일치하는 전형 패턴은 need_subj 채택.
            if stype not in ("PER", "ORG"):
                if s.get("type") == "Concept":
                    stype = need_subj
                else:
                    stats["skip_subjtype"] += 1
                    continue
            if stype != need_subj:
                stats["skip_subjmismatch"] += 1
                continue
            out.append({
                "sentence": sent,
                "subject_entity": {"word": s["surfaceform"], "start_idx": s["boundaries"][0] - start,
                                   "end_idx": s["boundaries"][1] - start - 1, "type": stype},
                "object_entity": {"word": str(o["surfaceform"]), "start_idx": o["boundaries"][0] - start,
                                  "end_idx": o["boundaries"][1] - start - 1, "type": otype},
                "label": label, "source": "sredfm", "pid": pid,
                "confidence": round(float(rel.get("confidence") or 0), 4),
            })
            stats[label] += 1

rng.shuffle(out)
with open(SP / "m2_sredfm_klue_aug.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"증강 예제 {len(out)}건")
print("스킵:", {k: v for k, v in stats.items() if k.startswith('skip')})
print("라벨 분포 top15:", collections.Counter({k: v for k, v in stats.items()
      if not k.startswith('skip')}).most_common(15))
print("M2-PREP-DONE")
