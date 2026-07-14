"""KLUE-RE 30 관계 라벨 — HF datasets/klue/klue README 공식 순서 (2026-07-14 확인).

타입 제약(TYPE_CONSTRAINT)은 라벨 '의미 정의'에서 결정적으로 유도한 것(train 통계 아님).
per:date_of_birth 는 정의상 subj=PER, obj=날짜(DAT). 이는 TACRED 계열 표준 관행.
"""

LABELS = [
    "no_relation",                          # 0
    "org:dissolved",                        # 1
    "org:founded",                          # 2
    "org:place_of_headquarters",            # 3
    "org:alternate_names",                  # 4
    "org:member_of",                        # 5
    "org:members",                          # 6
    "org:political/religious_affiliation",  # 7
    "org:product",                          # 8
    "org:founded_by",                       # 9
    "org:top_members/employees",            # 10
    "org:number_of_employees/members",      # 11
    "per:date_of_birth",                    # 12
    "per:date_of_death",                    # 13
    "per:place_of_birth",                   # 14
    "per:place_of_death",                   # 15
    "per:place_of_residence",               # 16
    "per:origin",                           # 17
    "per:employee_of",                      # 18
    "per:schools_attended",                 # 19
    "per:alternate_names",                  # 20
    "per:parents",                          # 21
    "per:children",                         # 22
    "per:siblings",                         # 23
    "per:spouse",                           # 24
    "per:other_family",                     # 25
    "per:colleagues",                       # 26
    "per:product",                          # 27
    "per:religion",                         # 28
    "per:title",                            # 29
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# subj 타입은 라벨 접두(org:/per:)가 강제. obj 타입 제약은 라벨 의미에서 유도.
# KLUE 개체 타입: PER, ORG, LOC, DAT, POH(기타 고유명), NOH(수량 아닌 기타 수사)
_ANY = None  # 제약 없음
TYPE_CONSTRAINT = {
    "org:dissolved":                       ("ORG", {"DAT"}),
    "org:founded":                         ("ORG", {"DAT"}),
    "org:place_of_headquarters":           ("ORG", {"LOC", "POH", "ORG"}),
    "org:alternate_names":                 ("ORG", {"ORG", "POH"}),
    "org:member_of":                       ("ORG", {"ORG", "POH", "LOC"}),
    "org:members":                         ("ORG", {"ORG", "POH", "LOC", "PER"}),
    "org:political/religious_affiliation": ("ORG", {"ORG", "POH"}),
    "org:product":                         ("ORG", {"POH", "ORG"}),
    "org:founded_by":                      ("ORG", {"PER", "ORG"}),
    "org:top_members/employees":           ("ORG", {"PER"}),
    "org:number_of_employees/members":     ("ORG", {"NOH"}),
    "per:date_of_birth":                   ("PER", {"DAT"}),
    "per:date_of_death":                   ("PER", {"DAT"}),
    "per:place_of_birth":                  ("PER", {"LOC", "POH", "ORG"}),
    "per:place_of_death":                  ("PER", {"LOC", "POH", "ORG"}),
    "per:place_of_residence":              ("PER", {"LOC", "POH", "ORG"}),
    "per:origin":                          ("PER", {"LOC", "POH", "ORG", "DAT"}),
    "per:employee_of":                     ("PER", {"ORG", "POH", "PER"}),
    "per:schools_attended":                ("PER", {"ORG", "POH"}),
    "per:alternate_names":                 ("PER", {"PER", "POH"}),
    "per:parents":                         ("PER", {"PER"}),
    "per:children":                        ("PER", {"PER"}),
    "per:siblings":                        ("PER", {"PER"}),
    "per:spouse":                          ("PER", {"PER"}),
    "per:other_family":                    ("PER", {"PER"}),
    "per:colleagues":                      ("PER", {"PER", "ORG", "POH"}),
    "per:product":                         ("PER", {"POH", "ORG"}),
    "per:religion":                        ("PER", {"ORG", "POH"}),
    "per:title":                           ("PER", {"POH", "ORG", "PER"}),
}


def type_ok(label: str, subj_type: str, obj_type: str) -> bool:
    """라벨 의미 정의 기반 타입 게이트. no_relation 은 항상 허용."""
    if label == "no_relation":
        return True
    st, ot = TYPE_CONSTRAINT[label]
    return subj_type == st and (ot is _ANY or obj_type in ot)
