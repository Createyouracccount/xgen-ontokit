"""KLUE-RE 결정적(LLM-free) 관계 분류기 — 한국어 통사·구조 신호 채널 v2.

계층 89 달성의 원리 이식: "표면 문자열이 아니라 한국어 문법 구조가 의미를
노출하는 지점"을 타게팅. 뉴스·위키 문장의 관계는 동사 SVO 가 아니라(연결력
실측 0.8%) 아래 구조에 산다:

  ① 괄호 구조   "기독교방송(CBS)" → alternate_names
                "김은호(金殷鎬, 1892년 6월 24일 ~ 1979년…)" → date_of_birth/death
  ② 직함 병치   "금호고속 이덕연 사장"(ORG,PER→top_members)
                "김정은 조선로동당 위원장"(PER,ORG→employee_of)
                "애플 CEO 팀 쿡"(PER,POH→title)
  ③ 관형격 방향 "나치 독일의 독일 국방군" → member_of (X의 Y: Y∈X)
                "A의 아버지 B" → parents (B=A의 아버지)
  ④ 서술 트리거 "1996년 문을 연" / "글래스고에 기반한" → founded / headquarters
  ⑤ 계사 보문   "…는 대한민국의 소설가이다" → title + origin

모든 채널은 문법 구조 + 유한 폐어휘(직함·친족·설립류 — 도메인 무관 일반어).
LLM/임베딩/학습 0회. 타입 게이트(labels.TYPE_CONSTRAINT)로 오발화 차단.
튜닝은 tune 분할에서만(홀드아웃 미접촉 — 누수 차단 프로토콜).
"""
import re

from labels import LABEL2ID, type_ok

# ── 유한 폐어휘(일반어 사전 — 특정 도메인·고유명 금지) ──
_TITLE = ("대통령|총리|수상|장관|차관|총장|부총장|학장|총재|의장|부의장|의원|시장|군수|"
          "구청장|지사|대사|사무총장|사무국장|위원장|회장|부회장|사장|부사장|대표이사|대표|"
          "이사장|이사|전무|상무|본부장|단장|팀장|실장|국장|처장|청장|원장|소장|점장|"
          "감독|코치|단주|구단주|CEO|CTO|CFO|COO|회주|주교|대주교|추기경|교황|목사|신부|"
          "스님|주지|왕|여왕|황제|왕비|왕세자|공작|백작|장군|제독|원수|대령|교수|박사|"
          "후보|선수|투수|타자|골키퍼|수비수|공격수|미드필더|센터|가드|포워드|멤버|리더|"
          "보컬|메인보컬|래퍼|서기|비서|차장|과장|주장|부주장|매니저|앵커|캐스터|위원|연구원")
_TITLE_RE = re.compile(rf"(?:{_TITLE})")
_JOB = ("소설가|작가|시인|화가|가수|배우|래퍼|감독|프로듀서|작곡가|작사가|연주가|지휘자|"
        "성우|개그맨|코미디언|아나운서|기자|언론인|정치인|정치가|기업인|사업가|의사|"
        "변호사|검사|판사|교수|과학자|물리학자|수학자|철학자|역사가|선수|축구선수|"
        "야구선수|농구선수|골퍼|바둑기사|승려|목회자|성직자|군인|외교관|관료|공무원|"
        "아동문학가|극작가|번역가|평론가|만화가|디자이너|모델|무용가|안무가|피아니스트|"
        "바이올리니스트|첼리스트|성악가|국악인|요리사|셰프|황족|왕족|귀족|수도사|수녀|"
        "연출가|극작가|방송인|연예인|음악가|미술가|조각가|건축가|사진가|해설가|캐스터|"
        "독립운동가|운동가|혁명가|사상가|교육자|법조인|은행가|실업가|경제학자|사회학자|"
        "심리학자|인류학자|고고학자|천문학자|생물학자|화학자|지리학자|신학자|언어학자|"
        "탐험가|비행사|우주인|성리학자|유학자|문신|무신|문관|무관|학자|명장|재상|공신")
_JOB_RE = re.compile(rf"(?:{_JOB})")
_KIN_PARENT = re.compile(r"아버지|어머니|부친|모친|아빠|엄마|생부|생모|양부|양모|친부|친모")
_KIN_CHILD = re.compile(r"아들|딸|장남|차남|삼남|사남|장녀|차녀|삼녀|막내|자녀|외아들|외동딸|소생")
_KIN_SIBLING = re.compile(r"동생|누나|언니|오빠|남동생|여동생|형제|자매|남매|쌍둥이|친형|친누나|친동생")
_KIN_SIBLING_FORM = re.compile(r"(?:^|[\s(])형(?:[\s),이과와은는의]|$)")  # '형'은 단독 어절만
_KIN_SPOUSE = re.compile(r"아내|남편|부인|배우자|와이프|부군|왕비가?\s?되|황후가?\s?되")
_KIN_OTHER = re.compile(r"할아버지|할머니|조부|조모|외조부|외조모|삼촌|외삼촌|이모|고모|"
                        r"조카|손자|손녀|사위|며느리|시아버지|시어머니|장인|장모|사촌|"
                        r"큰아버지|작은아버지|증조|외손|친척|처남|매형|매제|형수|제수|올케|시누이|"
                        r"의붓|이복|외숙|숙부|백부|고조|선조|후손|자손|손아래|손위")
_FOUND = re.compile(r"설립(?!자)|창립(?!자)|창설(?!자)|창건(?!자)|창단(?!주)|창간(?!인)|"
                    r"창업(?!자)|창당|창회|건립|개교|개국|개원|개장|"
                    r"출범|발족|결성|조직되|세워|세운|문을\s?연")
_FOUNDER = re.compile(r"설립자|창립자|창업자|창설자|창건자|창단주|창간인|설립하|창립하|창당하|"
                      r"창설하|세운|창업하|건립하")
_DISSOLVE = re.compile(r"해체|해산|폐지|폐교|폐국|폐업|소멸|와해|문을\s?닫")
_HQ = re.compile(r"본부|본사|본점|본거지|근거지|소재|위치한|위치해|자리한|자리\s?잡|기반[을한]|"
                 r"연고|둥지|거점")
_BORN = re.compile(r"태어났|태어난|출생|출생지")
_DIED = re.compile(r"사망|별세|서거|타계|숨졌|숨진|세상을\s?떠|눈을\s?감|전사|순직|처형|암살")
_RESIDE = re.compile(r"거주|거처|살았|살고|이주|정착|머물|은거|칩거")
_SCHOOL = re.compile(r"학교|대학|학원|학당|스쿨|고교")
_SCHOOL_V = re.compile(r"졸업|입학|재학|수학|편입|중퇴|진학|유학|출신|학사|석사|박사")
_RELIGION = re.compile(r"불교|기독교|개신교|천주교|가톨릭|이슬람|유대교|힌두교|천도교|"
                       r"원불교|성공회|정교회|장로교|감리교|침례교")
_PRODUCT_V = re.compile(r"출시|발매|출간|출판|발표한|발행|제작|개발|생산|선보인|내놓|만든")
_MEMBER_HINT = re.compile(r"소속|산하|계열|가맹|가입|합류|편입")
_EMPLOY_V = re.compile(r"활약|근무|재직|재임|복무|데뷔|이적|입단|영입|은퇴|방출|뛰었|뛰고|"
                       r"입사|취임|부임|임명|선임|발탁|기용|경질|해임|사임|사퇴|은퇴")
_COLLEAGUE = re.compile(r"(?:함께|같이)\s?(?:활동|결성|공연|작업|출연|우승|경기)|"
                        r"공동\s?(?:통치|수상|작업|우승|집권|창업|연구)|"
                        r"동료|듀오|콤비|같은\s?(?:팀|소속사|그룹)|선후배|파트너")
_ALIAS_HINT = re.compile(r"본명|예명|일명|별칭|별명|애칭|이하|개명|필명|법명|아명|호는|약칭")
_RENAME = re.compile(r"전신|후신|개칭|개명하|개편되|사명[을은]|법인명[을은]|이름[을은]\s?바꾸|"
                     r"[로으]로\s?바뀌|현재의|바뀌기\s?전")
_ORIGIN_HINT = re.compile(r"출신|국적|태생|출생")
_NOH_MEMBER = re.compile(r"명|인|회원|직원|사원|조합원|당원|단원|선수")
_WORK_QUOTE = re.compile(r"[《〈『「'\"]")

_DATE_RANGE = re.compile(r"[~∼–—]|\s[-−]\s")


def _best_spans(sent: str, sw: str, ow: str, ss: int, os_: int):
    """엔티티가 문장에 여러 번 나오면 최근접 출현쌍으로 재정렬.

    KLUE gold 스팬이 인용문 속 재출현을 가리키는 경우가 있어("강은탁 소속사 X …
    \"강은탁 어머니가…\"" 에서 두 번째 강은탁), 증거가 있는 최근접 쌍을 쓴다.
    결정적(문자열 검색), 미출현 시 주어진 스팬 유지.
    """
    def occs(w, default):
        out, i = [], sent.find(w)
        while i != -1:
            out.append(i)
            i = sent.find(w, i + 1)
        return out or [default]
    best, bd = None, None
    for a in occs(sw, ss):
        for b in occs(ow, os_):
            if a == b:
                continue
            d = abs(a - b)
            if bd is None or d < bd:
                bd, best = d, (a, b)
    if best is None:
        return ss, os_
    return best


def _paren_after(sent: str, end: int):
    """엔티티 직후에 여는 괄호가 있으면 그 (열림, 닫힘) 인덱스."""
    i = end + 1
    while i < len(sent) and sent[i] in "  ":
        i += 1
    if i >= len(sent) or sent[i] not in "((":
        return None
    depth = 0
    for j in range(i, len(sent)):
        if sent[j] in "((":
            depth += 1
        elif sent[j] in "))":
            depth -= 1
            if depth == 0:
                return (i, j)
    return (i, len(sent) - 1)


def predict(row) -> int:
    """(sentence, subj, obj) → 라벨 id. 발화 없으면 0(no_relation)."""
    sent = row["sentence"]
    s, o = row["subject_entity"], row["object_entity"]
    sw, ow = s["word"], o["word"]
    st, ot = s["type"], o["type"]
    ss, os_ = _best_spans(sent, sw, ow, s["start_idx"], o["start_idx"])
    se, oe = ss + len(sw) - 1, os_ + len(ow) - 1

    def ok(label):
        return type_ok(label, st, ot)

    def lid(label):
        return LABEL2ID[label]

    between = sent[min(se, oe) + 1: max(ss, os_)]
    gap = len(between)
    subj_first = ss < os_
    right_of_obj = sent[oe + 1: oe + 12]
    right_of_subj = sent[se + 1: se + 12]
    win = sent[max(0, os_ - 8): min(len(sent), oe + 14)]  # obj 주변 창

    # ── ① 괄호 구조: subj 직후 괄호 안에 obj ──
    par = _paren_after(sent, se)
    if par and par[0] <= os_ and oe <= par[1]:
        inner = sent[par[0] + 1: par[1]]
        if ot == "DAT":
            tilde = _DATE_RANGE.search(inner)
            obj_off = os_ - (par[0] + 1)
            if tilde:
                if obj_off < tilde.start() and ok("per:date_of_birth"):
                    return lid("per:date_of_birth")
                if obj_off > tilde.start() and ok("per:date_of_death"):
                    return lid("per:date_of_death")
            if _DIED.search(inner[:obj_off]) and ok("per:date_of_death"):
                return lid("per:date_of_death")
            if ok("per:date_of_birth"):
                return lid("per:date_of_birth")
        elif ot == "LOC" and _BORN.search(inner) and ok("per:place_of_birth"):
            return lid("per:place_of_birth")
        elif _TITLE_RE.search(sent[par[0]: os_]) and st == "ORG" and ot == "PER" \
                and ok("org:top_members/employees"):
            return lid("org:top_members/employees")  # "경남대학교(총장 박재규)"
        elif st == "ORG" and ok("org:alternate_names"):
            return lid("org:alternate_names")
        elif st == "PER" and ok("per:alternate_names"):
            return lid("per:alternate_names")

    # ①-보강: obj 직후 괄호 안에 subj — "금성사(현 LG전자)"
    #   단 괄호 안이 subj 단독일 때만("다비치(이해리 강민경)"=그룹 멤버 나열, 별칭 아님)
    par_o = _paren_after(sent, oe)
    if par_o and par_o[0] <= ss and se <= par_o[1]:
        inner_o = sent[par_o[0] + 1: par_o[1]].strip()
        alias_alone = inner_o == sw or re.fullmatch(rf"(?:현|구|옛|舊)\s*{re.escape(sw)}", inner_o)
        if alias_alone and st == "ORG" and ot in ("ORG", "POH") and ok("org:alternate_names"):
            return lid("org:alternate_names")
        if alias_alone and st == "PER" and ot in ("PER", "POH") and ok("per:alternate_names"):
            return lid("per:alternate_names")

    # 별칭 힌트("본명: X", "이하 X") — 괄호 밖 포함
    if gap <= 14 and _ALIAS_HINT.search(sent[max(0, min(ss, os_) - 10): max(se, oe) + 4]):
        if st == "PER" and ot in ("PER", "POH") and ok("per:alternate_names"):
            return lid("per:alternate_names")
        if st == "ORG" and ok("org:alternate_names"):
            return lid("org:alternate_names")

    # 개칭·전신: "중앙정보국의 전신인 전략사무국" / "법인명을 조선방송협회로 개칭"
    if st == "ORG" and ot in ("ORG", "POH") and gap <= 30 and \
            _RENAME.search(between + " " + right_of_obj[:8] + " " + right_of_subj[:8]) and \
            ok("org:alternate_names"):
        return lid("org:alternate_names")

    # ── ③-가 친족 (PER-PER 최우선 — 직함·동료보다 정밀) ──
    if st == "PER" and ot == "PER":
        seg = between if gap <= 14 else ""
        kin_zone = seg + " " + right_of_obj[:7] + " " + right_of_subj[:7]
        left_of_subj = sent[max(0, ss - 5): ss]
        left_of_obj = sent[max(0, os_ - 5): os_]
        # "subj와 obj의 아들/사이/소생" → spouse
        if subj_first and gap <= 4 and re.search(r"[와과]", between + sent[se + 1: se + 3]) and \
                re.match(r"의?\s*(?:아들|딸|사이|소생|슬하|자녀)", right_of_obj.lstrip()) and \
                ok("per:spouse"):
            return lid("per:spouse")
        # "아버지는 A … 어머니는 B" → A,B는 부부
        if re.search(r"아버지|부친", left_of_subj + right_of_subj[:4]) and \
                re.search(r"어머니|모친", left_of_obj + right_of_obj[:4]) and ok("per:spouse"):
            return lid("per:spouse")
        if re.search(r"어머니|모친", left_of_subj + right_of_subj[:4]) and \
                re.search(r"아버지|부친", left_of_obj + right_of_obj[:4]) and ok("per:spouse"):
            return lid("per:spouse")
        if (_KIN_SPOUSE.search(kin_zone) or re.search(r"결혼|혼인|재혼|혼례", seg)) and \
                ok("per:spouse"):
            return lid("per:spouse")
        if (_KIN_SIBLING.search(kin_zone) or _KIN_SIBLING_FORM.search(kin_zone)) and \
                ok("per:siblings"):
            return lid("per:siblings")
        # 방향성 친족: "X의 <친족> Y" = Y는 X의 <친족> / "<친족> X" 직전 수식
        if _KIN_PARENT.search(seg):
            if subj_first and ok("per:parents"):
                return lid("per:parents")      # "subj의 아버지 obj" → obj=부모
            if not subj_first and ok("per:children"):
                return lid("per:children")     # "obj의 아버지 subj" → subj=부모
        if _KIN_CHILD.search(seg):
            if subj_first and ok("per:children"):
                return lid("per:children")
            if not subj_first and ok("per:parents"):
                return lid("per:parents")
        if _KIN_PARENT.search(left_of_obj) and ok("per:parents"):
            return lid("per:parents")          # "아버지 obj" 직전 수식
        if _KIN_PARENT.search(left_of_subj) and ok("per:children"):
            return lid("per:children")         # "아버지 subj" → subj=부모
        if _KIN_OTHER.search(kin_zone) and ok("per:other_family"):
            return lid("per:other_family")

    # PER-POH: obj 자체가 친족 표기("어머니 김성애")
    if st == "PER" and ot in ("POH", "PER"):
        if _KIN_PARENT.search(ow) and ok("per:parents"):
            return lid("per:parents")
        if _KIN_SPOUSE.search(ow) and ok("per:spouse"):
            return lid("per:spouse")

    # ── ② 직함 병치 ──
    clean_between = "," not in between and "、" not in between  # 나열 경계 넘김 방지
    if st == "ORG" and ot == "PER" and ok("org:top_members/employees"):
        # "금호고속 이덕연 사장" / "보건복지부 차관 김강립"
        if gap <= 22 and clean_between and (_TITLE_RE.match(right_of_obj.strip(" ,")) or
                                            _TITLE_RE.search(between)):
            return lid("org:top_members/employees")
        if _FOUNDER.search(win) and ok("org:founded_by"):
            return lid("org:founded_by")
    if st == "PER" and ot == "ORG" and ok("per:employee_of"):
        # "김정은 조선로동당 위원장" / "바른정당 대선 후보였던 유승민"
        if gap <= 22 and clean_between and (_TITLE_RE.match(right_of_obj.strip(" ,")) or
                                            _TITLE_RE.search(between)):
            return lid("per:employee_of")
    if st == "PER" and ot == "POH" and ok("per:title"):
        if _TITLE_RE.search(ow) or _JOB_RE.search(ow):
            if gap <= 3:                        # "총리 존 디펜베이커" / "팀 쿡 CEO"
                return lid("per:title")
            tail = sent[oe + 1: oe + 40]
            if re.match(r"\s*(이다|였다|이었|이며|이고|이자|겸\s|다\.|로\s|를\s?지|을\s?지"
                        r"|가\s?되|이\s?되|로\s?임명|로\s?선임|로\s?취임|로\s?선출|에\s?올"
                        r"|로서|로\s?활동|로\s?재직)", tail):
                return lid("per:title")
            # 나열 계사: "소설가, 아동문학가이다" / "포지션은 공격수"
            if re.match(r"\s*[,、]", tail) and re.search(r"이다|였다", tail):
                return lid("per:title")
            if re.search(r"포지션은|직책은|직위는|계급은", sent[max(0, os_ - 10): os_]):
                return lid("per:title")

    # ── ③-나 조직 포함(관형격 방향) ──
    genitive_oc = (not subj_first and 0 <= gap <= 8 and
                   re.match(r"^의[\s]", sent[oe + 1: oe + 3] + " "))
    genitive_so = (subj_first and 0 <= gap <= 8 and
                   re.match(r"^의[\s]", sent[se + 1: se + 3] + " "))
    if st == "ORG" and ot in ("ORG", "POH", "LOC"):
        # 행정구역 접미 계층: "화순읍 ⊂ 화순군" (한국 행정 접미는 유한 문법 집합)
        if sw and ow and sw[-1] in "읍면동리" and ow[-1] in "군시구도" and \
                ok("org:member_of"):
            return lid("org:member_of")
        if genitive_oc:
            if ot == "LOC" and ok("org:place_of_headquarters"):
                return lid("org:place_of_headquarters")  # "그리스의 축구 클럽"
            if ok("org:member_of"):
                return lid("org:member_of")    # "나치 독일의 독일 국방군"
        if gap <= 16 and _MEMBER_HINT.search(between) and ok("org:member_of"):
            return lid("org:member_of")
        if genitive_so and ot != "LOC" and ok("org:members"):
            return lid("org:members")
        # LOC 병치 직전: "상하이 고려공산당" / "사우디 아람코"
        if not subj_first and gap <= 1 and ot == "LOC" and \
                ok("org:place_of_headquarters"):
            return lid("org:place_of_headquarters")
        # 리그·연맹 참가: "분데스리가 우승/참가/승격"
        if re.search(r"리그|연맹|협회|연합|디비전|컨퍼런스", ow) and \
                re.search(r"우승|참가|승격|강등|소속|가맹|경기|라운드", right_of_obj + between[-10:]) and \
                ok("org:member_of"):
            return lid("org:member_of")

    # ── ④ 서술 트리거 ──
    if st == "ORG":
        if ot == "DAT":
            win_d = sent[max(0, os_ - 8): min(len(sent), oe + 30)]
            if _DISSOLVE.search(win_d) and ok("org:dissolved"):
                return lid("org:dissolved")
            if _FOUND.search(win_d) and ok("org:founded"):
                return lid("org:founded")
        if ot == "PER" and (_FOUND.search(win) or _FOUNDER.search(win)) and \
                ok("org:founded_by"):
            return lid("org:founded_by")
        if ot in ("LOC", "POH") and _HQ.search(win) and ok("org:place_of_headquarters"):
            return lid("org:place_of_headquarters")
        if ot == "NOH" and _NOH_MEMBER.search(sent[oe + 1: oe + 5]) and \
                ok("org:number_of_employees/members"):
            return lid("org:number_of_employees/members")
        if ot in ("POH",) and ok("org:product"):
            if _PRODUCT_V.search(win) or \
                    re.match(r"\s?(?:생산|제조|제작)", right_of_obj):
                return lid("org:product")
    if st == "PER":
        if ot == "DAT":
            if _DIED.search(win) and ok("per:date_of_death"):
                return lid("per:date_of_death")
            if _BORN.search(win) and ok("per:date_of_birth"):
                return lid("per:date_of_birth")
        if ot in ("LOC", "POH", "ORG"):
            if _DIED.search(win) and ok("per:place_of_death"):
                return lid("per:place_of_death")
            if _BORN.search(win) and ok("per:place_of_birth"):
                return lid("per:place_of_birth")
            if _RESIDE.search(win) and ok("per:place_of_residence"):
                return lid("per:place_of_residence")
        if ot in ("ORG", "POH") and _SCHOOL.search(ow) and ok("per:schools_attended"):
            if _SCHOOL_V.search(win) or genitive_oc or gap <= 6:
                return lid("per:schools_attended")
        if _RELIGION.search(ow) and ok("per:religion") and \
                re.search(r"신자|신도|신앙|개종|귀의|믿|독실|세례|종교", win):
            return lid("per:religion")
        if ot in ("POH", "ORG") and ok("per:product"):
            # "마이클 잭슨의 《Thriller》" / "경희대학교 설립자 조영식"
            if _FOUNDER.search(win):
                return lid("per:product")
            if subj_first and gap <= 6 and genitive_so and _WORK_QUOTE.search(between + sent[os_ - 1: os_]):
                return lid("per:product")
            if _PRODUCT_V.search(right_of_obj) and subj_first and gap <= 20:
                return lid("per:product")

    # ── ⑤ origin / employee / colleagues ──
    if st == "PER":
        if ot in ("LOC", "POH", "ORG", "DAT") and ok("per:origin"):
            if _ORIGIN_HINT.search(right_of_obj[:6]):
                return lid("per:origin")
            after_obj = sent[oe + 1: oe + 20].lstrip()
            # "obj의 <직업>" 계사 술어("…은 대한민국의 배구 선수였다") — 어순 무관
            if after_obj.startswith("의") and \
                    (_TITLE_RE.search(after_obj[:18]) or _JOB_RE.search(after_obj[:18])):
                return lid("per:origin")
            # "obj <직함> subj" 병치("러시아 대통령 메드베데프")
            if not subj_first and gap <= 16 and (
                    _TITLE_RE.match(after_obj) or _JOB_RE.match(after_obj)):
                return lid("per:origin")
            # "obj의 subj" 직결 관형격(국가 obj 한정): "제국의 프란츠 대공"
            if not subj_first and ot == "LOC" and genitive_oc:
                return lid("per:origin")
            if re.match(r"계\s", sent[oe + 1: oe + 3] + " "):  # "독일계"
                return lid("per:origin")
        if ot in ("ORG", "POH") and ok("per:employee_of") and \
                not (ot == "POH" and (_TITLE_RE.search(ow) or _JOB_RE.search(ow))):
            if _MEMBER_HINT.search(between if gap <= 16 else "") or \
                    _MEMBER_HINT.search(right_of_subj[:8]) or \
                    _EMPLOY_V.search(right_of_obj) or \
                    (not subj_first and gap <= 14 and
                     (_TITLE_RE.search(between) or _JOB_RE.search(between))):
                return lid("per:employee_of")
        if ot == "PER" and ok("per:colleagues"):
            zone = (between if gap <= 30 else "") + " " + right_of_obj + " " + right_of_subj
            if _COLLEAGUE.search(zone):
                return lid("per:colleagues")

    return 0
