"""한국어 관계(objectProperty) 추출 — 조사 기반 SVO 트리플. LLM 0회.

한국어는 격조사가 문법역할을 명시(head-final SOV):
  주어=JKS(이/가)·JX(은/는)·JKC, 목적어=JKO(을/를), 여격 폴백=JKB(에/에게),
  술어=동사성명사(NNG)+XSV(하/되/시키) 또는 일반동사 VV(명사형화 ㅁ/음).
  "금융위원회는 은행을 감독한다" → (금융위원회, 감독, 은행).

XGEN kg_builder 소비 스키마에 맞춰 {subject, predicate, object, predicate_type,
source_chunks} dict 를 반환. 하위(kg_builder)가 subject/object 를 인스턴스 노드로
자동 등록하므로 NER 등록과 독립.

정밀도 우선 설계: 절에 주어·목적어(또는 여격)·술어가 모두 있을 때만 추출.
v0.7(0711, GT81 오류분석 기반 — 전부 품사 구조 규칙, 수동 단어목록 금지 원칙 준수):
  ①존재구문 가드("우려가 있다고"의 '우려'가 주어를 덮어쓰던 FP 제거)
  ②관형격 '의' 명사구 연결("채무의 지급" — 절단 방지, 공백 너머 연결)
  ③법령참조(XPN/SN/NNB) 버퍼 통과("상법 제391조를"의 '상법' 보존)
  ④여격 목적어 폴백("금융위원회에 등록") + 주제 승격("보험안내자료에는")
  ⑤금지 규범(부정 꼬리 '아니 된다/못한다/수 없다/말' → 술어+' 금지')
  ⑥VV 술어(따른다→따름, 정한다→정함 — 기능·보조동사 폐집합 제외)
  ⑦관형절 억제("인정되는 조합" 억제, 단 '…하는 행위'류 형식명사 head 는 본술어 취급)
  ⑧문장 간 주제 캐리(법령 항·호 열거 — inferred_subject 태깅으로 신뢰도 구분)
"""
from __future__ import annotations
import os
import re

from ..morphology.kiwi_nouns import STOP_HEAD

# 문장 분리 — 종결어미(SF) 또는 개행 기준. 절 경계를 넘는 오연결 방지.
_SENT_SPLIT = re.compile(r'(?<=[.!?。\n])\s+|(?<=다\.)\s*')

_HANGUL = re.compile(r"[가-힣]{2,}")
# 법령 열거 마커(호·목: "1."/"가.") — 주제 캐리(⑧) 수신 대상. ⚠️항(①②…)은 자체
# 주어를 가진 일반 문장이라 제외 — 포함하면 항이 캐리를 갱신하지 못해 캐리 전멸(0711).
_ENUM = re.compile(r"^\s*(?:\d+(?:의\d+)?\)\s|[가-힣]\)\s)")
_ARG = re.compile(r"[가-힣의]{2,}")  # '의' 연결 명사구 허용
# 접두 수사 포함 명사구 — "제3세계"·"제2공화국"처럼 XPN/SN 접두가 어휘의 일부인
# 표면(③' 분기가 버퍼에 포함). 한글-선도 + 한글/숫자 혼합만 허용(숫자-선도는 컷:
# "3세계"·"747" 오살 방지, 0712 숫자토큰 한글한정 원칙과 동일 계열의 닫힌 문자클래스 규칙).
# 알려진 트레이드오프(심판 채택 판정의 기록 조건): 숫자-선도 부착 복합어는 인자
# 전체가 소멸한다 — "2002월드컵을"은 구코드가 (…, 월드컵)을 냈지만 신코드는 무방출.
# 코퍼스 실측 발현 8건은 대부분 '20세기→세기'류 유령 인자의 정당한 제거였고 정당
# 트리플 소멸 0건(정밀도-우선 수용). 필요 시 "XPN-선도 런만 버퍼 포함"으로 완화 가능.
_ARG_NUMMIX = re.compile(r"[가-힣][가-힣0-9의]+")

# ── 문법화된 폐집합(닫힌 문법형태소·기능동사) — "수동 목록 확장 금지" 원칙의 허용예외 ──
_FUNC_VV = {"관하", "대하", "위하", "의하", "인하", "불구하", "비하", "반하"}  # 복합후치사화 동사
_AUX_VV = {"하", "되", "이", "있", "없", "같", "아니하", "못하", "말", "지"}   # 보조·계사류
_DAT_FORMS = {"에", "에게"}                                                  # 여격 조사


def _nominalize(stem: str) -> str:
    """VV 어간 → 명사형(ㅁ/음). 따르→따름, 알리→알림, 정하→정함, 막→막음."""
    if not stem:
        return stem
    code = ord(stem[-1])
    if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 == 0:  # 받침 없음
        return stem[:-1] + chr(code + 16)                        # ㅁ 받침(종성 16)
    return stem + "음"


class KoreanRelationExtractor:
    """조사 기반 한국어 SVO 관계 추출. Kiwi 인스턴스 주입(없으면 생성)."""

    MAX_ARG_LEN = 30  # 주어/목적어 명사구 최대 글자수('의' 연결 허용해 20→30)

    def __init__(self, kiwi=None, enable_carry: bool = True):
        """enable_carry=False 면 문장 간 주제 캐리(⑧)·열거 분배(⑩)를 끈다 —
        캐리 트리플은 inferred_subject 태깅되지만(미본 조문 실측 P≈0.50)
        하류 필터 없이 고정밀 용도로 쓸 때 소스 차단 옵션."""
        if kiwi is None:
            from kiwipiepy import Kiwi  # lazy — extras[korean]
            kiwi = Kiwi()
        self.kiwi = kiwi
        self.enable_carry = enable_carry

    def _flush_noun(self, buf: list[str]) -> str:
        """명사 버퍼 → 명사구 표면형(2자 이상, 불용어/길이/한글 컷).

        '의' 처리는 **버퍼 원소 단위** — buf 의 "의" 단독 원소는 JKG 부착 시에만
        생성되므로(_extract_sentence 의 JKG 분기), 명사 자체의 끝음절 '의'(회의·
        동의·결의 — 법령 최빈어군)는 형태소 원소 내부에 있어 절대 걸리지 않는다.
        이전 구현은 join 된 문자열에 endswith("의")/rpartition("의")를 적용해
        '회의'→'회'(2자 미달 소멸), '주주의 동의'→'주주의동'(오염 노드가 그래프에
        INSERT), '주주총회결의절차'→'주주총회결' 절단을 만들었다(0711 재심사
        CONFIRMED — 관계 recall 무성 손실 + 가시적 데이터 오염)."""
        if not buf:
            return ""
        # 꼬리 JKG 마커 제거 — "주주의"처럼 관형격에서 끊긴 채 flush 되는 경우
        while buf and buf[-1] == "의":
            buf = buf[:-1]
        # "…의 경우/때" 형식명사 꼬리 제거 — JKG 마커 + STOP_HEAD 원소일 때만.
        # "기금의 경우에는" buf=[기금,의,경우] → [기금]. (STOP_HEAD 재사용 — 새 목록 아님)
        if len(buf) >= 3 and buf[-2] == "의" and buf[-1] in STOP_HEAD:
            buf = buf[:-2]
        surf = "".join(buf).strip()
        if " " in surf:
            # P0-2 정제(둘 다 기존 자산 재사용, 새 목록 없음):
            # ① 선도 불용 세그먼트 스트립 — "대부분 러시아어"→"러시아어",
            #    "다음 대나무 꼬챙이"→"대나무 꼬챙이" (관형사성 명사 오연접 컷)
            # ② 인접 중복 세그먼트 병합 — "간호사 간호사"→"간호사"
            segs = surf.split(" ")
            while len(segs) > 1 and segs[0] in STOP_HEAD:
                segs = segs[1:]
            segs = [s for k, s in enumerate(segs) if k == 0 or s != segs[k - 1]]
            surf = " ".join(segs)
        # P0-2: 공백 보존 연접 시 검증은 공백 제거형(core)으로, 반환은 원표면(surf)으로 —
        # 정규식/불용어 게이트를 우회하지 않으면서 원문 그대로의 명사구를 보존한다.
        core = surf.replace(" ", "")
        if len(core) < 2 or len(surf) > self.MAX_ARG_LEN:
            return ""
        if core in STOP_HEAD or not (_ARG.fullmatch(core) or _ARG_NUMMIX.fullmatch(core)):
            return ""
        return surf

    def _extract_sentence(self, sent: str, init_subject: str = "",
                          is_enum: bool = False, init_prohibit: bool = False):
        """한 문장에서 SVO 트리플 추출. init_subject=이전 문장 주제(캐리, ⑧),
        is_enum=열거 호 라인, init_prohibit=부모 문장이 금지 규범.

        반환: (트리플, 주제캐리, pending_pred — 목적어 없어 미완 술어, final_np — 문말 명사구)."""
        toks = self.kiwi.tokenize(sent)
        n = len(toks)
        # 캐리 주어는 '추정'(subject_used=True) — 트리플에 inferred_subject 태깅됨
        subject, subject_used = (init_subject, True) if init_subject else ("", False)
        obj = dat = ""
        # P0-1(0715 심판): dat 이 '에 의해' 행위자구인지 — JKB 직후 의하/VV 인접
        # *위치 결속*으로만 참(문장 부분문자열 검사 금지 — dat 이 다른 '에'에 결속된
        # 케이스 4~6/19 실측, 스왑 FP 원인). 피동(되/XSV)에서 방향 스왑 트리거.
        dat_agent = False
        noun_buf: list[str] = []
        out: list[dict] = []
        prev_end: int | None = None
        gen_active = False   # 직전 JKG('의') — 공백 너머 명사구 연결(②)
        last_role: str | None = None  # 'dat' 추적 — "에는" 주제 승격(④)
        topic_subject = init_subject  # 캐리 후보 — JX(은/는) 주제 주어만(⑧)
        saved_subject, saved_used = "", True  # 관형절 주어 복원용(JKS 직전 값)
        adn_anchor = ""      # 관형절이 소비한 여격 — "…에 종사하는 자는"의 주어 앵커
        pending_pred = ""    # 목적어 없어 미완성된 본술어 — 열거 분배용(호 라인에 적용)

        def neg_after(i: int) -> bool:
            """술어 뒤 부정 꼬리 → 금지 규범(⑤). 전부 닫힌 문법형태소.
            단 부정 뒤 '수 있'(…하지 아니할 수 있다=재량규정)은 금지 아님."""
            prev_form, neg = "", False
            for t in toks[i + 1:i + 10]:
                # '말'은 금지("~지 말다")일 때만 — 완료상 "~고 말다"(처형되고 만다)를
                # 금지로 오탐하던 버그(R6 심판 적발). 선행 연결어미 '지'를 요구하는
                # 위치 결속으로 분리(둘 다 폐집합 형태소, 어휘 지식 불요).
                hit = ((t.tag == "VX" and (t.form == "못하"
                                           or (t.form == "말" and prev_form == "지")))
                       or (t.tag == "MAG" and t.form == "아니")
                       or t.form in ("않", "아니하")
                       or (t.tag == "VA" and t.form == "없" and prev_form == "수"))
                if hit:
                    neg = True
                elif neg and t.tag == "ETM":
                    return False           # "할 수 없을 때/못한 채무" = 관형·조건, 금지 아님
                elif neg and t.form == "있" and prev_form == "수":
                    return False           # "…아니할 수 있다" = 재량규정, 금지 아님
                elif t.tag == "SF":
                    break
                prev_form = t.form
            return neg

        def is_adnominal(i: int) -> bool:
            """술어 뒤(선어말어미 EP 통과) ETM = 관형절(⑦) — 단 '수/NNB'(모달)나
            '행위' head("~하는 행위" 금지규범 호 열거)는 본술어로 취급."""
            j = i + 1
            while j < n and toks[j].tag == "EP":   # 았/었/시 등 통과
                j += 1
            if j >= n or toks[j].tag != "ETM":
                return False
            if j + 1 < n:
                nx = toks[j + 1]
                if nx.tag == "NNB" and nx.form == "수":
                    return False           # ~할 수 있다/없다 = 모달 본술어
                if is_enum and nx.tag == "NNG" and nx.form == "행위":
                    return False           # "~하는 행위" 호 열거에서만 본술어 취급
            return True

        def emit(pred: str, i: int) -> bool:
            nonlocal obj, dat, subject_used
            o = obj or dat                 # 여격 폴백(④)
            if not (subject and o and pred) or subject == o:
                return False
            p = pred + (" 금지" if (neg_after(i) or init_prohibit) else "")
            tri = {"subject": subject, "predicate": p, "object": o}
            if subject_used:
                tri["inferred_subject"] = True
            subject_used = True
            out.append(tri)
            obj = dat = ""                 # 주어는 생략 대비 유지, 목적어·여격 리셋
            return True

        i = 0
        while i < n:
            t = toks[i]
            tag, form = t.tag, t.form
            if tag in ("NNG", "NNP") or (tag == "XSN" and form != "들" and noun_buf):
                # 띄어쓰기 경계 리셋(오결합 방지) — 단 '의' 연결 중엔 유지(②)
                # XSN(파생접미사 계/별/용)은 붙여서 연장 — "선임계리사"의 계/XSN에서
                # 버퍼가 끊겨 "리사"로 파손되던 결함(0711) 수정. 단 복수 '들'은
                # 어휘 복합의 일부가 아니라 제외(자유텍스트 "백성들" 주어 FP 방지)
                if prev_end is not None and t.start > prev_end and noun_buf and not gen_active:
                    # P0-2 실험: 공백 하나 건너 NN 이 이어지면 리셋 대신 공백 보존
                    # 연접("동방 지역"→'동방 지역'). 기존 리셋은 수식명사를 버려
                    # 파편 인자("지역")를 만들었다(R6 정크 지배 원인).
                    if os.environ.get("ONTOKIT_NN_JOIN", "") == "1" and t.start == prev_end + 1:
                        noun_buf.append(" ")
                    else:
                        noun_buf = []
                gen_active = False
                noun_buf.append(form)
                prev_end = t.start + t.len
                i += 1
                continue
            if tag == "JKG" and noun_buf:                    # ② 관형격 '의'
                # ②' '의' 연쇄 깊이 상한(슬라이딩 ≤1) — 무제한 연접이
                # "소수의거듭제곱의상" 류 유령 노드를 만들어 그래프 연결성을
                # 파편화한다(0714 심판 확정). 두 번째 '의'부터 가장 오래된
                # 세그먼트를 버리고 최근 세그먼트만 유지 — "종의 원소의 원자량"
                # → "원소의원자량". 순수 문법 폐규칙(목록 없음, LLM-free 불변식).
                # 알려진 비용(심판 채택 판정의 기록 조건): '의'를 품은 고유명사가
                # 연쇄에 끼면 참수된다 — "그림의 법칙의 예외"→"법칙의예외"
                # (191건 채점 중 2건). 어휘 지식 없이 회피 불가 = 목록 도입이라
                # 불변식 위반이므로 수용.
                if "의" in noun_buf:
                    noun_buf = noun_buf[noun_buf.index("의") + 1:]
                noun_buf.append("의")
                gen_active = True
                prev_end = t.start + t.len
                i += 1
                continue
            if tag in ("XPN", "SN"):
                # ③' 접두 판별 — XPN/SN 런의 바로 뒤가 *붙어 있는* NNG/NNP면
                # 어휘 접두이므로 버퍼에 포함한다: "제3세계"가 "세계"로,
                # "신세계"가 "세계"로 절단 방출되던 결함(0714 심판: 유령 절단
                # 변이가 노드 파편화의 주범) 수리. 꼬리가 NNB(제391조·제2항)면
                # 법령 참조 — 원설계 ③ 그대로 무음 통과("상법 제391조" 주어는
                # 상법 유지, 회귀 금지 조건).
                j = i
                while j < n and toks[j].tag in ("XPN", "SN"):
                    j += 1
                nxt_tok = toks[j] if j < n else None
                run_end = toks[j - 1].start + toks[j - 1].len
                if (nxt_tok is not None and nxt_tok.tag in ("NNG", "NNP")
                        and nxt_tok.start == run_end):
                    # 일반 명사 분기와 동일한 띄어쓰기 경계 리셋을 적용
                    if prev_end is not None and t.start > prev_end and noun_buf and not gen_active:
                        noun_buf = []
                    gen_active = False
                    noun_buf.extend(toks[k].form for k in range(i, j))
                    prev_end = run_end
                    i = j
                    continue
                prev_end = run_end
                i = j
                continue
            if tag == "NNB":                                 # ③ 의존명사 통과
                prev_end = t.start + t.len
                i += 1
                continue
            if form in ("(", "（"):                          # 삽입구 "(이하 …라 한다)" 스킵
                depth, j = 1, i + 1
                while j < n and depth:
                    if toks[j].form in ("(", "（"):
                        depth += 1
                    elif toks[j].form in (")", "）"):
                        depth -= 1
                    j += 1
                prev_end = toks[j - 1].start + toks[j - 1].len if j > i else prev_end
                i = j
                continue
            if form in ("「", "」", "『", "』", "\u201c", "\u201d"):  # 인용부호 통과(내용 보존)
                prev_end = t.start + t.len
                i += 1
                continue
            prev_end = t.start + t.len
            gen_active = False

            if tag in ("JKS", "JKC") or (tag == "JX" and form in ("은", "는")):
                if (tag == "JX" and not noun_buf and not subject
                        and not (last_role == "dat" and dat) and adn_anchor):
                    # ⑨ "…에 종사하는 자는" — 관형절이 소비한 명사구를 주어 앵커로
                    subject, subject_used = adn_anchor, False
                    topic_subject = subject
                    adn_anchor = ""
                elif (tag == "JX" and not noun_buf and last_role == "dat" and dat
                        and not subject):
                    # ④ "…에는/에 관하여는" 주제 승격 — 기존 주어가 없을 때만
                    #   ("조합에 대해서는"이 문두 주어를 덮어쓰던 FP 차단)
                    subject, subject_used = dat, False
                    topic_subject = subject
                    dat = ""
                else:
                    # ① 존재/판단 내포절 가드 — "우려가 있다"의 '우려'는 주어 아님
                    nxt = toks[i + 1] if i + 1 < n else None
                    exist_guard = (tag in ("JKS", "JKC") and nxt is not None
                                   and nxt.form in ("있", "없")
                                   and nxt.tag in ("VV", "VA", "VX"))
                    cand = self._flush_noun(noun_buf)
                    if cand and not exist_guard:
                        if tag in ("JKS", "JKC"):
                            saved_subject, saved_used = subject, subject_used
                        subject, subject_used = cand, False
                        dat = ""           # 새 주어 = 새 절 — 이전 절 여격 잔류 차단
                        if tag == "JX":
                            topic_subject = subject  # 주제격만 캐리 후보(⑧)
                            saved_subject, saved_used = "", True
                last_role = "subj"
            elif tag == "JKO":
                cand = self._flush_noun(noun_buf)
                if cand:
                    obj = cand
                last_role = "obj"
            elif tag == "JKB" and form in _DAT_FORMS:        # ④ 여격 후보
                cand = self._flush_noun(noun_buf)
                if cand:
                    dat = cand
                    last_role = "dat"
                    # P0-1: JKB *바로 뒤* 의하/VV = 명시적 행위자구("X에 의해").
                    # 인접 위치 결속 — 문장 어딘가의 '에 의해'로 판정하면 안 됨(심판 B).
                    nxt = toks[i + 1] if i + 1 < n else None
                    dat_agent = nxt is not None and nxt.tag == "VV" and nxt.form == "의하"
            elif tag == "XSV":                               # 명사+하/되/시키 술어
                if noun_buf:
                    pred_cand = noun_buf[-1]
                    if 2 <= len(pred_cand) <= 20 and pred_cand not in STOP_HEAD \
                            and _HANGUL.fullmatch(pred_cand):
                        if not is_adnominal(i):
                            # P0-1 피동 3분기(0715 심판 조건부 채택) — 되/받/당하는
                            # 태(voice) 표지인데 무시되어 방향 오류가 무효의 최다
                            # 단일 유형이었다(ctrl 6/28 실측). 전부 폐집합 형태소.
                            if form == "되":
                                if dat and dat_agent and subject:
                                    # "S가 O에 의해 V되다" → 능동 (O, V, S) 스왑 방출.
                                    # 잔여 FP(근거/수단형 "규정에 의해") 10~20% 실측 —
                                    # 현행 100% 방향오류 대비 순개선으로 수용(심판 B).
                                    p = pred_cand + (" 금지" if neg_after(i) else "")
                                    out.append({"subject": dat, "predicate": p,
                                                "object": subject})
                                    obj = dat = ""
                                    dat_agent = False
                                # 되+JKO: 되-피동은 대격 불가 — obj 는 구조적으로 이전
                                # 절 잔류물 → 억제. 되+여격(에/에게): R6 에서 피동 표지
                                # 보존 방출((주어, 술어됨, 여격) — 방향 중의 회피)을
                                # 실측했으나 블라인드 합의유효 28%(정크 지배 원인은
                                # 명사구 연접 미비의 인자 파편) → 심판 판정 "설계는
                                # 옳으나 시기상조, P0-2(명사구 연접) 선행 후 재도전".
                                # 그때까지 억제 유지(유효손실 ~96건 기록 조건 유지).
                                else:
                                    obj = dat = ""
                            elif form in ("받", "당하"):
                                # 수동 수혜/피해: 관계는 참, 술어만 왜곡 — 피동 접사를
                                # 보존해 방출(선고받음/점령당함). 어휘 지식 불요.
                                sfx = "받음" if form == "받" else "당함"
                                if not emit(pred_cand + sfx, i):
                                    pending_pred = pred_cand + sfx + (
                                        " 금지" if neg_after(i) else "")
                            elif not emit(pred_cand, i):
                                pending_pred = pred_cand + (" 금지" if neg_after(i) else "")
                        # 관형절 문맥 복원은 ETM 분기에서 일괄 처리
                last_role = None
            elif tag == "VV" and form in _FUNC_VV:
                pass  # 기능동사(에 관하여는…) — last_role 유지해 주제 승격 경로 보존
            elif tag == "VV" and form not in _AUX_VV:        # ⑥ 일반동사 술어
                # 본술어 게이트: 종결어미(EF) 또는 의무 연결(어야/아야/여야)일 때만.
                # "~에 따라/를 거쳐" 같은 부사적 연결 용법(EC 어/아/고)이 술어로
                # 오인되던 FP(따름/거침 폭발) 차단 — GT의 "따라야 한다"는 유지.
                nxt = toks[i + 1] if i + 1 < n else None
                finite = nxt is not None and (
                    nxt.tag == "EF" or (nxt.tag == "EC" and nxt.form in ("어야", "아야", "여야")))
                pred_cand = _nominalize(form)
                if (not is_adnominal(i) and finite and 2 <= len(pred_cand) <= 20
                        and pred_cand not in STOP_HEAD and _HANGUL.fullmatch(pred_cand)):
                    if not emit(pred_cand, i):
                        pending_pred = pred_cand + (" 금지" if neg_after(i) else "")
                last_role = None
            elif tag == "ETM":
                nxt = toks[i + 1] if i + 1 < n else None
                if not (nxt is not None and nxt.tag == "NNB" and nxt.form == "수"):
                    # 관형절 폐쇄 — 절이 소비한 주어/여격/목적어를 바깥 절로 복원(⑦)
                    if saved_subject:
                        subject, subject_used = saved_subject, saved_used
                        saved_subject = ""
                    if dat:
                        adn_anchor = dat   # "…에 종사하는 자는" 주어 앵커(⑨)
                    dat = obj = ""
            elif tag == "EC":
                # P0-3(0715 심판 채택): 절 경계에서 목적어 잔류물 차단 — 비방출
                # 용언을 건너뛴 obj 가 다음 술어에 오귀속되던 누출("사신을 보내
                # 봉헌" → 봉헌의 목적어=사신)이 무효 최다 복합유형(ctrl 12/28).
                # 리셋 조건: EC 직전(선어말어미 EP 통과)이 *실질동사 VV* 일 때만 —
                # 그 동사가 obj 의 주인인데 방출 없이 절이 닫힌 경우다. 무조건
                # 리셋하면 부사화 EC("부당하'게' 소멸시켜서는")가 정당한 obj 를
                # 파괴한다(스모크 실측 회귀 → 조건 정밀화). 목적어 공유 접속문
                # ("개발하고 판매")은 emit 이 이미 obj 를 리셋해 비용 ≈0(심판 D).
                # ⚠️ dat 은 유지 — '에 의해'(의하/VV+어/EC)·'에 관하여'(FUNC_VV+EC)
                # 경로가 EC 를 통과하므로 dat 리셋은 P0-1·④주제승격 파괴(심판 적발).
                j = i - 1
                while j >= 0 and toks[j].tag == "EP":
                    j -= 1
                if (j >= 0 and toks[j].tag == "VV"
                        and toks[j].form not in _AUX_VV
                        and toks[j].form not in _FUNC_VV):
                    obj = ""
            else:
                last_role = None
            noun_buf = []
            i += 1

        return out, topic_subject, pending_pred, self._flush_noun(noun_buf)

    def extract(self, text: str, *, source_chunks: list[str]) -> list[dict]:
        """청크 텍스트 → 관계 dict 리스트(kg_builder 소비 스키마).

        반환: [{subject, predicate, object, predicate_type='ObjectProperty',
                source_chunks, (inferred_subject)}, ...] — 같은 (s,p,o) 중복 제거.
        ⑧ 문장 간 주제 캐리: 법령 항·호 열거("…자는 다음 행위를 하여서는 아니 된다.
        1. …하는 행위")에서 앞 문장 주제를 다음 문장에 잇는다. 캐리로 만든 트리플은
        inferred_subject=True 로 태깅되어 하위가 신뢰도를 구분할 수 있다."""
        if not text or not text.strip():
            return []
        # 법령 편집 메타태그(<개정 2014. 1. 14.> 등) 제거 — 날짜 마침표가 문장분리를
        # 파편내 주제 캐리(⑧)를 끊는다(0711 실측: 85c7b36a 금지 캐리 전멸 원인)
        text = re.sub(r"<[^<>]{0,100}>", " ", text)
        # 호 마커("5. "/"가. ")의 마침표를 ")"로 정규화 — 문장분리기가 마커를 본문에서
        # 잘라내 캐리가 마커 단독문장에 붙고 본문(비열거)이 캐리를 리셋하던 결함(0711)
        text = re.sub(r"(?m)^(\s*(?:\d+(?:의\d+)?|[가-힣]))\.(\s)", r"\1)\2", text)
        seen: set = set()
        out: list[dict] = []
        carry, carry_neg, carry_pred = "", False, ""
        for sent in _SENT_SPLIT.split(text):
            if not sent.strip():
                continue
            # ⑧ 캐리는 열거 마커("1.", "①", "가.")로 시작하는 문장에만 —
            #   일반 문장까지 캐리하면 엉뚱한 주어 오염(FP 폭발, 0711 실측)
            enum = bool(_ENUM.match(sent))
            init = carry if (enum and self.enable_carry) else ""
            tris, new_carry, pending, final_np = self._extract_sentence(
                sent, init_subject=init, is_enum=enum,
                init_prohibit=(enum and carry_neg))
            if enum and not tris and init and carry_pred and final_np:
                # ⑩ 열거 분배 — "다음 각 호의 사항을 적어야 한다" + 명사구 호 라인:
                #   부모 미완 술어를 호 명사구에 분배(추정 주어 태깅)
                tris = [{"subject": init, "predicate": carry_pred,
                         "object": final_np, "inferred_subject": True}]
            if not enum:
                # 새 일반 문장 — 캐리 문맥 갱신(금지 여부는 문장 문자열로 판정, 폐어구)
                carry = new_carry
                carry_neg = bool(re.search(r"(아니\s?된다|못한다|수\s?없다)", sent))
                carry_pred = pending
            for tri in tris:
                if tri["subject"] == tri["object"]:
                    # 자기루프 컷(R6 심판) — "자기장 표시됨 자기장"류. 주어==목적어
                    # 관계는 조사 재부착/절 잔류물의 구조적 산물로 유효 사례 실측 0.
                    continue
                key = (tri["subject"], tri["predicate"], tri["object"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({**tri, "predicate_type": "ObjectProperty",
                            "source_chunks": source_chunks})
        return out
