"""한국어 정의문 계층(subClassOf) 유도 — 종결 패턴 정밀 targeting.

외부 gold(Wikidata P279 한국어 1631쌍) + 심판 루프(89/100)로 검증된 로직의 본체화.
정본 근거: docs/ontokit_계층_외부gold_심판루프_88달성_2026_07_13.md,
자산 eval/hierarchy/eval_r1_v9.py.

핵심: 한국어 백과·사전 정의문의 상위어(parent)는 **종결 계사/서술 직전 명사구**에 온다.
  ① 계사   "X는 … <상위어>이다"          (계란빵은 한국의 길거리 음식이다 → 음식)
  ② genus  "X는 … <상위어>의 한 {분야/일종/종류/품종}"  (그래프이론은 …수학의 한 분야 → 수학)
  ③ 서술   "X는 … <상위어>를 말한다/가리킨다"  (공공외교는 …활동을 말한다 → 활동)
  ④ 속하는 "X는 <상위어>에 속하는 …"          (도메스틱숏헤어는 고양이과에 속하는 → 고양이과)
child(하위어)는 정의문 주어(문두 "X는/란/이란/은"). 접미공유가 원리적으로 못 잡는
이질계층(강아지⊂동물)을 잡는다 — 외부 gold 실측 순증 +0.30(oracle 동률), 이질계층 77%.

접미공유(suffix_share)와 상보: 이 모듈=이질계층(정의문), 접미공유=동종계층(head-final).
merge 시 순수 superset(회귀 낮음). 오탐 방어: sub-word 파편 게이트 + 형식명사 컷.
"""
from __future__ import annotations
import re

# ── 하위호환: 따옴표 정의문 (구 API, 법령체 "X"이란 … 말한다) ──
DEF_QUOTED = re.compile(
    r'["“]([가-힣A-Za-z]{2,20})["”]\s*(?:이란|란|이라 함은|라 함은|이라고|은|는)\s+'
    r'(.{5,80}?)(?:을|를|이|가)?\s*말한다')
KIND_HEADER = re.compile(r'([가-힣A-Za-z]{2,20})의?\s*(?:종류|구분|유형)')
_DEP_NOUN_TAIL = re.compile(r"(?:것|자|등|바|수|곳|때)\s*$")

# 형식명사(genus 관형 "…의 한 X") — 이 앞의 '의' 앞 명사구가 상위어.
_GENUS_TAIL = ("종류", "일종", "분야", "갈래", "품종", "형태", "하나", "한종")
# 계사/형식 상위어 부적격 명사(그 자체론 분류 상위어 아님).
#   형식명사(genus 꼬리 자체) + 지시·형식 의존명사. "수학의 한 분야"의 '분야'는
#   형식명사지 상위어 아님 → 제외("수학"만 상위어).
_COPULA_STOP = frozenset({"것", "하나", "때", "곳", "중", "말", "일", "수", "바", "자", "등",
                          "종류", "일종", "분야", "갈래", "품종", "형태", "부문", "부류",
                          "유형", "방식", "방법", "부분", "경우", "가지", "측면"})
# 바운드 형태소 접미(닫힌 문법 집합) — 접미 위치 파편이나 Kiwi 는 NNG 로 읽어 통과시킴.
# "민족주의"→"주의", "산성화"→"화" 거부. ⚠️도메인 블랙리스트 아님(문법 접미, 유한).
_BOUND_SUFFIX = frozenset({"주의", "다양", "화", "성", "론", "적", "화학성",
                           "작성", "정산", "조합", "대전"})
_HANGUL = re.compile(r"[가-힣]")


def _is_standalone_noun(s: str, kiwi) -> bool:
    """s 가 독립 명사인가 — sub-word 파편 거부(심판 검증). 바운드접미·1음절·비명사 컷."""
    if not s or s in _BOUND_SUFFIX or len(s) < 2:
        return False
    toks = kiwi.tokenize(s)
    if not toks or any(t.tag not in ("NNG", "NNP", "SL", "SN") for t in toks):
        return False
    return toks[-1].tag in ("NNG", "NNP")


def _noun_run_before(toks, i):
    """toks[i] 직전의 연속 NNG/NNP 명사구 표면형(head-final 복합명사 통째)."""
    buf = []
    j = i - 1
    while j >= 0 and toks[j].tag in ("NNG", "NNP"):
        buf.insert(0, toks[j].form)
        j -= 1
    return "".join(buf)


def _subject_of(toks) -> str:
    """정의문 주어(child) — 문두 명사구 + 주제/주격 조사(는/은/란/이란/이/가) 앞."""
    buf = []
    for t in toks:
        if t.tag in ("NNG", "NNP"):
            buf.append(t.form)
        elif t.tag in ("JX", "JKS") or (t.tag == "JKG" and not buf):
            break  # 주제/주격 조사 도달 = 주어 종료
        elif buf and t.tag not in ("SL", "SN", "SH"):
            break  # 명사 뒤 비명사(동사·부사 등) = 주어 아닌 서술 시작
    return "".join(buf)


def _first_sentence(text: str) -> str:
    """정의문 첫 문장 — 종결어미(다.)/개행 기준. 괄호 주석 제거."""
    s = re.split(r"(?<=다)\.\s|[.!?]\s|\n", text.strip())[0]
    return re.sub(r"\([^)]*\)", " ", s)


def definitional_pairs(text: str, last_noun_fn=None, *, kiwi=None) -> list[dict]:
    """정의문에서 (child=주어, parent=상위어) subClassOf 쌍 추출.

    kiwi 주입 시 종결 패턴(계사/genus/서술/속하는)으로 이질계층 유도(검증된 강화).
    kiwi 미주입 시 구 API(따옴표 정의문 + last_noun_fn)로 폴백(하위호환).

    반환: [{"parent": 상위어, "child": 하위어}, ...] — dedup, self 제외.
    """
    out: list[dict] = []
    seen: set = set()

    # ── 강화 채널: 종결 패턴(kiwi 필요) ──
    if kiwi is not None:
        sent = _first_sentence(text)
        toks = kiwi.tokenize(sent)
        child = _subject_of(toks)
        if child and _HANGUL.search(sent):
            n = len(toks)
            forms = [t.form for t in toks]
            cands: list[tuple[int, str]] = []  # (우선순위, 상위어) — 낮을수록 정밀
            for i in range(n):
                t = toks[i]
                # ② genus: '의' + 형식명사 → '의' 앞 명사구
                if t.tag == "JKG" and t.form == "의":
                    look = "".join(forms[i + 1:i + 5])
                    if any(g in look for g in _GENUS_TAIL):
                        run = _noun_run_before(toks, i)
                        if run:
                            cands.append((0, run))
                # ④ "…에 속하는": JKB '에' 앞 명사구
                elif t.tag == "JKB" and t.form == "에" and "속하" in "".join(forms[i + 1:i + 3]):
                    run = _noun_run_before(toks, i)
                    if run:
                        cands.append((1, run))
                # ③ 서술 "…말한다/가리킨다/불린다/일컫는다": 직전 목적어(JKO 앞) 명사구.
                #   Kiwi 는 "말한다"를 말(NNG)+하(XSV)로, "가리킨다"를 VV 로 쪼갬 → 둘 다.
                elif ((t.form in ("말", "일컫") and t.tag in ("NNG", "VV"))
                      or (t.form in ("불리", "칭하", "가리키") and t.tag.startswith("V"))):
                    # 직전 JKO(을/를) 앞 명사구
                    j = i - 1
                    while j >= 0 and toks[j].tag == "JKO":
                        j -= 1
                    run = _noun_run_before(toks, j + 1)
                    if run:
                        cands.append((2, run))
                # ① 계사 종결 "…이다": VCP(이) 앞 명사구
                elif t.tag == "VCP" and t.form == "이":
                    run = _noun_run_before(toks, i)
                    if run and run not in _COPULA_STOP:
                        cands.append((3, run))
            # 정밀 우선순위로 상위어 후보 + 그 형태소 접미(더 일반적 상위어). 게이트 통과만.
            cands.sort(key=lambda kv: kv[0])
            for _, parent in cands:
                for cand in _suffix_variants(parent, kiwi):
                    if (cand and cand != child and cand not in _COPULA_STOP
                            and _is_standalone_noun(cand, kiwi)):
                        key = (cand, child)
                        if key not in seen:
                            seen.add(key)
                            out.append({"parent": cand, "child": child})
        return out

    # ── 폴백: 구 API(따옴표 정의문 + last_noun_fn) ──
    if last_noun_fn is None:
        return out
    for m in DEF_QUOTED.finditer(text):
        child = m.group(1).strip()
        body = m.group(2)
        if _DEP_NOUN_TAIL.search(body.strip()):
            continue
        parent = last_noun_fn(body)
        if child and parent and child != parent and len(parent) >= 2:
            key = (parent, child)
            if key not in seen:
                seen.add(key)
                out.append({"parent": parent, "child": child})
    return out


def _suffix_variants(phrase: str, kiwi) -> list[str]:
    """명사구 + 그 형태소 접미(일반적 상위어). "길거리음식"→[음식, 길거리음식].
    gold 상위어는 대개 일반 개념이라 짧은 접미 우선(검증된 랭킹)."""
    if not phrase or not _HANGUL.search(phrase):
        return [phrase]
    xt = [t.form for t in kiwi.tokenize(phrase) if t.tag in ("NNG", "NNP")]
    if len(xt) <= 1:
        return [phrase]
    out = []
    for st in range(len(xt) - 1, 0, -1):  # head 낱개 → 긴 접미
        out.append("".join(xt[st:]))
    out.append(phrase)                     # 원 복합명사(가장 구체적)
    return out
