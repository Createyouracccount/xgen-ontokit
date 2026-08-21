"""R13-2 스팬-형태소 경계 정렬 + 인접 동클래스 인명 병합 (결정론, LLM 0콜).

실측 실패 모드(mixed20k 미탐 추적):
- 토큰 중간 절단: '찰스턴'→'찰스', '진양호'→'진양' — NER 서브워드 경계가 Kiwi
  토큰 중간에서 끊김 → **토큰 끝까지 확장**. '한국측/서울역' 오결합 없음(스팬이
  완전한 토큰 경계에서 끝나면 미확장 — 측/역은 별도 토큰).
- (기각) 인명 병합은 G-A 채점에서 인명 나열 오결합 실증 — 제거.
"""
from __future__ import annotations

import os

# R3(a) 한국어 경계 복원 스위치 — **기본 off(롤백 상태)**.
# 사전 공시 `eval_runs/bench/demo_roster/r3a_predeclare.md` §5 의 롤백 임계
# "오살률 > 3%(분모=표층형이 바뀐 라벨 전량)" 가 실측 4.35%(1/23)로 **위반**됐고,
# 독립 심판 3인이 전원 REJECT 했다(0821). 순효과도 +1.37pp·McNemar p=0.50 으로
# 0과 구별되지 않는다. K-1 대로 결과를 본 뒤 임계를 고치지 않고 **롤백**한다.
# 코드는 차기 라운드 재실험을 위해 플래그 뒤에 남긴다(기본 미발화).
_SPAN_REPAIR_KO = os.environ.get("ONTOKIT_KO_SPAN_REPAIR", "").lower() in (
    "on", "true", "1")

# R4 좌측 복합명사 완성 — **기본 off · 심판 3인 전원 REJECT(0821) 상태**.
# 공시 `demo_roster/r4_predeclare.md`, 결과 `demo_roster/r4_result.md`.
#
# ⛔ 초판 주석의 안전성 주장("NNG/NNP 조건이 sg1 기각 사유를 **구조적으로** 막는다")은
#    **반증됐다**. XPN/MM 경계는 의미가 아니라 **Kiwi 어휘 등재 여부**일 뿐이다:
#      반정부·비정부·신정부(XPN) · 전정부·현정부(MM)  → 차단됨
#      괴뢰정부·역대정부·자국정부·현지정부·기존정부(NNG) → **그대로 걷는다**
#    같은 실패 유형이 태그만 다르게 통과한다. 구조적 보장이 아니라 우연이다.
#
# ⛔ 더 나쁜 것 — 아래 **스냅은 태그 검사도 인접 검사도 없다**. Kiwi 가 접두형을 한 토큰으로
#    붙여 내면(반미·대북·북미·남북한 = NNG/NNP 단일 토큰) 스냅이 그 시작으로 점프해
#    `미`→`반미`, `북한`→`남북한` 을 만든다. sg1 기각 사유 그 자체다.
#
# ⛔ 좌측 걷기에는 **길이 상한이 없다**(우측에는 있다). 실측 최대 +22자.
#
# 재도입하려면: 스냅 대상 토큰에 태그 가드 · 좌측 확장 길이 상한 · 수식어 폐집합 스톱리스트 ·
# 표층 공백 검사(토큰 접합이 아니라 text[ns:st] 에 공백이 있는지) · 전용 테스트.
_LEFT_COMPLETE_KO = os.environ.get("ONTOKIT_KO_LEFT_COMPLETE", "").lower() in (
    "on", "true", "1")
_NOUN_TAGS = ("NNG", "NNP")
_LEFT_MAX_CHARS = 12        # R5 §2-⑤ 좌측 확장 길이 상한

# R5 §2-③ 수식어 폐집합 — 관형적으로 쓰이는 명사·시간 명사. 개체명의 일부가 아니다.
# ⚠️ 이 목록은 `임시정부`·`세계문화유산`·`국제원자력기구` 같은 **정당한 복합명사도 차단**한다.
#    의도된 보수적 선택이며, 차단으로 잃는 참 라벨 수를 결과에 공시한다.
_MODIFIER_STOP = frozenset("""
기존 신규 일부 주요 관련 당시 현지 역대 자국 괴뢰 과도 연립 군사 차기 임시
전국 국내 해외 국제 세계 최고 최대 최초 최근 향후 현재 이번 다음 지난
올해 내년 작년 지난해 금년 명년 매년 연간 월간 주간 일간
""".split())


def _complete_left(text, toks, st):
    """스팬 시작점을 복합명사의 왼쪽 끝까지 되돌린다. 못 걸으면 st 그대로.

    R5 — 사전 공시 `demo_roster/r5_predeclare.md` §2 의 5개 부품을 **전부** 구현한다.
    R4 기각의 제1 사유가 미공시 부품이었으므로, 여기 있는 것 말고는 아무것도 하지 않는다.
    """
    idx = {}
    for i, t in enumerate(toks):
        idx.setdefault(t.start, i)      # 같은 start 중복 시 **첫** 토큰(R4는 last 라 lossy)
    i = idx.get(st)
    if i is None:
        # ① 스냅 — 단 소유 토큰이 NNG/NNP 일 때만(R4는 태그 검사가 없어 `미`→`반미` 발생)
        for j2, t2 in enumerate(toks):
            if t2.start < st < t2.end:
                if t2.tag not in _NOUN_TAGS:
                    return st
                i = j2
                break
        if i is None:
            return st
    start0 = st
    st = toks[i].start
    j = i
    while j > 0:
        prev = toks[j - 1]
        if prev.tag not in _NOUN_TAGS:      # ② XPN(반·비·신)·MM(전·현) 등에서 정지
            break
        if prev.end != toks[j].start:       # ② 토큰 접합 끊김
            break
        if prev.form in _MODIFIER_STOP:     # ③ 수식어 폐집합에서 정지
            break
        j -= 1
    ns = toks[j].start
    if ns >= start0:
        return start0
    # ④ 표층 공백 검사 — 토큰 접합 검사로는 Kiwi 다어절 토큰을 못 막는다(R4 실증)
    if any(ch.isspace() for ch in text[ns:start0]):
        return start0
    # ⑤ 길이 상한 — 우측 규칙의 +8 과 대칭적 취지
    if start0 - ns > _LEFT_MAX_CHARS:
        return start0
    return ns


def align_spans(text: str, ents: list[dict], kiwi=None) -> list[dict]:
    """NER 엔티티 스팬을 Kiwi 토큰 경계로 정렬 + 인접 인명 병합. in-place 아님."""
    if kiwi is None or not ents or not text:
        return ents
    try:
        toks = kiwi.tokenize(text)
    except Exception:
        return ents
    # 토큰 [start, end) 경계 목록.
    # ⚠️ `t.start + len(t.form)` 을 쓰면 안 된다 — `form` 은 **원형(lemma)** 이라 표층과
    #   길이가 다르다. 실측(mixed20k 901,057 토큰): 1,749건(0.194%) 불일치.
    #   '지었다'→form='짓'(len1)인데 표층은 2자 · '델타항공'→form='델타 항공'(원문에 없는
    #   공백 삽입, len5)인데 표층은 4자. 후자는 확장이 조사까지 삼켜
    #   'AT T 델타항공' → 'AT T 델타항공에' 를 만들고 위생 게이트에서 라벨이 통째 소멸했다.
    #   Kiwi 는 `t.end` 를 직접 준다.
    bounds = [(t.start, t.end) for t in toks]
    out: list[dict] = []
    for e in ents:
        st, en = e.get("start"), e.get("end")
        if isinstance(st, int) and isinstance(en, int):
            for ts, te in bounds:
                # 스팬 끝이 토큰 내부에서 끊김 → 토큰 끝까지 확장 (찰스턴)
                # R3(a)(기본 off): `st >= ts` 가드 제거 실험.
                # ⚠️ 초판 주석의 "발화 확률 0" 은 **실측으로 거짓**이었다(심판 3, 0821).
                #   구 가드는 ui_news100 538청크에서 **100회 발화**한다 — '행안'→'행안부',
                #   '기재'→'기재부', '하나'→'하나은행'. 한국어 주요 기관명 상당수가 Kiwi
                #   단일 토큰이라(단일 NNP 런 99.7%) st == ts 가 흔히 성립한다.
                #   '찰스턴'은 예외 사례가 아니라 **다수 사례**였다.
                # 참인 서술: 이 가드는 **절단 지점이 스팬 시작 토큰 밖에 있을 때만** 수리를
                #   막는다. 그 부분집합은 실재하며(신규 확장 41건·구 전용 0건 = 순증가),
                #   '산업통상자원부와'가 그 예다(산업/통상/자원부 로 쪼개져 st=0 < ts=4).
                # ⚠️ 사전 공시 §4의 방어 논거("종료점이 토큰 경계면 미발화")도 **거짓**이다.
                #   Kiwi 는 '서울시장'을 단일 토큰으로 잡아 '서울'→'서울시장' 이 두 팔 모두에서
                #   발화한다. 이 논거로 안전을 주장할 수 없다.
                if _SPAN_REPAIR_KO:
                    hit = ts < en < te and te - ts <= (en - st) + 8
                else:
                    hit = ts < en < te and st >= ts - 0 and te - ts <= (en - st) + 8
                if hit:
                    ext = text[st:te].strip()
                    if ext and len(ext) > len(e.get("entity", "")):
                        e = dict(e, entity=ext, end=te)
                    break
        # R4: 좌측 복합명사 완성 — 우측 확장과 **독립**으로 적용한다.
        if _LEFT_COMPLETE_KO:
            st2, en2 = e.get("start"), e.get("end")
            if isinstance(st2, int) and isinstance(en2, int):
                ns = _complete_left(text, toks, st2)
                if ns < st2:
                    ext = text[ns:en2].strip()
                    if ext and len(ext) > len(e.get("entity", "")):
                        e = dict(e, entity=ext, start=ns)
        out.append(e)
    # (제거됨, 0717 G-A) 인접 인명 병합은 나열된 서로 다른 인물을 오결합
    # ('R.A.디키'+'놀런', '데릭 로'+'마크' — 블라인드 2인 실증). 공백 인접만으로는
    # 단일 인명과 인명 나열을 구분 불가 — 판별 기제 없이 재도입 금지.
    return out
