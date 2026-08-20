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
        out.append(e)
    # (제거됨, 0717 G-A) 인접 인명 병합은 나열된 서로 다른 인물을 오결합
    # ('R.A.디키'+'놀런', '데릭 로'+'마크' — 블라인드 2인 실증). 공백 인접만으로는
    # 단일 인명과 인명 나열을 구분 불가 — 판별 기제 없이 재도입 금지.
    return out
