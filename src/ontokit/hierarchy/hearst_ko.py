"""한국어 Hearst 패턴 (정의문/열거) — subClassOf 보조 유도.

finreg 489 실측: 정의문("X이란…말한다") 96건, "X의 종류" 55건이 법령문 신호.
단 "마지막 명사=상위"는 서술구조라 오탐 많음 → 현재 접미공유가 주 엔진이고
이 모듈은 확장 여지(정의문 정밀화, KorLex 검증 결합)로 보류. 장르=정의문일 때 89.7% 계열.
"""
from __future__ import annotations
import re

# "생명보험업"이란 ... 을 말한다  (정의 대상=따옴표 안이 클래스)
DEF_QUOTED = re.compile(
    r'["“]([가-힣A-Za-z]{2,20})["”]\s*(?:이란|란|이라 함은|라 함은|이라고|은|는)\s+'
    r'(.{5,80}?)(?:을|를|이|가)?\s*말한다')
# "X의 종류/구분/유형" 표제
KIND_HEADER = re.compile(r'([가-힣A-Za-z]{2,20})의?\s*(?:종류|구분|유형)')


def definitional_pairs(text: str, last_noun_fn) -> list[dict]:
    """정의문에서 (child=정의대상, parent=피정의 상위개념) 추출.
    ⚠️ 실측상 오탐 있어 보수적 사용 권장(KorLex 검증 결합 시 정밀도↑).
    last_noun_fn: 구절→마지막 명사 (KiwiNounExtractor.last_noun 주입).

    이 함수는 법령체(따옴표 정의문 `"X"이란 … 말한다`)만 커버한다. 위키/일반
    텍스트의 계사 정의문(`X는 … Y이다`)은 아래 copula_pairs 를 쓴다."""
    out = []
    for m in DEF_QUOTED.finditer(text):
        child = m.group(1).strip()
        parent = last_noun_fn(m.group(2))
        if child and parent and child != parent and len(parent) >= 2:
            out.append({"parent": parent, "child": child})
    return out


# 계사 정의문에서 상위어로 부적격한 일반명사(오탐 컷). "X는 것이다/때이다" 등.
# Kiwi 는 '마찬가지·반대말·획기적' 도 NNG 로 태깅해 형태소로 못 거르므로(실측 확인),
# 추상·비교·관계·형용사성 명사를 어휘 목록으로 컷한다. 위키 30문서 실측서 나온
# 오탐 상위어 반영.
_STOP_HYPER = {
    # 의존/형식 명사
    "것", "때", "바", "수", "자", "곳", "점", "면", "중", "등", "경우",
    "하나", "모두", "이것", "그것", "무엇", "누구", "이유", "때문",
    # 비교·동일성(계사가 is-a 아닌 동일성/비유를 표현하는 케이스)
    "마찬가지", "반대말", "동일", "동의어", "유사", "차이", "관계",
    # 추상·메타(진짜 상위 개념 아님)
    "대상", "결과", "실정", "여부", "주체", "모음", "종류", "일종",
    "형태", "방식", "방법", "기준", "특징", "성질", "상태", "현상",
    # 형용사성(NNG 로 태깅되나 상위개념 아님)
    "획기적", "대표적", "일반적", "전형적", "필수적",
}


def copula_pairs(text: str, kiwi) -> list[dict]:
    """계사(긍정지정사 VCP='이다') 정의문에서 (child=주어, parent=상위개념) 추출.

    위키/백과사전체 정의문 `X는 … Y이다` 를 형태소로 정밀 매칭한다:
      - 하위어(child) = 문두 주어 = 첫 주격/보조사(JKS 이/가, JX 은/는) 직전 명사구.
      - 상위어(parent) = 계사(VCP '이') 직전 명사 = 술어 명사(피정의 상위개념).
      예: "수학은 수를 다루는 학문이다" → 수학 ⊂ 학문,
          "위키백과는 자유 백과사전이다" → 위키백과 ⊂ 백과사전.

    형태소 기반이라 정규식 계사 매칭의 고질적 오탐을 상당 부분 막는다:
      - 형용사 서술(`문제는 어렵다`)은 VA 라 VCP 아님 → 걸러짐(is-a 아닌 것 제외).
      - 계사 직전이 명사가 아니면(수식어/조사 등) 상위어 미채택.
    ⚠️ 남은 한계: 계사는 동일성·역할·은유도 표현하므로("그는 천재이다"=은유),
      _STOP_HYPER 로 일반명사 상위어를 컷하되 완벽하지 않다. 문장 단위로 첫 계사
      1개만 취해(정의는 보통 주문장) 열거·부가절의 계사 오탐을 줄인다.

    kiwi: Kiwi 인스턴스(토큰의 tag/위치로 역할 판별).
    """
    out: list[dict] = []
    # 문장 분리(종결부호/개행) 후 문장별 1쌍. 정의는 보통 첫 계사에서 성립.
    import re as _re
    for sent in _re.split(r'(?<=[.!?\n])\s+', text):
        if "이" not in sent and "다" not in sent:
            continue
        toks = kiwi.tokenize(sent)
        # ── child: 첫 주어(JKS/JX 직전 명사구) ──
        child = ""
        buf: list[str] = []
        for t in toks:
            if t.tag in ("NNG", "NNP"):
                buf.append(t.form)
            elif t.tag in ("JKS", "JX") and buf:
                child = "".join(buf)
                break
            else:
                buf = []
        if not child or len(child) < 2:
            continue
        # ── parent: 첫 VCP(계사 '이') 직전 명사 ──
        parent = ""
        for i, t in enumerate(toks):
            if t.tag == "VCP":
                # 계사 바로 앞의 명사 토큰(수식·조사 사이 없이 인접)만 상위어로.
                if i > 0 and toks[i - 1].tag in ("NNG", "NNP"):
                    parent = toks[i - 1].form
                break
        if (parent and len(parent) >= 2 and parent != child
                and parent not in _STOP_HYPER and child not in _STOP_HYPER):
            out.append({"parent": parent, "child": child})
    return out
