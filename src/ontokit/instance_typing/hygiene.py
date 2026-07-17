"""T4 라벨 위생 게이트 — 파편 라벨 컷 (폐집합 문법 판정만, 콘텐츠 어휘 0).

실측 파편(mixed20k): '는 신화다'(선도 조사+어미), '이소라 이소'(절단 중복 토큰),
'프랑스와'(조사 잔류). 위생은 **명백한 비명사 파편만** 컷하고 애매하면 통과
(정밀도 우선이되 리콜 오살 방지 — 판정 불가 시 무해 방향).
"""
from __future__ import annotations
import re

# 어절 단위 절단-중복: "이소라 이소" — 뒤 어절이 앞 어절의 진접두(또는 역). 공백 라벨만 해당.
_WS = re.compile(r"\s+")

# 선도/말미에 오면 파편인 품사(폐집합 문법형태소): 조사(J*), 어미(E*), 접사(XS*), 계사(VCP/VCN)
_BAD_EDGE_PREFIX = ("J", "E", "XS", "VCP", "VCN")
# 라벨을 구성할 수 있는 내용 품사(하나는 있어야 함)
_CONTENT_PREFIX = ("N", "SL", "SN", "SH", "XR")


def label_ok(label: str, kiwi=None) -> bool:
    """라벨이 개체명으로 성립하는가 — False 면 추출단에서 드랍.

    kiwi 미주입 시 형태소 판정은 생략하고 어절 중복 검사만(폴백 한계 공시).
    """
    if not label or len(label.replace(" ", "")) < 2:
        return False
    words = [w for w in _WS.split(label.strip()) if w]
    # 절단 중복: 인접 어절이 서로의 접두 — 진접두('이소라 이소') + 동일 반복('가능역 가능역', R14c)
    for a, b in zip(words, words[1:]):
        if a.startswith(b) or b.startswith(a):
            return False
    if kiwi is None:
        return True
    try:
        toks = kiwi.tokenize(label)
    except Exception:
        return True  # 분석 실패 = 판정 불가 → 통과(오살 방지)
    if not toks:
        return False
    # 선도 토큰이 조사/어미/접사/계사 = 절단 파편 ('는 신화다')
    if toks[0].tag.startswith(_BAD_EDGE_PREFIX):
        return False
    # 말미 토큰이 조사/어미 = 조사 잔류 ('프랑스와' 는 Kiwi 가 프랑스+와(JC) 로 분석)
    if toks[-1].tag.startswith(("J", "E", "VCP", "VCN")):
        return False
    # 내용 품사가 하나도 없으면 라벨 아님
    if not any(t.tag.startswith(_CONTENT_PREFIX) for t in toks):
        return False
    return True
