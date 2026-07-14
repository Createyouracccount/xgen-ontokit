"""복합명사 접미 공유 기반 subClassOf 계층 유도 (한국어·영어).

finreg 489 실측 핵심 발견: 한국어는 head-final이라 복합명사의 뒤가 상위 개념.
`생명보험업`의 뒤 `보험업`이 상위 → 생명보험업 ⊂ 보험업. 대주주⊂주주, 자회사⊂회사.
실측 순도 높음(825→1710건). 정의문 "마지막명사=상위"는 서술구조라 오탐 많아 제외.

영어(v0.6): 영어 복합명사도 head-final(`life insurance business`의 head는 뒤) —
같은 원리가 **단어 단위**로 적용된다. 공백 포함 이름은 단어 접미(뒤쪽 단어열)로,
무공백 이름(한국어·단일토큰)은 기존 문자 접미로 매칭. 대소문자는 소문자 정규화로
무시하되 출력은 원 표면형 유지. ⚠️이전(v0.5)엔 MAX_LEN=20자 문자기준이라 영어
다단어 복합명사(`Life insurance business`=23자)가 통째로 배제 → 영어 계층 0건.

성능·정밀도(v0.4):
- **인덱스화**: 이전 O(N²) pairwise endswith 를 접미 인덱스 조회 O(N·L²)로 교체
  (L=이름 길이 ≤ MAX_LEN 이라 사실상 선형). 800만 문서 → 수십만 클래스에서
  이전 이중 루프는 수조 비교로 멈췄음 — 라이브러리의 존재 이유(대용량 LLM-free)를
  살리려면 계층 유도도 O(N) 여야 한다.
- **허브 필터**: parent 후보가 MIN_CHILDREN 개 이상 child 의 접미로 등장할 때만 상위로
  인정. `대학⊂학`, `국가⊂가` 같은 형태소 경계 무시 오탐(순수 문자열 endswith 의 약점)을
  빈도 임계로 제거. `보험업`처럼 여러 하위어를 거느린 진짜 상위개념만 남는다.
"""
from __future__ import annotations
import re
from collections import defaultdict
from ..morphology.kiwi_nouns import STOP_HEAD
from ..morphology.en_nouns import STOP_HEAD_EN

_HANGUL = re.compile(r"[가-힣]")

MAX_LEN = 20         # 무공백 이름 최대 글자수 (문자 접미 인덱스 상한 — L² 항 방어)
MAX_LEN_SPACED = 40  # 공백형(영어 다단어) 최대 글자수 — en_nouns.MAX_LEN 과 정합
MIN_SUFFIX_LEN = 2   # 상위개념 후보 최소 길이 (1글자 접미 '학'/'가' 배제)
MIN_CHILDREN = 2     # 허브 임계 — 이만큼의 child 접미로 등장해야 상위로 인정


def _morph_suffix_ok(child: str, parent: str, kiwi) -> bool:
    """parent 가 child 의 **형태소 경계** 접미인가 (문자 파편 거부).

    빈도 임계(MIN_CHILDREN)만으론 못 막는 파편 상위어를 컷한다. mixed20k 실측:
    '대한민국'→'민국'은 Kiwi 가 대한민국을 단일 형태소(NNP)로 봐 '민국'이 형태소
    경계에 없음 → 거부. '생명보험업'→'보험업'은 [생명, 보험업] 경계 일치 → 인정.
    ⚠️ Kiwi 가 child 를 단일 형태소로 보면(고등학교) 접미가 경계에 없어 놓칠 수
    있으나, 파편 노이즈('민국·국·업')가 그래프를 오염시키는 손해가 훨씬 크다
    (노이즈 계층은 GraphRAG 가 그래프를 회피하게 만듦, B 축 A/B 실측 triples_used=0).
    """
    forms = [t.form for t in kiwi.tokenize(child)]
    acc = ""
    for f in reversed(forms):
        acc = f + acc
        if acc == parent:      # 형태소 경계에서 parent 와 정확 일치
            return True
        if len(acc) > len(parent):
            break
    return False


def _parent_is_proper_noun(parent: str, kiwi) -> bool:
    """parent(상위어 후보)의 핵심어(마지막 토큰)가 고유명사(NNP)인가.

    접미공유 계층은 **동종 하위개념 ⊂ 상위범주**(보험업·문법학파·전문학교=보통명사)를
    유도한다. 국가·기관·인명 같은 고유명사(대한민국·일본·한밭대학교)는 상위 *범주*가
    아니라 개체이므로 `X ⊂ 대한민국` 은 계층적으로 성립하지 않는다(그런 소속은
    member_of 관계의 몫). mixed20k 실측: '대한민국'이 15개 접합 클래스(문화재대한민국·
    고등학교대한민국)의 허브가 돼 그래프를 오염 → 상위어 head 가 NNP 면 허브 거부.
    ⚠️ 형태소 게이트(_morph_suffix_ok)가 못 막는 케이스 — child 가 [문화재,대한민국]
    처럼 진짜 형태소 경계에서 갈리면 게이트는 통과시키지만, parent 가 고유명사라
    애초에 상위범주 부적격. 심판 에이전트 검증: NNP head 는 진짜 복합어 상위어
    (보험업/학파/전문학교=NNG)와 직교라 회귀 0(진짜 계층 오탈락 없음).
    """
    forms = kiwi.tokenize(parent)
    return bool(forms) and forms[-1].tag == "NNP"


def induce_suffix_hierarchy(class_names: set[str],
                            min_children: int = MIN_CHILDREN, *, kiwi=None) -> list[dict]:
    """클래스 집합에서 접미 공유 subClassOf 유도 (인덱스화 + 허브 필터, 한·영).

    child 가 parent 로 끝나고 더 길면 child subClassOf parent — 단, parent 가
    min_children 개 이상의 서로 다른 child 의 접미일 때만(허브) 인정.
    무공백 이름=문자 접미(한국어), 공백 포함 이름=단어 접미(영어 다단어).
    대소문자 무시 매칭, 출력은 원 표면형. 전역 1회 호출(청크 경계 무관)이 정확.

    복잡도: O(N·L²) — 각 이름의 접미 후보(≤L개)를 집합 조회. L≤상한이라 실질 선형.
    """
    # 소문자 키 → 원 표면형 (한국어는 lower no-op). 언어별 길이상한·불용어 필터.
    # ⚠️정렬 순회 — 대소문자만 다른 이름("Insurance Business"/"insurance business")이
    # 충돌하면 사전순 최소 표면형이 승자. set 순회(해시시드 의존)로 두면 실행마다
    # 다른 표면형이 살아남아 비결정적 출력(0711 실측: PYTHONHASHSEED 별로 상이).
    # 매칭 키만 공백 정규화(앞뒤·중복 공백)+소문자 — " 보험업" 같은 외부 유입 이름이
    # 분기 오판으로 조용히 계층 탈락하던 것 방지(0711 리뷰 실측). ⚠️방출 표면형은
    # **원형 유지** — 정규화형으로 방출하면 classes 목록·OWL class_uris(원 표면형 키)와
    # lookup 이 어긋나 subClassOf 가 조용히 탈락한다(0711 재검증서 발견된 회귀).
    names: dict[str, str] = {}
    for c in sorted(class_names):
        if not c or not c.strip():
            continue
        norm = " ".join(c.split())
        key = norm.lower()
        if key in names:
            continue  # 사전순 먼저 온 표면형 유지 (결정적)
        if " " in norm:
            if MIN_SUFFIX_LEN <= len(norm) <= MAX_LEN_SPACED and key not in STOP_HEAD_EN:
                names[key] = c  # 원 표면형
        else:
            if MIN_SUFFIX_LEN <= len(norm) <= MAX_LEN and norm not in STOP_HEAD:
                names[key] = c  # 원 표면형

    # 1) parent 후보별로 그 후보를 접미로 갖는 child 수집 (인덱스 조회, pairwise 아님).
    parent_to_children: dict[str, set[str]] = defaultdict(set)
    for child in names:
        if " " in child:
            # 공백형: 단어 접미 (뒤쪽 단어열) — "life insurance business" → "insurance business", "business"
            words = child.split()
            for i in range(1, len(words)):
                parent = " ".join(words[i:])
                if parent in names and parent not in STOP_HEAD_EN:
                    parent_to_children[parent].add(child)
        elif _HANGUL.search(child):
            # 무공백·한글 포함: 문자 접미 (한국어 복합명사는 형태소 연접이라 유효)
            n = len(child)
            for i in range(1, n - MIN_SUFFIX_LEN + 1):
                parent = child[i:]
                if parent in names and parent not in STOP_HEAD and parent not in STOP_HEAD_EN:
                    # kiwi 주입 시 형태소 경계 검증 — '민국' 류 문자 파편 상위어 컷.
                    # 원 표면형(names[child]/names[parent])으로 형태소 분석(소문자키 아님).
                    if kiwi is not None and not _morph_suffix_ok(names[child], names[parent], kiwi):
                        continue
                    parent_to_children[parent].add(child)
        # 순수 라틴 단일토큰: 문자 접미 생성 안 함 — 영어 단일단어는 형태소 경계가
        # 없어 placement⊂cement 류 오탐(0711 리뷰 실측). 공백형 다단어의 단어접미
        # parent 로는 여전히 참여한다.

    # 2) 허브 필터 — min_children 이상 거느린 parent 만 상위로 인정.
    #    정렬 방출 — set 순회의 해시시드 의존 순서를 제거(출력 순서도 결정적).
    out = []
    for parent in sorted(parent_to_children):
        children = parent_to_children[parent]
        if len(children) < min_children:
            continue
        # 상위어 head 가 고유명사(NNP)면 허브 거부 — 국가·기관·인명은 상위범주 부적격
        # (X⊂대한민국 류 접합 클래스 오염 차단). parent 1회 판정(child 루프 밖).
        if kiwi is not None and _parent_is_proper_noun(names[parent], kiwi):
            continue
        for child in sorted(children):
            # 중복 접합(child == parent+parent, 예: 최양업최양업·외래어표기법외래어표기법)
            # 거부 — NNP 게이트가 못 잡는 보통명사 상위어의 자기중복(Kiwi 오분절 포함).
            if names[child] == names[parent] + names[parent]:
                continue
            out.append({"parent": names[parent], "child": names[child]})
    return out
