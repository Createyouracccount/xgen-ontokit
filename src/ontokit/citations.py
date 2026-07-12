"""문서 간 인용쌍 결정적 추출 + :cites 트리플 방출 — 정규식, LLM 0회, O(N).

배경(0712 실증): 온톨로지 그래프가 검색 sources 에 물리 미연결이던 구조를,
doc-level 인용 온톨로지(:cites)로 연결해 multi-hop 완전회수를 유의 개선.
수치는 주석에 박지 않는다(부패 방지) — 단일 진실원천은 xgen-documents
benchmarks/ontology_vs_synaptic/regression_gate.py 실행 결과와 정본 문서
(docs/그래프sources_인용온톨로지_연결_PoC실증_2026_07_12.md §6-2)다.
스코프 주의: 평가 GT 는 인용쌍에서 생성된 준-오라클 — 수치는 "명시적 동일법
인용형 multi-hop"의 상한이지 자연 multi-hop 일반화 성능이 아니다.
이 모듈은 PoC(poc_cite_graph.py)의 라이브러리화 + 적대검증 개선(거짓엣지 마스킹).

범용 설계: 인용 표기 pattern 과 문서의 (group, key) 메타데이터를 주입받는다.
- group: 인용 해석 스코프. 같은 그룹 안에서만 키를 문서로 해석한다
  (예: 법령명 — "제3조"는 같은 법의 제3조. 스코프 없이 해석하면 오연결).
- key: 그 문서가 "가리켜지는" 표기 (예: article_no "제3조").
한국어 법령 프리셋 KO_LAW_ARTICLE 제공. 스코프 한계는 명시적 동일그룹 인용형 —
암시적 참조·그룹 간 인용(「」)은 미지원(0712 정본 문서 §5 공시와 동일).
key 메타데이터가 없는 문서는 인용을 "할" 수는 있어도 "받을" 수는 없다 —
전 문서 key 부재 시 pairs()=[] 로 자연 무력화(오연결보다 무동작이 안전).
"""
from __future__ import annotations
import re
from typing import Callable, Iterable, Mapping
from urllib.parse import quote

# 한국어 법령 조문 인용 표기 — "제3조", "제12조의2". PoC gen_holdout.py 와 동일.
KO_LAW_ARTICLE = re.compile(r"제\d+조(?:의\d+)?")

# 타 스코프 인용 마스크 — "「은행법」 제4조" 는 같은 법 제4조가 아니다. 마스킹 없이는
# 같은 그룹의 제4조로 *거짓 엣지*가 생긴다(finreg489 실측: 「」+제N조 230건 중
# 타법 229 vs 자기법 1 — 마스킹이 압도적 순이득). 자기법을 「」로 자기참조하는
# 희귀 참 엣지 1건 손실은 수용(거짓 엣지 > 누락).
# "같은 법/동법(시행령·시행규칙) 제N조"도 동일 클래스 — 시행령 문서의 "같은 법
# 제4조"는 모법(직전 인용된 법)의 제4조라 자기 스코프로 해석하면 거짓 엣지
# (finreg489 실측 9건, R2 적대검증 적발). 확실히 해석 불가한 별칭은 마스킹.
KO_QUOTED_SCOPE_REF = re.compile(
    r"「[^」]{1,80}」\s*(?:제\d+조(?:의\d+)?)?"
    r"|(?:같은\s*법|동\s*법)(?:\s*시행령|\s*시행규칙)?\s*제\d+조(?:의\d+)?"
)

DEFAULT_CITES_PREDICATE = "cites"


class CitationCollector:
    """스트리밍 빌드용 인용 수집기 — 청크 단위 add(), 끝에서 pairs().

    본문은 문서당 경계 캐리 꼬리(TAIL_CARRY자)만 유지하고 그 외 텍스트는 버린다 —
    직전 문서 꼬리는 다음 문서 시작 시 즉시 해제(수만~수백만 청크 메모리 안전.
    단, 캐리는 한 문서의 청크가 연속으로 add 되는 스트림 전제 — XGEN 두 경로 모두 해당).
    같은 문서의 청크를 여러 번 add 해도 안전(키 집합 누적, 결정적 순서 유지).
    mask_pattern 구간은 참조 추출에서 제외(타 스코프 인용의 거짓 엣지 차단).
    key 가 없는 청크의 참조는 스코프 "" 로 해석된다 — 메타 부분결측 코퍼스에서는
    해당 참조가 조용히 유실될 수 있다(거짓 엣지 대신 누락을 택한 설계).
    """

    # 청크 경계 캐리 — 직전 청크 꼬리를 이어붙여, 경계에서 갈라진 표기("「은행법」|제4조",
    # "제4|조")의 마스킹 실패·매칭 누락을 막는다. 캐리 길이는 마스크 최장 표기
    # (「…80자」+제N조 ≈ 95자)보다 길게 — 경계 걸침 표기의 「 문맥이 항상 캐리에 담긴다.
    TAIL_CARRY = 100

    def __init__(self, pattern: re.Pattern = KO_LAW_ARTICLE,
                 mask_pattern: re.Pattern | None = KO_QUOTED_SCOPE_REF):
        self.pattern = pattern
        self.mask_pattern = mask_pattern
        self._own: dict[str, tuple[str, str]] = {}      # doc_id -> (group, key)
        self._refs: dict[str, dict[tuple[str, str], None]] = {}  # doc_id -> 참조키 ordered-set
        self._tails: dict[str, str] = {}                # doc_id -> 직전 청크 꼬리 (현 문서만)
        self._last_doc: str | None = None

    def add(self, doc_id: str, *, group: str = "", key: str = "", text: str = "") -> None:
        if not doc_id:
            return
        if key and doc_id not in self._own:
            # 키 충돌(같은 그룹 같은 키 2문서)은 선착 우선 — 결정적 유지.
            self._own[doc_id] = (group, key)
        if self._last_doc is not None and self._last_doc != doc_id:
            self._tails.pop(self._last_doc, None)  # 직전 문서 꼬리 해제(메모리 상한)
        self._last_doc = doc_id
        refs = self._refs.setdefault(doc_id, {})
        raw = text or ""
        carry = self._tails.get(doc_id, "")
        text = carry + raw
        self._tails[doc_id] = raw[-self.TAIL_CARRY:] if raw else carry
        if self.mask_pattern is not None:
            # 등길이 공백 치환 — 오프셋 보존(아래 캐리 경계 판정이 위치 기반이라 필수)
            text = self.mask_pattern.sub(lambda m: " " * (m.end() - m.start()), text)
        for m in self.pattern.finditer(text):
            # 캐리 구간 안에서 *완결*되는 매치는 스킵 — 직전 add 가 온전한 좌측 문맥
            # (마스킹 포함)으로 이미 판정했다. 캐리 절단점이 「…」 내부에 떨어지면
            # 마스킹이 무력화돼 거짓 엣지가 생기는 것(R3 적대검증 A' repro)을 차단.
            # 경계에 *걸치는* 매치(start<len(carry)<end)만 캐리의 존재 이유이므로 유지.
            if m.end() <= len(carry):
                continue
            refs.setdefault((group, m.group(0)))

    def pairs(self) -> list[tuple[str, str]]:
        """(인용하는 doc_id, 인용받는 doc_id) 쌍 — 중복·자기인용 제외, 결정적 순서."""
        by_key: dict[tuple[str, str], str] = {}
        for d, gk in self._own.items():
            by_key.setdefault(gk, d)
        out: list[tuple[str, str]] = []
        for src, refs in self._refs.items():
            seen: set[str] = set()
            for gk in refs:
                dst = by_key.get(gk)
                if dst and dst != src and dst not in seen:
                    seen.add(dst)
                    out.append((src, dst))
        return out


def extract_citation_pairs(
    docs: Iterable[Mapping],
    *,
    pattern: re.Pattern = KO_LAW_ARTICLE,
    mask_pattern: re.Pattern | None = KO_QUOTED_SCOPE_REF,
    doc_id: Callable[[Mapping], str] = lambda d: d["doc_id"],
    group: Callable[[Mapping], str] = lambda d: d.get("law", ""),
    key: Callable[[Mapping], str] = lambda d: str(d.get("article_no", "")).strip(),
    text: Callable[[Mapping], str] = lambda d: d.get("text", ""),
) -> list[tuple[str, str]]:
    """일괄 편의 API — docs 전체를 CitationCollector 에 태워 pairs 반환."""
    c = CitationCollector(pattern=pattern, mask_pattern=mask_pattern)
    for d in docs:
        c.add(doc_id(d), group=group(d), key=key(d), text=text(d))
    return c.pairs()


def doc_uri(namespace: str, doc_id: str) -> str:
    """doc 노드 URI — 클래스 온톨로지와 분리된 인스턴스 층(<ns>doc/<id>).

    doc_id 는 URI 경로 성분이라 percent-encode(한글 파일명·공백 안전).
    빌드(트리플 방출)와 검색 leg(SPARQL VALUES)가 반드시 이 함수를 공유해야
    인코딩 불일치로 인한 조인 실패가 없다.
    """
    return f"{namespace}doc/{quote(str(doc_id), safe='')}"


def citations_to_ttl(
    pairs: Iterable[tuple[str, str]],
    *,
    namespace: str = "https://w3id.org/xgen-domain#",
    predicate: str = DEFAULT_CITES_PREDICATE,
) -> str:
    """인용쌍 → Turtle 문자열. 기존 OWL 업로드 경로(TTL POST)로 그대로 실림."""
    pred = f"{namespace}{predicate}"
    lines = [
        f"<{doc_uri(namespace, s)}> <{pred}> <{doc_uri(namespace, o)}> ."
        for s, o in pairs
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def citations_insert_update(
    pairs: Iterable[tuple[str, str]],
    graph_uri: str,
    *,
    namespace: str = "https://w3id.org/xgen-domain#",
    predicate: str = DEFAULT_CITES_PREDICATE,
    drop_first: bool = True,
    batch_size: int = 5000,
) -> list[str]:
    """인용쌍 → SPARQL Update 문 리스트 (DROP SILENT + INSERT DATA, 멱등 재빌드).

    반환이 리스트인 이유: 호출측이 문 단위로 실행·로깅하게 한다(실패 지점 가시화).
    pairs 가 비면 DROP 만 — 재빌드 시 옛 인용 그래프 잔존 방지.
    INSERT 는 batch_size 쌍 단위로 분할 — 대용량(수십만 쌍)서 단일 요청 폭주 방지
    (kg_builder 트리플 업로드 40k 배치와 같은 이유).
    """
    pred = f"{namespace}{predicate}"
    stmts: list[str] = []
    if drop_first:
        stmts.append(f"DROP SILENT GRAPH <{graph_uri}>")
    pairs = list(pairs)
    for i in range(0, len(pairs), batch_size):
        vals = " ".join(
            f"<{doc_uri(namespace, s)}> <{pred}> <{doc_uri(namespace, o)}> ."
            for s, o in pairs[i:i + batch_size]
        )
        stmts.append(f"INSERT DATA {{ GRAPH <{graph_uri}> {{ {vals} }} }}")
    return stmts
