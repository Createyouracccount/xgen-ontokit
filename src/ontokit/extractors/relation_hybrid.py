"""관계 hybrid top-up — 규칙(LLM 0) 우선, 규칙이 못 잡은 청크만 LLM 보강.

배경: 조사 규칙(relation_ko)은 고정밀·저재현(GT81 pooling 실측 F1 0.309, R 0.158
— "~한다" 동사문만 걸리고 의무·금지·준용·무주어 규범을 놓침). LLM 은 F1 0.654 지만
전 청크 호출은 LLM-free 의 비용·재현 이점을 잃는다. 해법(업계 표준 selective
fallback + budget cap): **규칙이 관계 0건인 청크만** LLM 후보로 올리고, 사용자
예산 가드레일(청크 비율 상한 ∨ 달러 상한, 먼저 닿는 것에서 정지) 안에서만 호출.

핵심 불변식(LLM-free 가치 보호):
- 예산 0(max_chunk_pct=0 ∧ max_usd=0) → LLM 호출 0 → **순수 규칙 결과와 동일**.
- 예산 상한 초과 청크 → 규칙 결과(0건)로 남김 = 안전 폴백.
- **청크% 상한 = 엄격 하드 상한**(순수 카운트). **달러 상한 = LLM max_tokens 물리
  제한 시 하드, 없으면 최대 1호출 실제출력만큼 초과 가능**(유한 — BudgetGuard 참조).

주입(코어 의존성 0): LLM 은 ontokit.protocols.LLM(generate(prompt, system, timeout))
프로토콜만 요구. 토큰 회계는 usage 콜백(있으면)·문자수 근사(폴백) 이중.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .relation_ko import KoreanRelationExtractor

# 기본 LLM 추출 프롬프트 — score_rel_gt.py 의 검증된 LLM_SYS 와 동형(법령 SVO).
DEFAULT_LLM_SYSTEM = (
    "당신은 법령·규정 조문에서 관계 트리플을 추출하는 전문가입니다.\n"
    "(주어 | 술어 | 목적어) SVO 트리플을 JSON으로 추출합니다.\n"
    "- 주어=행위 주체, 술어=행위(동사성, 명사형 정규화: 선임/제출/감독 등), 목적어=행위 대상.\n"
    "- 금지규범은 술어 '~ 금지', 준용은 '준용', 간주는 '간주', 적용배제는 '적용 배제'.\n"
    "- '대통령령으로 정한다'류 단순 위임은 제외. 조문에 명시/명백히 함축된 관계만.\n"
    "- 관계가 없으면 빈 배열.\n"
    '반드시 {"triples":[{"s":"...","p":"...","o":"..."}, ...]} 형태 JSON으로만 답하세요.'
)

# 입력 요금(달러/1K 토큰) 기본값 — 호출측이 실제 모델 요율로 덮어쓸 것.
# 토큰≈문자/3(한국어 근사). 회계는 usage 콜백 우선, 없으면 이 근사.
_CHARS_PER_TOKEN = 3.0


def _nkey(s: str) -> str:
    """관계 dedup 정규화 키 — 공백 제거 + 소문자(규칙·LLM 표기차 흡수)."""
    return re.sub(r"\s+", "", str(s)).lower()


@dataclass
class BudgetGuard:
    """LLM 호출 예산 가드레일 — 청크 비율 상한 ∨ 달러 상한(먼저 닿는 것에서 정지).

    둘 다 None 이면 무제한(전량 top-up = LLM 상한선 측정용). 둘 다 0 이면 호출 0
    (순수 LLM-free).

    상한 강도(R2 적대검증 반영 — 정직한 공시):
    - **청크% 상한**: 순수 카운트라 **엄격한 하드 상한**(초과 불가).
    - **달러 상한**: 사전판정(can_call)은 출력 토큰을 호출 *전* 에 알 수 없어
      max_output_chars 로 가정한다. 출력이 이 가정을 넘으면 **마지막 1호출 실제
      출력만큼 초과 가능**(초과 즉시 이후 전량 거부되므로 유한, 무한증가 없음).
      ⚠️ usage 콜백은 *사후* 회계만 정확하게 할 뿐 사전판정은 여전히 근사라
      첫 초과를 못 막는다. **진짜 하드하게 하려면 LLM 호출에 max_tokens 를
      max_output_chars/토큰환산 값으로 물리 제한**해 출력을 상한 안에 가둘 것.
      엄격 상한이 최우선이면 청크% 상한을 쓰는 게 가장 안전하다.
    """
    max_chunk_pct: Optional[float] = 0.10   # 전체 청크의 비율 상한(0~1)
    max_usd: Optional[float] = None          # 누적 달러 상한
    price_per_1k_input: float = 0.005        # 입력 요금(달러/1K 토큰) — 모델별 덮어쓰기
    price_per_1k_output: float = 0.015
    max_output_chars: int = 1200             # 사전판정 출력 가정 = LLM max_tokens 와 맞출 것
    # ── 달러캡 하드화(0713 R2 실빌드에서 44% 초과 발견 → 보수적 사전판정) ──
    # 실측 결과: char/3 근사가 한국어 입력 토큰을 과소계상 + 시스템프롬프트 오버헤드
    # 누락으로 can_call 이 실제보다 싸게 봐 상한을 넘겼다. 사전판정은 '절대 과소추정
    # 안 함' 원칙으로: (1) 보수적 토큰환산(한국어 ≈2자/토큰), (2) 고정 오버헤드 가산,
    # (3) 안전마진(1콜 최악비용) 예약. 회계(record)는 실 usage 우선.
    est_chars_per_token: float = 2.0         # 사전판정용 보수적 환산(과소추정 방지)
    per_call_overhead_tokens: int = 400      # 시스템프롬프트+프리픽스 고정 오버헤드
    safety_margin_calls: float = 1.0         # 상한 근처에서 예약할 '최악 1콜' 배수

    # 실시간 회계(내부 상태)
    spent_usd: float = 0.0
    called: int = 0
    _budget_chunks: Optional[int] = field(default=None, repr=False)

    def plan(self, total_chunks: int) -> None:
        """전체 청크 수를 받아 비율 상한을 절대 청크 수로 확정(사전 계산)."""
        if self.max_chunk_pct is None:
            self._budget_chunks = None
        else:
            self._budget_chunks = int(total_chunks * self.max_chunk_pct)

    def can_call(self, est_input_chars: int,
                 est_output_chars: Optional[int] = None) -> bool:
        """이 청크에 LLM 을 호출해도 상한 내인가(사전 판정, 보수적).

        '절대 과소추정 안 함' — 입력을 보수적 토큰환산(est_chars_per_token, 한국어
        ≈2자/토큰) + 고정 오버헤드로 크게 잡고, 상한 근처엔 안전마진(최악 1콜)을
        예약한다. 이래야 실측 비용이 max_usd 를 넘지 않는다(R2 44% 초과 수정)."""
        if self._budget_chunks is not None and self.called >= self._budget_chunks:
            return False
        if self.max_usd is not None:
            out = est_output_chars if est_output_chars is not None else self.max_output_chars
            est = self._est_cost(est_input_chars, out)
            margin = self.safety_margin_calls * self._worst_call_cost()
            if self.spent_usd + est + margin > self.max_usd:
                return False
        return True

    def _est_cost(self, in_chars: int, out_chars: int) -> float:
        """사전판정용 보수적 비용 — 과소추정 방지(작은 환산 + 오버헤드 가산)."""
        in_tok = (in_chars / self.est_chars_per_token / 1000.0
                  + self.per_call_overhead_tokens / 1000.0)
        out_tok = out_chars / self.est_chars_per_token / 1000.0
        return in_tok * self.price_per_1k_input + out_tok * self.price_per_1k_output

    def _worst_call_cost(self) -> float:
        """안전마진용 — 최악(입력 상한 6000자 + 출력 상한) 1콜 비용."""
        return self._est_cost(6000, self.max_output_chars)

    def _cost(self, in_chars: int, out_chars: int) -> float:
        in_tok = in_chars / _CHARS_PER_TOKEN / 1000.0
        out_tok = out_chars / _CHARS_PER_TOKEN / 1000.0
        return in_tok * self.price_per_1k_input + out_tok * self.price_per_1k_output

    def record(self, in_chars: int, out_chars: int,
               usage: Optional[dict] = None) -> None:
        """실제 사용량 회계 — usage(모델 반환 토큰수) 우선, 없으면 문자수 근사."""
        self.called += 1
        if usage and "input_tokens" in usage:
            it = usage["input_tokens"] / 1000.0
            ot = usage.get("output_tokens", 0) / 1000.0
            self.spent_usd += it * self.price_per_1k_input + ot * self.price_per_1k_output
        else:
            self.spent_usd += self._cost(in_chars, out_chars)


@dataclass
class HybridReport:
    total_chunks: int = 0
    rule_covered: int = 0        # 규칙이 ≥1건 뽑아 top-up 불요
    topup_candidates: int = 0    # 규칙 0건(후보)
    llm_called: int = 0          # 실제 LLM 호출
    budget_skipped: int = 0      # 후보였으나 예산 상한으로 규칙(0건) 유지
    rule_triples: int = 0
    llm_triples: int = 0
    spent_usd: float = 0.0

    def as_dict(self) -> dict:
        return {
            "total_chunks": self.total_chunks, "rule_covered": self.rule_covered,
            "topup_candidates": self.topup_candidates, "llm_called": self.llm_called,
            "budget_skipped": self.budget_skipped, "rule_triples": self.rule_triples,
            "llm_triples": self.llm_triples, "spent_usd": round(self.spent_usd, 4),
            "llm_call_pct": round(self.llm_called / max(self.total_chunks, 1), 4),
        }


def _parse_llm_json(raw: str) -> list[dict]:
    """LLM 응답 → [{s,p,o}] — 코드펜스·잡텍스트 관용 파싱."""
    if not raw:
        return []
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    arr = obj.get("triples", []) if isinstance(obj, dict) else obj
    if not isinstance(arr, list):
        return []
    out = []
    for t in arr:
        if isinstance(t, dict) and t.get("s") and t.get("p") and t.get("o"):
            out.append({"s": str(t["s"]), "p": str(t["p"]), "o": str(t["o"])})
    return out


class HybridRelationExtractor:
    """규칙 우선 + 규칙 0건 청크만 예산 내 LLM top-up.

    사용:
        h = HybridRelationExtractor(llm=my_llm, budget=BudgetGuard(max_chunk_pct=0.1))
        rels, report = await h.extract_corpus(chunks)   # chunks=[{chunk_id,chunk_text}]
    llm=None 이면 순수 규칙(top-up 0) — LLM-free 완전 동치.
    """

    def __init__(self, *, llm=None, budget: Optional[BudgetGuard] = None,
                 kiwi=None, rule_extractor: Optional[KoreanRelationExtractor] = None,
                 llm_system: str = DEFAULT_LLM_SYSTEM,
                 topup_when: str = "empty", sparse_chars_per_rel: int = 400,
                 usage_extractor: Optional[Callable[[object], Optional[dict]]] = None):
        """topup_when: 어떤 청크를 LLM top-up 후보로 올릴지.
        - 'empty'(기본): 규칙이 관계 0건인 청크만. 밀도 낮은 코퍼스(위키·영어권)에
          유효하나, 법령류(규칙이 항상 ≥1건)에선 거의 발화 안 함(GT12 실측).
        - 'sparse': 0건 ∨ 규칙 관계 밀도가 낮은 청크(len(rels) < len(text)/
          sparse_chars_per_rel). 규칙이 '일부만 뽑고 놓친' 조문까지 회수 —
          법령류 재현율 갭의 실제 원인을 커버. 대신 후보·비용 증가.
        """
        if topup_when not in ("empty", "sparse"):
            raise ValueError(f"topup_when must be 'empty'|'sparse', got {topup_when!r}")
        self.rule = rule_extractor or KoreanRelationExtractor(kiwi=kiwi)
        self.llm = llm
        self.budget = budget or BudgetGuard()
        self.llm_system = llm_system
        self.topup_when = topup_when
        self.sparse_chars_per_rel = sparse_chars_per_rel
        # 주입 LLM 이 usage(토큰수)를 별 경로로 노출할 때 뽑는 콜백(선택).
        self.usage_extractor = usage_extractor

    def _is_candidate(self, rels: list, text: str) -> bool:
        """이 청크를 LLM top-up 후보로 올릴지(트리거 판정)."""
        if not rels:
            return True
        if self.topup_when == "sparse":
            expected = len(text) / max(self.sparse_chars_per_rel, 1)
            return len(rels) < expected
        return False

    async def extract_corpus(self, chunks: list[dict]) -> tuple[list[dict], dict]:
        """청크 리스트 → (관계 dict 리스트, 리포트).

        chunks: [{chunk_id, chunk_text}, ...]. 반환 관계는 relation_ko 스키마
        ({subject,predicate,object,predicate_type,source_chunks})와 동형 +
        LLM 산출은 origin='llm_topup' 태깅(하위가 신뢰도 구분 가능).
        """
        report = HybridReport(total_chunks=len(chunks))
        self.budget.plan(len(chunks))
        all_rels: list[dict] = []
        candidates: list[dict] = []

        # 1패스: 규칙 전량 실행(값쌈). 규칙 결과는 항상 유지하고, 트리거(0건 ∨
        # sparse) 걸린 청크만 top-up 후보로. sparse 는 규칙이 뽑은 것 위에 LLM 이
        # 놓친 것을 더한다(dedup).
        for ch in chunks:
            cid = ch.get("chunk_id")
            text = ch.get("chunk_text", "")
            sc = [cid] if cid else []
            rels = self.rule.extract(text, source_chunks=sc)
            if rels:
                report.rule_covered += 1
                report.rule_triples += len(rels)
                all_rels.extend(rels)
            if self._is_candidate(rels, text):
                candidates.append((ch, rels))
        report.topup_candidates = len(candidates)

        # 2패스: 후보를 예산 내에서만 LLM top-up. LLM 없으면 스킵(순수 규칙).
        if self.llm is not None:
            # 짧은 청크 우선(관계당 비용 최소 = relations-per-dollar 극대화).
            # R2 실빌드 교훈: 긴 청크 우선은 표·헤더 등 저수율 청크에 예산을 먼저
            # 태워 상한 소진 후 생산적 짧은 청크(검사로그 등)에 도달 못 함(85콜/$0.43
            # → 관계 0). 짧은 청크부터면 예산 소진 시에도 우아하게 저하된다.
            candidates.sort(key=lambda cr: len(cr[0].get("chunk_text", "")))
            for ch, rule_rels in candidates:
                text = ch.get("chunk_text", "")
                cid = ch.get("chunk_id")
                sc = [cid] if cid else []
                if not self.budget.can_call(len(text)):
                    report.budget_skipped += 1
                    continue
                triples, usage = await self._llm_call(text)
                self.budget.record(len(text), sum(len(t["p"]) + len(t["s"]) + len(t["o"])
                                                  for t in triples) or 100, usage)
                # sparse: 규칙이 이미 뽑은 (s,p,o)는 중복 방출 안 함(정규화 키).
                seen = {(_nkey(r["subject"]), _nkey(r["predicate"]), _nkey(r["object"]))
                        for r in rule_rels}
                for t in triples:
                    k = (_nkey(t["s"]), _nkey(t["p"]), _nkey(t["o"]))
                    if k in seen:
                        continue
                    seen.add(k)
                    all_rels.append({
                        "subject": t["s"], "predicate": t["p"], "object": t["o"],
                        "predicate_type": "ObjectProperty", "source_chunks": sc,
                        "origin": "llm_topup",
                    })
                    report.llm_triples += 1
        else:
            report.budget_skipped = len(candidates)

        report.llm_called = self.budget.called
        report.spent_usd = self.budget.spent_usd
        return all_rels, report.as_dict()

    async def _llm_call(self, text: str) -> tuple[list[dict], Optional[dict]]:
        prompt = f"조문:\n{text[:6000]}"
        # 출력 물리 상한 — max_usd 를 하드 상한으로 만드는 핵심(BudgetGuard docstring).
        # can_call 은 출력을 max_output_chars 로 가정하는데, 실제 출력이 이를 넘으면
        # 마지막 1호출이 상한을 넘는다. LLM max_tokens 를 이 가정과 맞춰 물리 제한하면
        # 초과가 원천 차단된다. 토큰≈문자/3(한국어 근사, _CHARS_PER_TOKEN).
        max_tokens = max(1, int(self.budget.max_output_chars / _CHARS_PER_TOKEN))
        try:
            try:
                res = self.llm.generate(prompt, system=self.llm_system,
                                        timeout=60, max_tokens=max_tokens)
            except TypeError:
                # max_tokens 미지원 어댑터 — 폴백(이 경우 달러캡은 소프트, docstring 경고).
                res = self.llm.generate(prompt, system=self.llm_system, timeout=60)
            if isinstance(res, Awaitable):
                res = await res
        except Exception:
            return [], None
        raw = res if isinstance(res, str) else getattr(res, "text", str(res))
        usage = self.usage_extractor(res) if self.usage_extractor else None
        return _parse_llm_json(raw), usage
