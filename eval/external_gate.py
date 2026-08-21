# -*- coding: utf-8 -*-
"""외부 좌표 게이트 — 0822 선행조사 근거를 라운드 판정에 집행한다.

왜 코드인가: 판례 7(측정 로직은 전부 코드에) + 판례 27(iii) 실증 2회 —
문서에 써 놓은 규범은 지켜지지 않았다. 게이트가 자동으로 물어야 한다.

⛔ 이 파일의 상수는 **외부 문헌·타 시스템 실측치**다. 우리 라운드 결과로 고치지 않는다.
   고쳐야 할 근거가 생기면 출처와 함께 고치고 ledger 에 기록한다(판례 29: 원값 병기).

근거 원본: eval_runs/bench/RESEARCH_2026_08_22_외부선행조사.md
프로토콜:  eval_runs/bench/VERIFY_PROTOCOL.md
"""
from __future__ import annotations
import argparse, sys

# ── 닫힌 경로 (RESEARCH §5). 키워드가 걸리면 BLOCK. ────────────────────────────
CLOSED = {
    "C1": dict(
        name="어휘 스톱리스트 확장",
        kw=["스톱리스트", "불용어", "어휘 목록", "단어 목록", "보통명사 필터",
            "직함 목록", "친족어", "식재료", "접미사 목록", "블랙리스트", "화이트리스트"],
        why="semantic drift(1999~) — 목록 확대가 악화 요인. 모호명사 목록 자동생성도 실패"
            "(pairwise recall <1/3, 필터 강화 시 recall 1%대 붕괴)",
        src="Curran et al. 2007 / Preiss & Stevenson *SEM 2013"),
    "C2": dict(
        name="모델 신뢰도 임계로 정밀도 매수",
        kw=["신뢰도 임계", "min_score", "confidence 임계", "score 임계", "임계값 상향"],
        why="우리 실측(n=147): 0.40→30.6% · 0.60→33.7% · 0.90→21.1%. "
            "정밀도가 신뢰도와 함께 오르지 않고 0.6 넘으면 하락. "
            "score 는 'TTA 개체인가'를 재지 '온톨로지 인스턴스인가'를 못 잰다",
        src="0822 로컬 실측"),
    "C3": dict(
        name="다중 증거·중복 확인",
        kw=["다중 증거", "corroboration", "중복 확인", "출현 빈도 임계", "여러 문서 합의"],
        why="연결도 [1,2) 구간 AUC 이득 −0.001/−0.002. 뉴스는 극단적 롱테일. "
            "우리 실측도 빈도 5에서 커버리지 6.1% 붕괴",
        src="IBM arXiv:1908.08104 Table 3 / 0822 로컬"),
    "C4": dict(
        name="CRF·BIO 전이 제약 디코딩",
        kw=["CRF", "전이 제약", "constrained decoding", "viterbi 제약"],
        why="우리 오류는 합법 BIO 전이다. '##나'=B-, '##19'=I- 는 문법 위반이 아님. "
            "CRF 를 붙여도 '나19' 는 그대로 나온다",
        src="0822 스팬 레인"),
    "C5": dict(
        name="GLiNER 계열 도입",
        kw=["GLiNER", "gliner"],
        why="사내 m1 이미 기각 — gliner2 신규 오탐 15.3%[10.7,21.5]. 조사 포함 방출",
        src="eval_runs/typing/m1_judge_verdict.md"),
    "C6": dict(
        name="사후 규칙으로 절단 수리",
        kw=["트림으로 절단", "사후 규칙으로 경계", "후처리로 절단"],
        why="off-morpheme 11.26% → 11.30%(트림 적용 후). 트림은 줄이는 연산인데 "
            "절단은 시작점이 틀려 복구 불가",
        src="0822 스팬 레인 실측"),
    "C7": dict(
        name="LLM 에게 규칙·labeling function 생성",
        kw=["LLM 이 규칙", "LLM 규칙 생성", "labeling function 생성", "LF 생성"],
        why="NER 전용 실측 최고 F1 16.66 · 최저 0. LLM 은 어노테이터로만 쓸 것",
        src="ACM IKDD CODS 2026"),
    "C8": dict(
        name="순진한 분기 엔트로피",
        kw=["분기 엔트로피", "branching entropy", "accessor variety"],
        why="정밀도 하락(임계 0.0→27.9% … 1.5→0.0%). '테슬라'도 H_L=0 — 공백이 신호를 삼킴. "
            "좌측이 '단어문자'일 것을 요구하는 정련형(left_closed)만 유효",
        src="0822 로컬 실측"),
    "C9": dict(
        name="엔티티 링킹을 정밀도 게이트로",
        kw=["엔티티 링킹 필터", "entity linking 게이트", "EL 필터", "NIL 로 거부"],
        why="영어+영어KB+성숙 시스템(DBpedia Spotlight) 뉴스 best F1 56.0% — "
            "게이트가 44% 오차를 새로 들여온다. 한국어 Wikidata 라벨 커버리지 1.61%. "
            "NIL 은 'KB 에 없음'이지 '개체 아님'이 아니다",
        src="Mendes et al. 2011 / Wikidata 통계"),
    "C10": dict(
        name="P31 존재로 instance 판정",
        kw=["P31 로 개체", "P31 있으면", "instance of 로 판정"],
        why="Wikidata 구조 오염 도메인별 40~100%. 2차 클래스 239만이 1차로도 분류. "
            "split-order 6,379 클래스. 비대칭(P279⇒class)만 유효",
        src="arXiv:2411.15550 / arXiv:2511.04926 / Wikidata WikiProject Ontology"),
    "C11": dict(
        name="aggregation average/max",
        kw=["average 집계", "max 집계", "aggregation_strategy=\"average\"",
            "aggregation_strategy=\"max\""],
        why="엔티티 −43%/−40%. 조사 서브워드(O)가 평균을 끌어내려 임계 탈락. "
            "max 는 '혁신 STAR상'→'혁신' 파손",
        src="0822 스팬 레인 실측"),
    "C12": dict(
        name="형태소인지 토크나이저 교체",
        kw=["형태소인지 토크나이저", "morpheme-aware tokeniz", "mecab vocab"],
        why="이미 소진 — KoELECTRA v3 는 Mecab+WordPiece vocab. 그리고 유일한 스팬 과제"
            "(KorQuAD)에서 형태소인지가 순수 BPE 에 패배",
        src="monologg/KoELECTRA / kortok AACL 2020"),
}

# ── 타입 수 ↔ 정밀도 곡선 (동일 BERT-Tagger 완전지도). RESEARCH §7 ──────────────
TYPE_CURVE = [(4, 90.62), (18, 90.00), (66, 65.56), (10331, 52.40)]
HUMAN_KAPPA = 76.44          # Few-NERD 숙련 어노테이터 2인 일치
GOOGLE_KG_BAR = 99.0         # KV 가 미달로 탈락한 요구선
KV_RAW = 30.0                # KV 원시 추출 정밀도
KLUE_BOUNDARY_TAX = 6.45     # 6타입에서도 Entity-F1 vs Char-F1 격차

# ── 우리 기준선 (0822 실측). 라운드마다 갱신하되 출처를 남긴다 ──────────────────
OUR = dict(types=1100, precision=62.14, recall=63.51,
           bucket={"spurious": 62.3, "partial": 32.1, "incorrect_type": 3.8, "merge": 1.9},
           conf_curve=[(0.40, 30.6), (0.60, 33.7), (0.70, 28.1), (0.90, 21.1)],
           gold_n=74, gold_mde_pp=8.11, gold_power=8)

# ── G3 평가 설계 조건 ─────────────────────────────────────────────────────────
G3 = {
    "대응설계(McNemar/paired bootstrap)": False,   # ⛔ 재현율 골드는 비대응
    "분석셋/잠금테스트셋 물리 분리":      False,   # ⛔ 미분리
    "심판 벤더 다양화":                   False,   # ⛔ 판례 21 4라운드 미해소
}
MIN_GOLD_N = 150   # Card et al. 이 은퇴 권고한 WNLI(147)보다 커야 한다


def interp(types: int) -> float:
    """타입 수 곡선의 로그선형 보간."""
    import math
    pts = TYPE_CURVE
    if types <= pts[0][0]:
        return pts[0][1]
    if types >= pts[-1][0]:
        return pts[-1][1]
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t0 <= types <= t1:
            w = (math.log(types) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return p0 + w * (p1 - p0)
    return pts[-1][1]


def check_path(text: str) -> list[str]:
    """처치 설명이 닫힌 경로에 걸리는지. 걸린 항목 코드 목록 반환."""
    low = text.lower()
    return [c for c, d in CLOSED.items()
            if any(k.lower() in low for k in d["kw"])]


def main() -> int:
    ap = argparse.ArgumentParser(description="외부 좌표 게이트 (0822)")
    ap.add_argument("--check-path", metavar="TEXT", help="처치 설명 — 닫힌 경로 대조")
    ap.add_argument("--target", type=float, help="목표 정밀도 (0~1 또는 0~100)")
    ap.add_argument("--types", type=int, default=OUR["types"], help="클래스 수")
    ap.add_argument("--gold-n", type=int, help="이 라운드 골드 표본 수")
    ap.add_argument("--report", action="store_true", help="전 좌표 출력")
    a = ap.parse_args()
    blocked = False

    if a.check_path:
        hits = check_path(a.check_path)
        print(f"\n=== G0 처치 제안 게이트 ===\n입력: {a.check_path}")
        if hits:
            blocked = True
            for c in hits:
                d = CLOSED[c]
                print(f"\n⛔ BLOCK [{c}] {d['name']}\n   근거: {d['why']}\n   출처: {d['src']}")
            print("\n재론하려면 위 근거를 무효화하는 신규 실측을 제출하고 ledger 에 기록하라.")
        else:
            print("✅ PASS — 닫힌 경로 미해당. (통과가 곧 타당성은 아니다)")

    if a.target is not None:
        t = a.target * 100 if a.target <= 1.0 else a.target
        base = interp(a.types)
        print(f"\n=== G1 목표 수치 게이트 ===")
        print(f"클래스 {a.types}종 → 곡선 보간 기대 정밀도 {base:.2f}%")
        print(f"우리 현재 {OUR['precision']:.2f}% · 목표 {t:.2f}%")
        gap = t - base
        print(f"곡선 대비 초과분 {gap:+.2f}pp")
        if gap >= 10:
            blocked = True
            print(f"⛔ BLOCK — 공개 SOTA 를 {gap:.1f}pp 앞서겠다는 목표다. 근거 제시 의무.")
        if t > HUMAN_KAPPA:
            blocked = True
            print(f"⛔ BLOCK — 인간 합의 상한 κ {HUMAN_KAPPA}% 초과 "
                  f"(Few-NERD: 70명+전문가10·1인32h·배치95%미달 시 전량 재작업 조건에서의 2인 일치)")
        if not blocked:
            print("✅ PASS")
        print(f"참고: Google KG 요구선 {GOOGLE_KG_BAR}% — Knowledge Vault 가 미달로 탈락. "
              f"KV 원시 추출 {KV_RAW}%")

    if a.gold_n is not None:
        print(f"\n=== G3 검정력 게이트 ===")
        print(f"골드 n={a.gold_n} · 하한 {MIN_GOLD_N}")
        if a.gold_n < MIN_GOLD_N:
            blocked = True
            print(f"⛔ BLOCK — Card et al.(EMNLP 2020)이 은퇴 권고한 WNLI(147건, MDE 5.26%)"
                  f"보다 작거나 비슷하다. 채택 판정 불가.")
        else:
            print("✅ PASS")
        for k, v in G3.items():
            print(f"  {'✅' if v else '⛔'} {k}")
        if not all(G3.values()):
            print("  → G3 미충족 항목이 있으면 채택 판정을 낼 수 없다(0822 이후 구속).")

    if a.report:
        print("\n=== 외부 좌표 전체 ===")
        print(f"타입 수 곡선: " + " → ".join(f"{t}종 {p}" for t, p in TYPE_CURVE))
        print(f"우리 {OUR['types']}종 {OUR['precision']}% (보간 기대 {interp(OUR['types']):.2f}%)")
        print(f"인간 합의 상한 κ {HUMAN_KAPPA}% · Google KG 바 {GOOGLE_KG_BAR}% · KV 원시 {KV_RAW}%")
        print(f"KLUE 경계세(6타입에서도) {KLUE_BOUNDARY_TAX}pp")
        print(f"\n오류 버킷: " + " · ".join(f"{k} {v}%" for k, v in OUR["bucket"].items()))
        print(f"신뢰도 곡선(무효 실증): " + " · ".join(f"{t}→{p}%" for t, p in OUR["conf_curve"]))
        print(f"골드 n={OUR['gold_n']} MDE {OUR['gold_mde_pp']}pp 검정력 {OUR['gold_power']}%")
        print(f"\n닫힌 경로 {len(CLOSED)}개: " + ", ".join(f"{c}={d['name']}" for c, d in CLOSED.items()))

    if not any([a.check_path, a.target is not None, a.gold_n is not None, a.report]):
        ap.print_help()
        return 0
    print()
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
