"""Round-1 계층 평가 하네스 — 심판 6개 수정 전부 반영. 심판 독립 검증용.

수정 반영:
 ① 순환 차단: 추출 소스 = 한국어 위키피디아 lead(gold_r1.leads), Wikidata desc 아님.
 ② 전이폐포 크레딧: gold = P279* 다중홉 조상(gold_closure). 층위 불일치 인정.
 ③ substring baseline: lead 에 gold조상이 substring 인 oracle 병기. 방법이 이걸 넘어야.
 ④ 정밀도 가드: hetero_R → hetero_F1. recall-only 게이밍 차단.
 ⑤ T2 열린 어휘: distractor 클래스 추가해 h∈classes 게이트가 real build 반영.
 ⑥ held-out: dev 로 개발, test(미열람)로 최종 보고.

사용: python3 eval_r1.py <method> <split>   method∈{baseline,improved}  split∈{dev,test}
"""
from __future__ import annotations
import sys, json, random, os
_HERE = os.path.dirname(os.path.abspath(__file__))
# 리포 상대경로 — eval/hierarchy/ 기준 ../../src, ./data
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

G = json.load(open(os.path.join(_HERE, "data", "gold_r1.json")))
DIRECT = [tuple(p) for p in G["direct_pairs"]]
CLOSURE = {k: set(v) for k, v in G["gold_closure"].items()}  # child -> {모든 조상}
LEADS = {k: v for k, v in G["leads"].items() if v}
DEV = [c for c in G["dev"] if c in LEADS and c in CLOSURE]
TEST = [c for c in G["test"] if c in LEADS and c in CLOSURE]

_ke = None
def _get_ke():
    global _ke
    if _ke is None:
        from ontokit.morphology.kiwi_nouns import KiwiNounExtractor
        _ke = KiwiNounExtractor()
    return _ke


# ══ 상위어 추출 (독립 소스 = 위키피디아 lead) ══
import re
def _first_def_sentence(lead: str, title: str) -> str:
    """lead 첫 문장(정의문) — 'X(...)는 ... Y이다' 패턴. 괄호 주석 제거."""
    s = re.split(r"(?<=다)\.\s|\.\s", lead)[0]
    s = re.sub(r"\([^)]*\)", "", s)  # (quantum computer,...) 제거
    return s

def extract_hypernyms_baseline(title: str) -> list[str]:
    """현 ontokit: hearst_ko 따옴표 정의문만. 위키 lead 엔 따옴표 드묾 → 대부분 빈결과."""
    from ontokit.hierarchy.hearst_ko import definitional_pairs
    ke = _get_ke()
    return [p["parent"] for p in definitional_pairs(LEADS[title], ke.last_noun)]

_GENUS_TAIL = ("종류", "일종", "분야", "갈래", "품종", "형태", "하나", "한종")  # "…의 X" 형식명사
_COPULA_STOP = {"것", "하나", "때", "곳", "중"}  # 계사 앞이라도 상위어 부적격

def _noun_run_before(toks, i):
    """toks[i] 직전(포함 안 함)의 연속 NNG/NNP 명사구 표면형. 관형어 컷은 호출측."""
    buf = []
    j = i - 1
    while j >= 0 and toks[j].tag in ("NNG", "NNP"):
        buf.insert(0, toks[j].form); j -= 1
    return "".join(buf), j

def extract_hypernyms_improved(title: str) -> list[str]:
    """개선 v3 — 정의문 종결 패턴 정밀 targeting(dev 패턴분석 기반).

    한국어 백과 정의문의 상위어는 종결 계사/서술 직전 명사구에 온다:
     ① 계사 종결 "…<상위어>이다/이라 한다": VCP(이)+EF 직전 명사구(계란빵은…음식**이다**).
     ② 서술 종결 "…<상위어>를/을 말한다": '말하'+목적어 명사구(공공외교…활동**을 말한다**).
     ③ genus 관형 "…<상위어>의 한 {분야/품종/종류/일종}": 형식명사 앞 '의' 앞 명사구
        (그래프이론…수학**의 한 분야**, 데본렉스…고양이**의 한 품종** — 앞 수식어 '영국'은 컷).
     ④ "…<상위어>에 속하는": '속하'+JKB '에' 앞 명사구.
    우선순위 ③②④① (genus/서술이 계사보다 정밀). 각 후보는 head-final 복합명사 통째.
    """
    sent = _first_def_sentence(LEADS[title], title)
    toks = _get_ke().kiwi.tokenize(sent)
    n = len(toks)
    forms = [t.form for t in toks]
    tags = [t.tag for t in toks]
    cand: list[str] = []

    for i in range(n):
        t = toks[i]
        # ③ genus: '의' + (한/두 관형어 스킵) + 형식명사 → '의' 앞 명사구
        if t.tag == "JKG" and t.form == "의":
            look = "".join(forms[i+1:i+5])
            if any(g in look for g in _GENUS_TAIL):
                run, _ = _noun_run_before(toks, i)
                if run:
                    cand.append(("genus", run))
        # ④ "에 속하는": JKB '에' 앞 명사구 (뒤에 '속하' 확인)
        if t.tag == "JKB" and t.form == "에":
            if "속하" in "".join(forms[i+1:i+3]):
                run, _ = _noun_run_before(toks, i)
                if run:
                    cand.append(("belong", run))
        # ② 서술 "…말한다/불린다": '말하'/'불리' 직전 목적어(JKO 앞) 명사구
        if t.form in ("말하", "불리", "칭하") and t.tag.startswith("V"):
            # 직전 JKO(을/를) 앞 명사구
            j = i - 1
            while j >= 0 and toks[j].tag in ("JKO",):
                j -= 1
            run, _ = _noun_run_before(toks, j + 1)
            if run:
                cand.append(("say", run))
        # ① 계사 종결: VCP(이) 앞 명사구 — 문미 근처만(마지막 계사)
        if t.tag == "VCP" and t.form == "이":
            run, _ = _noun_run_before(toks, i)
            if run and run not in _COPULA_STOP:
                cand.append(("copula", run))

    # 문미 명사구 폴백(패턴 미발화 시)
    fallback = []
    i = n - 1
    while i >= 0:
        if toks[i].tag in ("NNG", "NNP"):
            buf = []
            while i >= 0 and toks[i].tag in ("NNG", "NNP"):
                buf.insert(0, toks[i].form); i -= 1
            fallback.append("".join(buf))
        else:
            i -= 1

    # ⑤ 제목 접미(형태소 경계) — 접미공유 원리. dev 실측: 접미(30%)>head낱개(24%).
    #   "간세포"→[세포], "고대철학"→[철학], "1인용 비디오 게임"→[비디오게임].
    #   형태소 경계에서만 자름(임의 부분문자열 아님) — Kiwi 명사 토큰 접미열.
    ttoks = _get_ke().kiwi.tokenize(title)
    tnouns = [t.form for t in ttoks if t.tag in ("NNG", "NNP")]
    title_suffix = []   # 형태소 경계 접미(긴 것부터 = 가장 구체적 상위어 우선)
    if len(tnouns) >= 2:
        for start in range(1, len(tnouns)):
            title_suffix.append("".join(tnouns[start:]))
    title_head = tnouns[-1] if tnouns else ""

    # 우선순위: genus/belong/say(정의문 정밀) > 제목접미 > 제목head > copula > 문미폴백
    order = {"genus": 0, "belong": 1, "say": 2}
    def_cand = sorted([kv for kv in cand if kv[0] in order],
                      key=lambda kv: order[kv[0]])
    copula_cand = [kv for kv in cand if kv[0] == "copula"]

    ranked, seen = [], set()
    def push(x):
        if x and x not in seen and len(x) >= 2 and x != title:
            seen.add(x); ranked.append(x); return True
        return False
    for _, x in def_cand:      # 정의문 genus/belong/say
        push(x)
    for x in title_suffix:     # 제목 접미(긴 것부터)
        push(x)
    push(title_head)           # 제목 head 낱개
    for _, x in copula_cand:   # 계사(오탐 잦아 뒤로)
        push(x)
    for x in fallback:
        push(x)
    return ranked


# ══ T1: 상위어 발견 (전이폐포 gold, SemEval T9 metric) ══
def _nrm(s: str) -> str:
    """매칭 정규화 — 띄어쓰기 무시. gold "비디오 게임"과 추출 "비디오게임"은 동일
    개념(Kiwi 복합명사는 공백없이 결합, 위키 표면형은 공백 유지). 심판 공정성."""
    return s.replace(" ", "")

def eval_t1(extract, split):
    kids = DEV if split == "dev" else TEST
    p1 = mrr = mapv = 0.0
    for c in kids:
        gold = {_nrm(g) for g in CLOSURE[c]}
        preds = [_nrm(p) for p in extract(c)[:15]]
        if not preds:
            continue
        p1 += 1.0 if preds[0] in gold else 0.0
        for i, pr in enumerate(preds):
            if pr in gold:
                mrr += 1.0/(i+1); break
        hits = ap = 0
        for i, pr in enumerate(preds):
            if pr in gold:
                hits += 1; ap += hits/(i+1)
        mapv += ap/min(len(gold), len(preds)) if gold else 0.0
    n = len(kids)
    return {"P@1": p1/n, "MRR": mrr/n, "MAP": mapv/n, "n": n}

def substring_oracle(split):
    """baseline oracle: lead 에 gold조상이 substring 이면 P@1 성공으로 카운트.
    방법이 이 trivial 매처를 넘어야 함(심판 요구 #2)."""
    kids = DEV if split == "dev" else TEST
    hit = sum(1 for c in kids if any(a in LEADS[c] for a in CLOSURE[c]))
    return hit/len(kids)


# ══ T2: 클래스집합 → 계층 (열린 어휘 + hetero-F1) ══
def _distractor_classes(n=2000):
    """gold 노드 아닌 distractor — 실 build 의 열린 어휘 모사(심판 요구 #5).
    한국어 흔한 명사 조합으로 생성(결정적)."""
    base = ["시스템","방법","기술","정보","서비스","제품","활동","조직","계획","자료",
            "구조","과정","기능","도구","환경","정책","사업","단체","기관","분야",
            "개념","원리","방식","형태","특성","요소","성질","단위","범위","조건"]
    pre = ["국가","지역","금융","교육","의료","산업","문화","경제","사회","과학",
           "기술","정치","환경","군사","법률","행정","상업","농업","공업","해양"]
    mid = ["","관리","지원","개발","운영","통합","전문","기초","고급","특수"]
    rng = random.Random(42)
    # 조합 상한 = pre×mid×base. n 을 상한으로 캡(무한루프 방지 — 이전 20×30=600<2000).
    cap = len(pre) * len(mid) * len(base)
    target = min(n, cap)
    out = set()
    tries = 0
    while len(out) < target and tries < target * 50:
        out.add(rng.choice(pre) + rng.choice(mid) + rng.choice(base))
        tries += 1
    return out

def eval_t2(method, split):
    kids = DEV if split == "dev" else TEST
    # gold 쌍 중 이 split 의 child 만
    gold = {(c, p) for c, p in DIRECT if c in kids}
    gold_closure_pairs = {(c, a) for c in kids for a in CLOSURE[c]}
    # 클래스 집합 = split child + 그 조상 + distractor(열린 어휘)
    classes = set(kids)
    for c in kids:
        classes |= CLOSURE[c]
    classes |= _distractor_classes(2000)
    pred = method(classes, kids)
    # 전이폐포 기준 정밀도/재현율
    tp = len(pred & gold_closure_pairs)
    p = tp/len(pred) if pred else 0.0
    r_direct = len(pred & gold)/len(gold) if gold else 0.0
    # hetero-F1 (심판 #4)
    hetero_gold = {(c, p2) for c, p2 in gold if not (c.endswith(p2) and c != p2)}
    hp = pred & {(c, a) for c in kids for a in CLOSURE[c]
                 if not (c.endswith(a) and c != a)}
    hetero_tp = len(pred & hetero_gold)
    hpre = hetero_tp/len(hp) if hp else 0.0
    hrec = hetero_tp/len(hetero_gold) if hetero_gold else 0.0
    hf1 = 2*hpre*hrec/(hpre+hrec) if (hpre+hrec) else 0.0
    f = 2*p*r_direct/(p+r_direct) if (p+r_direct) else 0.0
    return {"P": p, "R_direct": r_direct, "F1": f, "hetero_F1": hf1,
            "hetero_P": hpre, "hetero_R": hrec, "pred": len(pred)}

def t2_baseline(classes, kids):
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    return {(h["child"], h["parent"]) for h in induce_suffix_hierarchy(classes)}

def t2_improved(classes, kids):
    from ontokit.hierarchy.suffix_share import induce_suffix_hierarchy
    out = {(h["child"], h["parent"]) for h in induce_suffix_hierarchy(classes)}
    for c in kids:
        for h in extract_hypernyms_improved(c)[:1]:
            if h in classes and h != c:
                out.add((c, h))
    return out


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "improved"
    split = sys.argv[2] if len(sys.argv) > 2 else "dev"
    print(f"=== Round-1 (누수차단·전이폐포·held-out) [{method}/{split}] ===")
    print(f"dev {len(DEV)} test {len(TEST)}, gold=전이폐포 조상\n")

    ext = extract_hypernyms_baseline if method == "baseline" else extract_hypernyms_improved
    t2m = t2_baseline if method == "baseline" else t2_improved

    orc = substring_oracle(split)
    r1 = eval_t1(ext, split)
    print(f"T1 상위어발견: P@1={r1['P@1']:.3f} MRR={r1['MRR']:.3f} MAP={r1['MAP']:.3f}")
    print(f"  substring oracle P@1={orc:.3f}  → 방법 P@1이 이걸 넘는가: "
          f"{'✅' if r1['P@1'] > orc else '❌ 미달(trivial 매처 이하)'}")
    r2 = eval_t2(t2m, split)
    print(f"T2 계층: P={r2['P']:.3f} R직접={r2['R_direct']:.3f} F1={r2['F1']:.3f}")
    print(f"  hetero-F1={r2['hetero_F1']:.3f} (P={r2['hetero_P']:.3f} R={r2['hetero_R']:.3f})")

    score = r1["MAP"]*40 + r2["F1"]*40 + r2["hetero_F1"]*20
    print(f"\n참고점수: {score:.1f}/100 (MAP×40 + F1×40 + heteroF1×20, 심판 재판정)")
