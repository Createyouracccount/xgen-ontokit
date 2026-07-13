"""의존파스 관계 PoC — 조사SVO 규칙(현 ontokit) vs stanza 의존파스.

실증 질문: 현 조사SVO 규칙이 구조적으로 못 잡는 관계 —
  ①연결어미절("A를 감독하고 B를 규제한다"의 두 번째 절)
  ②관형절("은행을 감독하는 금융위원회")
  ③수동/계사
을 stanza 의존파스가 실제로 잡는가? 같은 문장셋으로 통제 비교.

의존파스 → SVO: nsubj(주어)·obj(목적어)·의존관계로 논항을 트리에서 읽음.
Kiwi 는 의존파싱 불가(LGPL, 형태소만) → 규칙은 인접 조사만 보고 추측.
"""
from __future__ import annotations
import re
import sys
sys.path.insert(0, "/Users/kimdu/company/xgen-levelup/xgen-ontokit/src")

# ── 테스트 문장 + GT (사람 판정) ──
# 각 문장의 참 관계쌍(subject, object) — predicate 는 완화매칭
SENTENCES = [
    # ① 연결어미절 — 규칙은 종결절만, 첫 절 관계 놓침
    ("금융위원회는 은행을 감독하고 보험사를 규제한다",
     [("금융위원회", "은행"), ("금융위원회", "보험사")]),
    # ② 관형절 — 규칙은 관형절 억제
    ("예금을 보호하는 예금보험공사가 설립되었다",
     [("예금보험공사", "예금")]),
    # ③ 다중 연결
    ("회사는 자료를 제출하며 내용을 보고하고 결과를 공개한다",
     [("회사", "자료"), ("회사", "내용"), ("회사", "결과")]),
    # ④ 단순 종결(규칙도 잡아야 — 대조군)
    ("정부는 정책을 발표했다",
     [("정부", "정책")]),
    # ⑤ 주어 공유 연결
    ("이사회가 안건을 심의하고 승인한다",
     [("이사회", "안건")]),
]

def norm(s):
    return s.replace(" ", "")

def match_pairs(pred, gold):
    """(subj,obj) 완화매칭 — 부분문자열 허용(명사구 경계차 흡수)."""
    g = set()
    for ps, po in pred:
        for gs, go in gold:
            if (gs in ps or ps in gs) and (go in po or po in go):
                g.add((gs, go)); break
    tp = len(g)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f


# ── 방법 1: 현 ontokit 조사SVO 규칙 (실코드) ──
def rule_extract(sent):
    from ontokit.extractors.relation_ko import KoreanRelationExtractor
    ext = KoreanRelationExtractor()
    rels = ext.extract(sent, source_chunks=["c1"])
    return [(r["subject"], r["object"]) for r in rels]


# ── 방법 2: stanza 의존파스 → SVO ──
_nlp = None
def dep_extract(sent):
    global _nlp
    if _nlp is None:
        import stanza
        _nlp = stanza.Pipeline("ko", processors="tokenize,pos,lemma,depparse",
                               verbose=False, download_method=None)
    doc = _nlp(sent)
    out = []
    for s in doc.sentences:
        # word.head(1-idx), word.deprel. 동사(root/conj)마다 nsubj·obj 수집.
        words = s.words
        by_head = {}
        for w in words:
            by_head.setdefault(w.head, []).append(w)
        # 술어 후보 = VERB (root 또는 conj 로 연결된 병렬동사)
        verbs = [w for w in words if w.upos in ("VERB", "ADJ")]
        # 주어는 종종 최상위 동사에만 달리고 conj 동사엔 생략 → 공유주어 전파
        # 각 동사의 subtree 에서 nsubj/obj 를 찾고, 없으면 head 체인 따라 주어 상속
        # 한국어 stanza: 주제 은/는 주어를 nsubj 아닌 dislocated 로 태깅(실측).
        # nsubj/nsubj:pass/dislocated 모두 주어 후보로. 공유주어는 head 체인 상속.
        SUBJ_REL = ("nsubj", "nsubj:pass", "dislocated", "csubj")
        def find_subj(vid):
            seen = set()
            cur = vid
            while cur and cur not in seen:
                seen.add(cur)
                for w in by_head.get(cur, []):
                    if w.deprel in SUBJ_REL:
                        return w.text
                cw = words[cur-1] if 1 <= cur <= len(words) else None
                cur = cw.head if cw else 0
            return None
        for v in verbs:
            subj = find_subj(v.id)
            objs = [w.text for w in by_head.get(v.id, [])
                    if w.deprel in ("obj", "iobj")]
            for o in objs:
                if subj:
                    out.append((subj, o))
    # 조사 흔적 제거 — stanza 어절 토큰은 "예금보험공사가"처럼 조사 포함.
    # 간단 정규화: 끝 조사 1음절 제거(은/는/이/가/을/를/에/의)
    def strip_josa(w):
        return re.sub(r"(?:은|는|이|가|을|를|에|의|와|과|로|도)$", "", w)
    return [(strip_josa(s), strip_josa(o)) for s, o in out]


if __name__ == "__main__":
    print("의존파스 로딩 중...", flush=True)
    # 워밍업
    dep_extract("정부는 정책을 발표했다")
    print("로딩 완료\n")

    tot_rule = [0.0, 0.0, 0.0]
    tot_dep = [0.0, 0.0, 0.0]
    for sent, gold in SENTENCES:
        rp = rule_extract(sent)
        dp = dep_extract(sent)
        rprf = match_pairs(rp, gold)
        dprf = match_pairs(dp, gold)
        for i in range(3):
            tot_rule[i] += rprf[i]; tot_dep[i] += dprf[i]
        print(f"문장: {sent}")
        print(f"  GT: {gold}")
        print(f"  [규칙]   {[(s,o) for s,o in rp]}  R={rprf[1]:.2f}")
        print(f"  [의존파스] {[(s,o) for s,o in dp]}  R={dprf[1]:.2f}")
        print()

    n = len(SENTENCES)
    print("=== 평균 ===")
    print(f"[조사SVO 규칙]  P={tot_rule[0]/n:.3f} R={tot_rule[1]/n:.3f} F1={tot_rule[2]/n:.3f}")
    print(f"[stanza 의존파스] P={tot_dep[0]/n:.3f} R={tot_dep[1]/n:.3f} F1={tot_dep[2]/n:.3f}")
    print("\n=== 판정 ===")
    if tot_dep[1] > tot_rule[1]:
        print(f"✅ 의존파스가 규칙이 놓친 관계(연결어미절·관형절)를 복원 "
              f"(recall +{(tot_dep[1]-tot_rule[1])/n:.3f})")
    else:
        print("❌ 의존파스 이득 미실증")
