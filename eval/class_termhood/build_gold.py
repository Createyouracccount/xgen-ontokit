"""termhood 외부 gold 빌드 — ACL RD-TEC 2.0 (valid term vs 비-term).

문제: LLM-free 빌더가 명사를 클래스 후보로 과생성한다. 유효 용어(한진택배·품질관리·
PCB)와 노이즈(PageNumber·ENDOFREPORT·PDF·Q4)를 common하게(도메인 블랙리스트 없이)
가르는 필터를 외부 gold 로 채점하기 위한 것.

gold = ACL RD-TEC 2.0 (CC BY 4.0, 상업 OK). 사람이 <term class="tech|other">로 마킹한
것 = 유효 용어(positive). 같은 문서의 명사구인데 term 미마킹 = 비-term(negative).
자체 합성 GT 금지(계층 합성 F1 0.96 → 외부 0.33 붕괴 실측) 원칙 준수.

⚠️ 언어: ACL RD-TEC 은 영어(전산언어학). shape 신호(대문자·확장자·숫자·camelcase)는
언어 독립적이라 영어 gold 로 원리 검증 유효. 한국어 파편(POS) 신호는 별도 검증 필요
(한국어는 우리말샘 보조 + 위키 독립 소스, 후속).

누수 차단: gold(사람 valid/invalid 판정) ⊥ 우리 신호(형태·통계). 필터는 라벨 안 봄.

실행: python3 build_gold.py  →  data/gold.json
  ACL_RD_TEC 경로는 ACL_RD_TEC env 또는 기본 /tmp/acl-rd-tec-2.0.
"""
import os, re, json, glob
from collections import Counter

ACL = os.getenv("ACL_RD_TEC", "/tmp/acl-rd-tec-2.0")
ANN = os.path.join(ACL, "distribution/annoitation_files/annotator1")
OUT = os.path.join(os.path.dirname(__file__), "data/gold.json")

_TERM = re.compile(r'<term class="(tech|other)">(.*?)</term>', re.S)
_TAG = re.compile(r'<[^>]+>')
# 영어 명사구 후보(negative 풀) — 대문자 시작 단어열 또는 소문자 명사열. 대략적
# 후보 생성(우리 필터가 걸러낼 대상). gold term 과 겹치면 positive 로 이동.
_CAND = re.compile(r'\b[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,3}\b')


def _clean(s):
    return re.sub(r'\s+', ' ', _TAG.sub('', s)).strip()


def build():
    positives = Counter()   # 유효 용어 (term 마킹) → df
    all_cands = Counter()   # 전체 명사구 후보 → df
    files = sorted(glob.glob(os.path.join(ANN, "*/*.xml")))
    if not files:
        raise SystemExit(f"gold 없음: {ANN} — ACL_RD_TEC env 확인")
    for fp in files:
        raw = open(fp, encoding="utf-8", errors="ignore").read()
        # positive: 이 문서의 마킹된 term (문서당 set = df 1 기여)
        terms = {_clean(m.group(2)).lower() for m in _TERM.finditer(raw)}
        terms = {t for t in terms if t}
        for t in terms:
            positives[t] += 1
        # 후보 풀: term 태그 제거한 평문에서 명사구 추출
        plain = _clean(raw)
        cands = {m.group(0).strip().lower() for m in _CAND.finditer(plain)}
        for c in cands:
            all_cands[c] += 1

    # negative = 후보 중 positive 아닌 것. 단 너무 흔한 기능어(the, of)는 후보에서 이미
    # 명사구 규칙상 다수 걸러짐. positive 는 df 유지.
    pos_set = set(positives)
    negatives = {c: df for c, df in all_cands.items() if c not in pos_set}

    gold = {
        "source": "ACL RD-TEC 2.0 (CC BY 4.0)",
        "positives": positives.most_common(),   # [(term, df), ...] 유효 용어
        "negatives": Counter(negatives).most_common(),  # 비-term 후보
        "n_docs": len(files),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(gold, open(OUT, "w"), ensure_ascii=False)
    print(f"gold: positive(유효 용어) {len(positives)}, negative(비-term) {len(negatives)}, "
          f"문서 {len(files)} → {OUT}")
    # 샘플
    print("positive 샘플:", [t for t, _ in positives.most_common(8)])
    print("negative 샘플:", [t for t, _ in Counter(negatives).most_common(8)])


if __name__ == "__main__":
    build()
