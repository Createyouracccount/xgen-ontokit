# -*- coding: utf-8 -*-
"""경계 위생 채점기 — 심판 없이 기계적으로 잴 수 있는 지표.

off-morpheme: 스팬의 start 또는 end 가 Kiwi 형태소 경계 집합에 **없는** 비율.
  = 형태소 내부를 갈라버린 절단 = **사후 트림으로 복구 불가능한 오류**.
  0822 실측(위키 코퍼스): simple 11.26% → first 2.91%, 트림 적용해도 11.26→11.30%.

josa_tail: 경계는 정합하지만 끝이 조사/어미 = **사후 규칙이 고칠 수 있는 오류**.
  simple 1.25% → first 44.05%. 즉 first 는 절단을 조사과다로 **교환**한다.

word_mismatch: entity["word"] != text[start:end]. HF 가 서브워드 재조립 시 공백 삽입.
  0822 뉴스 코퍼스 실측 3.36%(`140 %`·`4 · 5홀`·`newsis. com`).

이 셋은 골드가 없어도 산출되므로 G3(대응설계·셋분리·벤더다양화) 미충족 상태에서도
**기계적 사실**로 보고할 수 있다. 단 정밀도 주장은 별개다 — 여기서 정밀도를 말하지 않는다.
"""
from __future__ import annotations
import json

TAIL_TAGS = frozenset(
    "JKS JKC JKG JKO JKB JKV JKQ JX JC EF EC ETM ETN EP VCP XSV XSA".split())


def morph_bounds(text: str, kiwi) -> set[int]:
    """텍스트의 형태소 경계 오프셋 집합(시작·끝 전부)."""
    b = {0, len(text)}
    for t in kiwi.tokenize(text):
        b.add(t.start)
        b.add(t.start + t.len)
    return b


def score(spans, kiwi) -> dict:
    """spans: [(text, start, end, word)] — word 는 없으면 None.

    반환: 건수·비율 전부. 분모를 반드시 함께 낸다(판례 16).
    """
    n = off = tail = mism = 0
    off_ex, tail_ex, mism_ex = [], [], []
    for text, st, en, word in spans:
        if st is None or en is None or not (0 <= st < en <= len(text)):
            continue
        n += 1
        surf = text[st:en]
        bounds = morph_bounds(text, kiwi)
        if st not in bounds or en not in bounds:
            off += 1
            if len(off_ex) < 25:
                off_ex.append(surf)
        else:
            toks = kiwi.tokenize(surf)
            if toks and toks[-1].tag in TAIL_TAGS:
                tail += 1
                if len(tail_ex) < 25:
                    tail_ex.append(surf)
        if word is not None and word.replace("##", "").strip() != surf:
            mism += 1
            if len(mism_ex) < 25:
                mism_ex.append((word, surf))
    pct = lambda k: round(k / n * 100, 2) if n else 0.0
    return {
        "n": n,
        "off_morpheme": {"k": off, "pct": pct(off), "예": off_ex},
        "josa_tail": {"k": tail, "pct": pct(tail), "예": tail_ex},
        "word_mismatch": {"k": mism, "pct": pct(mism), "예": mism_ex},
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True, help="JSON: [[text,start,end,word],…]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from kiwipiepy import Kiwi
    spans = [tuple(r) for r in json.load(open(a.spans, encoding="utf-8"))]
    res = score(spans, Kiwi())
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: (v if k == "n" else {kk: vv for kk, vv in v.items() if kk != "예"})
                      for k, v in res.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
