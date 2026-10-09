"""README 그림 생성 — 측정 원자료(harness/results*)에서 다시 만든다. 손으로 그린 숫자 없음.

  python docs/img/make_figures.py      # 리포 루트에서 → docs/img/*-{light,dark}.svg

팔레트: dataviz 기준 팔레트 1·2번 슬롯(파랑·주황), 밝은/어두운 모드 각각 검증 통과(validate_palette.js).
GitHub README 는 <picture> 로 모드별 그림을 고른다. 각 그림 아래 README 에 같은 값의 표가 있다(표 보기).
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from harness.stats import STRUCT, boot_ci  # noqa: E402

OUT = "docs/img"
THEME = {
    "light": dict(bg="#fcfcfb", fg="#0b0b0b", fg2="#52514e", grid="#e4e3df", s1="#2a78d6", s2="#eb6834",
                  box="#f3f2ef", boxline="#c9c8c2", accent="#2a78d6"),
    "dark": dict(bg="#1a1a19", fg="#ffffff", fg2="#c3c2b7", grid="#34342f", s1="#3987e5", s2="#d95926",
                 box="#262624", boxline="#4a4a45", accent="#3987e5"),
}
FONT = "-apple-system, 'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif"


def _svg(w, h, t, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}"><rect width="{w}" height="{h}" rx="10" fill="{t["bg"]}"/>{body}</svg>')


def _txt(x, y, s, fill, size=13, anchor="start", weight="400"):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def load(path):
    return json.load(open(path))


def delta(res, arm, forms=STRUCT):
    d = [r["arms"][arm]["score"] - r["arms"]["VLLM"]["score"] for r in res["rows"] if r["form"] in forms]
    lo, hi = boot_ci(d)
    return sum(d) / len(d), lo, hi


# ── 그림 1: 활용 경로(빌드 → 저장 → 질의 → 병합) ──
def fig_pipeline(t):
    W, H = 900, 330
    b = []

    def box(x, y, w, h, title, sub, strong=False):
        stroke = t["accent"] if strong else t["boxline"]
        b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{t["box"]}" stroke="{stroke}" '
                 f'stroke-width="{2 if strong else 1}"/>')
        b.append(_txt(x + w / 2, y + 24, title, t["fg"], 14, "middle", "600"))
        for i, line in enumerate(sub):
            b.append(_txt(x + w / 2, y + 44 + i * 17, line, t["fg2"], 12, "middle"))

    def arrow(x1, y1, x2, y2):
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["fg2"]}" stroke-width="1.5" '
                 f'marker-end="url(#ah)"/>')
    b.append(f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{t["fg2"]}"/></marker></defs>')
    b.append(_txt(24, 32, "빌드 (문서가 들어올 때, LLM 0회)", t["fg2"], 12, "start", "600"))
    box(24, 44, 150, 74, "문서", ["파싱·청킹된 텍스트"])
    box(214, 44, 220, 74, "ontokit", ["개체·타입·관계·계층 추출", "로컬 모델(NER·관계)"], True)
    box(474, 44, 190, 74, "그래프 저장소", ["graphstore", "Fuseki(RDF) · Neo4j(LPG)"])
    arrow(174, 81, 212, 81)
    arrow(434, 81, 472, 81)
    b.append(_txt(24, 160, "질의 (질문이 들어올 때)", t["fg2"], 12, "start", "600"))
    box(24, 172, 150, 74, "질문", ["\"X 에 해당하는 것", "모두 나열해줘\""])
    box(214, 172, 220, 74, "계획기 → 컴파일러", ["질의계획(JSON) →", "SPARQL / Cypher"])
    box(474, 172, 190, 74, "그래프 질의 결과", ["하위 분류 폐포·역관계", "조건에 맞는 항목 목록"])
    box(704, 172, 172, 74, "답 병합", ["판독기 답 ∪ 그래프 결과", "(개수는 그래프)"], True)
    arrow(174, 209, 212, 209)
    arrow(434, 209, 472, 209)
    arrow(664, 209, 702, 209)
    arrow(569, 118, 569, 170)
    box(214, 266, 450, 48, "벡터 검색(상위 40청크) → 판독기(LLM)가 답", [])
    b.append(f'<path d="M 664 290 L 790 290 L 790 248" fill="none" stroke="{t["fg2"]}" stroke-width="1.5" '
             f'marker-end="url(#ah)"/>')
    arrow(99, 246, 99, 290)
    b.append(f'<line x1="99" y1="290" x2="212" y2="290" stroke="{t["fg2"]}" stroke-width="1.5" marker-end="url(#ah)"/>')
    return _svg(W, H, t, "".join(b))


# ── 그림 2: 효과 크기(벡터+LLM 대비 Δ, 95% CI) ──
def fig_effect(t, data):
    W, H = 900, 300
    x0, x1, xmin, xmax = 300, 860, -0.1, 0.8

    def X(v):
        return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)
    b = [_txt(24, 30, "벡터+LLM 대비 실제 답 점수 변화 (구조형 문항, 점 = 평균, 막대 = 95% 신뢰구간)", t["fg"], 14, "start", "600")]
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        b.append(f'<line x1="{X(v)}" y1="56" x2="{X(v)}" y2="{H - 46}" stroke="{t["grid"] if v else t["fg2"]}" '
                 f'stroke-width="{1 if v else 1.5}"/>')
        b.append(_txt(X(v), H - 28, f"{v:+.1f}" if v else "0 (효과 없음)", t["fg2"], 12, "middle"))
    rows = [("그래프 결과 병합", "ontokit, LLM 없이 구축", "MRG:ontokit_r5", "main"),
            ("정답 그래프 (상한)", "추출이 완벽했다면", "MRG:oracle", "ctrl"),
            ("가짜 그래프 (음성 대조)", "사실을 모두 뒤바꿈", "MRG:placebo2", "ctrl")]
    y = 92
    for name, sub, arm, kind in rows:
        b.append(_txt(24, y + 4, name, t["fg"], 13, "start", "600"))
        b.append(_txt(24, y + 22, sub, t["fg2"], 12))
        for k, (ds, col) in enumerate((("EVAL", t["s1"]), ("HOLDOUT", t["s2"]))):
            m, lo, hi = data[(ds, kind, arm)]
            yy = y - 6 + k * 18
            b.append(f'<line x1="{X(lo)}" y1="{yy}" x2="{X(hi)}" y2="{yy}" stroke="{col}" stroke-width="2" '
                     f'stroke-linecap="round"/>')
            b.append(f'<circle cx="{X(m)}" cy="{yy}" r="5" fill="{col}" stroke="{t["bg"]}" stroke-width="2"/>')
            b.append(_txt(X(hi) + 8, yy + 4, f"{m:+.3f}", t["fg2"], 12))
        y += 62
    lx = x0
    for ds, col, lab in (("EVAL", t["s1"], "EVAL — 개발에 쓴 3,000문서"), ("HOLDOUT", t["s2"], "HOLDOUT — 처음 보는 3,000문서")):
        b.append(f'<circle cx="{lx}" cy="46" r="5" fill="{col}"/>')
        b.append(_txt(lx + 10, 50, lab, t["fg2"], 12))
        lx += 260
    return _svg(W, H, t, "".join(b))


# ── 그림 3: 형태별(HOLDOUT) — 어디서 돕고 어디서 못 돕나 ──
def fig_forms(t, res):
    W, H = 900, 330
    forms = [("E1", "타입 열거"), ("E2", "상위 분류 열거"), ("R", "역관계 열거"), ("M", "다중 홉"),
             ("A", "개수 집계"), ("C", "교차 조건"), ("L", "단순 조회")]
    sc = {(f, a): sum(r["arms"][a]["score"] for r in res["rows"] if r["form"] == f) /
          max(1, sum(1 for r in res["rows"] if r["form"] == f))
          for f, _ in forms for a in ("VLLM", "MRG:ontokit_r5")}
    top, base, x0, gw = 70, H - 70, 70, 112
    ymax = 0.7

    def Y(v):
        return base - v / ymax * (base - top)
    b = [_txt(24, 30, "처음 보는 문서(HOLDOUT)에서 형태별 실제 답 점수 (1.0 만점)", t["fg"], 14, "start", "600")]
    for v in (0, 0.2, 0.4, 0.6):
        b.append(f'<line x1="{x0 - 10}" y1="{Y(v)}" x2="{W - 24}" y2="{Y(v)}" stroke="{t["grid"]}" stroke-width="1"/>')
        b.append(_txt(x0 - 16, Y(v) + 4, f"{v:.1f}", t["fg2"], 12, "end"))
    bw = 30
    for i, (f, name) in enumerate(forms):
        gx = x0 + 20 + i * gw
        for k, (a, col) in enumerate((("VLLM", t["s2"]), ("MRG:ontokit_r5", t["s1"]))):
            v = sc[(f, a)]
            x = gx + k * (bw + 2)
            h = max(0.0, base - Y(v))
            r = min(4, h)
            b.append(f'<path d="M{x},{base} L{x},{base - h + r} Q{x},{base - h} {x + r},{base - h} '
                     f'L{x + bw - r},{base - h} Q{x + bw},{base - h} {x + bw},{base - h + r} L{x + bw},{base} Z" fill="{col}"/>')
            b.append(_txt(x + bw / 2, base - h - 6, f"{v:.2f}", t["fg2"], 11, "middle"))
        b.append(_txt(gx + bw, base + 18, f, t["fg"], 12, "middle", "600"))
        b.append(_txt(gx + bw, base + 34, name, t["fg2"], 11, "middle"))
    lx = W - 360
    for col, lab in ((t["s2"], "벡터 검색 + LLM"), (t["s1"], "+ ontokit 그래프 결과 병합")):
        b.append(f'<rect x="{lx}" y="40" width="12" height="12" rx="2" fill="{col}"/>')
        b.append(_txt(lx + 18, 51, lab, t["fg2"], 12))
        lx += 150
    return _svg(W, H, t, "".join(b))


# ── 그림 4: 개선 과정(그래프 단독, 1~6차) ──
def fig_rounds(t):
    W, H = 900, 300
    pts = [("1차", "기준선", 0.018), ("2차", "영주어 복원", 0.048), ("3차", "정의문 채널", 0.096),
           ("4차", "위치 관계", 0.113), ("5차", "기관명 보충", 0.128), ("6차", "+로컬 LLM 추출", 0.302)]
    top, base, x0, x1 = 60, H - 70, 90, W - 60
    ymax = 0.35

    def Y(v):
        return base - v / ymax * (base - top)

    def X(i):
        return x0 + i * (x1 - x0) / (len(pts) - 1)
    b = [_txt(24, 30, "그래프만으로 답할 때의 점수 — 차수별 개선 (구조형, 1.0 만점)", t["fg"], 14, "start", "600")]
    for v in (0, 0.1, 0.2, 0.3):
        b.append(f'<line x1="{x0 - 10}" y1="{Y(v)}" x2="{x1 + 10}" y2="{Y(v)}" stroke="{t["grid"]}"/>')
        b.append(_txt(x0 - 16, Y(v) + 4, f"{v:.1f}", t["fg2"], 12, "end"))
    b.append(f'<line x1="{x0 - 10}" y1="{Y(0.162)}" x2="{x1 + 10}" y2="{Y(0.162)}" stroke="{t["s2"]}" '
             f'stroke-width="2" stroke-dasharray="6 4"/>')
    b.append(_txt(x0, Y(0.162) - 8, "벡터 검색 상한 0.162 (판독기가 완벽하다고 가정)", t["fg2"], 12))
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(i)},{Y(v)}" for i, (_, _, v) in enumerate(pts[:5]))
    b.append(f'<path d="{d}" fill="none" stroke="{t["s1"]}" stroke-width="2"/>')
    b.append(f'<line x1="{X(4)}" y1="{Y(0.128)}" x2="{X(5)}" y2="{Y(0.302)}" stroke="{t["s1"]}" stroke-width="2" '
             f'stroke-dasharray="3 4"/>')
    for i, (r, name, v) in enumerate(pts):
        b.append(f'<circle cx="{X(i)}" cy="{Y(v)}" r="5" fill="{t["s1"]}" stroke="{t["bg"]}" stroke-width="2"/>')
        near = abs(v - 0.162) < 0.05          # 기준선과 겹치면 점 아래에 쓴다
        b.append(_txt(X(i), Y(v) + 20 if near else Y(v) - 12, f"{v:.3f}", t["fg2"], 12, "middle"))
        b.append(_txt(X(i), base + 20, r, t["fg"], 12, "middle", "600"))
        b.append(_txt(X(i), base + 36, name, t["fg2"], 11, "middle"))
    return _svg(W, H, t, "".join(b))


def main():
    os.makedirs(OUT, exist_ok=True)
    ev, evc = load("harness/results/L2_eval_merge.json"), load("harness/results/L2_eval_ctrl_merge.json")
    ho, hoc = load("harness/results_sealed/H2_merge.json"), load("harness/results_sealed/H2_ctrl_merge.json")
    data = {}
    for ds, m_, c_ in (("EVAL", ev, evc), ("HOLDOUT", ho, hoc)):
        data[(ds, "main", "MRG:ontokit_r5")] = delta(m_, "MRG:ontokit_r5")
        for arm in ("MRG:oracle", "MRG:placebo2"):
            data[(ds, "ctrl", arm)] = delta(c_, arm)
    for mode, t in THEME.items():
        for name, svg in (("pipeline", fig_pipeline(t)), ("effect", fig_effect(t, data)),
                          ("forms", fig_forms(t, ho)), ("rounds", fig_rounds(t))):
            open(os.path.join(OUT, f"{name}-{mode}.svg"), "w").write(svg)
    for k, v in sorted(data.items()):
        print(k, tuple(round(x, 3) for x in v))


if __name__ == "__main__":
    main()
