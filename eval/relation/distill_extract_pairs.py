"""경로2 STEP1~3 — 새 원문 → 관계추출 후보 (subj,obj) 쌍 생성.

기존 ontokit 자산 재활용(재발명 금지):
  _split_sentences (Kiwi 문장분할) → KoElectraNER.entities (NER) → _pairs (문장스코프 쌍)
  → _mark (typed marker, sentence.find로 위치 자동계산 — teacher mark() 규약과 동일)

teacher 태깅은 다음 단계(distill_relabel_unlabeled.py 재활용, 입력만 이 산출물로 교체).
이 스크립트는 쌍 품질 육안검증용 — teacher 없이 순수 후보 생성.

산출: {sentence, subject_entity{word,type}, object_entity{word,type}, marked, src} jsonl
  — mark() 재적용 위해 word/type만 저장(start_idx는 teacher 단계서 find로 계산).

재현:
  EXTRACT_SRC=../../../eval_runs/typing/wiki2_3k.jsonl \
  EXTRACT_OUT=distill_pairs_wiki.jsonl EXTRACT_CAP=100 \
  /path/venv/bin/python distill_extract_pairs.py
env: EXTRACT_SRC(필수), EXTRACT_OUT(기본 distill_pairs.jsonl), EXTRACT_CAP(기본 0=전량),
     TEXT_KEY(기본 text), ONTOKIT_NER_MIN_SCORE(기본 0.40)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ontokit.extractors.relation_encoder_ko import (  # noqa: E402
    KoreanRelationEncoder, _mark, _split_sentences,
)
from ontokit.ner.koelectra import KoElectraNER  # noqa: E402
from ontokit.ner.span_align import align_spans  # noqa: E402

# _pairs 는 self 미사용 순수 로직(문장스코프 쌍생성, subj∈{PER,ORG} 제약).
# 인스턴스(teacher 로딩) 없이 언바운드로 호출 — 기존 로직 재발명 방지.
_pairs = KoreanRelationEncoder._pairs

# 입력품질 개선(소규모 육안검증 반영):
#  ① 줄바꿈 선분할 — 위키 목록/표가 한 "문장"으로 붙어 무관 엔티티가 쌍이 되는 오염 차단
#  ② 문장 길이 게이트 — 분할 실패 잔재(과장문)는 스킵
#  ③ align_spans — 조사포함·경계절단 수리(GLiNER 실측서 확인된 문제)
MAX_SENT_LEN = int(os.getenv("MAX_SENT_LEN") or "200")


def split_clean(text):
    """줄바꿈 선분할 후 Kiwi 문장분할, 과장문 스킵."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in _split_sentences(line):
            s = s.strip()
            if s and len(s) <= MAX_SENT_LEN:
                out.append(s)
    return out


def main():
    SRC = os.environ["EXTRACT_SRC"]
    OUT = os.getenv("EXTRACT_OUT", "distill_pairs.jsonl")
    CAP = int(os.getenv("EXTRACT_CAP") or "0")
    TEXT_KEY = os.getenv("TEXT_KEY", "text")

    ner = KoElectraNER()
    # align_spans 용 Kiwi — _split_sentences 가 만든 것 재사용(중복 생성 방지)
    from ontokit.extractors.relation_encoder_ko import _KIWI_HOLDER
    _split_sentences("")   # Kiwi 초기화 유발
    kiwi = _KIWI_HOLDER.get("kiwi")

    docs = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            txt = r.get(TEXT_KEY, "")
            if txt.strip():
                docs.append((r.get("doc_id") or r.get("id") or "?", txt))
            if CAP and len(docs) >= CAP:
                break

    print(f"src {SRC} docs {len(docs)}", flush=True)

    n_sent = n_ent = n_pair = n_marked = 0
    out_rows = []
    for doc_id, text in docs:
        sentences = split_clean(text)   # 줄바꿈 선분할 + 과장문 스킵
        n_sent += len(sentences)
        for sent in sentences:
            ents = ner.entities(sent, source_chunks=[doc_id])
            ents = align_spans(sent, ents, kiwi)   # 조사제거·경계수리
            n_ent += len(ents)
            pairs = _pairs(None, ents, [sent])   # 언바운드 호출(self 미사용)
            n_pair += len(pairs)
            for (sw, stype, ow, otype, s_cls, o_cls, s) in pairs:
                marked = _mark(s, sw, stype, ow, otype)
                if marked is None:   # 미출현·중첩 스킵
                    continue
                n_marked += 1
                out_rows.append({
                    "sentence": s,
                    "subject_entity": {"word": sw, "type": stype},
                    "object_entity": {"word": ow, "type": otype},
                    "marked": marked,
                    "src": doc_id,
                })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"문장 {n_sent} | 엔티티 {n_ent} | 후보쌍 {n_pair} | 마킹성공 {n_marked}", flush=True)
    print(f"saved → {OUT}", flush=True)
    print("EXTRACT-DONE", flush=True)


if __name__ == "__main__":
    main()
