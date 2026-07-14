# xgen-ontokit

[한국어](README.md) · **ENGLISH**

XGEN ontology **build & search improvement kit**. A library that bundles the ontology
improvements we author and inject into XGEN. Complementary to omnifuse (the search
library) — this side carries **build (LLM-free Korean·English extraction) + search
improvements** together.

## Language support matrix (v0.10, stated honestly)

| Axis | Korean | English | Mixed chunks |
|---|---|---|---|
| Class (compound nouns) | ✅ Kiwi | ✅ nltk POS (extras[english], **auto-wire**) | ✅ dual extraction (minority-language terms preserved) |
| Hierarchy (subClassOf) | ✅ character suffix-share | ✅ word suffix-share (case-insensitive) | ✅ |
| Entity (NER) | ✅ KoELECTRA (injected) | ✅ dslim BERT MIT (injected) | ⚠️ dominant language only (cost) |
| **Relation** | ✅ particle SVO (rules) + KLUE-RE encoder (opt-in) | ❌ **unsupported** — separate track once measurement infra exists | Korean only |
| OWL labels | `@ko` | `@en` (auto-detected) | mixed output |

⚠️ Quality-verification scope: Korean = measured on finreg 489, English = structural
tests only (corpus measurement incomplete).
Relation extraction: rule-based particle SVO (precision-oriented, KLUE linkage 0.8%) +
**KLUE-RE encoder channel (opt-in, external-gold judge loop 90/100, holdout micro-F1
0.5924)**. The encoder is enabled via env; when unset it falls back to rules — see
[Relation encoder](#relation-encoder-v012--klue-re-external-gold-judge-loop-90) below.
⚠️ Behavior change vs v0.5: `auto_english=True` is the default, so **English classes are
newly added to Korean corpora containing Latin acronyms** (pure-Hangul corpora produce
identical output — measured on finreg 489). Set `auto_english=False` to keep prior behavior.

## Philosophy
- **Zero core dependencies** — backends·models (kiwipiepy/transformers/httpx) are all extras.
- **Protocol injection** — XGEN injects only protocol implementations, no infra coupling.
  `Extractor`/`GraphStore`/`VectorStore`/`LLM`.
- **Single source** — improvements are managed in one library, not hardcoded inline in XGEN
  code. A/B via config switch.

## Install
```bash
pip install xgen-ontokit                 # core (zero dependencies)
pip install "xgen-ontokit[korean]"       # + Kiwi morphology
pip install "xgen-ontokit[ner]"          # + KoELECTRA NER
pip install "xgen-ontokit[relation-encoder]"  # + KLUE-RE relation encoder (opt-in)
pip install "xgen-ontokit[all]"          # everything
# Direct from GitHub:
pip install "git+https://github.com/<org>/xgen-ontokit.git"
```

## Build — LLM-free Korean·English extraction
```python
from ontokit import DeterministicKoreanExtractor

ext = DeterministicKoreanExtractor(domain_words=["여신전문금융업", "보험업"])
concepts, entities, relations, data = await ext.extract(documents)
# documents = {"filename": [{"chunk_id","chunk_text","chunk_index"}, ...]}
# concepts.class_hierarchy holds the subClassOf hierarchy (suffix-share), with source_chunks tagging
```
finreg 489 measured: **4.5s / $0** (vs gpt-4o 23min/$2), classes 3156·subClassOf 1710.
Search A/B: identical Recall@10 (0.947) to a gpt-4o build. ⚠️ But this metric is **carried
by the vector leg (embedding + FTS)**, so it *cannot measure the difference between build
methods* (LLM vs LLM-free indistinguishable). Hierarchy·relation quality must be measured
separately by hierarchy counts / full enumeration / relation GT (see roadmap).

## Citation ontology (v0.8) — doc-level `:cites`
```python
from ontokit.citations import CitationCollector, citations_insert_update, doc_uri

col = CitationCollector()            # streaming — chunk-boundary carry (TAIL_CARRY)
col.feed(file_name, chunk_text)      # 「law name」 Article N pattern, same-law/this-law masking
sparql = citations_insert_update(col.edges(), graph_uri + "__cites")
```
Emits cross-citations among statutes as document-level `:cites` edges → XGEN multi_turn_rag
5th leg (UNION SPARQL, minimal-displacement insert). mixed measured: full multihop recovery
dev 0.842/ho 0.700/te 0.767.
Basis: `docs/그래프sources_인용온톨로지_연결_PoC실증_2026_07_12.md`.

## Class-promotion filter (v0.9) — LLM-free over-generation cleanup
```python
from ontokit.filter import ClassPromotionFilter

f = ClassPromotionFilter(corpus_chunks=n_chunks)  # support gate auto-disabled if unknown(None) or small(<5000)
keep, reason = f.decide(label, df=df, has_rel=..., has_kid=..., has_inst=...)
```
Promotion criterion (termhood): promote to class only on reuse (df≥2) or structural
participation (relation·hierarchy-parent·instance). Junk rules use statistics + closed-class
grammatical function words only (no domain blacklist). mixed20k measured: 444,817→70,671
(-84.1%), relation triples 100% preserved. ⚠️ Also removes isolated df1 valid concepts
(intended cost; XGEN wiring is reversible via sidecar `<graph>__filtered`).
Basis: `docs/클래스승격필터_과생성해소_2026_07_12.md`.

## Co-occurrence weak relation (v0.10) — LLM-free relation-density boost (language-agnostic)
```python
from ontokit.cooccurrence import CooccurrenceCollector, make_korean_label_ok

col = CooccurrenceCollector(min_pair_df=3, lift_k=2.0, label_ok=make_korean_label_ok())
col.add_chunk(chunk_id, [(uri, label), ...])   # chunk streaming
edges = col.edges(exclude_pairs=svo_pairs)      # [(a, b, count)] — excludes SVO-linked pairs
```
Emits same-chunk co-occurring entity pairs as `coOccursWith` (mentioned-together) weak
relations — deterministically boosting relation density / English coverage that SVO
(Korean-only) can't fill. Selection is statistics only (pair df≥3 ∧ lift>2, zero lists);
label eligibility is by morphology·POS (calendar·punctuation·numeric-token·Latin-micro·
mixed-case fragment·particle-terminal·standalone-bound-noun·collision sink). mixed20k
measured: 1.75%→10.5%, SVO 100% preserved, displayed junk rate ~18%.
⚠️ Coarse relation (no type); consumers should prefer SVO + co-occ fallback slot.
Truncated·merged fragments (`대구광역`) are beyond morphological detection (upstream NER).
Basis: `docs/관계밀도_coOccursWith_확충_2026_07_12.md`.

## Relation encoder (v0.12) — KLUE-RE external-gold judge loop 90
```bash
pip install "xgen-ontokit[relation-encoder]"          # + transformers·torch
export ONTOKIT_RELATION_ENCODER_MODEL=/path/to/model_re   # this env is the on/off switch
```
A **local RE encoder** channel that surpasses rule-based particle SVO (KLUE linkage 0.8%).
Fine-tuned klue/roberta-small, **zero LLM API calls** (local inference, same family as NER).
External gold (KLUE-RE, CC BY-SA) judge loop 90/100, holdout micro-F1 0.5924 (consistent
with the official baseline 60.89).

- **Invariant**: if any of env-unset·extras-not-installed·bad-path·no-NER holds, it falls
  back to rule-based particle SVO. "It does not turn on unless installed·configured." The
  NER (KoELECTRA) supplies entities, which are paired → relation classification.
- **Model swap**: not tied to a specific model. ① change env ② retrain via
  `eval/relation/train_encoder.py` ③ inject `relation_extractor=` ④ `.extract()` wrapper —
  see the "Model swap" section in `eval/relation/README.md`.
- Reproduce·evaluate: `eval/relation/` (train_encoder.py·eval_encoder.py·gold). Weights are
  reproduced by script (not committed to git).
  Basis: `docs/ontokit_관계_KLUE-RE_인코더_심판루프_90_2026_07_14.md`.

## Search improvements
```python
from ontokit.search import class_instances_triple, blend_score
# #1 subClassOf* transitive closure — full enumeration of subclass instances (zero regression)
# #2 vscore missing-floor guard — restores keyword exact-match chunk ranking
```

## XGEN injection (1 edit site)
```python
# service/ontology/pipeline.py
# before:  self.doc_extractor = DocumentOntologyExtractor(self.llm)
# after:   self.doc_extractor = ExtractorFactory.create(config, llm=self.llm)
#          (config ONTOLOGY_EXTRACTOR=deterministic_ko → LLM-free switch)
```
`ExtractorFactory` replicates XGEN's existing `RerankerFactory` pattern
(PROVIDER_NAMES + importlib + config).

## Structure
```
src/ontokit/
├── protocols.py          # injection interfaces (Extractor/GraphStore/VectorStore/LLM)
├── extractors/           # deterministic_ko (core, ko·en dual extraction) + base (merge_concepts)
├── morphology/           # kiwi_nouns (Korean) + en_nouns (English nltk POS)
├── hierarchy/            # suffix_share (main engine, ko=char/en=word), hearst_ko (extension)
├── ner/                  # koelectra (ko) + english (dslim BERT MIT)
├── citations.py          # doc-level :cites citation collection·SPARQL emit (v0.8)
├── filter/               # class_promotion — termhood promotion gate (v0.9)
├── cooccurrence.py       # coOccursWith co-occurrence weak relation — density boost (v0.10)
└── search/               # improvements (subClassOf*, floor guard) — ⚠️XGEN-specific
```

## Basis
Measured: `docs/LLM-free_추출기_프로토타입_실측_2026_07_08.md`,
`docs/온톨로지검색_synaptic_vs_XGEN_실측종합_2026_07_07.md` (xgen-levelup/docs).

## XGEN deployment coupling (public repo)

The repo is public, so it installs **without authentication**:

```bash
pip install "git+https://github.com/Createyouracccount/xgen-ontokit.git"
# Pin the version (recommended): ...xgen-ontokit.git@v0.10.0
```

Add the URL above to XGEN's `pyproject.toml` dependencies or requirements.
Verified: in-container git clone → pip install → import·extract E2E works without auth.
