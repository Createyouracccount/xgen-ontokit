# xgen-ontokit

[한국어](README.md) · **ENGLISH**

A library that extracts an ontology (classes, hierarchy, entities, relations) from Korean
documents **without an LLM**.

**What it does** — feed it document chunks and it emits concepts (classes), a `subClassOf`
hierarchy, entities, and relation triples. Extraction runs on morphological analysis (Kiwi),
rules, statistics, and local encoders only — it **makes no LLM API calls on the default path**.
(There is one LLM top-up channel, but it only activates if you inject it explicitly, and with
a zero budget it behaves identically to pure rules.)

**Why LLM-free** — for cases that need these three things:
- **Cost**: ~$0 per document. (For reference, LLM extraction pipelines run in the tens of
  dollars per 1M tokens.)
- **Determinism·reproducibility**: same input → same output. Auditable, roll-back-able, A/B-able.
- **Data control**: all inference is local. Documents never leave for an external API.

**The limits are real too** — open-domain relation recall and implicit/heterogeneous hypernym
inference are better served by LLM extraction. This library trades that gap for the three
properties above. See [Quality evidence](#quality-evidence--only-what-you-can-reproduce)
for details and for what falls short.

```python
from ontokit import DeterministicKoreanExtractor

documents = {"file.pdf": [{"chunk_id": "c1", "chunk_text": "…", "chunk_index": 0}]}

ext = DeterministicKoreanExtractor()          # no args = 0 LLM calls, 0 model loads
concepts, entities, relations, data = await ext.extract(documents)
```
(Entity extraction requires injecting a NER — see [Install](#install) and the
[defaults table](#defaults--env-switches-at-a-glance).)

## Language support matrix (v0.13.1, stated honestly)

| Axis | Korean | English | Mixed chunks |
|---|---|---|---|
| Class (compound nouns) | ✅ Kiwi | ✅ nltk POS (extras[english], **auto-wire**) | ✅ dual extraction (minority-language terms preserved) |
| Hierarchy (subClassOf) | ✅ character suffix-share + **definitional (Hearst, on by default)** | ✅ word suffix-share (case-insensitive) | ✅ |
| Entity (NER) | ✅ KoELECTRA (injected) | ✅ dslim BERT MIT (injected or `ONTOKIT_NER_EN=auto`) | ⚠️ dominant language only (cost) |
| **Relation** | ✅ particle SVO (rules) + KLUE-RE encoder (opt-in) | ✅ **spaCy dependency SVO (opt-in, v0.13)** | Korean only |
| Instance typing | ✅ definitional + **occupation P106 lexicon (on by default)** | ❌ unsupported | Korean only |
| OWL labels | `@ko` | `@en` (auto-detected) | mixed output |

⚠️ Quality-verification scope: Korean = measured on finreg 489, English = structural
tests only (corpus measurement incomplete).
Relation extraction: the rule-based particle SVO channel is now **availability-fallback only**
(ensemble permanently rejected, B3) + **KLUE-RE encoder channel (opt-in, holdout micro-F1
0.6274 — external gold KLUE-RE)**. The encoder is enabled via env; when unset it falls back
to rules — see [Relation encoder](#relation-encoder-v013--klue-re--sredfm-ko-augmented-holdout-06274) below.
⚠️ Behavior change vs v0.5: `auto_english=True` is the default, so **English classes are
newly added to Korean corpora containing Latin acronyms** (pure-Hangul corpora produce
identical output — measured on finreg 489). Set `auto_english=False` to keep prior behavior.
⚠️ Behavior change vs v0.11: `enable_hearst=True` (definitional heterogeneous hierarchy) and
`enable_occupation=True` (occupation typing) are **on by default** — output differs from pure
suffix-share. Set each to `False` to revert.

### Defaults / env switches at a glance

What a no-arg `DeterministicKoreanExtractor()` turns on, vs what only env enables:

| Channel | Default | Switch | Loads a model? |
|---|---|---|---|
| Korean classes·suffix-share hierarchy | **on** | — | no (Kiwi) |
| English classes | **on** (if nltk installed) | `auto_english=False` | no (nltk POS) |
| Definitional hierarchy·typing (Hearst) | **on** | `enable_hearst=False` | no (rules) |
| Occupation typing (P106) | **on** | `enable_occupation=False` / `ONTOKIT_OCCUPATION_TYPING=off` | no (bundled lexicon) |
| Korean relations (particle SVO) | **on** | `enable_relations=False` | no (Kiwi) |
| Relation encoder (KLUE-RE) | off | `ONTOKIT_RELATION_ENCODER_MODEL` | transformers (local) |
| English NER | off | `ONTOKIT_NER_EN=auto` | transformers (local) |
| English relations (spaCy) | off | `ONTOKIT_RELATION_EN=auto` | spaCy |
| Auxiliary NER union | off | `ONTOKIT_NER_AUX_MODEL` | transformers (local) |
| Dictionary synonym merge | off | `ONTOKIT_SYNONYM_DICT` | no (TSV) |

**Invariant**: the default path makes **0 LLM calls and 0 transformers loads**. Every
model-backed channel is env opt-in, and the one new on-by-default channel (occupation typing)
only reads a lexicon bundled in the package (zero network). The sole LLM path is
`relation_hybrid.HybridRelationExtractor`, reachable only by **explicitly injecting**
`relation_extractor=`, and with a zero budget it is equivalent to pure rules.

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
pip install "xgen-ontokit[english-relations]" # + spaCy English dependency SVO (opt-in)
pip install "xgen-ontokit[all]"          # everything
# ⚠️ english-relations needs the spaCy model fetched separately (not a PyPI package):
#   python -m spacy download en_core_web_sm
# Direct from GitHub:
pip install "git+https://github.com/Createyouracccount/xgen-ontokit.git@v0.13.1"
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
dev 0.842/ho 0.700/te 0.767 (measured on an internal corpus — not external gold).

## Class-promotion filter (v0.9) — LLM-free over-generation cleanup
```python
from ontokit.filter import ClassPromotionFilter

f = ClassPromotionFilter(corpus_chunks=n_chunks)  # support gate auto-disabled if unknown(None) or small(<5000)
keep, reason = f.decide(label, df=df, has_rel=..., has_kid=..., has_inst=...)
```
Promotion criterion (termhood): promote to class only on reuse (df≥2) or structural
participation (relation·hierarchy-parent·instance). Junk rules use statistics + closed-class
grammatical function words only (no domain blacklist). mixed20k measured: 444,817→70,671
(-84.1%), relation triples 100% preserved (measured on an internal corpus). ⚠️ Also removes
isolated df1 valid concepts (intended cost; XGEN wiring is reversible via sidecar
`<graph>__filtered`).

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
The figures above are measured on an internal corpus (not external gold).

## Relation encoder (v0.13) — KLUE-RE + SREDFM-ko augmented, holdout 0.6274
```bash
pip install "xgen-ontokit[relation-encoder]"          # + transformers·torch
export ONTOKIT_RELATION_ENCODER_MODEL=/path/to/model_re   # this env is the on/off switch
```
A **local RE encoder** channel that surpasses rule-based particle SVO (KLUE linkage 0.8%).
Fine-tuned klue/roberta-small, **zero LLM API calls** (local inference, same family as NER).

**Adoption history** (external gold = KLUE-RE official validation 7,765, micro-F1):

| Version | holdout | Note |
|---|---|---|
| re-ko-v1 | 0.5924 | consistent with the official roberta-small baseline 60.85 |
| re-ko-aug-v1 | 0.6259 | SREDFM-ko augmentation |
| **re-ko-aug-v12 (current)** | **0.6274** | removed 721 unsupported P112 rows — `founded_by` 0.519→0.696 |

`eval/relation/MODEL_LOCK.json` is the single source of truth for the adopted build
(release_tag·sha256 pinned).
⚠️ The model is still **small** — below the official base 0.6666 / large 0.6959, so
**size upgrade remains an unspent lever** (achievable without any LLM).

- **Invariant**: if any of env-unset·extras-not-installed·bad-path·no-NER holds, it falls
  back to rule-based particle SVO. "It does not turn on unless installed·configured." The
  NER (KoELECTRA) supplies entities, which are paired → relation classification.
- **Rule-channel status** (B3): the particle-SVO channel is **availability-fallback only**.
  Ensembling it with the encoder is **permanently rejected** — measured on holdout, rules
  corrected 29 cases but contaminated 144 (net value negative).
- **Model swap**: not tied to a specific model. ① change env ② retrain via
  `eval/relation/train_encoder.py` ③ inject `relation_extractor=` ④ `.extract()` wrapper —
  see the "Model swap" section in `eval/relation/README.md`.
- Reproduce·evaluate: [`eval/relation/`](eval/relation/) — its README has the KLUE-RE
  download commands, `train_encoder.py` (retrain), `eval_encoder.py` (score), and
  `JUDGE_PROTOCOL.md` (criteria). Weights ship as GitHub Release assets (not committed;
  sha256 in `MODEL_LOCK.json`).

## Definitional hierarchy·typing (v0.12~) — heterogeneous hierarchy induction (on by default)
Induces the heterogeneous hierarchies that suffix-share is **structurally incapable** of
catching (강아지 ⊂ 동물, 신용공여 ⊂ 거래) via definitional sentence-ending patterns
(copula/genus/predicate/속하는). `enable_hearst=True` is the default.

```python
ext = DeterministicKoreanExtractor(enable_hearst=True)   # default
```
- **ABox↔TBox bridge**: when a definitional subject is a NER entity, it emits `rdf:type`
  instead of `subClassOf` — repairing isolated islands that made hierarchy reachability 0%.
- Entirely rule-based (Kiwi morphology + ending patterns). Zero LLM calls.
- ⚠️ **Evidence level**: the external-gold (Wikidata P279) judge loop 89/100 and the
  real-build 615 pairs·87% precision are **self-judged development-round records** and have
  **not yet landed as reproducible artifacts** under `eval/hierarchy/`. Read them on that basis.

## Occupation instance typing (v0.13) — P106 lexicon (on by default)
A build-time channel that assigns occupation classes to person entities.

```python
ext = DeterministicKoreanExtractor(enable_occupation=True)   # default
# disable: enable_occupation=False or ONTOKIT_OCCUPATION_TYPING=off
```
- Lexicon `data/occupation_lexicon_ko.json.gz` (4,121 pairs) — SREDFM-ko P106 surface forms +
  Wikidata candidates kept **only on blind two-rater agreement** (348, human-verified).
  **Bundled in the package = zero build-time network**, zero model loads, zero LLM calls.
- Gates: person-dominance cut (homonym defense) + evidence gate
  (`ONTOKIT_OCCUPATION_EVIDENCE=adj` default). Domain false positives 63.6%→0% measured
  (self-measured, not external gold).
- Multiple occupations (Galileo = physicist·mathematician) are emitted as additional records
  → naturally supports multiple `rdf:type`.

## Search improvements
```python
from ontokit.search import class_instances_triple, blend_score
# #1 subClassOf* transitive closure — full enumeration of subclass instances (zero regression)
# #2 vscore missing-floor guard — restores keyword exact-match chunk ranking
```

## XGEN injection (internal use only — external users can skip this)
XGEN is the internal product that consumes this library. The following is its wiring note.
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
│                         #   relation_ko (particle SVO) / relation_encoder_ko (KLUE-RE, opt-in)
│                         #   relation_en (spaCy dep SVO, opt-in) / relation_hybrid (⚠️LLM, injection-only)
├── morphology/           # kiwi_nouns (Korean) + en_nouns (English nltk POS)
├── hierarchy/            # suffix_share (main engine, ko=char/en=word), hearst_ko (definitional, on by default)
├── instance_typing/      # occupation (P106 lexicon, on by default) + evidence + hygiene (v0.13)
├── ner/                  # koelectra (ko) + english (dslim BERT MIT) + ensemble·span_align
├── dedup/                # deterministic (morphology) + synonym_dict (Urimalsaem, opt-in)
│                         #   class_synonyms (TBox candidate proposal — no merge, offline review)
├── citations.py          # doc-level :cites citation collection·SPARQL emit (v0.8)
├── filter/               # class_promotion — termhood promotion gate (v0.9)
├── cooccurrence.py       # coOccursWith co-occurrence weak relation — density boost (v0.10)
└── search/               # improvements (subClassOf*, floor guard) — ⚠️XGEN-specific
```

## Quality evidence — only what you can reproduce

Performance claims are measured **on external public datasets only**. Self-made synthetic GT
is banned, because on the hierarchy axis it collapsed from **synthetic-GT F1 0.96 to
external-gold 0.33**. Everything below is reproducible from within this repo.

| Axis | External gold | License | Result | Reproduce |
|---|---|---|---|---|
| **Relation** | KLUE-RE (official validation 7,765) | CC BY-SA 4.0 | holdout micro-F1 **0.6274** | [`eval/relation/`](eval/relation/) |
| **Hierarchy** | Wikidata P279 + Korean Wikipedia lead | CC0 | ⚠️ see caveat below | [`eval/hierarchy/`](eval/hierarchy/) |
| **Entity resolution (ER)** | Korean Wikipedia redirects | CC BY-SA 4.0 | balanced F1 **0.776** — **below** the 0.80 gate | [`eval/entity_resolution/`](eval/entity_resolution/) |
| Fine-grained typing | (self-measured) | — | **discarded** — 0.16% retyped, no effect | [`eval/instance_typing/`](eval/instance_typing/) |

Each directory's README carries the data-download commands, evaluation scripts, and
decision criteria. E.g. reproducing the relation axis:
```bash
cd eval/relation && cat README.md      # includes curl commands for the KLUE-RE parquet
python eval_encoder.py holdout
```

### ⚠️ Stated honestly — what is still weakly evidenced

- **Hierarchy 89/100 and definitional 615 pairs·87% precision** are **self-judged
  development-round records**. The result log in `eval/hierarchy/README.md` only contains
  R0 26/100 and "R1 in progress"; the artifacts backing 89/100 have **not landed yet**.
- **Every "NN/100 judge" score comes from an in-repo judge loop** (protocols in
  `eval/*/JUDGE_PROTOCOL.md`), not external re-scoring. Only two numbers are anchored
  directly to external gold: relation holdout (0.6274) and ER (0.776).
- **ER is not shipped.** Embeddings cannot separate synonymy from topical proximity
  (AUC ceiling ~0.81), so it missed the gate and was deliberately left unwired. Default dedup
  is morphology-based; dictionary merging is opt-in via `ONTOKIT_SYNONYM_DICT`.
- **English is structurally tested only** (no corpus measurement). Korean = finreg 489 measured.

## Adding as a dependency (install from GitHub)

The repo is public, so it installs **without authentication**:

```bash
pip install "git+https://github.com/Createyouracccount/xgen-ontokit.git@v0.13.1"
```
Pinning is recommended — on-by-default channels have changed across minor versions (see the
behavior-change notes above). Add the URL to your `pyproject.toml` dependencies or requirements.
