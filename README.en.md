# xgen-ontokit

[한국어](README.md) · **ENGLISH**

**Build an ontology graph (entities · types · relations · hierarchy) from Korean documents without an LLM, and use it to add
answers to the questions vector search cannot handle — "list them all · how many · who relates to whom".**

- **Build** — document chunks → entities, types, relations, `subClassOf` hierarchy. Morphology, rules and small local models only
  (0 LLM calls, nothing leaves the machine, same input → same graph).
- **Store** — graph-DB agnostic. The same query returns the same answers on Fuseki (RDF/SPARQL) and Neo4j (LPG/Cypher)
  (via [graphstore](https://github.com/Createyouracccount/xgen-graphstore)).
- **Use** — turn a question into a graph query and **merge the results into the vector-search answer**. This is the only way
  of using the graph that has been shown to improve real answers.

---

## At a glance — how it is used

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/pipeline-dark.svg">
  <img alt="Build: documents → ontokit → graph store. Query: question → planner/compiler → graph results → answer merge, joined by vector search + reader answer (labels in Korean)" src="docs/img/pipeline-light.svg" width="900">
</picture>

1. **Build** (when documents arrive): ontokit extracts entities, types, relations and hierarchy into the graph store.
2. **Query** (when a question arrives): a planner turns the question into a query plan; a compiler runs it as SPARQL/Cypher,
   using ontology semantics at query time — subclass closure ("musician" → singers, composers), inverse relations, aliases.
3. **Merge**: the graph results are merged into the answer of a reader LLM that read the top-40 vector-search chunks;
   count questions use the graph count.

> ⚠️ The XGEN product does not use ontokit today, and uses its graph only to fetch extra chunks (measured: no effect).
> The merge path above is **validated inside the measurement harness and not yet in the product**.

---

## Where it stands — an honest scorecard

Answers from **vector search + LLM** alone vs. the same answers **merged with ontokit graph results**, scored against gold.
3,000 documents used during development (EVAL) and 3,000 **never-seen** documents (HOLDOUT), 210 questions each.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/effect-dark.svg">
  <img alt="Change in real-answer score vs. vector+LLM: graph-result merge EVAL +0.081, HOLDOUT +0.057 / gold graph +0.699, +0.630 / fake graph +0.023, +0.009 (labels in Korean)" src="docs/img/effect-light.svg" width="900">
</picture>

| Condition (score change vs. vector+LLM, 95% CI) | EVAL | HOLDOUT (never-seen docs) |
|---|---|---|
| **Merge ontokit graph results** (built without an LLM) | **+0.081** [+0.049, +0.115] | **+0.057** [+0.030, +0.086] |
| Gold graph — if extraction were perfect (ceiling) | +0.699 [+0.599, +0.792] | +0.630 [+0.521, +0.736] |
| Fake graph — every fact swapped (negative control) | +0.023 [−0.024, +0.083] | +0.009 [−0.021, +0.048] |

**How to read it** — merging improves answers even on never-seen documents (CI above 0). The fake graph does nothing, so the
gain comes from the graph's **content**. But the gain is small — about 1/10 of the room the gold graph shows (+0.63).

### Where it helps, and where it does not

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/forms-dark.svg">
  <img alt="Real-answer score by question form on never-seen documents: up for type enumeration, inverse relation, multi-hop; no gain for counting and intersection; lookup unchanged (labels in Korean)" src="docs/img/forms-light.svg" width="900">
</picture>

| Question form (HOLDOUT) | vector + LLM | + ontokit merge | |
|---|---|---|---|
| E1 type enumeration — "all items that are X" | 0.048 | **0.173** | 3.6× |
| E2 superclass enumeration — "including subtypes" | 0.000 | **0.063** | |
| R inverse relation — "everyone born in Seoul" | 0.084 | **0.204** | 2.4× |
| M multi-hop — "the schools those people attended" | 0.051 | **0.088** | |
| A counting — "how many" | 0.067 | 0.033 | ❌ no gain |
| C intersection — "nationality X and occupation Y" | 0.039 | 0.037 | ❌ no gain |
| L lookup — "where was X born" | 0.656 | 0.648 | no regression (vector's home turf) |

### Why the scores are low

| Cause | Evidence |
|---|---|
| The test is hard by design | Every matching item across 3,000 docs must be named **exactly**; counts must be exact. Even a perfect reader caps vector search at ~0.25 on type enumeration |
| **Graph quality sets the ceiling** | ontokit fact recall — relations 16.7% · types 11.5% (gold graph 100%) |
| **Class names don't match the questions** | Question class names present in ontokit's type vocabulary — type enumeration 22/30, **superclass 9/30** |
| The reader is small | A local 8B model: wrong names, repetition, and it drops correct graph answers (which is why merging helps) |

Full process, failures and corrections: [`harness/docs/`](harness/docs/) (Korean) — verdict summary in
[`L02_루프탈출_판정.md`](harness/docs/L02_루프탈출_판정.md).

---

## What was done — graph-quality rounds

Score when answering from the graph **alone**. Each round fixed the single biggest loss and re-measured (all pre-registered).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/rounds-dark.svg">
  <img alt="Graph-only score: round 1 0.018, 2 0.048, 3 0.096, 4 0.113, 5 0.128, 6 (local-LLM extraction) 0.302; vector-search ceiling 0.162 (labels in Korean)" src="docs/img/rounds-light.svg" width="900">
</picture>

| Round | Treatment | Score | Result |
|---|---|---|---|
| 1 | Baseline | 0.018 | relation recall 1.6% diagnosed |
| 2 | Restore dropped subjects · per-sentence pair cap | 0.048 | ✅ significant |
| 3 | Definitional channel (encyclopedic only) | 0.096 | ✅ significant |
| 4 | Location-statement relations | 0.113 | ✅ significant |
| 5 | Add NER-missed institution names | 0.128 | ✅ significant — still below the vector ceiling (0.162) → **"graph alone beats vector without an LLM" failed** |
| 6 | + local LLM (Qwen3-8B) extraction | 0.302 | beats the ceiling, but 12 h for 3,000 docs and Wikipedia-contamination risk |

→ Since the graph alone could not beat vector search, the approach changed to **merging graph results into the vector answer** —
and that is what was validated (scorecard above).

---

## Quick start

```bash
pip install "xgen-ontokit[korean,ner]"                 # Kiwi morphology + KoELECTRA NER
pip install "xgen-ontokit[relation-encoder]"           # + local relation encoder (optional)
```

```python
import asyncio
from ontokit import DeterministicKoreanExtractor
from ontokit.ner.koelectra import KoElectraNER

documents = {"doc.txt": [{"chunk_id": "c1", "chunk_text": "김가수는 서울에서 태어난 가수이다.", "chunk_index": 0}]}
ext = DeterministicKoreanExtractor(ner=KoElectraNER())          # 0 LLM calls
concepts, entities, relations, _ = asyncio.run(ext.extract(documents))
# entities → {'doc.txt': [{'entity': '김가수', 'class': '인물', 'source_chunks': ['c1'], ...},
#                         {'entity': '서울', 'class': '지역', ...}, ...]}
# concepts → classes · class_hierarchy (subClassOf) / relations → [(subject, relation, object), ...]
# relations need the local relation encoder: export ONTOKIT_RELATION_ENCODER_MODEL=<model path>
# every output carries source_chunks (which chunk it came from)
```

Loading, querying and merging live in the measurement harness ([`harness/`](harness/), outside the package):
`harness.graph` (load) · `harness.query` (SPARQL) · `harness.lpg` (Cypher) · `harness.merge_eval` (merge).
Long runs use `harness.supervise` (resume-on-crash) and `harness.guard` (OS-agnostic memory guard).

---

## Components and models

| Role | What | Note |
|---|---|---|
| Morphology · classes · hierarchy | Kiwi + rules (suffix sharing · definitions · occupation lexicon) | no model |
| Entity recognition | KoELECTRA-small (local) | coarse types (person · org · location…) |
| Relation extraction | KLUE-RE roberta-small v13c (local, opt-in) | external-gold F1 0.6169; large 0.6726 is 2.2× slower, optional |
| Graph store | graphstore → Fuseki (RDF) · Neo4j (LPG) | same query, same answers (210/210 × 3 graphs) |
| Planner · reader (measurement) | Qwen3-8B 4-bit, local (MLX) | nothing leaves the machine; weaker than the product's large LLM |
| Vector search (measurement baseline) | XGEN product search API · text-embedding-3-small | product settings as-is |

**Defaults** — a no-arg constructor makes 0 LLM calls and loads 0 models. Every model-backed channel (NER, relation encoder,
English) is opt-in. Full switch table: [channel details](docs/CHANNELS.en.md#defaults--env-switches-at-a-glance).

---

## Limits and next steps

**Not measured** — Korean Wikipedia (encyclopedic register) only · local 8B reader only · gold = Wikidata facts (correct answers may be
scored wrong; no human audit) · templated questions.

| Priority | Task | Why |
|---|---|---|
| 1 | Put "graph-result merge" into the product RAG path | the only validated gain is not in the product yet |
| 2 | Align the class vocabulary (import a taxonomy — e.g. a product category tree) | 21/30 superclass questions use names absent from the graph |
| 3 | Gold sets on Lotte / news documents | results so far are encyclopedic only |
| 4 | Re-measure with a larger reader | does the gain survive at product scale |
| 5 | Extraction recall | relations 16.7% · types 11.5% set the ceiling |

---

## More

<details>
<summary><b>Per-version channels and behavior changes</b> (expand)</summary>

| Version | Change | Details |
|---|---|---|
| v0.16 | Four opt-in relation-encoder channels (dropped-subject restoration · per-sentence pair cap · location relations · institution-name candidates), concept-gate no-op warning. **No default behavior change** | [defaults · switches](docs/CHANNELS.en.md#defaults--env-switches-at-a-glance) |
| v0.15 | QDT ghost gate **on by default** (blocks quantity/date entities in no relation) | [language support · behavior changes](docs/CHANNELS.en.md#language-support-matrix-v0160-stated-honestly) |
| v0.14 | Definitional channel **off by default** (88.5% false on news) | [definitional hierarchy](docs/CHANNELS.en.md#definitional-hierarchytyping-v012--heterogeneous-hierarchy-induction-off-by-default-v014) |
| v0.13 | Relation encoder (KLUE-RE) · occupation typing (P106, on) · English spaCy relations | [relation encoder](docs/CHANNELS.en.md#relation-encoder-v013--klue-re--sredfm-ko-augmented-holdout-06169-v13c) · [occupation typing](docs/CHANNELS.en.md#occupation-instance-typing-v013--p106-lexicon-on-by-default) |
| v0.12 | Definitional hierarchy · typing | [definitional hierarchy](docs/CHANNELS.en.md#definitional-hierarchytyping-v012--heterogeneous-hierarchy-induction-off-by-default-v014) |
| v0.10 | Co-occurrence weak relation | [co-occurrence](docs/CHANNELS.en.md#co-occurrence-weak-relation-v010--llm-free-relation-density-boost-language-agnostic) |
| v0.9 | Class-promotion filter | [class promotion](docs/CHANNELS.en.md#class-promotion-filter-v09--llm-free-over-generation-cleanup) |
| v0.8 | Citation ontology (`:cites`) | [citations](docs/CHANNELS.en.md#citation-ontology-v08--doc-level-cites) |

</details>

<details>
<summary><b>Quality evidence — measured on external public data</b> (expand)</summary>

| Axis | External gold | Result |
|---|---|---|
| Relations | KLUE-RE official validation 7,765 | micro-F1 **0.6169** (v13c) · large 0.6726 |
| Entity resolution | Korean Wikipedia redirects | F1 0.776 — below the 0.80 gate, not shipped |
| Hierarchy | Wikidata P279 + Korean Wikipedia | reproducible artifacts not landed — in-repo judge records only |

How to reproduce and what is still weakly evidenced: [quality evidence](docs/CHANNELS.en.md#quality-evidence--only-what-you-can-reproduce)

</details>

<details>
<summary><b>Code layout</b> (expand)</summary>

```
src/ontokit/
├── protocols.py          # injection interfaces (Extractor/GraphStore/VectorStore/LLM)
├── extractors/           # deterministic_ko (core, ko·en dual extraction) + base (merge_concepts)
│                         #   relation_ko (particle SVO) / relation_encoder_ko (KLUE-RE, opt-in)
│                         #   relation_en (spaCy dep SVO, opt-in) / relation_hybrid (⚠️LLM, injection-only)
├── morphology/           # kiwi_nouns (Korean) + en_nouns (English nltk POS)
├── hierarchy/            # suffix_share (main engine), hearst_ko (definitional, off by default v0.14~)
├── instance_typing/      # occupation (P106 lexicon, on) + evidence + hygiene
├── ner/                  # koelectra (ko) + english (dslim BERT) + ensemble·span_align
│                         #   word_boundary·suffix_fragment (experimental, off)
├── dedup/                # deterministic (morphology) + synonym_dict (Urimalsaem, opt-in) + class_synonyms
├── citations.py          # doc-level :cites (v0.8)
├── filter/               # class_promotion (v0.9) · concept_gate (P279, off)
├── cooccurrence.py       # coOccursWith co-occurrence weak relation (v0.10)
└── search/               # improvements (subClassOf*, floor guard) — XGEN-specific

harness/                  # ontology harness — outside the package (not installed): bench · query · reading · scoring · controls
docs/                     # channel details (CHANNELS.en.md), README figures (img/, regenerate with make_figures.py)
```

Design principles — **zero core dependencies** (models and backends are extras) · **protocol injection**
(`Extractor`/`GraphStore`/`VectorStore`/`LLM`) · **single source** (improvements live in the library, A/B via switches).
Why deterministic and LLM-free (audit · reproducibility · data control · provenance), the input contract (already parsed and chunked text —
no PDF/HWP parsing), and all install extras: [overview · design rationale](docs/CHANNELS.en.md#overview--design-rationale-former-readme-intro).

</details>

---

## Install · add as a dependency

```bash
pip install xgen-ontokit                       # core (no dependencies)
pip install "xgen-ontokit[all]"                # everything (Kiwi · NER · relation encoder · English)
pip install "git+https://github.com/Createyouracccount/xgen-ontokit.git@v0.16.0"
```

⚠️ The latest tag on the remote is currently **v0.13.1** (v0.14.0–v0.16.0 not pushed) — until they are, the last command fails;
pin a commit SHA instead (`...xgen-ontokit.git@<commit>`). On-by-default channels have changed across minor versions, so pin a version.
