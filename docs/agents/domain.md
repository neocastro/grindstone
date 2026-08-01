# Domain Docs

Single context for grindstone's domain vocabulary. For how these terms map
to modules, data flow, and tests, see [`modules.md`](modules.md).

## Glossary

**Corpus**:
A curated, pinned collection of source documents (Rust books, TLA+ source)
that the pipeline indexes. Corpus-agnostic by design — adding a corpus must
cost about the same as adding any other.
_Avoid_: "the data", "the docs" when referring to a specific indexed set.

**Manifest**:
The pinned list of corpus sources (URL, version, license, content hash). The
single source of truth: same manifest → same corpus state (deterministic
rebuilds).
_Avoid_: "config" when referring to the pinned sources.
Sources carry a `format` (`html` = rendered doc fetched by URL, `text` = a
local source checkout assembled by ingest — the TLA+ Java corpus) and a
**trust tier**.

**Chunk**:
A heading-aware slice of a document (~500 tokens with overlap), with stable
IDs and provenance metadata (source, license, trust tier).
_Avoid_: "paragraph" — chunks are structural units, not prose units.

**Embedding**:
The vector representation of a chunk, produced by `nomic-embed-text` via the
local Ollama server. The embedder is a **seam**: production uses
`ollama_embed`, tests inject a deterministic fake so the determinism contract
holds without Ollama.

**Index build**:
The library module that owns the whole pipeline walk end-to-end —
manifest → corpus files → chunks → embeddings → vector store
(`indexbuild::build_index`). The determinism contract lives here: same
manifest + same corpus → identical index artifacts, bit-for-bit. The CLI
stays per-stage (`gs chunk`, `gs embed`) but delegates the walks to Index
build; retrieval consumers load through `indexbuild::load_index` (store +
sparse index) or `indexbuild::load_vector_store` (store only).
_Avoid_: "the pipeline" when the library's orchestration path is meant — the
CLI's per-stage commands are entry points, not the walk.

**Searcher seam**:
The fulltext baseline's pluggable adapter, `fulltext::Searcher` (corpus dir +
query → ranked hits): `RipgrepSearcher` when `rg` is on PATH,
`InProcessSearcher` (deterministic pure-Rust scan) otherwise — so the
baseline runs anywhere, including CI. `fulltext::default_searcher()` picks
per environment. See ADR-0002.

**Strategy seam**:
The retrieval-quality pluggable path, `eval::Strategy` (query → ranked doc
ids). The library owns the adapters as constructors — `fulltext_strategy`,
`cosine_strategy`, `hybrid_strategy` — and the CLI only picks a name. Every
strategy must measurably beat the previous on the eval harness. See ADR-0004.

**Vector store**:
The persisted chunk + embedding + metadata index (deterministic JSON artifacts + in-memory cosine at this scale — see RAG-4).
_Avoid_: "database" when referring to the index.

**Retrieval strategy**:
A pluggable query path behind the **strategy seam**: full-text baseline →
cosine → BM25 hybrid (cross-encoder rerank is the roadmap rung after that).
Each rung must measurably beat the previous on the eval harness.
_Avoid_: "search mode" — strategies are measured, not selected by taste.

**Eval harness**:
A set of needle queries with expected-hit annotations, scored by recall@k.
The before/after instrument for every retrieval change.
_Avoid_: "tests" when referring to eval — it is a retrieval-quality gate, not
a correctness test.

**Trust tier**:
Per-chunk provenance rank (pinned-source > docs-wiki > navigational). Retrieval
may filter or bias by tier.

**Prompt injection**:
The threat model this repo is designed against: issue bodies are
attacker-controlled text on a public tracker, and must never carry
instructions to the agent. The untrusted-data frame in `gs build-prompt` is
the mechanism.

## Relationships

- A **corpus** is described by a **manifest**
- A **manifest** produces **chunks** via the chunker
- **Index build** walks a **manifest** and **corpus** to **chunks**, then to
  **embeddings**, assembling the **vector store**
- **Chunks** are embedded and stored in the **vector store**
- The **searcher seam** serves the full-text baseline over the **corpus**
- A **retrieval strategy** queries the **vector store** (or the corpus, for
  the full-text baseline), all behind the **strategy seam**
- The **eval harness** scores every **retrieval strategy**
- `gs build-prompt` consumes untrusted issue text and emits a prompt with the
  **untrusted-data frame**, then grounds the implementer by injecting top-k
  **chunks** for the issue between the frame and the working rules
  (retrieve-then-inject, RAG-5)
