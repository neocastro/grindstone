# Domain Docs

Single context for grindstone's domain vocabulary.

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

**Chunk**:
A heading-aware slice of a document (~500 tokens with overlap), with stable
IDs and provenance metadata (source, license, trust tier).
_Avoid_: "paragraph" — chunks are structural units, not prose units.

**Embedding**:
The vector representation of a chunk, produced by `nomic-embed-text` via the
local Ollama server.

**Vector store**:
The persisted chunk + embedding + metadata index (deterministic JSON artifacts + in-memory cosine at this scale — see RAG-4).
_Avoid_: "database" when referring to the index.

**Retrieval strategy**:
A pluggable query path: full-text baseline → cosine → BM25 hybrid →
cross-encoder rerank. Each rung must measurably beat the previous on the eval
harness.
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
- **Chunks** are embedded and stored in the **vector store**
- A **retrieval strategy** queries the **vector store**
- The **eval harness** scores every **retrieval strategy**
- `gs build-prompt` consumes untrusted issue text and emits a prompt with the
  **untrusted-data frame**
