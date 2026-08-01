# Module Map

The doc an agent reads first when picking up a new issue: what each module
does, how data flows through the pipeline, where the seams are, and where the
tests live. If you are about to touch retrieval, chunking, or the CLI, start
here — and read the domain glossary (`domain.md`) before proposing or naming
anything.

## Dependency flow

```
manifest.json ──► ingest ──► corpus/ files
   │                             │
   └───────────────► chunk ──────┘
                        │
                        ▼
                   chunks.json ──► embed ──► embeddings.json
                                          │
                                          ▼
                                  vector store (load: chunks + embeddings)
                                          │
                    ┌─────────────────────┼────────────────────┐
                    ▼                     ▼                    ▼
               fulltext              cosine               hybrid
              (corpus dir)      (store, query vec)   (store + BM25, RRF)
                    │                     │                    │
                    └─────────────────────┼────────────────────┘
                                          ▼
                                       eval harness
                                          │
                                          ▼
                                      gs CLI (build-prompt, query, eval, ...)
```

The one orchestrated walk is **Index build** (`indexbuild`): it loads the
manifest, chunks every source, embeds incrementally, and persists both
artifacts — the CLI delegates to it per stage. Retrieval consumers load
through `indexbuild::load_index` (store + sparse index) or
`indexbuild::load_vector_store` (store only); nobody reloads `chunks.json`
and rebuilds the sparse index by hand.

## Modules (`crates/grindstone/src/`)

### `manifest.rs` — the pinned source list
The single source of truth for the corpus: each source's URL, license,
content hash, `format` (`html` | `text`), and trust tier. `Manifest::load`
reads `corpus/manifest.json`; the hash function (`sha256_hex`) pins content
for deterministic rebuilds and ingest's corruption check.

### `ingest.rs` — fetch sources to disk
Fetches every source whose corpus file is missing or hash-mismatched and
writes the resolved manifest back. The **fetcher seam**: callers inject
`http_fetcher` (network) or `local_text_fetcher` (`file://` — the TLA+
source checkout) via the `Fetcher` callback.

### `chunk.rs` — heading-aware chunking (RAG-9)
One entry, `chunk(content, source)`: dispatches on `SourceFormat` to the
format-specific section producer (`sections_from_html` /
`sections_from_text`), then feeds one shared merge core — ~500-token target
with overlap, oversized-section splitting, dedup, content-addressed IDs,
provenance stamping. Deterministic: same content → same chunks, bit-for-bit
(see ADR-0001).

### `embed.rs` — embeddings via Ollama
The **embedder seam**: an `Embedder` is `FnMut(&[String]) -> Result<Vec<Vec<f32>>>`,
with `ollama_embed` as the production implementation (`nomic-embed-text`,
hard timeout, batched). `embed_chunks_incremental` reuses vectors for
unchanged chunk IDs and prunes stale ones — reuse-counting lives here, and
`embed_corpus` in Index build reports it. Tests use a deterministic fake
embedder so the determinism contract is testable without Ollama.

### `indexbuild.rs` — Index build, the pipeline walk (RAG-12a)
Owns the whole manifest → chunk → embed → store walk (see ADR-0005):
- `chunk_corpus(corpus_dir, on_source)` — the walk through `chunk::chunk`;
- `embed_corpus(chunks, index_dir, embed, on_progress)` — incremental embed
  + reuse count, via `embed::embed_chunks_incremental`;
- `build_index(...)` — the full walk, persisting `chunks.json` +
  `embeddings.json`, returning an `IndexBuildReport`;
- `load_index(index_dir)` — store + sparse BM25 index, one call;
- `load_vector_store(index_dir)` — store only (the cosine path; skips the
  sparse build).

The determinism contract — same manifest + corpus → identical artifacts,
bit-for-bit — is stated and tested at this module's interface.

### `vector.rs` — the vector store (RAG-4)
In-memory chunk + embedding + metadata store assembled from the two JSON
artifacts; `search` ranks by cosine with deterministic tie-breaking, optional
trust-tier filter, and returns at most `k` hits. At this scale it is a
JSON + cosine scan, no vector server (see the module docs).

### `bm25.rs` — sparse retrieval index
`Bm25Index::build(chunks)` + `tokenize`. Pure lexical ranking; the sparse
half of hybrid search.

### `fulltext.rs` — the fulltext baseline (RAG-10)
The deliberately-dumb baseline every retrieval change must measurably beat.
The **searcher seam**: `Searcher` (corpus dir + query → ranked hits) with two
adapters — `RipgrepSearcher` (used when `rg` is on PATH) and
`InProcessSearcher` (deterministic pure-Rust fallback, so the baseline runs
anywhere including CI). `default_searcher()` picks per environment; `search`
is the convenience wrapper (see ADR-0002).

### `hybrid.rs` — BM25 + cosine fusion (RAG-7)
`hybrid_search` fuses sparse + dense rankings via reciprocal-rank fusion
(RRF), so lexical matches promote over pure cosine order.

### `eval.rs` — the eval harness (RAG-11a/11b)
The retrieval-quality gate: a set of needle queries with expected-hit
annotations (`EvalSet`), scored by recall@k (`recall_at_k`, per-query and
overall). The harness owns its result — `run_eval` computes `corpus_hash`
from the manifest itself (ADR-0003). The **strategy seam**: the library owns
the strategy adapters as constructors — `fulltext_strategy`, `cosine_strategy`,
`hybrid_strategy` — all returning the one `Strategy` trait object (ADR-0004);
the CLI only picks a name and prints the report.

### `lib.rs` — prompt construction + retrieval-to-prompt
`build_prompt(issue, repo)` / `build_prompt_with_context(...)` emit the
hardened agent prompt: the issue body is framed as untrusted data inside
explicit delimiters (the injection defense), working rules live outside the
frame. `retrieve_context(index_dir, issue, k, embed)` embeds the issue
title+body and returns the top-k chunks for grounding (retrieve-then-inject,
RAG-5).

## Seams at a glance

- **Fetcher** (`ingest`) — inject `http_fetcher` / `local_text_fetcher`
- **Embedder** (`embed`) — inject `ollama_embed` or a deterministic fake
- **Searcher** (`fulltext`, ADR-0002) — `RipgrepSearcher` /
  `InProcessSearcher` behind `default_searcher()`
- **Strategy** (`eval`, ADR-0004) — library constructors behind one
  `Strategy` trait object; CLI picks by name

These are the extension points. New retrieval paths plug in as strategies;
new sources plug in via the fetcher seam and the manifest.

## CLI (`crates/gs/src/main.rs`)

| Command | What it does |
| --- | --- |
| `gs build-prompt [REPO] [--no-rag] [--index DIR] [--ollama URL]` | issue JSON on stdin → grounded agent prompt (stdout is exactly the prompt; diagnostics go to stderr; retrieval failure degrades to ungrounded, exit 0) |
| `gs ingest [CORPUS_DIR]` | fetch missing/hash-mismatched sources; re-run with unchanged manifest is a no-op |
| `gs chunk [CORPUS_DIR] [INDEX_DIR]` | delegates to `indexbuild::chunk_corpus` |
| `gs embed [INDEX_DIR] [OLLAMA_URL]` | delegates to `indexbuild::embed_corpus`; reports reuse |
| `gs query [--embed\|--hybrid] [--tier TIER] QUERY [DIR] [OLLAMA_URL]` | fulltext baseline (default), cosine (`--embed`, via `load_vector_store`), hybrid (`--hybrid`, via `load_index`) |
| `gs eval [--strategy fulltext\|cosine\|hybrid] [CORPUS_DIR] [EVAL_SET]` | run a strategy over the eval set, print recall@k, persist `eval/results/<strategy>.json` |

Exit codes: usage/manifest errors → 2; runtime errors (missing file, Ollama
down, store load failure) → 1.

## Why — the ADRs

The "why" behind the architecture lives in `docs/adr/`:

- [ADR-0001](../adr/0001-unified-section-merge-core.md) — one section-merge
  core for every source format (RAG-9)
- [ADR-0002](../adr/0002-fulltext-searcher-seam.md) — two-adapter fulltext
  searcher seam so the baseline runs without `rg` (RAG-10)
- [ADR-0003](../adr/0003-eval-harness-owns-its-result.md) — the harness
  computes `corpus_hash` itself; no CLI patch-back (RAG-11a)
- [ADR-0004](../adr/0004-library-owns-eval-strategies.md) — the library owns
  the strategy adapters; the CLI picks by name (RAG-11b)
- [ADR-0005](../adr/0005-index-build-module.md) — Index build owns the
  pipeline walk; the CLI stays per-stage but delegates (RAG-12a)

## Test surface

- **Unit + integration tests** live inline in each module
  (`crates/grindstone/src/*.rs`, `#[cfg(test)] mod tests`). Run everything:
  `cargo test` (also `cargo fmt --check`, `cargo clippy --all-targets` —
  these gate merges).
- **Index build determinism** is tested at its interface with a tiny fixture
  corpus and a deterministic fake embedder (`indexbuild::tests`).
- **Eval harness** (`eval/`) is a retrieval-quality gate, not a correctness
  test: `eval/evalset.json` carries the needle queries, `eval/results/` the
  per-strategy numbers. Every retrieval change must measurably beat the
  previous strategy — run `gs eval` and compare.
- **CLI behavior** (output-identical delegation, exit codes) is verified by
  running the `gs` commands against a real corpus/index with local Ollama;
  see the RAG-12b PR for the comparison procedure.
