# ADR-0004: The library owns the eval strategy adapters

Until RAG-11b there were two parallel strategy abstractions: the library's
`eval::Strategy` (query → ranked doc ids) and the CLI's private `EvalStrategy`
box — an identical trait-object type that existed only because the CLI had to
build each strategy itself. `cmd_eval` inlined ~80 lines of construction: it
loaded stores, indexes, and chunks, wrapped the embed call in closures, and
wired each strategy's `doc_ids` function, with the fulltext branch reaching
directly into `fulltext::default_searcher()`.

The strategy adapters now live in the library as owning constructors, all
returning the single `eval::Strategy` abstraction:

- `fulltext_strategy(corpus_dir, searcher)` — takes the RAG-10 searcher seam,
  so the baseline runs offline (in-process adapter) when `rg` is absent;
- `cosine_strategy(index_dir, embed)` — loads the vector store itself;
- `hybrid_strategy(index_dir, embed)` — loads the vector store, chunks, and
  BM25 index itself.

The embedder is injected as an owned `Box<Embedder<'static>>` (a fake in
tests, `embed::ollama_embedder()` in production — the new factory that
replaces inline embed closures). Constructor load failures surface through a
new `EvalError::Load { what, detail }` variant instead of the CLI's inline
`eprintln!` + `exit(1)`.

`cmd_eval` is now a pure name → strategy lookup: three arms, each pairing a
result path with a constructor call, plus a tiny `build` helper that unwraps
the constructor result. The CLI's `EvalStrategy` type is deleted; there is
exactly one strategy abstraction in the codebase, and it is `eval::Strategy`.

**Status**: accepted

**Consequences**: eval strategies are unit-testable in the library without a
CLI (four new tests: each constructor's query path, plus load-error
reporting), and any future strategy — BM25-only, rerank, hybrid variants —
is a new constructor plus a lookup arm. Error messages for missing index
artifacts change slightly (wrapped in `EvalError::Load`), but the exit
behavior (status 1) is unchanged. Recall on the committed eval set is
identical: the same `doc_ids` functions drive the same queries, and the
regenerated fulltext baseline is byte-for-byte unchanged.
