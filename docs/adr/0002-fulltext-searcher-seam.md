# ADR-0002: Two-adapter fulltext searcher seam with an in-process fallback

The fulltext baseline is the eval harness's reference score — every retrieval
strategy must measurably beat it. Until RAG-10 that baseline was a single
concrete `rg` invocation: one function, one implementation, hard-wired to the
filesystem. Its only integration test self-skipped when `rg` was absent, and
CI has no `rg` — so the score every strategy must beat could never be
measured in CI, and `gs eval --strategy fulltext` failed outright on any
machine without ripgrep.

The fulltext path now sits behind a small searcher seam, `Searcher`
(corpus dir + query → ranked hits), with exactly two adapters:
`RipgrepSearcher` (the existing `rg`-based implementation, preferred when
`rg` is on PATH) and `InProcessSearcher` (a deterministic pure-Rust scan over
the corpus files, with the same file ordering and the same hit/dedup/ranking/
snippet semantics). `fulltext::default_searcher()` picks `rg` when present
and the in-process scan otherwise, and the eval harness accepts any
`Searcher`, so the baseline is measurable anywhere, including CI.

**Status**: accepted

**Considered options**: keeping a single `rg`-based function and just
documenting that CI cannot run the baseline (rejected — the baseline is the
reference score for the whole retrieval ladder; leaving it unmeasurable in CI
defeats the RAG-2 gate); adding the in-process scan as a test-only fake
(rejected — the issue demands a genuine production fallback, and a fake would
never be exercised by real `gs eval` runs); parameterizing the existing free
function with a boolean flag (adopted in essence — the seam is a trait so the
eval harness and CLI can accept *any* searcher without a growing flag
surface, and future adapters slot in without touching call sites).

**Consequences**: `gs query` and `gs eval --strategy fulltext` now work
without ripgrep; CI can measure the baseline and fail the gate when a
strategy regresses below it. The in-process adapter is a line-oriented,
case-insensitive literal scan (the Rust equivalent of `rg -F -i`); its
matching is deliberately simple and deterministic — same corpus + same query
→ identical hits, with stable tie-breaks — not a reimplementation of every
`rg` feature. Behavior with `rg` present is unchanged: the `rg` adapter keeps
its exact flag set, and the committed `eval/results/fulltext-baseline.json`
regenerates byte-for-byte. A cross-adapter agreement test pins the two
adapters to identical hits whenever both are available. `MissingRipgrep` is
retained as an error for callers that explicitly require the `rg` adapter;
the default path falls back instead of failing.
