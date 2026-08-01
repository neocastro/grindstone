# ADR-0003: The eval harness owns its result (corpus hash, no CLI patch-back)

`EvalResult` always carried a `corpus_hash`, but until RAG-11a the harness
never filled it: `run_eval` emitted `corpus_hash: String::new()` and the CLI
patched the real value back after the fact (`result.corpus_hash = corpus_hash;`).
Any other caller of the library eval API would silently persist a score
without any corpus-state pin — the very thing the field exists to guarantee.
The hash computation itself was also only reachable from the binary, so the
library's eval contract was incomplete.

The library now owns the hash end-to-end: `run_eval` takes a `manifest_path`,
reads the corpus `manifest.json`, and computes `corpus_hash` itself. The CLI
no longer computes or patches anything — it passes the manifest path and
writes the result the harness produced. A missing manifest (corpus not built
yet) still yields an empty hash, preserving the pre-RAG-11a CLI behavior.

The same RAG-11a pass also removed two other triplicated code paths that
made the eval adapters drift from the library: the "embed the query, take the
first vector, error if empty" step (four copies — lib retrieval, cosine
adapter, hybrid CLI strategy, dense query — the last of which indexed `[0]`
unguarded and could panic on an empty embedder response) and the "dedupe
hits by source, preserving rank order" loop (two copies). Both are now single
library helpers: `embed::embed_query_vector` and the private
`eval::dedup_doc_ids`.

**Status**: accepted

**Consequences**: every caller of `run_eval` gets a correct `corpus_hash`
with no extra ceremony, and the persisted score is reproducible against the
manifest it was measured on. The CLI `gs eval` JSON output is unchanged —
same manifest, same hash, byte-identical result file. The shared helpers mean
the dense-query path no longer panics on an empty embedder response (it now
errors like every other caller), and the cosine/hybrid adapters cannot drift
apart in how they map hits to doc ids. CLI note: the hand-rolled argument
matching stays as-is until RAG-13, which will move the CLI to `clap`.
