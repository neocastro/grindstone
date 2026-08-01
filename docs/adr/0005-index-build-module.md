# ADR-0005: Index build — the library owns the pipeline walk

Until RAG-12a, the manifest → corpus → chunks → embeddings → vector store
walk existed only as ad-hoc CLI plumbing. `gs chunk` re-implemented the
manifest→file→chunk walk; `gs embed` duplicated the reuse-counting the
incremental embedder already does internally; the dense query path reloaded
`chunks.json` and rebuilt the BM25 index by hand. The library exposed stage
functions and artifacts, but no orchestrated path — so the determinism
contract ("same manifest → same corpus → same index") had no single owner,
and every new consumer re-learned the walk.

The library now owns the walk in one module, **Index build**
(`crates/grindstone/src/indexbuild.rs`):

- `chunk_corpus(corpus_dir, on_source)` — the manifest→file→chunk walk
  through the RAG-9 unified chunker dispatch;
- `embed_corpus(chunks, index_dir, embed, on_progress)` — reuse-counting +
  incremental embedding, returning the embeddings file and the reuse count
  (the counting the CLI used to duplicate);
- `build_index(corpus_dir, index_dir, embed, ...)` — the full walk, persisting
  `chunks.json` + `embeddings.json` and returning an `IndexBuildReport`;
- `load_index(index_dir)` — the assembled store + sparse BM25 index in one
  call, so consumers never reload chunks and rebuild by hand.

The determinism contract is stated at this module's interface and tested
there: same manifest + same corpus + same (deterministic) embedder →
bit-for-bit identical artifacts.

**Status**: accepted

**Consequences**: `gs chunk`, `gs embed`, and `gs query --embed`/`--hybrid`
now delegate to the library — the CLI stays per-stage (its progress/report
output is UI, not walk logic) but re-implements nothing. Error handling moves
into `IndexBuildError` (manifest → exit 2, everything else → exit 1, same
status codes as before); the manifest error message text changes slightly.
Reuse-counting is exactly the library's, so `gs embed` and `build_index`
cannot drift apart. The term **Index build** was added to the domain glossary
with its relationships.
