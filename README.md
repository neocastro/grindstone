# grindstone

Local RAG pipeline that grounds a weak coding agent on curated corpora
(Rust books, TLA+ source). Ingest → chunk → embed → retrieve → eval — the
grindstone the agent sharpens itself on.

Sibling repo to [tlarc](https://github.com/neocastro/tlarc): the weak local
model (Ollama `gpt-oss:20b` under `codewhale exec`) grinds GitHub issues, and
grindstone supplies the reference context it needs to write correct,
idiomatic code.

## Status

A working local RAG pipeline: ingest fetches pinned corpus sources, the
chunker cuts them into deterministic heading-aware chunks, the embedder
indexes them via local Ollama, retrieval ranks them (fulltext, cosine, or
BM25+cosine hybrid), and the eval harness scores every retrieval change.
`gs build-prompt` consumes untrusted issue text and grounds the agent on the
top-k chunks behind a hardened prompt-injection frame.

The architecture lives in the library: the **Index build** module
(`indexbuild`) owns the manifest → chunk → embed → store walk end-to-end,
and the CLI delegates to it — `gs chunk`, `gs embed`, and the dense query
paths are thin per-stage shells, not re-implementations. See the module map
in [`docs/agents/modules.md`](docs/agents/modules.md) and the "why" in
[`docs/adr/`](docs/adr/).

## Layout

- `crates/grindstone` — engine library: manifest, ingest, chunk, embed,
  indexbuild, vector store, bm25, fulltext, hybrid, eval, and the hardened
  prompt constructor (`build_prompt` / `retrieve_context`)
- `crates/gs` — CLI binary: `build-prompt`, `ingest`, `chunk`, `embed`,
  `query` (fulltext/cosine/hybrid), `eval`
- `corpus/` — the curated sources (manifest + fetched documents)
- `index/` — the persisted artifacts (`chunks.json`, `embeddings.json`)
- `eval/` — the eval set and per-strategy result files
- `docs/agents/` — agent-facing docs: module map, domain glossary,
  issue-tracker and triage-label conventions
- `docs/adr/` — architecture decision records

## Safety model

Issue bodies are attacker-controlled text on a public tracker. `gs
build-prompt` frames the body as untrusted data inside explicit delimiters —
instructions embedded in a hostile issue cannot escape the frame, and the
agent's working rules always live outside it. See the tests in
`crates/grindstone/src/lib.rs`.
