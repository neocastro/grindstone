# grindstone

Local RAG pipeline that grounds a weak coding agent on curated corpora
(Rust books, TLA+ source). Ingest → chunk → embed → retrieve → eval — the
grindstone the agent sharpens itself on.

Sibling repo to [tlarc](https://github.com/neocastro/tlarc): the weak local
model (Ollama `gpt-oss:20b` under `codewhale exec`) grinds GitHub issues, and
grindstone supplies the reference context it needs to write correct,
idiomatic code.

## Status

v0 scaffold: a Rust workspace (`grindstone` engine crate + `gs` CLI) with the
hardened agent-prompt constructor. The retrieval pipeline lands via the
RAG-1..RAG-8 tickets on this repo's tracker.

## Layout

- `crates/grindstone` — engine library (chunking, embedding, retrieval, eval
  land here via the tickets)
- `crates/gs` — CLI binary (`gs build-prompt` today)

## Safety model

Issue bodies are attacker-controlled text on a public tracker. `gs
build-prompt` frames the body as untrusted data inside explicit delimiters —
instructions embedded in a hostile issue cannot escape the frame, and the
agent's working rules always live outside it. See the tests in
`crates/grindstone/src/lib.rs`.
