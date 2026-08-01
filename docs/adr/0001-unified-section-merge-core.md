# ADR-0001: One section-merge core for every source format

The chunker serves two corpus formats: rendered HTML (heading-aware sections)
and text corpora assembled by ingest (FILE-marker sections). Until RAG-9 each
format carried its own near-identical flush helper — `flush_section` for HTML
(collapsing whitespace) and `flush_text_section` for text (preserving it) —
and the two paths could drift on the shared merge/flush logic. The
`SourceFormat` switch also lived in two places: the manifest layer's
per-source filename mapping and the CLI's per-format match in `gs chunk`.

The chunker now exposes a single entry, `chunk(content, source)`, that owns
the `SourceFormat` dispatch and feeds one shared section-merge core
(`chunk_sections`). Format-specific code is a thin producer
(`sections_from_html` / `sections_from_text`) that only decides how to cut
and normalize sections; everything after that — merging to ~500 tokens,
overlap, oversized-section splitting, dedup, provenance stamping — is one
path.

**Status**: accepted

**Considered options**: keeping the two flush helpers and just sharing
`chunk_sections` (left the drift risk in place — rejected because the
merge/flush behavior is the part most likely to change and must change once);
parameterizing the shared flush with a normalization flag (adopted — the
whitespace policy is the only real behavioral difference between the formats,
and it stays visible as an explicit `TextNormalization` choice at the
producer boundary rather than hidden in duplicated helpers).

**Consequences**: a new source format needs only a thin producer plus a
`SourceFormat` variant — the merge core and the CLI chunk command are
untouched. The token-estimate constants (`TARGET_TOKENS`, `OVERLAP_TOKENS`,
`CHARS_PER_TOKEN`) remain declared exactly once, in the chunker, and tests
import them rather than re-declaring. Output is byte-for-byte identical to
the pre-RAG-9 chunker (verified by regenerating the corpus and diffing the
committed `index/chunks.json`), so the determinism contract holds and no
re-embedding is required.
