//! Deterministic heading-aware chunker (RAG-3).
//!
//! Turns a corpus HTML document into chunks: structural sections split at
//! headings, merged into ~500-token units with overlap. Determinism is the
//! contract: same input → identical chunk ids and text, bit-for-bit. Chunk
//! ids are content-addressed (sha256 of the chunk text), so an unchanged
//! chunk keeps its id across runs even if neighbouring chunks shift.

use crate::manifest::{sha256_hex, Source, TrustTier};
use serde::{Deserialize, Serialize};

/// Target chunk size in (estimated) tokens.
pub const TARGET_TOKENS: usize = 500;
/// Overlap between adjacent chunks, in (estimated) tokens.
pub const OVERLAP_TOKENS: usize = 50;
/// Approximate characters per token used for deterministic estimation.
pub const CHARS_PER_TOKEN: usize = 4;

/// Marker line prefix separating files inside a text corpus document
/// (`===== FILE: <relpath> =====`). Text sources are assembled by ingest and
/// split on these markers by `sections_from_text`.
pub const FILE_MARKER: &str = "===== FILE: ";

/// One heading-aware slice of a document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Chunk {
    /// Content-addressed stable id: sha256 hex of `text`.
    pub id: String,
    /// Provenance: manifest source name (e.g. `rust-book`).
    pub source: String,
    /// Provenance: SPDX license expression.
    pub license: String,
    /// Provenance: trust tier.
    pub tier: TrustTier,
    /// The heading of the first section in this chunk (context aid).
    pub heading: String,
    /// Estimated token count (deterministic: chars / CHARS_PER_TOKEN).
    pub tokens: usize,
    /// The chunk text (tags stripped, entities decoded).
    pub text: String,
}

/// A section: text under one heading, before the next heading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Heading text (level-agnostic; `""` for preamble before any heading).
    heading: String,
    /// Text content of the section, tags stripped.
    text: String,
}

/// Split raw HTML into heading-aware sections.
///
/// Headings are `<h1>`..`<h6>` tags; the section following a heading runs to
/// the next heading. Text before the first heading is a preamble section with
/// an empty heading. Tags are stripped and the basic HTML entities
/// (`&amp; &lt; &gt; &quot; &#39; &nbsp;`) decoded. Deterministic: no
/// whitespace normalization beyond collapsing runs of whitespace.
pub fn sections_from_html(html: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut current_heading = String::new();
    let mut current_text = String::new();

    let mut rest = html;
    loop {
        match find_heading_tag(rest) {
            None => {
                current_text.push_str(&strip_tags(rest));
                break;
            }
            Some(start) => {
                // Text before the heading completes the current section.
                current_text.push_str(&strip_tags(&rest[..start]));
                let open_end = rest[start..]
                    .find('>')
                    .map(|i| start + i + 1)
                    .unwrap_or(rest.len());
                let close_start = rest[open_end..]
                    .find("</h")
                    .map(|i| open_end + i)
                    .unwrap_or(rest.len());
                let close_end = rest[close_start..]
                    .find('>')
                    .map(|i| close_start + i + 1)
                    .unwrap_or(rest.len());
                let heading_text = strip_tags(&rest[open_end..close_start]).trim().to_string();
                flush_section(&mut sections, &mut current_heading, &mut current_text);
                current_heading = heading_text;
                rest = &rest[close_end..];
            }
        }
    }
    flush_section(&mut sections, &mut current_heading, &mut current_text);
    sections
}

/// Index of the next `<hN>` (N in 1..=6) opening tag in `s`, or `None`.
fn find_heading_tag(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] == b'<' && bytes[i + 1] == b'h' && (b'1'..=b'6').contains(&bytes[i + 2]) {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn flush_section(sections: &mut Vec<Section>, heading: &mut String, text: &mut String) {
    // Take ownership FIRST so the caller's buffer is cleared: the pushed
    // section is a collapsed copy of the accumulated text, and the buffer
    // must not leak into the next section.
    let raw = std::mem::take(text);
    let collapsed = collapse_whitespace(&raw);
    if !collapsed.is_empty() || !heading.is_empty() {
        sections.push(Section {
            heading: std::mem::take(heading),
            text: collapsed,
        });
    }
}

/// Strip tags and decode the basic HTML entities.
fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

/// Collapse runs of whitespace into single spaces, trim.
fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Deterministic token estimate: chars / CHARS_PER_TOKEN, rounded up.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(CHARS_PER_TOKEN)
}

/// Chunk one source's HTML into deterministic chunks.
///
/// Sections are merged in order until the running token count reaches
/// `TARGET_TOKENS`; a chunk never splits a section unless that single section
/// already exceeds the target (then it is split by token windows). Each new
/// chunk carries the last `OVERLAP_TOKENS` of the previous chunk's text as
/// overlap, and records the heading of its first section.
/// Split a text corpus document (Java sources joined by
/// `===== FILE: <relpath> =====` marker lines) into sections: one per file,
/// with the file path as the heading. Text before the first marker is a
/// preamble section with an empty heading (normally absent from
/// ingest-built documents).
pub fn sections_from_text(text: &str) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();
    let mut current_heading = String::new();
    let mut current_text = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(FILE_MARKER) {
            if let Some(path) = rest.strip_suffix(" =====") {
                flush_text_section(&mut sections, &mut current_heading, &mut current_text);
                current_heading = path.trim().to_string();
                continue;
            }
        }
        current_text.push_str(line);
        current_text.push('\n');
    }
    flush_text_section(&mut sections, &mut current_heading, &mut current_text);
    sections
}

/// Push the accumulated text section unless it is empty (mirrors
/// `flush_section`, but preserves code whitespace — collapsing runs of
/// whitespace would destroy indentation in source files).
fn flush_text_section(sections: &mut Vec<Section>, heading: &mut String, text: &mut String) {
    let raw = std::mem::take(text);
    if !raw.trim().is_empty() || !heading.is_empty() {
        sections.push(Section {
            heading: std::mem::take(heading),
            text: raw,
        });
    } else {
        heading.clear();
    }
}

/// Chunk a text corpus document (see `sections_from_text`) with the source's
/// provenance metadata. Deterministic: same text + source → identical chunks.
pub fn chunk_text(text: &str, source: &Source) -> Vec<Chunk> {
    chunk_sections(&sections_from_text(text), source)
}

pub fn chunk_source(html: &str, source: &Source) -> Vec<Chunk> {
    chunk_sections(&sections_from_html(html), source)
}

/// Merge and split `sections` into ~500-token chunks with overlap; every
/// chunk carries the source's provenance. Shared by the HTML and text paths.
fn chunk_sections(sections: &[Section], source: &Source) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut pending = String::new();
    let mut pending_heading = String::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for section in sections {
        if section.text.trim().is_empty() {
            // Empty source file (e.g. several in the tlaplus checkout):
            // no content, no chunk.
            continue;
        }
        let text = section.text.clone();
        if estimate_tokens(&text) > TARGET_TOKENS {
            // Oversized single section: emit what's pending, then split the
            // section by token windows (no overlap within a split section).
            if !pending.is_empty() {
                push_unique(
                    &mut chunks,
                    &mut seen,
                    make_chunk(source, &pending_heading, &pending),
                );
                pending.clear();
            }
            let mut start = 0;
            let chars: Vec<char> = text.chars().collect();
            while start < chars.len() {
                let end = (start + TARGET_TOKENS * CHARS_PER_TOKEN).min(chars.len());
                let piece: String = chars[start..end].iter().collect();
                push_unique(
                    &mut chunks,
                    &mut seen,
                    make_chunk(source, &section.heading, &piece),
                );
                start = end;
            }
            pending_heading.clear();
            continue;
        }
        if pending.is_empty() {
            pending_heading = section.heading.clone();
        }
        let merged_tokens = estimate_tokens(&pending) + estimate_tokens(&section.text);
        if !pending.is_empty() && merged_tokens > TARGET_TOKENS {
            let overlap = overlap_of(&pending);
            push_unique(
                &mut chunks,
                &mut seen,
                make_chunk(source, &pending_heading, &pending),
            );
            pending = overlap;
            pending_heading = section.heading.clone();
            pending.push(' ');
            pending.push_str(&text);
        } else {
            if !pending.is_empty() {
                pending.push(' ');
            }
            pending.push_str(&text);
        }
    }
    if !pending.is_empty() {
        push_unique(
            &mut chunks,
            &mut seen,
            make_chunk(source, &pending_heading, &pending),
        );
    }
    chunks
}

/// Push a chunk unless its content-addressed id was already emitted —
/// identical text in different files/sections collapses to one chunk.
fn push_unique(
    chunks: &mut Vec<Chunk>,
    seen: &mut std::collections::HashSet<String>,
    chunk: Chunk,
) {
    if seen.insert(chunk.id.clone()) {
        chunks.push(chunk);
    }
}

/// The last `OVERLAP_TOKENS` of `text` (as estimated tokens), for overlap.
fn overlap_of(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let keep = OVERLAP_TOKENS * CHARS_PER_TOKEN;
    if chars.len() <= keep {
        text.to_string()
    } else {
        chars[chars.len() - keep..].iter().collect()
    }
}

fn make_chunk(source: &Source, heading: &str, text: &str) -> Chunk {
    let text = text.trim().to_string();
    Chunk {
        id: sha256_hex(text.as_bytes()),
        source: source.name.clone(),
        license: source.license.clone(),
        tier: source.tier,
        heading: heading.to_string(),
        tokens: estimate_tokens(&text),
        text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Source {
        Source {
            name: "rust-book".into(),
            license: "MIT OR Apache-2.0".into(),
            url: "https://example.invalid/rust-book".into(),
            hash: None,
            tier: TrustTier::PinnedSource,
            format: crate::manifest::SourceFormat::Html,
        }
    }

    #[test]
    fn estimate_tokens_rounds_up() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn strips_tags_and_decodes_entities() {
        assert_eq!(strip_tags("<p>a &amp; b</p><br><em>c</em>"), "a & bc");
    }

    #[test]
    fn sections_split_at_headings() {
        let html = "<h1>Intro</h1><p>preamble body</p><h2>Deep Dive</h2><p>details</p>";
        let sections = sections_from_html(html);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading, "Intro");
        assert!(sections[0].text.contains("preamble body"));
        assert_eq!(sections[1].heading, "Deep Dive");
        assert!(sections[1].text.contains("details"));
    }

    #[test]
    fn preamble_before_first_heading_has_empty_heading() {
        let html = "<p>lone preamble</p><h1>Main</h1><p>body</p>";
        let sections = sections_from_html(html);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading, "");
        assert!(sections[0].text.contains("lone preamble"));
    }

    #[test]
    fn chunking_is_deterministic() {
        let html = "<h1>Ownership</h1><p>The borrow checker is central.</p>".repeat(40);
        let a = chunk_source(&html, &source());
        let b = chunk_source(&html, &source());
        let ids_a: Vec<&str> = a.iter().map(|c| c.id.as_str()).collect();
        let ids_b: Vec<&str> = b.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(a, b);
        assert_eq!(ids_a, ids_b);
    }

    #[test]
    fn ids_are_content_addressed() {
        let html = "<h1>X</h1><p>stable text</p>";
        let mut s = source();
        let c1 = chunk_source(html, &s);
        s.name = "other-source".into();
        let c2 = chunk_source(html, &s);
        // Same content → same id regardless of source name.
        assert_eq!(c1[0].id, c2[0].id);
        // Different content → different id.
        let c3 = chunk_source("<h1>X</h1><p>different text</p>", &source());
        assert_ne!(c1[0].id, c3[0].id);
    }

    #[test]
    fn chunks_carry_provenance() {
        let s = source();
        let chunks = chunk_source("<h1>T</h1><p>text</p>", &s);
        assert_eq!(chunks[0].source, "rust-book");
        assert_eq!(chunks[0].license, "MIT OR Apache-2.0");
        assert_eq!(chunks[0].tier, TrustTier::PinnedSource);
        assert_eq!(chunks[0].heading, "T");
    }

    #[test]
    fn chunks_are_bounded_around_target() {
        // 30 sections of ~40 tokens each → each chunk ~500 tokens, not
        // unbounded, and never merging everything into one.
        let html = (0..30)
            .map(|i| format!("<h2>S{i}</h2><p>{} needle words</p>", "word ".repeat(120)))
            .collect::<String>();
        let chunks = chunk_source(&html, &source());
        assert!(
            chunks.len() >= 3,
            "expected several chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.tokens <= TARGET_TOKENS * 2,
                "chunk {} has {} tokens (over budget)",
                c.id,
                c.tokens
            );
        }
    }

    #[test]
    fn adjacent_chunks_overlap() {
        // Enough small sections to produce >= 2 chunks; the second chunk must
        // start with the tail of the first (overlap window).
        let html = (0..20)
            .map(|i| {
                format!(
                    "<h3>S{i}</h3><p>{} overlap test words</p>",
                    "word ".repeat(100)
                )
            })
            .collect::<String>();
        let chunks = chunk_source(&html, &source());
        assert!(chunks.len() >= 2, "expected overlap to require >= 2 chunks");
        let first_tail: String = chunks[0]
            .text
            .chars()
            .skip(chunks[0].text.chars().count() - OVERLAP_TOKENS * CHARS_PER_TOKEN)
            .collect();
        assert!(
            chunks[1].text.starts_with(first_tail.trim()),
            "chunk 2 should begin with chunk 1's overlap tail"
        );
    }

    #[test]
    fn sections_do_not_leak_accumulated_text() {
        // Regression: flush_section used to shadow its `text` param, so the
        // caller's buffer was never cleared and every section accumulated all
        // previous text (O(n^2) growth). Each section must contain only its
        // own content.
        let html = "<h1>First</h1><p>UNIQUE_ALPHA content here</p>\
            <h2>Second</h2><p>UNIQUE_BETA content here</p>\
            <h3>Third</h3><p>UNIQUE_GAMMA content here</p>";
        let sections = sections_from_html(html);
        assert_eq!(sections.len(), 3);
        assert!(sections[0].text.contains("UNIQUE_ALPHA"));
        assert!(!sections[0].text.contains("UNIQUE_BETA"));
        assert!(sections[1].text.contains("UNIQUE_BETA"));
        assert!(!sections[1].text.contains("UNIQUE_ALPHA"));
        assert!(!sections[1].text.contains("UNIQUE_GAMMA"));
        assert!(sections[2].text.contains("UNIQUE_GAMMA"));
    }

    #[test]
    fn large_document_chunk_count_is_bounded() {
        // A big doc must yield roughly chars/target chunks, not a multiple of
        // the document size (the accumulation bug amplified ~190x).
        let body = format!(
            "<h1>T</h1>{}{}",
            "<p>word </p>".repeat(0),
            "word ".repeat(TARGET_TOKENS * CHARS_PER_TOKEN * 20)
        );
        let chunks = chunk_source(&body, &source());
        let upper = 3 * (body.len() / (TARGET_TOKENS * CHARS_PER_TOKEN)) + 5;
        assert!(
            chunks.len() <= upper,
            "chunk count {} exceeds bounded upper {}",
            chunks.len(),
            upper
        );
    }

    #[test]
    fn oversized_section_is_split_by_windows() {
        let html = format!(
            "<h1>Huge</h1><p>{}</p>",
            "word ".repeat(TARGET_TOKENS * CHARS_PER_TOKEN + 50)
        );
        let chunks = chunk_source(&html, &source());
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].heading, "Huge");
    }

    // --- RAG-6: text corpus documents (FILE markers) ---

    #[test]
    fn sections_from_text_splits_on_file_markers() {
        let text = "===== FILE: a/A.java =====\nclass A {}\n===== FILE: b/B.tla =====\nMODULE B\n";
        let sections = sections_from_text(text);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading, "a/A.java");
        assert!(sections[0].text.contains("class A {}"));
        assert_eq!(sections[1].heading, "b/B.tla");
        assert!(sections[1].text.contains("MODULE B"));
    }

    #[test]
    fn sections_from_text_preamble_has_empty_heading() {
        let text = "stray preamble\n===== FILE: a.java =====\ncode\n";
        let sections = sections_from_text(text);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading, "");
        assert!(sections[0].text.contains("stray preamble"));
        assert_eq!(sections[1].heading, "a.java");
    }

    #[test]
    fn chunk_text_carries_provenance_and_file_headings() {
        let text = "===== FILE: tla2sany/OpApplNode.java =====\npackage tla2sany;\nclass OpApplNode { /* CHOOSE application */ }\n";
        let chunks = chunk_text(text, &source());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].source, "rust-book"); // source() helper's name
        assert_eq!(chunks[0].license, "MIT OR Apache-2.0");
        assert_eq!(chunks[0].tier, TrustTier::PinnedSource);
        assert_eq!(chunks[0].heading, "tla2sany/OpApplNode.java");
        assert!(chunks[0].text.contains("class OpApplNode"));
    }

    #[test]
    fn chunk_text_is_deterministic() {
        let text = "===== FILE: a.java =====\nAAAA\n===== FILE: b.java =====\nBBBB\n";
        let s1 = source();
        let s2 = source();
        let c1 = chunk_text(text, &s1);
        let c2 = chunk_text(text, &s2);
        assert_eq!(c1.len(), c2.len());
        assert_eq!(c1[0].id, c2[0].id);
        assert_eq!(c1[0].text, c2[0].text);
    }
    #[test]
    fn chunk_text_skips_empty_files() {
        // An empty source file (marker with no content) must not become a
        // chunk — the tlaplus checkout has several (PcalTLAGen.java etc).
        let text = "===== FILE: a.java =====\ncode\n===== FILE: empty.java =====\n===== FILE: b.java =====\nmore\n";
        let chunks = chunk_text(text, &source());
        assert!(!chunks.is_empty());
        // The empty file must not produce a chunk, and no chunk is empty
        // (small sections may merge, so assert the invariants, not a count).
        assert!(chunks.iter().all(|c| !c.text.trim().is_empty()));
        assert!(chunks.iter().all(|c| c.heading != "empty.java"));
    }

    #[test]
    fn chunk_text_dedups_identical_file_content() {
        // Two files with byte-identical, oversized content (e.g. the real
        // PlusCal.tla / PlusCal2.tla copies) produce identical split pieces
        // whose content-addressed ids must collapse to unique chunks.
        let body = "LINE ".repeat(600); // 3000 chars ≈ 750 tokens → oversized split
        let text = format!(
            "===== FILE: PlusCal.tla =====\n{body}\n===== FILE: PlusCal2.tla =====\n{body}\n"
        );
        let chunks = chunk_text(&text, &source());
        assert!(!chunks.is_empty());
        let ids: std::collections::HashSet<&String> = chunks.iter().map(|c| &c.id).collect();
        assert_eq!(
            ids.len(),
            chunks.len(),
            "duplicate content must not produce duplicate chunk ids"
        );
        assert!(chunks.iter().any(|c| c.heading == "PlusCal.tla"));
    }
}

/// Persisted chunks file: the chunking output consumed by `embed` and the
/// vector store (RAG-4).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunksFile {
    /// Schema version of this chunks-file format.
    pub version: u32,
    /// The chunks, in deterministic order (source, then section order).
    pub chunks: Vec<Chunk>,
}

impl ChunksFile {
    /// Load a chunks file from JSON.
    pub fn load(path: &std::path::Path) -> Result<Self, ChunksError> {
        let text = std::fs::read_to_string(path).map_err(ChunksError::Io)?;
        serde_json::from_str(&text).map_err(ChunksError::Json)
    }

    /// Persist as pretty JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<(), ChunksError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ChunksError::Io)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(ChunksError::Json)?;
        std::fs::write(path, text).map_err(ChunksError::Io)
    }
}

/// Errors produced while loading/saving chunks files.
#[derive(Debug)]
pub enum ChunksError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for ChunksError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunksError::Io(e) => write!(f, "chunks I/O error: {e}"),
            ChunksError::Json(e) => write!(f, "chunks parse error: {e}"),
        }
    }
}

impl std::error::Error for ChunksError {}
