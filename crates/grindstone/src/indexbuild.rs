//! **Index build** — the deep pipeline orchestrator (RAG-12a).
//!
//! The single library path that owns the whole index walk end-to-end:
//! manifest → corpus files → chunks → embeddings → vector store. Stage
//! functions exist (`chunk_corpus`, `embed_corpus`) so per-stage CLI
//! commands can delegate instead of re-implementing walks, and
//! [`build_index`] composes them into one deterministic pipeline.
//!
//! **Determinism contract**: same manifest + same corpus → identical index
//! artifacts, bit-for-bit. The chunker is deterministic (RAG-9), the
//! embedder is injectable (a deterministic embedder — Ollama or a fake —
//! yields identical vectors for identical chunk text), and the artifacts
//! (`chunks.json`, `embeddings.json`) are persisted as sorted, content-keyed
//! files. This contract is tested at this module's interface.

use std::path::Path;

/// Per-source progress callback: source name and chunk count.
pub type SourceProgress<'a> = dyn FnMut(&str, usize) + 'a;

/// Embedding progress callback: chunks done and total chunks.
pub type EmbedProgress<'a> = dyn FnMut(usize, usize) + 'a;

/// The deterministic walk: load the corpus manifest, chunk every source
/// (through the RAG-9 unified chunker dispatch), and return the assembled
/// `ChunksFile`. `on_source` fires per source for progress reporting.
///
/// Same manifest + corpus → identical `ChunksFile`, bit-for-bit.
pub fn chunk_corpus<'a>(
    corpus_dir: &Path,
    mut on_source: Option<&'a mut SourceProgress<'a>>,
) -> Result<crate::chunk::ChunksFile, IndexBuildError> {
    let manifest = crate::manifest::Manifest::load(&corpus_dir.join("manifest.json"))
        .map_err(IndexBuildError::Manifest)?;

    let mut chunks = Vec::new();
    for source in &manifest.sources {
        let path = corpus_dir.join(source.filename());
        let content = std::fs::read_to_string(&path).map_err(|e| IndexBuildError::Read {
            file: path.display().to_string(),
            detail: e.to_string(),
        })?;
        let doc_chunks = crate::chunk::chunk(&content, source);
        if let Some(cb) = on_source.as_mut() {
            cb(&source.name, doc_chunks.len());
        }
        chunks.extend(doc_chunks);
    }

    Ok(crate::chunk::ChunksFile { version: 1, chunks })
}

/// Embed every chunk, reusing existing vectors for unchanged chunks
/// (content-addressed ids) and pruning stale ones. The reuse-counting lives
/// here — callers never duplicate it. Returns the embeddings file plus the
/// number of vectors reused from the previous build.
pub fn embed_corpus<'a>(
    chunks: &[crate::chunk::Chunk],
    index_dir: &Path,
    embed: &mut crate::embed::Embedder,
    on_progress: Option<&'a mut EmbedProgress<'a>>,
) -> Result<(crate::embed::EmbeddingsFile, usize), IndexBuildError> {
    let existing = crate::embed::EmbeddingsFile::load(&index_dir.join("embeddings.json")).ok();
    let reused = existing
        .as_ref()
        .map(|e| {
            e.vectors
                .keys()
                .filter(|id| chunks.iter().any(|c| &c.id == *id))
                .count()
        })
        .unwrap_or(0);

    let embeddings = crate::embed::embed_chunks_incremental(
        chunks,
        crate::embed::DEFAULT_EMBED_MODEL,
        crate::embed::DEFAULT_BATCH_SIZE,
        embed,
        existing.as_ref(),
        on_progress,
    )
    .map_err(IndexBuildError::Embed)?;

    Ok((embeddings, reused))
}

/// Load an assembled index for querying: the vector store plus the sparse
/// BM25 index, both derived from the same `chunks.json`. The one library
/// path for consuming the index — callers never reload chunks and rebuild
/// the sparse index by hand.
pub fn load_index(
    index_dir: &Path,
) -> Result<(crate::vector::VectorStore, crate::bm25::Bm25Index), IndexBuildError> {
    let store = crate::vector::VectorStore::load(index_dir).map_err(IndexBuildError::Vector)?;
    let chunks = crate::chunk::ChunksFile::load(&index_dir.join("chunks.json"))
        .map_err(IndexBuildError::Chunks)?;
    let bm25 = crate::bm25::Bm25Index::build(&chunks.chunks);
    Ok((store, bm25))
}

/// Summary of one full index build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexBuildReport {
    /// Total chunks produced from the corpus.
    pub chunks: usize,
    /// Total vectors in the persisted embeddings file.
    pub vectors: usize,
    /// Vectors reused from a previous build (unchanged chunks).
    pub reused: usize,
    /// Vector dimensionality.
    pub dim: usize,
}

/// The full manifest → chunk → embed → store walk, end to end.
///
/// Persists `INDEX_DIR/chunks.json` and `INDEX_DIR/embeddings.json` (the
/// vector store derives from both on load), reusing unchanged chunk
/// embeddings incrementally. Deterministic: same manifest + same corpus +
/// same embedder → identical artifacts, bit-for-bit.
pub fn build_index<'a>(
    corpus_dir: &Path,
    index_dir: &Path,
    embed: &mut crate::embed::Embedder,
    on_source: Option<&'a mut SourceProgress<'a>>,
    on_embed_progress: Option<&'a mut EmbedProgress<'a>>,
) -> Result<IndexBuildReport, IndexBuildError> {
    let chunks_file = chunk_corpus(corpus_dir, on_source)?;

    let chunks_path = index_dir.join("chunks.json");
    chunks_file
        .save(&chunks_path)
        .map_err(IndexBuildError::Chunks)?;

    let (embeddings, reused) =
        embed_corpus(&chunks_file.chunks, index_dir, embed, on_embed_progress)?;

    let embeddings_path = index_dir.join("embeddings.json");
    embeddings
        .save(&embeddings_path)
        .map_err(IndexBuildError::Embed)?;

    Ok(IndexBuildReport {
        chunks: chunks_file.chunks.len(),
        vectors: embeddings.vectors.len(),
        reused,
        dim: embeddings.dim,
    })
}

/// Errors produced by the index build walk.
#[derive(Debug)]
pub enum IndexBuildError {
    /// The corpus manifest could not be loaded or parsed.
    Manifest(crate::manifest::ManifestError),
    /// A corpus file could not be read.
    Read { file: String, detail: String },
    /// A chunks artifact could not be loaded or persisted.
    Chunks(crate::chunk::ChunksError),
    /// Embedding failed (e.g. Ollama unreachable).
    Embed(crate::embed::EmbedError),
    /// The vector store could not be loaded.
    Vector(crate::vector::VectorError),
    /// Generic filesystem I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for IndexBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexBuildError::Manifest(e) => write!(f, "manifest error: {e}"),
            IndexBuildError::Read { file, detail } => {
                write!(f, "cannot read {file}: {detail}")
            }
            IndexBuildError::Chunks(e) => write!(f, "chunks error: {e}"),
            IndexBuildError::Embed(e) => write!(f, "embed error: {e}"),
            IndexBuildError::Vector(e) => write!(f, "vector store error: {e}"),
            IndexBuildError::Io(e) => write!(f, "index build I/O error: {e}"),
        }
    }
}

impl std::error::Error for IndexBuildError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny two-source fixture corpus (one HTML doc, one text doc) with a
    /// matching manifest, written into `dir` (created fresh).
    fn write_fixture_corpus(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            br#"{"version":1,"sources":[
                {"name":"rust-book","license":"MIT OR Apache-2.0","url":"https://doc.rust-lang.org/book/print.html","hash":"abc","tier":"pinned-source","format":"html"},
                {"name":"tla-model","license":"MIT","url":"file:///tmp/tla","hash":"def","tier":"pinned-source","format":"text"}
            ]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("rust-book.html"),
            b"<html><body><h1>Ownership</h1><p>The borrow checker is the core of ownership.</p><h1>Lifetimes</h1><p>Every reference has a lifetime.</p></body></html>",
        )
        .unwrap();
        std::fs::write(
            dir.join("tla-model.txt"),
            b"MODULE Model\nVARIABLES x\nInit == x = 0\nNext == x' = x + 1\n",
        )
        .unwrap();
    }

    /// Deterministic fake embedder: identical text → identical vector, so a
    /// rebuild produces byte-identical artifacts (the determinism contract).
    fn fake_embedder() -> Box<crate::embed::Embedder<'static>> {
        Box::new(|inputs: &[String]| {
            Ok(inputs
                .iter()
                .map(|t| {
                    let mut h = 0.0_f32;
                    for b in t.as_bytes() {
                        h = h * 1.01 + *b as f32;
                    }
                    vec![h, t.len() as f32, t.chars().count() as f32]
                })
                .collect())
        })
    }

    #[test]
    fn chunk_corpus_walks_manifest_sources_in_order() {
        let dir = std::env::temp_dir().join(format!("gs-ib-chunk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_fixture_corpus(&dir);

        let mut order = Vec::new();
        let file = chunk_corpus(
            &dir,
            Some(&mut |name: &str, n: usize| {
                order.push((name.to_string(), n));
            }),
        )
        .unwrap();

        // Sources walked in manifest order, each producing ≥1 chunk.
        assert_eq!(order.len(), 2);
        assert_eq!(order[0].0, "rust-book");
        assert!(order[0].1 > 0);
        assert_eq!(order[1].0, "tla-model");
        assert!(order[1].1 > 0);
        assert_eq!(
            file.chunks.len(),
            order.iter().map(|(_, n)| n).sum::<usize>()
        );
        // Every chunk carries its source's provenance.
        assert!(file
            .chunks
            .iter()
            .all(|c| c.source == "rust-book" || c.source == "tla-model"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn build_index_end_to_end_and_queries_back() {
        let dir = std::env::temp_dir().join(format!("gs-ib-e2e-{}", std::process::id()));
        let index = dir.join("index");
        let _ = std::fs::remove_dir_all(&dir);
        write_fixture_corpus(&dir);

        let mut call = fake_embedder();
        let report = build_index(&dir, &index, call.as_mut(), None, None).unwrap();

        assert_eq!(report.chunks, report.vectors); // nothing reused on first build
        assert_eq!(report.reused, 0);
        assert!(report.chunks >= 2);
        assert_eq!(report.dim, 3);

        // The store is assembled from the persisted artifacts and queries
        // back through the library's load path.
        let store = crate::vector::VectorStore::load(&index).unwrap();
        assert_eq!(store.len(), report.chunks);
        let hits = store.search(&[1.0, 1.0, 1.0], 5, None);
        assert!(!hits.is_empty());
        assert!(hits
            .iter()
            .any(|h| h.chunk.source == "rust-book" || h.chunk.source == "tla-model"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn build_index_is_deterministic_bit_for_bit() {
        let dir = std::env::temp_dir().join(format!("gs-ib-det-{}", std::process::id()));
        let index_a = dir.join("index-a");
        let index_b = dir.join("index-b");
        let _ = std::fs::remove_dir_all(&dir);
        write_fixture_corpus(&dir);

        let mut call_a = fake_embedder();
        build_index(&dir, &index_a, call_a.as_mut(), None, None).unwrap();
        let mut call_b = fake_embedder();
        build_index(&dir, &index_b, call_b.as_mut(), None, None).unwrap();

        // The determinism contract at this module's interface: same manifest
        // + same corpus → identical index artifacts, bit-for-bit.
        let chunks_a = std::fs::read(index_a.join("chunks.json")).unwrap();
        let chunks_b = std::fs::read(index_b.join("chunks.json")).unwrap();
        assert_eq!(chunks_a, chunks_b);
        let emb_a = std::fs::read(index_a.join("embeddings.json")).unwrap();
        let emb_b = std::fs::read(index_b.join("embeddings.json")).unwrap();
        assert_eq!(emb_a, emb_b);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn embed_corpus_reuses_unchanged_chunks() {
        let dir = std::env::temp_dir().join(format!("gs-ib-reuse-{}", std::process::id()));
        let index = dir.join("index");
        let _ = std::fs::remove_dir_all(&dir);
        write_fixture_corpus(&dir);

        // First build embeds everything.
        let mut call = fake_embedder();
        let first = build_index(&dir, &index, call.as_mut(), None, None).unwrap();
        assert_eq!(first.reused, 0);

        // Second build with an embedder that refuses to embed: every chunk
        // must be reused from the previous artifacts, nothing re-embedded.
        let mut panicky = Box::new(
            |_: &[String]| -> Result<Vec<Vec<f32>>, crate::embed::EmbedError> {
                panic!("re-embedding unchanged chunks; reuse is broken")
            },
        ) as Box<crate::embed::Embedder<'static>>;
        let second = build_index(&dir, &index, panicky.as_mut(), None, None).unwrap();
        assert_eq!(second.reused, second.chunks);
        assert_eq!(second.vectors, first.vectors);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_index_returns_store_and_sparse_index() {
        let dir = std::env::temp_dir().join(format!("gs-ib-load-{}", std::process::id()));
        let index = dir.join("index");
        let _ = std::fs::remove_dir_all(&dir);
        write_fixture_corpus(&dir);

        let mut call = fake_embedder();
        build_index(&dir, &index, call.as_mut(), None, None).unwrap();

        let (store, bm25) = load_index(&index).unwrap();
        assert_eq!(store.len(), bm25.len());
        let hits = crate::hybrid::hybrid_search(
            &store,
            &bm25,
            &[1.0, 1.0, 1.0],
            "borrow checker",
            5,
            None,
        );
        assert!(!hits.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn build_index_reports_missing_manifest() {
        let dir = std::env::temp_dir().join(format!("gs-ib-err-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut call = fake_embedder();
        let err = build_index(&dir, &dir.join("index"), call.as_mut(), None, None).unwrap_err();
        assert!(matches!(err, IndexBuildError::Manifest(_)));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
