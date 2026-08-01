//! Vector store (RAG-4): chunk + embedding + metadata index with cosine
//! retrieval.
//!
//! The store is assembled from the two deterministic index artifacts —
//! `chunks.json` (RAG-3 chunker) and `embeddings.json` (RAG-3 embedder) — so
//! the same manifest + corpus rebuilds the identical store (determinism is
//! inherited from the artifact chain). At this scale (a few thousand chunks)
//! an in-memory cosine scan needs no vector server; the RAG-4 issue explicitly
//! permits "a numpy array + cosine at this scale", and the JSON + cosine form
//! satisfies every RAG-4 acceptance criterion with zero new dependencies.
//!
//! Retrieval is fully offline after ingestion: the index artifacts are local,
//! and embedding the query talks only to the local Ollama server (never the
//! network).

use crate::chunk::{Chunk, ChunksError, ChunksFile};
use crate::embed::{EmbedError, EmbeddingsFile};
use crate::manifest::TrustTier;
use std::collections::BTreeMap;
use std::path::Path;

/// Default number of hits returned by `search`.
pub const DEFAULT_TOP_K: usize = 10;

/// One ranked retrieval hit: a chunk with its cosine score.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The retrieved chunk (carries source, license, tier, heading, text).
    pub chunk: Chunk,
    /// Cosine similarity between the query and the chunk embedding, in [-1, 1].
    pub score: f64,
}

/// The in-memory vector store: chunk metadata + embeddings keyed by chunk id.
#[derive(Debug, Clone)]
pub struct VectorStore {
    /// Embedder model that produced the vectors.
    pub model: String,
    /// Vector dimensionality (768 for nomic-embed-text).
    pub dim: usize,
    /// chunk id → chunk (metadata + text).
    chunks: BTreeMap<String, Chunk>,
    /// chunk id → embedding (as persisted).
    vectors: BTreeMap<String, Vec<f32>>,
}

impl VectorStore {
    /// Assemble the store from `INDEX_DIR/chunks.json` + `embeddings.json`.
    ///
    /// Validates consistency: every chunk has an embedding and vice versa,
    /// and every vector has the declared dimension. A broken artifact pair is
    /// an error, never a silently partial store.
    pub fn load(index_dir: &Path) -> Result<Self, VectorError> {
        let chunks = ChunksFile::load(&index_dir.join("chunks.json"))?;
        let embeddings = EmbeddingsFile::load(&index_dir.join("embeddings.json"))?;

        let mut chunk_map = BTreeMap::new();
        for c in &chunks.chunks {
            chunk_map.insert(c.id.clone(), c.clone());
        }
        for id in embeddings.vectors.keys() {
            if !chunk_map.contains_key(id) {
                return Err(VectorError::Inconsistent(format!(
                    "embeddings reference chunk '{id}' that is not in chunks.json"
                )));
            }
        }
        for id in chunk_map.keys() {
            if !embeddings.vectors.contains_key(id) {
                return Err(VectorError::Inconsistent(format!(
                    "chunk '{id}' has no embedding in embeddings.json"
                )));
            }
        }
        for (id, v) in &embeddings.vectors {
            if v.len() != embeddings.dim {
                return Err(VectorError::DimensionMismatch {
                    chunk: id.clone(),
                    expected: embeddings.dim,
                    got: v.len(),
                });
            }
        }

        Ok(VectorStore {
            model: embeddings.model.clone(),
            dim: embeddings.dim,
            chunks: chunk_map,
            vectors: embeddings.vectors,
        })
    }

    /// Number of chunks in the store.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// True when the store has no chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Rank chunks by cosine similarity to `query`, filtered by `tier`.
    ///
    /// Deterministic: ties are broken by chunk id (ascending), so the same
    /// store + query always yields the same ranking. Returns at most `k` hits.
    /// A query whose dimension differs from `dim` yields no hits (a malformed
    /// query must not silently scan with a partial dot product).
    pub fn search(&self, query: &[f32], k: usize, tier: Option<TrustTier>) -> Vec<Hit> {
        if k == 0 || query.len() != self.dim {
            return Vec::new();
        }
        let q_norm = norm(query);
        let mut scored: Vec<(&Chunk, f64)> = Vec::with_capacity(self.vectors.len());
        for (id, v) in &self.vectors {
            let Some(chunk) = self.chunks.get(id) else {
                continue;
            };
            if let Some(t) = tier {
                if chunk.tier != t {
                    continue;
                }
            }
            scored.push((chunk, cosine(v, query, q_norm)));
        }
        scored.sort_by(|(a, a_score), (b, b_score)| {
            b_score
                .partial_cmp(a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        scored
            .into_iter()
            .take(k)
            .map(|(c, score)| Hit {
                chunk: c.clone(),
                score,
            })
            .collect()
    }
}

/// Cosine similarity between persisted vector `a` and query `b`.
///
/// The query's norm is passed in so it is computed once per search rather
/// than once per chunk. A zero vector has no direction and scores 0.0.
fn cosine(a: &[f32], b: &[f32], b_norm: f64) -> f64 {
    let a_norm = norm(a);
    if a_norm == 0.0 || b_norm == 0.0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
    }
    dot / (a_norm * b_norm)
}

/// Euclidean norm of `v`.
fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt()
}

/// Errors produced while loading or searching the vector store.
#[derive(Debug)]
pub enum VectorError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Chunks(ChunksError),
    Embed(EmbedError),
    /// The artifacts disagree about which chunks exist or are embedded.
    Inconsistent(String),
    /// A vector's length differs from the declared dimension.
    DimensionMismatch {
        chunk: String,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorError::Io(e) => write!(f, "vector store I/O error: {e}"),
            VectorError::Json(e) => write!(f, "vector store JSON error: {e}"),
            VectorError::Chunks(e) => write!(f, "{e}"),
            VectorError::Embed(e) => write!(f, "{e}"),
            VectorError::Inconsistent(msg) => write!(f, "vector store inconsistent: {msg}"),
            VectorError::DimensionMismatch {
                chunk,
                expected,
                got,
            } => write!(
                f,
                "chunk '{chunk}' embedding has dimension {got}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for VectorError {}

impl From<ChunksError> for VectorError {
    fn from(e: ChunksError) -> Self {
        VectorError::Chunks(e)
    }
}

impl From<EmbedError> for VectorError {
    fn from(e: EmbedError) -> Self {
        VectorError::Embed(e)
    }
}

impl From<std::io::Error> for VectorError {
    fn from(e: std::io::Error) -> Self {
        VectorError::Io(e)
    }
}

impl From<serde_json::Error> for VectorError {
    fn from(e: serde_json::Error) -> Self {
        VectorError::Json(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::TrustTier::{DocsWiki, PinnedSource};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// Unique, self-cleaning temp dir (tests run in parallel).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let p = std::env::temp_dir().join(format!("gs-vec-{tag}-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&p).expect("create temp dir");
            TempDir(p)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn chunk(id: &str, source: &str, tier: TrustTier, text: &str) -> Chunk {
        Chunk {
            id: id.into(),
            source: source.into(),
            license: "MIT".into(),
            tier,
            heading: "H".into(),
            tokens: 1,
            text: text.into(),
        }
    }

    /// Persist a chunks file + embeddings file pair in `dir`.
    fn write_artifacts(dir: &TempDir, chunks: &[Chunk], vectors: &[(&str, Vec<f32>)], dim: usize) {
        let file = ChunksFile {
            version: 1,
            chunks: chunks.to_vec(),
        };
        file.save(&dir.path().join("chunks.json")).unwrap();

        let mut map = BTreeMap::new();
        for (id, v) in vectors {
            map.insert(id.to_string(), v.clone());
        }
        let emb = EmbeddingsFile {
            model: "fake-model".into(),
            dim,
            vectors: map,
        };
        emb.save(&dir.path().join("embeddings.json")).unwrap();
    }

    #[test]
    fn load_joins_chunks_and_embeddings() {
        let dir = TempDir::new("join");
        let chunks = vec![
            chunk("a", "rust-book", PinnedSource, "text a"),
            chunk("b", "rust-reference", PinnedSource, "text b"),
        ];
        write_artifacts(
            &dir,
            &chunks,
            &[("a", vec![1.0, 0.0]), ("b", vec![0.0, 1.0])],
            2,
        );

        let store = VectorStore::load(dir.path()).unwrap();
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
        assert_eq!(store.dim, 2);
        assert_eq!(store.model, "fake-model");
    }

    #[test]
    fn load_rejects_embedding_without_chunk() {
        let dir = TempDir::new("ghost");
        let chunks = vec![chunk("a", "rust-book", PinnedSource, "text a")];
        write_artifacts(&dir, &chunks, &[("a", vec![1.0]), ("ghost", vec![0.5])], 1);

        let err = VectorStore::load(dir.path()).unwrap_err();
        assert!(matches!(err, VectorError::Inconsistent(_)));
    }

    #[test]
    fn load_rejects_chunk_without_embedding() {
        let dir = TempDir::new("orphan");
        let chunks = vec![
            chunk("a", "rust-book", PinnedSource, "text a"),
            chunk("b", "rust-reference", PinnedSource, "text b"),
        ];
        write_artifacts(&dir, &chunks, &[("a", vec![1.0])], 1);

        let err = VectorStore::load(dir.path()).unwrap_err();
        assert!(matches!(err, VectorError::Inconsistent(_)));
    }

    #[test]
    fn load_rejects_dimension_mismatch() {
        let dir = TempDir::new("dim");
        let chunks = vec![chunk("a", "rust-book", PinnedSource, "text a")];
        // Declared dim 2 but the vector has length 3.
        write_artifacts(&dir, &chunks, &[("a", vec![1.0, 0.0, 0.0])], 2);

        let err = VectorStore::load(dir.path()).unwrap_err();
        assert!(matches!(
            err,
            VectorError::DimensionMismatch {
                expected: 2,
                got: 3,
                ..
            }
        ));
    }

    #[test]
    fn cosine_ranks_by_similarity() {
        let dir = TempDir::new("rank");
        let chunks = vec![
            chunk("a", "rust-book", PinnedSource, "axis x"),
            chunk("b", "rust-book", PinnedSource, "axis y"),
            chunk("c", "rust-book", PinnedSource, "axis z"),
        ];
        write_artifacts(
            &dir,
            &chunks,
            &[
                ("a", vec![1.0, 0.0, 0.0]),
                ("b", vec![0.0, 1.0, 0.0]),
                ("c", vec![0.0, 0.0, 1.0]),
            ],
            3,
        );
        let store = VectorStore::load(dir.path()).unwrap();

        let hits = store.search(&[0.0, 1.0, 0.5], 3, None);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].chunk.id, "b");
        assert_eq!(hits[1].chunk.id, "c");
        assert_eq!(hits[2].chunk.id, "a");
        // b: dot = 1, |b| = 1, |q| = sqrt(1.25) → 1 / sqrt(1.25) ≈ 0.8944
        assert!((hits[0].score - (1.0f64 / 1.25f64.sqrt())).abs() < 1e-9);
        // c: dot = 0.5, |c| = 1 → 0.5 / sqrt(1.25) ≈ 0.4472
        assert!((hits[1].score - (0.5f64 / 1.25f64.sqrt())).abs() < 1e-9);
        assert_eq!(hits[2].score, 0.0);
    }

    #[test]
    fn search_respects_top_k() {
        let dir = TempDir::new("topk");
        let chunks = vec![
            chunk("a", "rust-book", PinnedSource, "x"),
            chunk("b", "rust-book", PinnedSource, "y"),
            chunk("c", "rust-book", PinnedSource, "z"),
        ];
        write_artifacts(
            &dir,
            &chunks,
            &[
                ("a", vec![1.0, 0.0]),
                ("b", vec![0.0, 1.0]),
                ("c", vec![0.5, 0.5]),
            ],
            2,
        );
        let store = VectorStore::load(dir.path()).unwrap();

        assert_eq!(store.search(&[1.0, 0.0], 1, None).len(), 1);
        assert_eq!(store.search(&[1.0, 0.0], 2, None).len(), 2);
        assert_eq!(store.search(&[1.0, 0.0], 0, None).len(), 0);
    }

    #[test]
    fn search_tie_breaks_by_chunk_id_deterministically() {
        let dir = TempDir::new("tie");
        let chunks = vec![
            chunk("b-id", "rust-book", PinnedSource, "same vec"),
            chunk("a-id", "rust-book", PinnedSource, "same vec"),
        ];
        write_artifacts(
            &dir,
            &chunks,
            &[("b-id", vec![1.0, 0.0]), ("a-id", vec![1.0, 0.0])],
            2,
        );
        let store = VectorStore::load(dir.path()).unwrap();

        let first = store.search(&[1.0, 0.0], 2, None);
        let second = store.search(&[1.0, 0.0], 2, None);
        assert_eq!(first, second); // deterministic across runs
        assert_eq!(first.len(), 2);
        // Equal scores → ascending chunk id.
        assert_eq!(first[0].chunk.id, "a-id");
        assert_eq!(first[1].chunk.id, "b-id");
        assert_eq!(first[0].score, first[1].score);
    }

    #[test]
    fn search_filters_by_trust_tier() {
        let dir = TempDir::new("tier");
        let chunks = vec![
            chunk("a", "rust-book", PinnedSource, "pinned"),
            chunk("b", "rust-wiki", DocsWiki, "wiki"),
        ];
        write_artifacts(
            &dir,
            &chunks,
            &[("a", vec![1.0, 0.0]), ("b", vec![1.0, 0.0])],
            2,
        );
        let store = VectorStore::load(dir.path()).unwrap();

        let pinned = store.search(&[1.0, 0.0], 10, Some(PinnedSource));
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0].chunk.id, "a");
        assert_eq!(pinned[0].chunk.tier, PinnedSource);

        let wiki = store.search(&[1.0, 0.0], 10, Some(DocsWiki));
        assert_eq!(wiki.len(), 1);
        assert_eq!(wiki[0].chunk.id, "b");

        // No filter → both.
        assert_eq!(store.search(&[1.0, 0.0], 10, None).len(), 2);
    }

    #[test]
    fn zero_query_vector_scores_zero_without_panicking() {
        let dir = TempDir::new("zero");
        let chunks = vec![chunk("a", "rust-book", PinnedSource, "x")];
        write_artifacts(&dir, &chunks, &[("a", vec![1.0, 1.0])], 2);
        let store = VectorStore::load(dir.path()).unwrap();

        let hits = store.search(&[0.0, 0.0], 10, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn wrong_dimension_query_yields_no_hits() {
        let dir = TempDir::new("wrongdim");
        let chunks = vec![chunk("a", "rust-book", PinnedSource, "x")];
        write_artifacts(&dir, &chunks, &[("a", vec![1.0, 1.0])], 2);
        let store = VectorStore::load(dir.path()).unwrap();

        assert!(store.search(&[1.0], 10, None).is_empty());
        assert!(store.search(&[], 10, None).is_empty());
    }

    #[test]
    fn hit_carries_provenance_metadata() {
        let dir = TempDir::new("meta");
        let chunks = vec![chunk("a", "clippy", DocsWiki, "lint docs")];
        write_artifacts(&dir, &chunks, &[("a", vec![1.0, 0.0])], 2);
        let store = VectorStore::load(dir.path()).unwrap();

        let hit = &store.search(&[1.0, 0.0], 1, None)[0];
        assert_eq!(hit.chunk.source, "clippy");
        assert_eq!(hit.chunk.license, "MIT");
        assert_eq!(hit.chunk.tier, DocsWiki);
        assert_eq!(hit.chunk.text, "lint docs");
    }
}
