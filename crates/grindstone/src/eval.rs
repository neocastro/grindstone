//! Evaluation harness (RAG-2): needle queries + recall@k scorer.
//!
//! The measurement instrument for the whole retrieval ladder. A committed
//! eval set (`eval/evalset.json`) pins needle queries with expected-hit
//! document ids; `gs eval` runs the current retrieval strategy over it and
//! reports recall@k (k=5, k=10) per query and overall. The full-text
//! baseline's score is persisted to `eval/results/` so every later strategy
//! (cosine, hybrid, rerank) has a number to beat on the same eval set.
//!
//! Expected hits are document ids (manifest `Source.name`, e.g. `rust-book`,
//! or a future corpus id such as `tlaplus`) — never file paths. Queries whose
//! expected docs are not yet in the corpus score 0 until that corpus lands.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A needle query with the document ids that must be retrievable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalQuery {
    /// Stable query id, e.g. `rust-borrow-checker`.
    pub id: String,
    /// The needle: the string a retrieval strategy should return hits for.
    pub query: String,
    /// Document ids that genuinely answer the query (curated, non-empty).
    pub expected: Vec<String>,
}

/// The committed eval set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalSet {
    /// Schema version of this eval-set format.
    pub version: u32,
    /// The needle queries, in eval order.
    pub queries: Vec<EvalQuery>,
}

impl EvalSet {
    /// Load an eval set from a JSON file.
    pub fn load(path: &Path) -> Result<Self, EvalError> {
        let text = std::fs::read_to_string(path).map_err(EvalError::Io)?;
        serde_json::from_str(&text).map_err(EvalError::Json)
    }

    /// Validate invariants: unique query ids, non-empty expected sets.
    /// Returns a list of validation problems (empty when valid).
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for q in &self.queries {
            if !seen.insert(q.id.as_str()) {
                problems.push(format!("duplicate query id: {}", q.id));
            }
            if q.query.trim().is_empty() {
                problems.push(format!("query {} has an empty needle", q.id));
            }
            if q.expected.is_empty() {
                problems.push(format!("query {} has an empty expected set", q.id));
            }
        }
        problems
    }
}

/// recall@k for one query: how much of the expected set appears in the top-k
/// hits. Returns 0.0 when `expected` is empty (unscorable).
pub fn recall_at_k(hits: &[String], expected: &[String], k: usize) -> f64 {
    if expected.is_empty() {
        return 0.0;
    }
    let in_top_k = hits
        .iter()
        .take(k)
        .collect::<std::collections::HashSet<_>>();
    let found = expected.iter().filter(|e| in_top_k.contains(e)).count();
    found as f64 / expected.len() as f64
}

/// Per-query eval score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryScore {
    pub id: String,
    pub query: String,
    pub expected: Vec<String>,
    /// Ranked doc ids returned by the strategy (top-10 kept).
    pub hits: Vec<String>,
    pub recall_5: f64,
    pub recall_10: f64,
}

/// Full eval run result, persisted for later strategies to beat.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalResult {
    /// Strategy name, e.g. `fulltext`.
    pub strategy: String,
    /// sha256 of the corpus manifest this run measured against
    /// (deterministic; no wall-clock timestamps — reproducibility).
    pub corpus_hash: String,
    pub queries: Vec<QueryScore>,
    pub overall_recall_5: f64,
    pub overall_recall_10: f64,
}

/// A retrieval strategy: query → ranked document ids.
pub type Strategy<'a> = dyn FnMut(&str) -> Result<Vec<String>, EvalError> + 'a;

/// Run `strategy` over every query in `set` and score recall@5 and recall@10.
pub fn run_eval(
    strategy_name: &str,
    strategy: &mut Strategy,
    set: &EvalSet,
) -> Result<EvalResult, EvalError> {
    let mut scores = Vec::with_capacity(set.queries.len());
    for q in &set.queries {
        let hits = strategy(&q.query)?;
        let hits: Vec<String> = hits.into_iter().take(10).collect();
        scores.push(QueryScore {
            id: q.id.clone(),
            query: q.query.clone(),
            expected: q.expected.clone(),
            recall_5: recall_at_k(&hits, &q.expected, 5),
            recall_10: recall_at_k(&hits, &q.expected, 10),
            hits,
        });
    }
    Ok(EvalResult {
        strategy: strategy_name.to_string(),
        corpus_hash: String::new(),
        overall_recall_5: mean(&scores, |s| s.recall_5),
        overall_recall_10: mean(&scores, |s| s.recall_10),
        queries: scores,
    })
}

/// Mean of `f` over the per-query scores (0.0 for an empty set).
fn mean(scores: &[QueryScore], f: impl Fn(&QueryScore) -> f64) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().map(f).sum::<f64>() / scores.len() as f64
}

impl EvalResult {
    /// Load a persisted eval result from JSON.
    pub fn load(path: &Path) -> Result<Self, EvalError> {
        let text = std::fs::read_to_string(path).map_err(EvalError::Io)?;
        serde_json::from_str(&text).map_err(EvalError::Json)
    }
}

/// Persist an eval result as pretty JSON.
pub fn save_result(path: &Path, result: &EvalResult) -> Result<(), EvalError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(EvalError::Io)?;
    }
    let text = serde_json::to_string_pretty(result).map_err(EvalError::Json)?;
    std::fs::write(path, text).map_err(EvalError::Io)
}

/// Full-text strategy adapter: search `corpus_dir` and map hits to doc ids
/// (file stem, e.g. `rust-book.html` → `rust-book`).
pub fn fulltext_doc_ids(corpus_dir: &Path, query: &str) -> Result<Vec<String>, EvalError> {
    let hits = crate::fulltext::search(corpus_dir, query).map_err(EvalError::Fulltext)?;
    Ok(hits
        .into_iter()
        .map(|h| {
            h.file
                .rsplit_once('.')
                .map(|(s, _)| s)
                .unwrap_or(&h.file)
                .to_string()
        })
        .collect())
}

/// Cosine strategy adapter: embed `query` via the injectable embedder, rank
/// chunks through the vector store, and map the top-k hits to doc ids
/// (deduped source names, rank order preserved).
///
/// A document with many matching chunks appears once, at its best position —
/// the same doc-id granularity the full-text baseline is scored on.
pub fn cosine_doc_ids(
    store: &crate::vector::VectorStore,
    embed: &mut crate::embed::Embedder,
    query: &str,
) -> Result<Vec<String>, EvalError> {
    let embeddings = embed(&[query.to_string()]).map_err(EvalError::Embed)?;
    let q = embeddings.first().ok_or_else(|| {
        EvalError::Embed(crate::embed::EmbedError::Http(
            "embedder returned no vectors".into(),
        ))
    })?;
    let hits = store.search(q, crate::vector::DEFAULT_TOP_K, None);
    let mut seen = std::collections::HashSet::new();
    let mut doc_ids = Vec::new();
    for hit in hits {
        if seen.insert(hit.chunk.source.clone()) {
            doc_ids.push(hit.chunk.source);
        }
    }
    Ok(doc_ids)
}

/// Hybrid strategy adapter: fuse BM25 + cosine for `query` (the embedding is
/// passed in — the caller owns the embed call), map the top-k fused chunks to
/// doc ids (deduped source names, rank order preserved).
pub fn hybrid_doc_ids(
    store: &crate::vector::VectorStore,
    bm25: &crate::bm25::Bm25Index,
    query: &str,
    query_embedding: &[f32],
) -> Vec<String> {
    let hits = crate::hybrid::hybrid_search(
        store,
        bm25,
        query_embedding,
        query,
        crate::vector::DEFAULT_TOP_K,
        None,
    );
    let mut seen = std::collections::HashSet::new();
    let mut doc_ids = Vec::new();
    for hit in hits {
        if seen.insert(hit.chunk.source.clone()) {
            doc_ids.push(hit.chunk.source);
        }
    }
    doc_ids
}

/// Errors produced by the eval harness.
#[derive(Debug)]
pub enum EvalError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Fulltext(crate::fulltext::FulltextError),
    Embed(crate::embed::EmbedError),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Io(e) => write!(f, "eval I/O error: {e}"),
            EvalError::Json(e) => write!(f, "eval JSON error: {e}"),
            EvalError::Fulltext(e) => write!(f, "fulltext error: {e}"),
            EvalError::Embed(e) => write!(f, "embed error: {e}"),
        }
    }
}

impl std::error::Error for EvalError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_set() -> EvalSet {
        EvalSet {
            version: 1,
            queries: vec![
                EvalQuery {
                    id: "q1".into(),
                    query: "borrow checker".into(),
                    expected: vec!["rust-book".into(), "rust-reference".into()],
                },
                EvalQuery {
                    id: "q2".into(),
                    query: "propagating errors".into(),
                    expected: vec!["rust-book".into()],
                },
            ],
        }
    }

    #[test]
    fn recall_counts_expected_in_top_k() {
        let hits: Vec<String> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let expected: Vec<String> = ["c", "x"].iter().map(|s| s.to_string()).collect();
        // top-2 = [a,b] → 0/2; top-3 = [a,b,c] → 1/2; top-5 → 1/2 (x never hits)
        assert_eq!(recall_at_k(&hits, &expected, 2), 0.0);
        assert_eq!(recall_at_k(&hits, &expected, 3), 0.5);
        assert_eq!(recall_at_k(&hits, &expected, 5), 0.5);
    }

    #[test]
    fn recall_with_empty_expected_is_zero() {
        let hits: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let expected: Vec<String> = vec![];
        assert_eq!(recall_at_k(&hits, &expected, 5), 0.0);
    }

    #[test]
    fn recall_handles_k_beyond_hit_count() {
        let hits: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let expected: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(recall_at_k(&hits, &expected, 10), 2.0 / 3.0);
        assert_eq!(recall_at_k(&hits, &expected, 2), 2.0 / 3.0);
    }

    #[test]
    fn run_eval_scores_per_query_and_overall() {
        let set = sample_set();
        // Fake strategy: q1 → [rust-book, other], q2 → [unrelated]
        let mut strategy = |q: &str| -> Result<Vec<String>, EvalError> {
            Ok(match q {
                "borrow checker" => vec!["rust-book".into(), "other".into()],
                "propagating errors" => vec!["unrelated".into()],
                _ => vec![],
            })
        };
        let result = run_eval("fake", &mut strategy, &set).unwrap();

        // q1: 1/2 expected in top-5 and top-10 → 0.5
        // q2: 0/1 → 0.0
        assert_eq!(result.queries.len(), 2);
        assert_eq!(result.queries[0].recall_5, 0.5);
        assert_eq!(result.queries[0].recall_10, 0.5);
        assert_eq!(result.queries[1].recall_5, 0.0);
        // overall = mean of per-query recall
        assert_eq!(result.overall_recall_5, 0.25);
        assert_eq!(result.overall_recall_10, 0.25);
        assert_eq!(result.strategy, "fake");
    }

    #[test]
    fn validation_rejects_duplicate_ids_and_empty_expected() {
        let mut set = sample_set();
        assert!(set.validate().is_empty());

        set.queries[1].id = "q1".into();
        let problems = set.validate();
        assert!(problems.iter().any(|p| p.contains("duplicate")));

        set = sample_set();
        set.queries[0].expected.clear();
        assert!(set.validate().iter().any(|p| p.contains("empty")));
    }

    #[test]
    fn committed_eval_set_is_valid_and_covers_both_domains() {
        // The real committed eval set must satisfy RAG-2's acceptance
        // criteria: >= 10 queries, valid, covering TLA+ semantics and Rust
        // idioms (TLA+ expected hits reference the future `tlaplus` doc).
        let json = include_str!("../../../eval/evalset.json");
        let set: EvalSet = serde_json::from_str(json).expect("evalset.json must parse");
        assert!(set.validate().is_empty(), "{:?}", set.validate());
        assert!(set.queries.len() >= 10, "need >= 10 queries");
        let ids: Vec<&str> = set.queries.iter().map(|q| q.id.as_str()).collect();
        assert!(
            ids.iter().any(|id| id.starts_with("rust-")),
            "need rust-id queries"
        );
        assert!(
            ids.iter().any(|id| id.starts_with("tla-")),
            "need tla-id queries"
        );
        assert!(
            set.queries
                .iter()
                .any(|q| q.expected.contains(&"tlaplus".to_string())),
            "TLA+ queries must reference the tlaplus doc id"
        );
    }

    #[test]
    fn result_json_roundtrip() {
        let result = EvalResult {
            strategy: "fulltext".into(),
            corpus_hash: "deadbeef".into(),
            queries: vec![QueryScore {
                id: "q1".into(),
                query: "needle".into(),
                expected: vec!["rust-book".into()],
                hits: vec!["rust-book".into()],
                recall_5: 1.0,
                recall_10: 1.0,
            }],
            overall_recall_5: 1.0,
            overall_recall_10: 1.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: EvalResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back, result);
    }

    // --- integration test against the real fulltext baseline (needs rg) ---

    fn ripgrep_present() -> bool {
        std::process::Command::new("rg")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn fulltext_eval_on_tiny_corpus() {
        if !ripgrep_present() {
            eprintln!("skipping: rg not installed");
            return;
        }
        let dir = std::env::temp_dir().join(format!("gs-eval-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // "rust-book.html" with two needles, "other.html" with one.
        std::fs::write(
            dir.join("rust-book.html"),
            b"borrow checker explained here\npropagating errors with the question mark operator\n",
        )
        .unwrap();
        std::fs::write(dir.join("other.html"), b"nothing relevant\n").unwrap();

        let set = EvalSet {
            version: 1,
            queries: vec![EvalQuery {
                id: "q1".into(),
                query: "borrow checker".into(),
                expected: vec!["rust-book".into()],
            }],
        };
        let mut strategy = |q: &str| fulltext_doc_ids(&dir, q);
        let result = run_eval("fulltext", &mut strategy, &set).unwrap();
        assert_eq!(result.queries[0].hits, vec!["rust-book".to_string()]);
        assert_eq!(result.queries[0].recall_5, 1.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // --- cosine strategy adapter tests (fake embedder + tiny store) ---

    /// Persist a 3-chunk store (2 rust-book chunks + 1 clippy chunk) into `dir`.
    fn write_tiny_store(dir: &std::path::Path) {
        use crate::chunk::{Chunk, ChunksFile};
        use crate::embed::EmbeddingsFile;
        use crate::manifest::TrustTier::PinnedSource;
        use std::collections::BTreeMap;

        let mk = |id: &str, source: &str, text: &str| Chunk {
            id: id.into(),
            source: source.into(),
            license: "MIT".into(),
            tier: PinnedSource,
            heading: "H".into(),
            tokens: 1,
            text: text.into(),
        };
        ChunksFile {
            version: 1,
            chunks: vec![
                mk("r1", "rust-book", "borrow checker"),
                mk("r2", "rust-book", "borrow checker again"),
                mk("c1", "clippy", "restriction lints"),
            ],
        }
        .save(&dir.join("chunks.json"))
        .unwrap();

        let mut vectors = BTreeMap::new();
        vectors.insert("r1".to_string(), vec![1.0f32, 0.0]);
        vectors.insert("r2".to_string(), vec![1.0f32, 0.0]);
        vectors.insert("c1".to_string(), vec![0.0f32, 1.0]);
        EmbeddingsFile {
            model: "fake".into(),
            dim: 2,
            vectors,
        }
        .save(&dir.join("embeddings.json"))
        .unwrap();
    }

    #[test]
    fn cosine_doc_ids_dedups_sources_in_rank_order() {
        let dir = std::env::temp_dir().join(format!("gs-eval-cos-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_tiny_store(&dir);
        let store = crate::vector::VectorStore::load(&dir).unwrap();

        let mut embed = |_: &[String]| -> Result<Vec<Vec<f32>>, crate::embed::EmbedError> {
            Ok(vec![vec![1.0, 0.0]])
        };
        let ids = cosine_doc_ids(&store, &mut embed, "borrow checker").unwrap();
        // Two rust-book chunks tie (both score 1.0); doc ids dedupe in rank
        // order: rust-book first (two chunks), then clippy.
        assert_eq!(ids, vec!["rust-book".to_string(), "clippy".to_string()]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hybrid_doc_ids_promotes_lexical_match_to_first() {
        let dir = std::env::temp_dir().join(format!("gs-eval-hyb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_tiny_store(&dir); // r1/r2 rust-book "borrow checker", c1 clippy "restriction lints"
        let store = crate::vector::VectorStore::load(&dir).unwrap();
        let chunks_file = crate::chunk::ChunksFile::load(&dir.join("chunks.json")).unwrap();
        let bm25 = crate::bm25::Bm25Index::build(&chunks_file.chunks);

        // Embedding points at rust-book (cosine-first is rust-book), but the
        // query text "restriction lints" only matches the clippy chunk — the
        // fusion must promote clippy to the front.
        let ids = hybrid_doc_ids(&store, &bm25, "restriction lints", &[1.0, 0.0]);
        assert_eq!(ids, vec!["clippy".to_string(), "rust-book".to_string()]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cosine_doc_ids_propagates_embed_error() {
        let dir = std::env::temp_dir().join(format!("gs-eval-cos-err-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_tiny_store(&dir);
        let store = crate::vector::VectorStore::load(&dir).unwrap();

        let mut embed = |_: &[String]| -> Result<Vec<Vec<f32>>, crate::embed::EmbedError> {
            Err(crate::embed::EmbedError::Http("ollama down".into()))
        };
        let err = cosine_doc_ids(&store, &mut embed, "x").unwrap_err();
        assert!(matches!(err, EvalError::Embed(_)));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
