//! Hybrid search (RAG-7): reciprocal-rank fusion (RRF) of sparse BM25 and
//! dense cosine scores, fused at the DOCUMENT level.
//!
//! Each retriever ranks chunks; a document's rank in a list is its best
//! chunk's rank. RRF merges the two doc rankings (a doc both agree on
//! outranks one only one retriever likes), and each top doc is re-expanded
//! to its best chunk. Doc-level fusion matches the eval harness's unit of
//! recall and gives the agent doc-diverse context. Fully offline and
//! deterministic: same store + BM25 index + query → same fusion.

use crate::bm25::Bm25Index;
use crate::manifest::TrustTier;
use crate::vector::{Hit, VectorStore};
use std::collections::BTreeMap;

/// The standard RRF constant (k = 60).
pub const RRF_K: f64 = 60.0;

/// Fuse BM25 + cosine retrieval for `query` via document-level RRF.
///
/// The returned hits carry the doc's RRF score (not a cosine or BM25 score);
/// ties are broken by doc name (ascending), so the result is deterministic.
/// Chunks are taken from a pool of `max(k, 10) * 2` per retriever.
pub fn hybrid_search(
    store: &VectorStore,
    bm25: &Bm25Index,
    query_embedding: &[f32],
    query: &str,
    k: usize,
    tier: Option<TrustTier>,
) -> Vec<Hit> {
    if k == 0 {
        return Vec::new();
    }
    let pool = k.max(10) * 2;
    let cosine_hits = store.search(query_embedding, pool, tier);
    let bm25_hits = bm25.search(query, pool);

    // Doc rank per retriever = its best chunk's rank.
    let mut cosine_rank: BTreeMap<String, usize> = BTreeMap::new();
    for (i, hit) in cosine_hits.iter().enumerate() {
        cosine_rank.entry(hit.chunk.source.clone()).or_insert(i);
    }
    let mut bm25_rank: BTreeMap<String, usize> = BTreeMap::new();
    for (i, hit) in bm25_hits.iter().enumerate() {
        let Some(chunk) = store.chunk(&hit.chunk_id) else {
            continue;
        };
        if let Some(t) = tier {
            if chunk.tier != t {
                continue;
            }
        }
        bm25_rank.entry(chunk.source.clone()).or_insert(i);
    }

    // RRF over the two doc rankings.
    let mut rrf: BTreeMap<String, f64> = BTreeMap::new();
    for (doc, rank) in &cosine_rank {
        *rrf.entry(doc.clone()).or_default() += 1.0 / (RRF_K + *rank as f64 + 1.0);
    }
    for (doc, rank) in &bm25_rank {
        *rrf.entry(doc.clone()).or_default() += 1.0 / (RRF_K + *rank as f64 + 1.0);
    }

    let mut ranked: Vec<(String, f64)> = rrf.into_iter().collect();
    ranked.sort_by(|(a, a_score), (b, b_score)| {
        b_score
            .partial_cmp(a_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });

    // Re-expand each top doc to its best chunk (first cosine hit, else the
    // first BM25 hit for that doc).
    ranked
        .into_iter()
        .take(k)
        .filter_map(|(doc, score)| {
            let chunk = cosine_hits
                .iter()
                .find(|h| h.chunk.source == doc)
                .map(|h| h.chunk.clone())
                .or_else(|| {
                    bm25_hits.iter().find_map(|h| {
                        store
                            .chunk(&h.chunk_id)
                            .filter(|c| c.source == doc)
                            .cloned()
                    })
                });
            chunk.map(|chunk| Hit { chunk, score })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, ChunksFile};
    use crate::embed::EmbeddingsFile;
    use crate::manifest::TrustTier::PinnedSource;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let p =
                std::env::temp_dir().join(format!("gs-hybrid-{tag}-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn chunk(id: &str, source: &str, text: &str) -> Chunk {
        Chunk {
            id: id.into(),
            source: source.into(),
            license: "MIT".into(),
            tier: PinnedSource,
            heading: "H".into(),
            tokens: 1,
            text: text.into(),
        }
    }

    fn make_store(dir: &TempDir, chunks: &[Chunk], vectors: &[(&str, Vec<f32>)]) -> VectorStore {
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
            model: "fake".into(),
            dim: 2,
            vectors: map,
        };
        emb.save(&dir.path().join("embeddings.json")).unwrap();
        VectorStore::load(dir.path()).unwrap()
    }

    #[test]
    fn fusion_ranks_doc_in_both_lists_first() {
        let dir = TempDir::new("both");
        let chunks = vec![
            chunk("a", "a-doc", "model values explained"),
            chunk("b", "b-doc", "borrow checker ownership"),
            chunk("c", "a-doc", "model values too"),
        ];
        let store = make_store(
            &dir,
            &chunks,
            &[
                ("a", vec![1.0, 0.0]),
                ("b", vec![0.0, 1.0]),
                ("c", vec![0.9, 0.1]),
            ],
        );
        let bm25 = Bm25Index::build(&chunks);

        // "model values": a-doc is in both lists (its chunks rank high in
        // cosine and match in BM25); b-doc matches neither strongly.
        let hits = hybrid_search(&store, &bm25, &[1.0, 0.0], "model values", 3, None);
        assert_eq!(
            hits[0].chunk.source, "a-doc",
            "doc in both lists must rank first"
        );
        // One hit per doc (doc-level fusion), so a-doc appears once.
        assert_eq!(hits.len(), 2);
        let sources: Vec<&str> = hits.iter().map(|h| h.chunk.source.as_str()).collect();
        assert_eq!(sources, vec!["a-doc", "b-doc"]);
    }

    #[test]
    fn hybrid_promotes_lexical_match_over_pure_cosine() {
        let dir = TempDir::new("lex");
        let chunks = vec![
            chunk("a", "a-doc", "CHOOSE operator semantics"),
            chunk("b", "b-doc", "unrelated completely different text"),
        ];
        // Cosine: b-doc is actually closer to the query vector (adversarial);
        // BM25: only a-doc contains "CHOOSE".
        let store = make_store(
            &dir,
            &chunks,
            &[("a", vec![0.6, 0.4]), ("b", vec![1.0, 0.0])],
        );
        let bm25 = Bm25Index::build(&chunks);

        let cosine_first = store.search(&[1.0, 0.0], 2, None)[0].chunk.source.clone();
        assert_eq!(cosine_first, "b-doc");

        let hits = hybrid_search(&store, &bm25, &[1.0, 0.0], "CHOOSE", 2, None);
        assert_eq!(
            hits[0].chunk.source, "a-doc",
            "lexical match must be promoted by fusion"
        );
    }

    #[test]
    fn hybrid_is_deterministic_and_respects_top_k() {
        let dir = TempDir::new("det");
        let chunks = vec![
            chunk("x", "x-doc", "the chooser the"),
            chunk("y", "y-doc", "the"),
            chunk("z", "z-doc", "chooser"),
        ];
        let store = make_store(
            &dir,
            &chunks,
            &[
                ("x", vec![0.3, 0.3]),
                ("y", vec![0.1, 0.1]),
                ("z", vec![0.2, 0.2]),
            ],
        );
        let bm25 = Bm25Index::build(&chunks);

        let a = hybrid_search(&store, &bm25, &[0.3, 0.3], "the chooser", 2, None);
        let b = hybrid_search(&store, &bm25, &[0.3, 0.3], "the chooser", 2, None);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        assert_eq!(
            hybrid_search(&store, &bm25, &[0.3, 0.3], "the chooser", 0, None).len(),
            0
        );
    }

    #[test]
    fn hybrid_empty_bm25_degrades_to_cosine_order() {
        let dir = TempDir::new("degrade");
        let chunks = vec![
            chunk("a", "a-doc", "aaa bbb"),
            chunk("b", "b-doc", "ccc ddd"),
        ];
        let store = make_store(
            &dir,
            &chunks,
            &[("a", vec![1.0, 0.0]), ("b", vec![0.0, 1.0])],
        );
        let bm25 = Bm25Index::build(&chunks);

        // Query text absent from the corpus → BM25 empty → fusion keeps the
        // cosine ranking.
        let hits = hybrid_search(&store, &bm25, &[1.0, 0.0], "zzz qqq", 2, None);
        assert_eq!(hits[0].chunk.source, "a-doc");
        assert_eq!(hits[1].chunk.source, "b-doc");
    }
}
