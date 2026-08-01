//! BM25 sparse retrieval over corpus chunks (RAG-7): the lexical half of
//! hybrid search.
//!
//! Deterministic: same chunks + same query → identical scores (no randomness,
//! stable tie-breaks by chunk id). Built from the same `chunks.json` the
//! vector store reads, so a hybrid strategy is fully offline.

use crate::chunk::Chunk;
use std::collections::BTreeMap;

/// BM25 term-frequency saturation.
pub const K1: f64 = 1.2;
/// BM25 length normalization.
pub const B: f64 = 0.75;

/// One BM25 hit: a chunk id with its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25Hit {
    pub chunk_id: String,
    pub score: f64,
}

/// Deterministic BM25 index over a chunk set.
#[derive(Debug)]
pub struct Bm25Index {
    /// chunk id → term → frequency.
    term_freqs: BTreeMap<String, BTreeMap<String, usize>>,
    /// term → number of chunks containing it.
    doc_freqs: BTreeMap<String, usize>,
    /// chunk id → token count.
    doc_len: BTreeMap<String, usize>,
    /// Number of chunks in the index.
    num_docs: usize,
    /// Average chunk length in tokens.
    avg_doc_len: f64,
}

impl Bm25Index {
    /// Build the index from chunks (tokenized deterministically).
    pub fn build(chunks: &[Chunk]) -> Self {
        let mut term_freqs = BTreeMap::new();
        let mut doc_freqs: BTreeMap<String, usize> = BTreeMap::new();
        let mut doc_len = BTreeMap::new();
        let mut total_len = 0usize;
        for chunk in chunks {
            let tokens = tokenize(&chunk.text);
            let mut tf: BTreeMap<String, usize> = BTreeMap::new();
            for t in &tokens {
                *tf.entry(t.clone()).or_default() += 1;
            }
            for t in tf.keys() {
                *doc_freqs.entry(t.clone()).or_default() += 1;
            }
            total_len += tokens.len();
            doc_len.insert(chunk.id.clone(), tokens.len());
            term_freqs.insert(chunk.id.clone(), tf);
        }
        let num_docs = chunks.len();
        let avg_doc_len = if num_docs == 0 {
            0.0
        } else {
            total_len as f64 / num_docs as f64
        };
        Bm25Index {
            term_freqs,
            doc_freqs,
            doc_len,
            num_docs,
            avg_doc_len,
        }
    }

    /// Rank chunks by BM25 score for `query`, top `k`, ties broken by chunk
    /// id (ascending).
    pub fn search(&self, query: &str, k: usize) -> Vec<Bm25Hit> {
        if k == 0 || self.num_docs == 0 || query.trim().is_empty() {
            return Vec::new();
        }
        let mut qtf: BTreeMap<String, usize> = BTreeMap::new();
        for t in tokenize(query) {
            *qtf.entry(t).or_default() += 1;
        }
        let mut scores: BTreeMap<String, f64> = BTreeMap::new();
        for (term, q_count) in &qtf {
            let Some(&df) = self.doc_freqs.get(term) else {
                continue; // term absent from the corpus
            };
            let idf = ((self.num_docs as f64 - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln();
            for (chunk_id, tf_map) in &self.term_freqs {
                let Some(&tf) = tf_map.get(term) else {
                    continue;
                };
                let dl = self.doc_len.get(chunk_id).copied().unwrap_or(0) as f64;
                let denom = tf as f64 + K1 * (1.0 - B + B * dl / self.avg_doc_len.max(1e-9));
                let score = idf * (tf as f64 * (K1 + 1.0)) / denom * *q_count as f64;
                *scores.entry(chunk_id.clone()).or_default() += score;
            }
        }
        let mut ranked: Vec<Bm25Hit> = scores
            .into_iter()
            .map(|(chunk_id, score)| Bm25Hit { chunk_id, score })
            .collect();
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
        });
        ranked.truncate(k);
        ranked
    }

    /// Number of chunks in the index.
    pub fn len(&self) -> usize {
        self.num_docs
    }

    /// True when the index has no chunks.
    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }
}

/// Split `text` into lowercase tokens: runs of alphanumerics + `_`, split at
/// camelCase boundaries (`OpDeclNode` → `op` `decl` `node`) so code
/// identifiers match across case conventions.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_alphanumeric() || c == '_' {
            // camelCase boundary: split before a lowercase→uppercase turn.
            let prev = i.checked_sub(1).and_then(|j| chars.get(j));
            if prev.is_some_and(|p| p.is_ascii_lowercase() && c.is_ascii_uppercase())
                && !current.is_empty()
            {
                tokens.push(current.to_lowercase());
                current.clear();
            }
            current.push(c);
        } else if !current.is_empty() {
            tokens.push(current.to_lowercase());
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(current.to_lowercase());
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::TrustTier::PinnedSource;

    fn chunk(id: &str, text: &str) -> Chunk {
        Chunk {
            id: id.into(),
            source: "tlaplus".into(),
            license: "MIT".into(),
            tier: PinnedSource,
            heading: "H".into(),
            tokens: 1,
            text: text.into(),
        }
    }

    #[test]
    fn tokenize_splits_and_lowercases() {
        assert_eq!(
            tokenize("Borrow Checker foo_bar"),
            vec!["borrow", "checker", "foo_bar"]
        );
        assert_eq!(
            tokenize("OpDeclNode.apply(a, 2)"),
            vec!["op", "decl", "node", "apply", "a", "2"]
        );
        assert_eq!(tokenize(""), Vec::<String>::new());
    }

    #[test]
    fn bm25_scores_only_matching_chunks() {
        let index = Bm25Index::build(&[
            chunk("a", "the borrow checker explains borrowing"),
            chunk("b", "the TLA+ model checker checks invariants"),
        ]);
        assert_eq!(index.len(), 2);
        let hits = index.search("borrow", 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, "a");
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn bm25_term_frequency_raises_score() {
        let index = Bm25Index::build(&[
            chunk("rare", "borrow borrow borrow"),
            chunk("single", "borrow"),
        ]);
        let hits = index.search("borrow", 2);
        assert_eq!(hits[0].chunk_id, "rare");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn bm25_respects_top_k_and_tie_breaks_by_id() {
        let index = Bm25Index::build(&[
            chunk("b-id", "model values"),
            chunk("a-id", "model values"),
            chunk("c-id", "model values"),
        ]);
        let hits = index.search("model values", 2);
        assert_eq!(hits.len(), 2);
        // Equal scores → ascending chunk id.
        assert_eq!(hits[0].chunk_id, "a-id");
        assert_eq!(hits[1].chunk_id, "b-id");
        // Deterministic across runs.
        let again = index.search("model values", 2);
        assert_eq!(hits, again);
    }

    #[test]
    fn bm25_empty_corpus_and_query_are_safe() {
        let empty = Bm25Index::build(&[]);
        assert!(empty.is_empty());
        assert!(empty.search("anything", 5).is_empty());

        let index = Bm25Index::build(&[chunk("a", "model values")]);
        assert!(index.search("", 5).is_empty());
        assert!(index.search("model", 0).is_empty());
    }

    #[test]
    fn bm25_common_term_scores_lower_than_rare_term() {
        // "the" appears in both chunks (high df → low idf); "chooser" in one.
        let index = Bm25Index::build(&[chunk("a", "the chooser the"), chunk("b", "the")]);
        let hits = index.search("the chooser", 2);
        assert_eq!(hits[0].chunk_id, "a");
        // Chunk a's score comes mostly from the rare "chooser".
        assert!(hits[0].score > 0.0);
    }
}
