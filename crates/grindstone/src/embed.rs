//! Embedder (RAG-3): `nomic-embed-text` via the local Ollama server.
//!
//! Embeds every chunk and persists `chunk id → vector` for the vector store
//! (RAG-4) to consume. Determinism: `nomic-embed-text` is deterministic for a
//! given input on a given model file, so the same chunks produce the same
//! embeddings. The HTTP call is injectable so tests run fully offline; a down
//! Ollama server yields a clear error, never a hang (hard timeout). Uses
//! reqwest (blocking, no TLS) — the embedder only talks to localhost.

use crate::chunk::Chunk;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Default local Ollama server URL.
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
/// The embedder model (pinned per the repo toolchain docs).
pub const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";
/// Chunks per /api/embed call.
pub const DEFAULT_BATCH_SIZE: usize = 16;
/// Hard timeout for one Ollama call, so a down server errors instead of hanging.
pub const EMBED_TIMEOUT_SECS: u64 = 60;

/// Persisted embeddings: model + dim + chunk-id → vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingsFile {
    /// Embedder model that produced the vectors.
    pub model: String,
    /// Vector dimensionality (768 for nomic-embed-text).
    pub dim: usize,
    /// Chunk id → embedding, ordered by id (deterministic).
    pub vectors: BTreeMap<String, Vec<f32>>,
}

impl EmbeddingsFile {
    /// Load an embeddings file from JSON.
    pub fn load(path: &Path) -> Result<Self, EmbedError> {
        let text = std::fs::read_to_string(path).map_err(EmbedError::Io)?;
        serde_json::from_str(&text).map_err(EmbedError::Json)
    }

    /// Persist as pretty JSON.
    pub fn save(&self, path: &Path) -> Result<(), EmbedError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(EmbedError::Io)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(EmbedError::Json)?;
        std::fs::write(path, text).map_err(EmbedError::Io)
    }
}

/// Embed `inputs` via Ollama's `/api/embed` endpoint (one batched call).
///
/// Fails with a clear, actionable error when the server is unreachable —
/// including a hard timeout so a wedged server cannot hang the pipeline.
pub fn ollama_embed(
    server_url: &str,
    model: &str,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(EMBED_TIMEOUT_SECS))
        .build()
        .map_err(|e| EmbedError::Http(format!("cannot build HTTP client: {e}")))?;
    let body = serde_json::json!({ "model": model, "input": inputs });
    let response = client
        .post(format!("{server_url}/api/embed"))
        .json(&body)
        .send()
        .map_err(|e| {
            EmbedError::Http(format!(
                "cannot reach Ollama at {server_url}: {e} — is `ollama serve` running?"
            ))
        })?;
    let parsed: serde_json::Value = response
        .json()
        .map_err(|e| EmbedError::Http(format!("cannot decode Ollama response: {e}")))?;
    let embeddings = parsed
        .get("embeddings")
        .and_then(|e| e.as_array())
        .ok_or_else(|| EmbedError::Http("Ollama response missing `embeddings`".into()))?;
    if embeddings.len() != inputs.len() {
        return Err(EmbedError::DimensionMismatch {
            expected: inputs.len(),
            got: embeddings.len(),
        });
    }
    embeddings
        .iter()
        .map(|v| {
            v.as_array()
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                .ok_or_else(|| EmbedError::Http("embedding is not a numeric array".into()))
        })
        .collect()
}

/// The injectable embedder call: inputs -> vectors (tests use a fake;
/// production uses `ollama_embed`).
pub type Embedder<'a> = dyn FnMut(&[String]) -> Result<Vec<Vec<f32>>, EmbedError> + 'a;

/// Embed every chunk in order, batched, keyed by chunk id.
///
/// `call` is the injectable embedder. Returns an `EmbeddingsFile` with a
/// consistent dimension.
pub fn embed_chunks(
    chunks: &[Chunk],
    model: &str,
    batch_size: usize,
    call: &mut Embedder,
) -> Result<EmbeddingsFile, EmbedError> {
    let mut vectors = BTreeMap::new();
    let mut dim: Option<usize> = None;
    for batch in chunks.chunks(batch_size.max(1)) {
        let inputs: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let embeddings = call(&inputs)?;
        if embeddings.len() != batch.len() {
            return Err(EmbedError::DimensionMismatch {
                expected: batch.len(),
                got: embeddings.len(),
            });
        }
        for (chunk, embedding) in batch.iter().zip(embeddings) {
            match dim {
                None => dim = Some(embedding.len()),
                Some(d) if d != embedding.len() => {
                    return Err(EmbedError::DimensionMismatch {
                        expected: d,
                        got: embedding.len(),
                    });
                }
                _ => {}
            }
            vectors.insert(chunk.id.clone(), embedding);
        }
    }
    Ok(EmbeddingsFile {
        model: model.to_string(),
        dim: dim.unwrap_or(0),
        vectors,
    })
}

/// Errors produced by the embedder.
#[derive(Debug)]
pub enum EmbedError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// Ollama/HTTP-level failure, with actionable guidance.
    Http(String),
    /// A batch returned the wrong number of vectors or a mismatched dimension.
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::Io(e) => write!(f, "embed I/O error: {e}"),
            EmbedError::Json(e) => write!(f, "embed JSON error: {e}"),
            EmbedError::Http(e) => write!(f, "{e}"),
            EmbedError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "embedding count mismatch: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for EmbedError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(n: usize) -> Vec<Chunk> {
        (0..n)
            .map(|i| Chunk {
                id: format!("chunk-{i}"),
                source: "rust-book".into(),
                license: "MIT OR Apache-2.0".into(),
                tier: crate::manifest::TrustTier::PinnedSource,
                heading: "H".into(),
                tokens: 10,
                text: format!("text {i}"),
            })
            .collect()
    }

    #[test]
    fn embed_chunks_batches_and_keys_by_id() {
        let mut fake = |inputs: &[String]| -> Result<Vec<Vec<f32>>, EmbedError> {
            Ok(inputs.iter().map(|s| vec![s.len() as f32, 0.5]).collect())
        };
        let file = embed_chunks(&chunks(5), "fake-model", 2, &mut fake).unwrap();
        assert_eq!(file.model, "fake-model");
        assert_eq!(file.dim, 2);
        assert_eq!(file.vectors.len(), 5);
        // Deterministic order by chunk id.
        let keys: Vec<&String> = file.vectors.keys().collect();
        assert_eq!(
            keys,
            vec!["chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"]
        );
        assert_eq!(file.vectors["chunk-0"], vec![6.0, 0.5]); // "text 0".len() == 6
    }

    #[test]
    fn embed_chunks_rejects_wrong_batch_count() {
        let mut fake = |inputs: &[String]| -> Result<Vec<Vec<f32>>, EmbedError> {
            // Return one fewer embedding than requested.
            Ok(inputs[..inputs.len() - 1]
                .iter()
                .map(|s| vec![s.len() as f32])
                .collect())
        };
        let err = embed_chunks(&chunks(2), "fake", 2, &mut fake).unwrap_err();
        assert!(matches!(err, EmbedError::DimensionMismatch { .. }));
    }

    #[test]
    fn embed_chunks_rejects_inconsistent_dimension() {
        let mut calls = 0;
        let mut fake = |inputs: &[String]| -> Result<Vec<Vec<f32>>, EmbedError> {
            calls += 1;
            Ok(inputs
                .iter()
                .map(|s| vec![s.len() as f32; calls as usize])
                .collect())
        };
        let err = embed_chunks(&chunks(3), "fake", 2, &mut fake).unwrap_err();
        assert!(matches!(err, EmbedError::DimensionMismatch { .. }));
    }

    #[test]
    fn embeddings_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("gs-embed-test-{}", std::process::id()));
        let path = dir.join("embeddings.json");
        let file = EmbeddingsFile {
            model: "nomic-embed-text".into(),
            dim: 2,
            vectors: BTreeMap::from([("c1".into(), vec![0.1, 0.2])]),
        };
        file.save(&path).unwrap();
        let loaded = EmbeddingsFile::load(&path).unwrap();
        assert_eq!(loaded, file);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn down_ollama_errors_clearly_not_hang() {
        // Point at a closed port on localhost: connection refused, fast.
        let err = ollama_embed("http://127.0.0.1:1", "nomic-embed-text", &["x".into()]);
        let err = err.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Ollama"),
            "expected actionable error, got: {msg}"
        );
    }
}
