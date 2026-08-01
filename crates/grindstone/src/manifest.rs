//! Corpus manifest: the pinned source-of-truth list of corpus documents.
//!
//! A manifest records, per source: name, license, pinned URL, and (once
//! ingested) the sha256 content hash. Same manifest → same corpus state.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// The single source of truth for a corpus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    /// Schema version of this manifest format.
    pub version: u32,
    /// The documents that make up the corpus, in ingestion order.
    pub sources: Vec<Source>,
}

/// One pinned corpus document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Source {
    /// Stable identifier, also used as the corpus file stem (e.g. `rust-book`).
    pub name: String,
    /// SPDX expression for the source's license (e.g. "MIT OR Apache-2.0").
    pub license: String,
    /// Pinned URL the document is fetched from.
    pub url: String,
    /// Hex sha256 of the fetched content; `None` until ingested.
    #[serde(default)]
    pub hash: Option<String>,
    /// Trust tier for retrieval ranking/filtering (defaults to
    /// `pinned-source`; `docs-wiki` and `navigational` arrive with RAG-6).
    #[serde(default)]
    pub tier: TrustTier,
}

/// Provenance rank for retrieval: `pinned-source > docs-wiki > navigational`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TrustTier {
    /// Pinned, hash-verified upstream content (the manifest's own sources).
    #[default]
    #[serde(rename = "pinned-source")]
    PinnedSource,
    /// Community wiki docs (e.g. docs.tlapl.us), below pinned source.
    #[serde(rename = "docs-wiki")]
    DocsWiki,
    /// Navigational prose only, never ground truth (e.g. DeepWiki).
    #[serde(rename = "navigational")]
    Navigational,
}

impl Source {
    /// Corpus file name this source is stored under (e.g. `rust-book.html`).
    pub fn filename(&self) -> String {
        format!("{}.html", self.name)
    }
}

/// sha256 of `bytes`, lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(digest)
}

impl Manifest {
    /// Load a manifest from a JSON file.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(ManifestError::Io)?;
        let manifest = serde_json::from_str(&text).map_err(ManifestError::Json)?;
        Ok(manifest)
    }

    /// Save the manifest as pretty JSON.
    pub fn save(&self, path: &Path) -> Result<(), ManifestError> {
        let text = serde_json::to_string_pretty(self).map_err(ManifestError::Json)?;
        std::fs::write(path, text).map_err(ManifestError::Io)
    }
}

/// Errors produced while loading/saving manifests.
#[derive(Debug)]
pub enum ManifestError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Io(e) => write!(f, "manifest I/O error: {e}"),
            ManifestError::Json(e) => write!(f, "manifest parse error: {e}"),
        }
    }
}

impl std::error::Error for ManifestError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_source() -> Source {
        Source {
            name: "rust-book".into(),
            license: "MIT OR Apache-2.0".into(),
            url: "https://doc.rust-lang.org/1.95.0/book/print.html".into(),
            hash: None,
            tier: TrustTier::PinnedSource,
        }
    }

    #[test]
    fn parses_manifest_json() {
        let json = r#"{
            "version": 1,
            "sources": [
                {"name": "rust-book", "license": "MIT OR Apache-2.0",
                 "url": "https://doc.rust-lang.org/1.95.0/book/print.html"}
            ]
        }"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.version, 1);
        assert_eq!(m.sources.len(), 1);
        assert_eq!(m.sources[0].name, "rust-book");
        // Missing hash field defaults to None.
        assert_eq!(m.sources[0].hash, None);
    }

    #[test]
    fn load_and_save_roundtrip() {
        let dir =
            std::env::temp_dir().join(format!("gs-manifest-test-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");
        let m = Manifest {
            version: 1,
            sources: vec![sample_source()],
        };
        m.save(&path).unwrap();
        let loaded = Manifest::load(&path).unwrap();
        assert_eq!(loaded, m);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_missing_file_errors() {
        let err = Manifest::load(Path::new("/nonexistent/manifest.json")).unwrap_err();
        assert!(matches!(err, ManifestError::Io(_)));
    }

    #[test]
    fn load_invalid_json_errors() {
        let dir = std::env::temp_dir().join(format!("gs-manifest-test-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, "not json").unwrap();
        let err = Manifest::load(&path).unwrap_err();
        assert!(matches!(err, ManifestError::Json(_)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sha256_known_vector() {
        // sha256("abc"), per FIPS 180-4.
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn filename_uses_source_name() {
        let s = sample_source();
        assert_eq!(s.filename(), "rust-book.html");
    }
}
