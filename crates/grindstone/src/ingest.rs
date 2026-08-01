//! Corpus ingestion: fetch pinned sources into a corpus directory.
//!
//! Determinism rules:
//! - A source whose corpus file already exists and matches the manifest's
//!   pinned hash is skipped — re-running ingest with an unchanged manifest
//!   is a no-op (no re-download).
//! - A pinned hash is a hard contract: if the fetched content does not hash
//!   to the pinned value, ingest fails instead of writing divergent content.
//! - Ingest always writes a resolved manifest (with content hashes filled
//!   in) to `corpus_dir/manifest.json`.

use crate::manifest::{sha256_hex, Manifest, ManifestError, Source};
use std::path::Path;

/// Fetch a URL's bytes. Injected so tests can run fully offline.
pub type Fetcher<'a> = dyn FnMut(&str) -> Result<Vec<u8>, IngestError> + 'a;

/// Fetch `url` over HTTPS via reqwest (the production fetcher; blocking,
/// rustls TLS — no system OpenSSL dependency).
pub fn http_fetcher(url: &str) -> Result<Vec<u8>, IngestError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| IngestError::Http(format!("cannot build HTTP client: {e}")))?;
    let response = client
        .get(url)
        .send()
        .map_err(|e| IngestError::Http(format!("{url}: {e}")))?;
    let bytes = response
        .bytes()
        .map_err(|e| IngestError::Http(format!("{url}: read failed: {e}")))?;
    Ok(bytes.to_vec())
}

/// Outcome for one source during an ingest run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceAction {
    /// File already on disk matched the pinned hash; nothing fetched.
    Skipped,
    /// Content was fetched and written (or re-written) to disk.
    Fetched,
}

/// Result of an ingest run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestReport {
    /// Per-source outcomes, in manifest order.
    pub actions: Vec<(Source, SourceAction)>,
}

impl IngestReport {
    /// How many sources were actually fetched over the network.
    pub fn fetched_count(&self) -> usize {
        self.actions
            .iter()
            .filter(|(_, a)| *a == SourceAction::Fetched)
            .count()
    }
}

/// Ingest `manifest` into `corpus_dir` using `fetcher` for downloads.
///
/// The resolved manifest (with hashes filled in) is written to
/// `corpus_dir/manifest.json`; per-source corpus files are written as
/// `<name>.html`.
pub fn ingest(
    manifest: &Manifest,
    corpus_dir: &Path,
    fetcher: &mut Fetcher,
) -> Result<IngestReport, IngestError> {
    std::fs::create_dir_all(corpus_dir).map_err(IngestError::Io)?;

    let mut resolved = manifest.clone();
    let mut actions = Vec::with_capacity(manifest.sources.len());

    for source in &manifest.sources {
        let path = corpus_dir.join(source.filename());

        // No-op path: file already on disk and matches the pinned hash.
        if let Some(expected) = &source.hash {
            if let Ok(bytes) = std::fs::read(&path) {
                if &sha256_hex(&bytes) == expected {
                    actions.push((source.clone(), SourceAction::Skipped));
                    continue;
                }
            }
        }

        // Fetch, verify against the pinned hash, write.
        let bytes = fetcher(&source.url)?;
        let actual = sha256_hex(&bytes);
        if let Some(expected) = &source.hash {
            if &actual != expected {
                return Err(IngestError::HashMismatch {
                    source: source.name.clone(),
                    expected: expected.clone(),
                    actual,
                });
            }
        }
        std::fs::write(&path, &bytes).map_err(IngestError::Io)?;
        for r in &mut resolved.sources {
            if r.name == source.name {
                r.hash = Some(actual.clone());
            }
        }
        actions.push((source.clone(), SourceAction::Fetched));
    }

    resolved.save(&corpus_dir.join("manifest.json"))?;
    Ok(IngestReport { actions })
}

/// Errors produced by ingestion.
#[derive(Debug)]
pub enum IngestError {
    /// Network fetch failure.
    Http(String),
    /// Fetched content does not match the manifest's pinned hash.
    HashMismatch {
        source: String,
        expected: String,
        actual: String,
    },
    /// Local file system error.
    Io(std::io::Error),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::Http(e) => write!(f, "fetch failed: {e}"),
            IngestError::HashMismatch {
                source,
                expected,
                actual,
            } => write!(
                f,
                "content hash mismatch for {source}: expected {expected}, got {actual} \
                 (upstream content changed; update the manifest to re-pin)"
            ),
            IngestError::Io(e) => write!(f, "corpus I/O error: {e}"),
        }
    }
}

impl std::error::Error for IngestError {}

impl From<ManifestError> for IngestError {
    fn from(e: ManifestError) -> Self {
        IngestError::Io(std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::TrustTier;
    use std::collections::{HashMap, HashSet};

    fn source(name: &str, url: &str, hash: Option<&str>) -> Source {
        Source {
            name: name.into(),
            license: "MIT OR Apache-2.0".into(),
            url: url.into(),
            hash: hash.map(String::from),
            tier: TrustTier::PinnedSource,
        }
    }

    fn manifest(sources: Vec<Source>) -> Manifest {
        Manifest {
            version: 1,
            sources,
        }
    }

    /// A fetcher backed by a map of url → bytes, counting calls.
    struct FakeFetcher {
        contents: HashMap<String, Vec<u8>>,
        calls: Vec<String>,
    }

    impl FakeFetcher {
        fn new(contents: HashMap<String, Vec<u8>>) -> Self {
            Self {
                contents,
                calls: Vec::new(),
            }
        }
        fn call(&mut self, url: &str) -> Result<Vec<u8>, IngestError> {
            self.calls.push(url.to_string());
            self.contents
                .get(url)
                .cloned()
                .ok_or_else(|| IngestError::Http(format!("no fake content for {url}")))
        }
        fn fetched_urls(&self) -> HashSet<&str> {
            self.calls.iter().map(String::as_str).collect()
        }
    }

    fn corpus_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("gs-ingest-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn fetches_all_sources_and_writes_resolved_manifest() {
        let dir = corpus_dir("all");
        let m = manifest(vec![
            source("a", "https://x/a", None),
            source("b", "https://x/b", None),
        ]);
        let mut fetcher = FakeFetcher::new(HashMap::from([
            ("https://x/a".to_string(), b"content a".to_vec()),
            ("https://x/b".to_string(), b"content b".to_vec()),
        ]));

        let report = ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap();

        assert_eq!(report.fetched_count(), 2);
        assert_eq!(fetcher.fetched_urls().len(), 2);
        assert_eq!(std::fs::read(dir.join("a.html")).unwrap(), b"content a");
        assert_eq!(std::fs::read(dir.join("b.html")).unwrap(), b"content b");

        // Resolved manifest has hashes filled in and round-trips.
        let resolved = Manifest::load(&dir.join("manifest.json")).unwrap();
        assert_eq!(
            resolved.sources[0].hash.as_deref(),
            Some(sha256_hex(b"content a").as_str())
        );
        assert_eq!(
            resolved.sources[1].hash.as_deref(),
            Some(sha256_hex(b"content b").as_str())
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rerun_with_unchanged_manifest_is_a_noop() {
        let dir = corpus_dir("noop");
        let m = manifest(vec![
            source("a", "https://x/a", Some(sha256_hex(b"content a").as_str())),
            source("b", "https://x/b", Some(sha256_hex(b"content b").as_str())),
        ]);
        let mut fetcher = FakeFetcher::new(HashMap::from([
            ("https://x/a".to_string(), b"content a".to_vec()),
            ("https://x/b".to_string(), b"content b".to_vec()),
        ]));

        let first = ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap();
        assert_eq!(first.fetched_count(), 2);

        let second = ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap();
        assert_eq!(second.fetched_count(), 0, "no re-download");
        assert!(
            fetcher.fetched_urls().is_empty() || fetcher.fetched_urls().len() <= 2,
            "second run must not fetch again"
        );
        // The fetch call counter must be unchanged by the second run.
        assert_eq!(fetcher.calls.len(), 2, "exactly two fetches total");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_file_is_redownloaded() {
        let dir = corpus_dir("missing");
        let m = manifest(vec![source(
            "a",
            "https://x/a",
            Some(&sha256_hex(b"content a")),
        )]);
        let mut fetcher = FakeFetcher::new(HashMap::from([(
            "https://x/a".to_string(),
            b"content a".to_vec(),
        )]));

        // First run fetches, then the file is deleted.
        let first = ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap();
        assert_eq!(first.fetched_count(), 1);
        std::fs::remove_file(dir.join("a.html")).unwrap();

        // Second run re-fetches because the file is gone.
        let second = ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap();
        assert_eq!(second.fetched_count(), 1);
        assert_eq!(fetcher.calls.len(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupted_file_is_rewritten_from_pinned_hash() {
        let dir = corpus_dir("corrupt");
        let m = manifest(vec![source(
            "a",
            "https://x/a",
            Some(&sha256_hex(b"content a")),
        )]);
        let mut fetcher = FakeFetcher::new(HashMap::from([(
            "https://x/a".to_string(),
            b"content a".to_vec(),
        )]));

        ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap();
        // Corrupt the file on disk; ingest must detect and repair it.
        std::fs::write(dir.join("a.html"), b"tampered").unwrap();

        let report = ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap();
        assert_eq!(report.fetched_count(), 1);
        assert_eq!(std::fs::read(dir.join("a.html")).unwrap(), b"content a");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pinned_hash_mismatch_fails_loudly() {
        let dir = corpus_dir("mismatch");
        let m = manifest(vec![source(
            "a",
            "https://x/a",
            Some(&sha256_hex(b"expected content")),
        )]);
        let mut fetcher = FakeFetcher::new(HashMap::from([(
            "https://x/a".to_string(),
            b"different content".to_vec(),
        )]));

        let err = ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap_err();
        match err {
            IngestError::HashMismatch { source, .. } => assert_eq!(source, "a"),
            other => panic!("expected HashMismatch, got {other:?}"),
        }
        // No file and no resolved manifest may be left behind.
        assert!(!dir.join("a.html").exists());
        assert!(!dir.join("manifest.json").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fetch_error_propagates() {
        let dir = corpus_dir("fetherr");
        let m = manifest(vec![source("a", "https://x/a", None)]);
        let mut fetcher = FakeFetcher::new(HashMap::new());

        let err = ingest(&m, &dir, &mut |url| fetcher.call(url)).unwrap_err();
        assert!(matches!(err, IngestError::Http(_)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn deterministic_across_identical_runs() {
        let dir1 = corpus_dir("det1");
        let dir2 = corpus_dir("det2");
        let m = manifest(vec![source("a", "https://x/a", None)]);
        let mut f1 = FakeFetcher::new(HashMap::from([(
            "https://x/a".to_string(),
            b"same content".to_vec(),
        )]));
        let mut f2 = FakeFetcher::new(HashMap::from([(
            "https://x/a".to_string(),
            b"same content".to_vec(),
        )]));

        ingest(&m, &dir1, &mut |url| f1.call(url)).unwrap();
        ingest(&m, &dir2, &mut |url| f2.call(url)).unwrap();

        let m1 = Manifest::load(&dir1.join("manifest.json")).unwrap();
        let m2 = Manifest::load(&dir2.join("manifest.json")).unwrap();
        assert_eq!(m1, m2, "same manifest → same corpus state");
        std::fs::remove_dir_all(&dir1).unwrap();
        std::fs::remove_dir_all(&dir2).unwrap();
    }

    // Re-export so the manifest error conversion is exercised.
    #[allow(dead_code)]
    fn _manifest_error_conversion(e: ManifestError) -> IngestError {
        e.into()
    }
}
