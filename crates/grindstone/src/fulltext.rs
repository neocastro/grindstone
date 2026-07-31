//! Full-text search baseline over the corpus (rg-based).
//!
//! This is the deliberately-dumb retrieval baseline: plain `rg` over the raw
//! corpus files, ranked by total match count. Every later retrieval strategy
//! (cosine top-k, BM25 hybrid, cross-encoder reranking) must measurably beat
//! this on the eval harness (RAG-2). Runs fully offline.

use std::path::Path;

/// One ranked hit: a corpus file with at least one query match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    /// Corpus file name, e.g. `rust-book.html`.
    pub file: String,
    /// Total number of query matches in the file.
    pub matches: usize,
    /// First matching line, trimmed and truncated, as evidence.
    pub snippet: String,
}

/// Search all `*.html` files under `corpus_dir` for `query`.
///
/// Ranking: total match count per file, descending; ties broken by file name
/// (deterministic). The query is passed to `rg` as a literal argument — never
/// through a shell — so it cannot inject flags or commands.
/// Maximum snippet length in characters (snippets come from raw HTML lines,
/// which can be very long).
const SNIPPET_MAX_CHARS: usize = 240;

pub fn search(corpus_dir: &Path, query: &str) -> Result<Vec<Hit>, FulltextError> {
    let query = query.trim();
    if query.is_empty() {
        return Ok(Vec::new());
    }
    if !ripgrep_present() {
        return Err(FulltextError::MissingRipgrep);
    }

    let mut files: Vec<_> = std::fs::read_dir(corpus_dir)
        .map_err(FulltextError::Io)?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "html"))
        .map(|e| e.path())
        .collect();
    files.sort();

    let mut hits = Vec::new();
    for path in &files {
        let count = match_count(path, query)?;
        if count == 0 {
            continue;
        }
        let snippet = first_match_snippet(path, query);
        hits.push(Hit {
            file: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            matches: count,
            snippet,
        });
    }

    hits.sort_by(|a, b| b.matches.cmp(&a.matches).then_with(|| a.file.cmp(&b.file)));
    Ok(hits)
}

/// Is `rg` available on PATH?
fn ripgrep_present() -> bool {
    std::process::Command::new("rg")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run `rg` with a fixed flag set. `query` is passed as a literal `--`
/// argument so it can never be interpreted as a flag; the pattern is a fixed
/// string (`-F`), so `gs query "fn main()"` matches that text literally.
fn run_rg(args: &[&str], file: &Path, query: &str) -> Result<(bool, String), FulltextError> {
    let output = std::process::Command::new("rg")
        .args(args)
        .arg("--")
        .arg(query)
        .arg(file)
        .output()
        .map_err(|e| FulltextError::Ripgrep {
            file: file.display().to_string(),
            detail: e.to_string(),
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    match output.status.code() {
        Some(0) => Ok((true, stdout)),
        Some(1) => Ok((false, stdout)), // no matches
        _ => Err(FulltextError::Ripgrep {
            file: file.display().to_string(),
            detail: format!("unexpected rg status: {}", output.status),
        }),
    }
}

/// Total number of query matches in `file` (0 if none).
fn match_count(file: &Path, query: &str) -> Result<usize, FulltextError> {
    let (matched, stdout) = run_rg(
        &["--count-matches", "-F", "-i", "-a", "--no-messages"],
        file,
        query,
    )?;
    if !matched {
        return Ok(0);
    }
    let count = stdout.trim().parse::<usize>().unwrap_or(0);
    Ok(count)
}

/// First matching line, line-number prefix stripped, trimmed and truncated.
fn first_match_snippet(file: &Path, query: &str) -> String {
    let (_, stdout) = run_rg(
        &["-F", "-i", "-a", "--no-messages", "-m", "1", "-n"],
        file,
        query,
    )
    .unwrap_or_default();
    let text = match stdout.split_once(':') {
        Some((lineno, rest)) if lineno.bytes().all(|b| b.is_ascii_digit()) => rest,
        _ => &stdout,
    };
    let text = text.trim();
    if text.chars().count() > SNIPPET_MAX_CHARS {
        let mut out: String = text.chars().take(SNIPPET_MAX_CHARS).collect();
        out.push('…');
        out
    } else {
        text.to_string()
    }
}

/// Errors produced by full-text search.
#[derive(Debug)]
pub enum FulltextError {
    /// `rg` is not installed or not on PATH.
    MissingRipgrep,
    /// The corpus directory could not be read.
    Io(std::io::Error),
    /// `rg` failed in an unexpected way for one file.
    Ripgrep { file: String, detail: String },
}

impl std::fmt::Display for FulltextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FulltextError::MissingRipgrep => write!(
                f,
                "ripgrep (`rg`) is required for the full-text baseline but was not found on PATH"
            ),
            FulltextError::Io(e) => write!(f, "corpus I/O error: {e}"),
            FulltextError::Ripgrep { file, detail } => {
                write!(f, "rg failed on {file}: {detail}")
            }
        }
    }
}

impl std::error::Error for FulltextError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("gs-fulltext-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Skip the test when `rg` is unavailable (e.g. running outside devbox).
    fn ripgrep_present() -> bool {
        std::process::Command::new("rg")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn ranks_files_by_match_count() {
        if !ripgrep_present() {
            eprintln!("skipping: rg not installed");
            return;
        }
        let dir = corpus_dir("rank");
        std::fs::write(dir.join("a.html"), b"one needle here\nplain line\n").unwrap();
        std::fs::write(
            dir.join("b.html"),
            b"needle needle needle\nmore needle\nnone here\n",
        )
        .unwrap();
        std::fs::write(dir.join("c.html"), b"nothing relevant\n").unwrap();

        let hits = search(&dir, "needle").unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].file, "b.html");
        assert!(hits[0].matches >= hits[1].matches);
        assert_eq!(hits[1].file, "a.html");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_matches_returns_empty() {
        if !ripgrep_present() {
            eprintln!("skipping: rg not installed");
            return;
        }
        let dir = corpus_dir("none");
        std::fs::write(dir.join("a.html"), b"nothing here\n").unwrap();

        assert!(search(&dir, "zebra").unwrap().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn snippet_points_at_matching_line() {
        if !ripgrep_present() {
            eprintln!("skipping: rg not installed");
            return;
        }
        let dir = corpus_dir("snippet");
        std::fs::write(
            dir.join("a.html"),
            b"intro line\nThe borrow checker is the core of ownership.\nend\n",
        )
        .unwrap();

        let hits = search(&dir, "borrow checker").unwrap();

        assert_eq!(hits.len(), 1);
        assert!(hits[0].snippet.contains("borrow checker"));
        assert!(hits[0].snippet.contains("ownership"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn matches_case_insensitively() {
        if !ripgrep_present() {
            eprintln!("skipping: rg not installed");
            return;
        }
        let dir = corpus_dir("case");
        std::fs::write(dir.join("a.html"), b"BORROW CHECKER explained\n").unwrap();

        let hits = search(&dir, "borrow checker").unwrap();
        assert_eq!(hits.len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_corpus_dir_errors() {
        let err = search(Path::new("/nonexistent/corpus"), "needle").unwrap_err();
        assert!(matches!(err, FulltextError::Io(_)));
    }

    #[test]
    fn empty_query_returns_no_hits() {
        if !ripgrep_present() {
            eprintln!("skipping: rg not installed");
            return;
        }
        let dir = corpus_dir("empty");
        std::fs::write(dir.join("a.html"), b"anything\n").unwrap();

        assert!(search(&dir, "   ").unwrap().is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ties_break_deterministically_by_name() {
        if !ripgrep_present() {
            eprintln!("skipping: rg not installed");
            return;
        }
        let dir = corpus_dir("tie");
        std::fs::write(dir.join("b.html"), b"needle once\n").unwrap();
        std::fs::write(dir.join("a.html"), b"needle once\n").unwrap();

        let hits = search(&dir, "needle").unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].file, "a.html");
        assert_eq!(hits[1].file, "b.html");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
