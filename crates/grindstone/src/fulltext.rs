//! Full-text search baseline over the corpus.
//!
//! This is the deliberately-dumb retrieval baseline: plain substring search
//! over the raw corpus files, ranked by total match count. Every later
//! retrieval strategy (cosine top-k, BM25 hybrid, cross-encoder reranking)
//! must measurably beat this on the eval harness (RAG-2). Runs fully offline.
//!
//! The search path sits behind a two-adapter seam (RAG-10): [`RipgrepSearcher`]
//! shells out to `rg` when it is on PATH, and [`InProcessSearcher`] is a
//! deterministic pure-Rust scan used whenever `rg` is absent — so the
//! baseline is measurable anywhere, including CI. [`search`] picks the
//! preferred adapter automatically.

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

/// Maximum snippet length in characters (snippets come from raw HTML lines,
/// which can be very long).
const SNIPPET_MAX_CHARS: usize = 240;

/// The fulltext searcher seam: corpus dir + query → ranked hits.
///
/// Exactly two adapters exist: [`RipgrepSearcher`] (preferred, shells out to
/// `rg` with a literal query) and [`InProcessSearcher`] (deterministic
/// pure-Rust fallback used whenever `rg` is absent). Both adapters share the
/// same semantics: rank by total match count per file (descending), ties
/// broken by file name; snippet = first matching line, trimmed and truncated.
pub trait Searcher {
    /// Rank all `*.html`/`*.txt` documents under `corpus_dir` for `query`.
    fn search(&self, corpus_dir: &Path, query: &str) -> Result<Vec<Hit>, FulltextError>;
}

/// `rg`-based adapter: the preferred fulltext path when ripgrep is on PATH.
///
/// The query is passed to `rg` as a literal `--` argument — never through a
/// shell — so it cannot inject flags or commands; the pattern is a fixed
/// string (`-F`), so `gs query "fn main()"` matches that text literally.
#[derive(Debug, Default, Clone, Copy)]
pub struct RipgrepSearcher;

impl Searcher for RipgrepSearcher {
    fn search(&self, corpus_dir: &Path, query: &str) -> Result<Vec<Hit>, FulltextError> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        // Line-oriented search (no `--multiline`): a newline-containing
        // pattern can never match. `rg` errors out on such a literal, so
        // short-circuit to the same empty result the in-process adapter
        // produces — semantics preserved without an error path.
        if query.contains('\n') {
            return Ok(Vec::new());
        }

        // Filesystem errors take precedence over the `rg` availability check so
        // a missing corpus dir is reported the same whether or not `rg` is
        // installed (CI has no devbox, so `rg` is absent there).
        let files = collect_files(corpus_dir)?;
        if !ripgrep_present() {
            return Err(FulltextError::MissingRipgrep);
        }

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
}

/// Pure-Rust in-process adapter: a deterministic scan over the corpus files.
///
/// A **real production fallback** — not a test fake. Used whenever `rg` is
/// absent, so the fulltext baseline (the score every retrieval strategy must
/// beat) is measurable anywhere, including CI. Same file ordering, same
/// hit/dedup semantics, same ranking and snippet rules as the `rg` adapter.
/// Matching is case-insensitive literal substring search per line (the Rust
/// equivalent of `rg -F -i`), so a multi-line pattern never matches, just as
/// with line-oriented `rg`.
#[derive(Debug, Default, Clone, Copy)]
pub struct InProcessSearcher;

impl Searcher for InProcessSearcher {
    fn search(&self, corpus_dir: &Path, query: &str) -> Result<Vec<Hit>, FulltextError> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let files = collect_files(corpus_dir)?;
        let query_lower = query.to_lowercase();

        let mut hits = Vec::new();
        for path in &files {
            // Mirrors `rg --no-messages`: an unreadable file is skipped, not fatal.
            let text = match std::fs::read(path) {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(_) => continue,
            };
            let count = in_process_match_count(&text, &query_lower);
            if count == 0 {
                continue;
            }
            let snippet = in_process_snippet(&text, &query_lower);
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
}

/// Pick the preferred searcher for this machine: `rg` when present, otherwise
/// the in-process scan. Both are production adapters.
pub fn default_searcher() -> Box<dyn Searcher> {
    if ripgrep_present() {
        Box::new(RipgrepSearcher)
    } else {
        Box::new(InProcessSearcher)
    }
}

/// Convenience entry: search `corpus_dir` for `query` via the preferred
/// searcher ([`default_searcher`]).
pub fn search(corpus_dir: &Path, query: &str) -> Result<Vec<Hit>, FulltextError> {
    // Preserve Io precedence: a missing corpus dir is reported the same
    // whether or not `rg` is installed.
    collect_files(corpus_dir)?;
    default_searcher().search(corpus_dir, query)
}

/// All `*.html`/`*.txt` files under `corpus_dir`, sorted by path (stable file
/// ordering shared by both adapters).
fn collect_files(corpus_dir: &Path) -> Result<Vec<std::path::PathBuf>, FulltextError> {
    let mut files: Vec<_> = std::fs::read_dir(corpus_dir)
        .map_err(FulltextError::Io)?
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "html" || ext == "txt")
        })
        .map(|e| e.path())
        .collect();
    files.sort();
    Ok(files)
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
    truncate_snippet(text)
}

/// Total number of (non-overlapping) case-insensitive matches of `query_lower`
/// across all lines of `text`. Line-oriented, mirroring `rg` without
/// `--multiline`.
fn in_process_match_count(text: &str, query_lower: &str) -> usize {
    text.lines()
        .map(|line| line.to_lowercase().matches(query_lower).count())
        .sum()
}

/// First line containing `query_lower`, trimmed and truncated.
fn in_process_snippet(text: &str, query_lower: &str) -> String {
    let line = text
        .lines()
        .find(|line| line.to_lowercase().contains(query_lower))
        .unwrap_or_default();
    truncate_snippet(line)
}

/// Trim and truncate a snippet line to [`SNIPPET_MAX_CHARS`], appending an
/// ellipsis when cut.
fn truncate_snippet(line: &str) -> String {
    let line = line.trim();
    if line.chars().count() > SNIPPET_MAX_CHARS {
        let mut out: String = line.chars().take(SNIPPET_MAX_CHARS).collect();
        out.push('…');
        out
    } else {
        line.to_string()
    }
}

/// Errors produced by full-text search.
#[derive(Debug)]
pub enum FulltextError {
    /// `rg` is not installed or not on PATH (only raised by direct
    /// [`RipgrepSearcher`] use; the default path falls back to
    /// [`InProcessSearcher`] instead).
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

    /// The adapters under test: always the in-process scan, plus `rg` when it
    /// is installed (its behavior must agree with the fallback).
    fn adapters() -> Vec<Box<dyn Searcher>> {
        let mut v: Vec<Box<dyn Searcher>> = vec![Box::new(InProcessSearcher)];
        if ripgrep_present() {
            v.push(Box::new(RipgrepSearcher));
        }
        v
    }

    #[test]
    fn ranks_files_by_match_count() {
        let dir = corpus_dir("rank");
        std::fs::write(dir.join("a.html"), b"one needle here\nplain line\n").unwrap();
        std::fs::write(
            dir.join("b.html"),
            b"needle needle needle\nmore needle\nnone here\n",
        )
        .unwrap();
        std::fs::write(dir.join("c.html"), b"nothing relevant\n").unwrap();
        std::fs::write(dir.join("d.txt"), b"needle in a txt file\n").unwrap();

        for searcher in adapters() {
            let hits = searcher.search(&dir, "needle").unwrap();

            assert_eq!(hits.len(), 3);
            assert_eq!(hits[0].file, "b.html");
            assert_eq!(hits[0].matches, 4);
            assert_eq!(hits[1].file, "a.html");
            assert_eq!(hits[1].matches, 1);
            assert_eq!(hits[2].file, "d.txt");
            assert_eq!(hits[2].matches, 1);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_matches_returns_empty() {
        let dir = corpus_dir("none");
        std::fs::write(dir.join("a.html"), b"nothing here\n").unwrap();

        for searcher in adapters() {
            assert!(searcher.search(&dir, "zebra").unwrap().is_empty());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn snippet_points_at_matching_line() {
        let dir = corpus_dir("snippet");
        std::fs::write(
            dir.join("a.html"),
            b"intro line\nThe borrow checker is the core of ownership.\nend\n",
        )
        .unwrap();

        for searcher in adapters() {
            let hits = searcher.search(&dir, "borrow checker").unwrap();

            assert_eq!(hits.len(), 1);
            assert!(hits[0].snippet.contains("borrow checker"));
            assert!(hits[0].snippet.contains("ownership"));
            assert!(!hits[0].snippet.contains("intro line"));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn matches_case_insensitively() {
        let dir = corpus_dir("case");
        std::fs::write(dir.join("a.html"), b"BORROW CHECKER explained\n").unwrap();

        for searcher in adapters() {
            let hits = searcher.search(&dir, "borrow checker").unwrap();
            assert_eq!(hits.len(), 1);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn counts_every_occurrence_per_line() {
        let dir = corpus_dir("count");
        std::fs::write(dir.join("a.html"), b"x needle y needle\nneedle\n").unwrap();

        for searcher in adapters() {
            let hits = searcher.search(&dir, "needle").unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].matches, 3);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn multiline_query_never_matches() {
        let dir = corpus_dir("multiline");
        std::fs::write(dir.join("a.html"), b"foo\nbar\n").unwrap();

        for searcher in adapters() {
            assert!(searcher.search(&dir, "foo\nbar").unwrap().is_empty());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_corpus_dir_errors() {
        let err = search(Path::new("/nonexistent/corpus"), "needle").unwrap_err();
        assert!(matches!(err, FulltextError::Io(_)));

        for searcher in adapters() {
            let err = searcher
                .search(Path::new("/nonexistent/corpus"), "needle")
                .unwrap_err();
            assert!(matches!(err, FulltextError::Io(_)));
        }
    }

    #[test]
    fn empty_query_returns_no_hits() {
        let dir = corpus_dir("empty");
        std::fs::write(dir.join("a.html"), b"anything\n").unwrap();

        for searcher in adapters() {
            assert!(searcher.search(&dir, "   ").unwrap().is_empty());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ties_break_deterministically_by_name() {
        let dir = corpus_dir("tie");
        std::fs::write(dir.join("b.html"), b"needle once\n").unwrap();
        std::fs::write(dir.join("a.html"), b"needle once\n").unwrap();

        for searcher in adapters() {
            let hits = searcher.search(&dir, "needle").unwrap();

            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0].file, "a.html");
            assert_eq!(hits[1].file, "b.html");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn in_process_and_rg_adapters_agree() {
        if !ripgrep_present() {
            eprintln!("skipping: rg not installed");
            return;
        }
        let dir = corpus_dir("agree");
        std::fs::write(
            dir.join("b.html"),
            b"Rust ownership and the borrow checker\nborrow checker borrow checker\n",
        )
        .unwrap();
        std::fs::write(dir.join("a.html"), b"only one borrow checker mention\n").unwrap();
        std::fs::write(dir.join("c.html"), b"unrelated text\n").unwrap();

        let via_rg = RipgrepSearcher.search(&dir, "borrow checker").unwrap();
        let via_scan = InProcessSearcher.search(&dir, "borrow checker").unwrap();

        assert_eq!(via_rg, via_scan);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn snippet_truncates_long_lines() {
        let dir = corpus_dir("trunc");
        let long = format!("{} needle at the end", "x".repeat(1000));
        std::fs::write(dir.join("a.html"), format!("{long}\n")).unwrap();

        for searcher in adapters() {
            let hits = searcher.search(&dir, "needle").unwrap();
            assert_eq!(hits.len(), 1);
            assert!(hits[0].snippet.ends_with('…'));
            assert!(hits[0].snippet.chars().count() <= SNIPPET_MAX_CHARS + 1);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
