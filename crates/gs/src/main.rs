//! gs — grindstone CLI.

use grindstone::{build_prompt, eval, fulltext, ingest, manifest, manifest::Manifest, Issue};
use std::io::Read;
use std::path::PathBuf;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_CORPUS_DIR: &str = "corpus";
const DEFAULT_EVAL_SET: &str = "eval/evalset.json";
const BASELINE_RESULT: &str = "eval/results/fulltext-baseline.json";

fn main() {
    let mut args = std::env::args().skip(1);
    let sub = args.next();

    // `gs --version` identifies the grindstone CLI (not Ghostscript, which
    // also installs a binary named `gs`). The runner checks this marker
    // before trusting the binary.
    if sub.as_deref() == Some("--version") || sub.as_deref() == Some("version") {
        println!("grindstone {VERSION}");
        return;
    }

    match sub.as_deref() {
        Some("build-prompt") => cmd_build_prompt(args),
        Some("ingest") => cmd_ingest(args),
        Some("query") => cmd_query(args),
        Some("eval") => cmd_eval(args),
        Some("-h" | "--help") => usage(),
        Some(other) => {
            eprintln!("gs: unknown subcommand `{other}`");
            usage();
            std::process::exit(2);
        }
        None => {
            usage();
            std::process::exit(2);
        }
    }
}

fn usage() {
    eprintln!("usage: gs build-prompt [REPO]   (issue JSON on stdin)");
    eprintln!("       gs ingest [CORPUS_DIR]   (default: corpus)");
    eprintln!("       gs query QUERY [CORPUS_DIR]   (default: corpus)");
    eprintln!("       gs eval [CORPUS_DIR] [EVAL_SET]   (defaults: corpus, eval/evalset.json)");
    eprintln!("       gs --version");
}

fn cmd_build_prompt(mut args: impl Iterator<Item = String>) {
    let repo = args.next().unwrap_or_else(|| "tlarc".to_string());

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("failed to read stdin");

    let issue: Issue = serde_json::from_str(input.trim())
        .expect("stdin must be issue JSON: {number, title, body}");

    print!("{}", build_prompt(&issue, &repo));
}

/// `gs ingest [CORPUS_DIR]` — fetch every source whose corpus file is missing
/// or does not match the manifest's pinned hash, then write the resolved
/// manifest (content hashes filled in) back to `CORPUS_DIR/manifest.json`.
/// Re-running with an unchanged manifest is a no-op: no re-download.
fn cmd_ingest(mut args: impl Iterator<Item = String>) {
    let corpus_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| DEFAULT_CORPUS_DIR.to_string()),
    );
    let manifest_path = corpus_dir.join("manifest.json");

    let manifest = match Manifest::load(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gs ingest: cannot load {}: {e}", manifest_path.display());
            std::process::exit(2);
        }
    };

    let mut fetcher = ingest::http_fetcher;
    match ingest::ingest(&manifest, &corpus_dir, &mut fetcher) {
        Ok(report) => {
            for (source, action) in &report.actions {
                let label = match action {
                    ingest::SourceAction::Fetched => "fetched",
                    ingest::SourceAction::Skipped => "skipped",
                };
                println!("{label:8} {}", source.name);
            }
            println!(
                "ingest complete: {} source(s) in {}",
                report.actions.len(),
                corpus_dir.display()
            );
        }
        Err(e) => {
            eprintln!("gs ingest: {e}");
            std::process::exit(1);
        }
    }
}

/// `gs query QUERY [CORPUS_DIR]` — the deliberately-dumb retrieval baseline:
/// plain `rg` full-text over the corpus, fully offline, ranked by match count
/// per file with an evidence snippet. Every later retrieval strategy must
/// measurably beat this.
fn cmd_query(mut args: impl Iterator<Item = String>) {
    let query = match args.next() {
        Some(q) => q,
        None => {
            eprintln!("gs query: missing QUERY argument");
            usage();
            std::process::exit(2);
        }
    };
    let corpus_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| DEFAULT_CORPUS_DIR.to_string()),
    );

    match fulltext::search(&corpus_dir, &query) {
        Ok(hits) => {
            if hits.is_empty() {
                println!("no hits for {query:?} in {}", corpus_dir.display());
                return;
            }
            for (i, hit) in hits.iter().enumerate() {
                let plural = if hit.matches == 1 { "" } else { "es" };
                println!("{}. {} ({} match{plural})", i + 1, hit.file, hit.matches);
                println!("   {}", hit.snippet);
            }
        }
        Err(e) => {
            eprintln!("gs query: {e}");
            std::process::exit(1);
        }
    }
}

/// `gs eval [CORPUS_DIR] [EVAL_SET]` — run the full-text baseline over the
/// eval set, print recall@k (k=5, k=10) per query and overall, and persist
/// the result to `eval/results/fulltext-baseline.json` so later retrieval
/// strategies have a number to beat on the same eval set. Fully offline.
fn cmd_eval(mut args: impl Iterator<Item = String>) {
    let corpus_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| DEFAULT_CORPUS_DIR.to_string()),
    );
    let eval_set_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_EVAL_SET.to_string()));

    let set = match eval::EvalSet::load(&eval_set_path) {
        Ok(set) => set,
        Err(e) => {
            eprintln!("gs eval: cannot load {}: {e}", eval_set_path.display());
            std::process::exit(2);
        }
    };
    let problems = set.validate();
    if !problems.is_empty() {
        for p in &problems {
            eprintln!("gs eval: invalid eval set: {p}");
        }
        std::process::exit(2);
    }

    // Corpus hash ties the recorded score to the corpus state (deterministic,
    // no wall-clock timestamps — reproducibility).
    let manifest_path = corpus_dir.join("manifest.json");
    let corpus_hash = std::fs::read(&manifest_path)
        .map(|bytes| manifest::sha256_hex(&bytes))
        .unwrap_or_default();

    // Warn about expected doc ids that are not in the corpus yet (e.g. the
    // TLA+ doc id before RAG-6 lands): they score 0 until that corpus exists.
    let known_docs: std::collections::HashSet<String> = Manifest::load(&manifest_path)
        .map(|m| m.sources.iter().map(|s| s.name.clone()).collect())
        .unwrap_or_default();
    for q in &set.queries {
        for expected in &q.expected {
            if !known_docs.is_empty() && !known_docs.contains(expected) {
                eprintln!(
                    "gs eval: warning: expected doc '{expected}' for '{}' is not in the corpus (scores 0 until it lands)",
                    q.id
                );
            }
        }
    }

    let mut strategy = |query: &str| eval::fulltext_doc_ids(&corpus_dir, query);
    let mut result = match eval::run_eval("fulltext", &mut strategy, &set) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gs eval: {e}");
            std::process::exit(1);
        }
    };
    result.corpus_hash = corpus_hash;

    for q in &result.queries {
        println!(
            "recall@5 {:>5.3}  recall@10 {:>5.3}  {:<24} {}",
            q.recall_5, q.recall_10, q.id, q.query
        );
    }
    println!(
        "overall recall@5 {:>5.3}  recall@10 {:>5.3}  (strategy: {})",
        result.overall_recall_5, result.overall_recall_10, result.strategy
    );

    let result_path = PathBuf::from(BASELINE_RESULT);
    match eval::save_result(&result_path, &result) {
        Ok(()) => println!("saved {}", result_path.display()),
        Err(e) => {
            eprintln!("gs eval: cannot write {}: {e}", result_path.display());
            std::process::exit(1);
        }
    }
}
