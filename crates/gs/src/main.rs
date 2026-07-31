//! gs — grindstone CLI.

use grindstone::{build_prompt, fulltext, ingest, manifest::Manifest, Issue};
use std::io::Read;
use std::path::PathBuf;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_CORPUS_DIR: &str = "corpus";

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
