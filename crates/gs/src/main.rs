//! gs — grindstone CLI.

use grindstone::{
    build_prompt, chunk, embed, eval, fulltext, ingest, manifest,
    manifest::{Manifest, TrustTier},
    vector::{self, VectorStore},
    Issue,
};
use std::io::Read;
use std::path::{Path, PathBuf};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_CORPUS_DIR: &str = "corpus";
const DEFAULT_EVAL_SET: &str = "eval/evalset.json";
const DEFAULT_INDEX_DIR: &str = "index";
const BASELINE_RESULT: &str = "eval/results/fulltext-baseline.json";
const COSINE_RESULT: &str = "eval/results/cosine.json";

/// A retrieval strategy for the eval harness: query → ranked doc ids.
type EvalStrategy = Box<dyn FnMut(&str) -> Result<Vec<String>, eval::EvalError>>;

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
        Some("chunk") => cmd_chunk(args),
        Some("embed") => cmd_embed(args),
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
    eprintln!("       gs query [--embed] [--tier TIER] QUERY [DIR] [OLLAMA_URL]");
    eprintln!("             --embed: cosine over INDEX_DIR (default: index); default: fulltext over CORPUS_DIR");
    eprintln!("             --tier: pinned-source|docs-wiki|navigational (cosine only)");
    eprintln!("       gs eval [--strategy fulltext|cosine] [CORPUS_DIR] [EVAL_SET]");
    eprintln!("       gs chunk [CORPUS_DIR] [INDEX_DIR]   (defaults: corpus, index)");
    eprintln!(
        "       gs embed [INDEX_DIR] [OLLAMA_URL]   (defaults: index, http://127.0.0.1:11434)"
    );
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

/// `gs query [--embed] [--tier TIER] QUERY [DIR] [OLLAMA_URL]` — two
/// retrieval strategies behind one flag:
/// - default: full-text `rg` over the corpus (the deliberately-dumb baseline,
///   fully offline, ranked by match count with an evidence snippet);
/// - `--embed`: embed the query via local Ollama and rank chunks by cosine
///   similarity through the vector store, with provenance metadata per hit
///   and an optional trust-tier filter.
fn cmd_query(mut args: impl Iterator<Item = String>) {
    let mut embed_mode = false;
    let mut tier: Option<TrustTier> = None;
    let mut positional: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--embed" => embed_mode = true,
            "--tier" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("gs query: --tier requires a value");
                    std::process::exit(2);
                });
                tier = Some(match parse_tier(&value) {
                    Ok(t) => t,
                    Err(msg) => {
                        eprintln!("gs query: {msg}");
                        std::process::exit(2);
                    }
                });
            }
            flag if flag.starts_with('-') && flag != "-" => {
                eprintln!("gs query: unknown flag {flag}");
                usage();
                std::process::exit(2);
            }
            pos => positional.push(pos.to_string()),
        }
    }

    let query = match positional.first() {
        Some(q) => q.clone(),
        None => {
            eprintln!("gs query: missing QUERY argument");
            usage();
            std::process::exit(2);
        }
    };

    if embed_mode {
        let index_dir = PathBuf::from(
            positional
                .get(1)
                .cloned()
                .unwrap_or_else(|| DEFAULT_INDEX_DIR.to_string()),
        );
        let server_url = positional
            .get(2)
            .cloned()
            .unwrap_or_else(|| embed::DEFAULT_OLLAMA_URL.to_string());
        cmd_query_cosine(&query, &index_dir, &server_url, tier);
        return;
    }

    if tier.is_some() {
        eprintln!("gs query: --tier only applies with --embed");
        std::process::exit(2);
    }
    let corpus_dir = PathBuf::from(
        positional
            .get(1)
            .cloned()
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

/// The cosine query path: load the vector store, embed the query, rank, and
/// print top-k hits with score + provenance metadata.
fn cmd_query_cosine(query: &str, index_dir: &Path, server_url: &str, tier: Option<TrustTier>) {
    let store = match VectorStore::load(index_dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gs query: {e}");
            std::process::exit(1);
        }
    };
    let call =
        |inputs: &[String]| embed::ollama_embed(server_url, embed::DEFAULT_EMBED_MODEL, inputs);
    let vectors = match call(&[query.to_string()]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("gs query: {e}");
            std::process::exit(1);
        }
    };
    let hits = store.search(&vectors[0], vector::DEFAULT_TOP_K, tier);
    if hits.is_empty() {
        println!(
            "no hits for {query:?} (store: {} chunks, dim {}, model {})",
            store.len(),
            store.dim,
            store.model
        );
        return;
    }
    for (i, hit) in hits.iter().enumerate() {
        println!(
            "{}. ({:.3}) {} — {}",
            i + 1,
            hit.score,
            hit.chunk.source,
            hit.chunk.heading
        );
        println!("   {}", snippet(&hit.chunk.text, 140));
        println!(
            "   license: {} | tier: {}",
            hit.chunk.license,
            tier_name(hit.chunk.tier)
        );
    }
}

/// Map a trust-tier name (the serde rename) to its enum value.
fn parse_tier(s: &str) -> Result<TrustTier, String> {
    match s {
        "pinned-source" => Ok(TrustTier::PinnedSource),
        "docs-wiki" => Ok(TrustTier::DocsWiki),
        "navigational" => Ok(TrustTier::Navigational),
        other => Err(format!(
            "unknown tier {other:?} (expected pinned-source|docs-wiki|navigational)"
        )),
    }
}

/// The serde rename of a trust tier (stable, user-facing).
fn tier_name(t: TrustTier) -> String {
    match serde_json::to_value(t) {
        Ok(serde_json::Value::String(s)) => s,
        _ => "unknown".to_string(),
    }
}

/// First `max` characters of `text` (whitespace-trimmed), ellipsized.
fn snippet(text: &str, max: usize) -> String {
    let t = text.trim();
    if t.chars().count() <= max {
        return t.to_string();
    }
    let cut: String = t.chars().take(max).collect();
    format!("{cut}…")
}

/// `gs eval [--strategy fulltext|cosine] [CORPUS_DIR] [EVAL_SET]` — run a
/// retrieval strategy over the eval set, print recall@k (k=5, k=10) per
/// query and overall, and persist the result to
/// `eval/results/<strategy>.json` so every strategy has a number to beat on
/// the same eval set. `fulltext` is the offline baseline; `cosine` embeds
/// each query via local Ollama and retrieves through the vector store, then
/// prints the delta against the persisted fulltext baseline.
fn cmd_eval(mut args: impl Iterator<Item = String>) {
    let mut strategy_name = "fulltext".to_string();
    let first = args.next();
    if first.as_deref() == Some("--strategy") {
        strategy_name = match args.next() {
            Some(s) => s,
            None => {
                eprintln!("gs eval: --strategy requires a value (fulltext|cosine)");
                std::process::exit(2);
            }
        };
    }
    let mut rest = first.into_iter().chain(args);
    let corpus_dir = PathBuf::from(
        rest.next()
            .unwrap_or_else(|| DEFAULT_CORPUS_DIR.to_string()),
    );
    let eval_set_path = PathBuf::from(rest.next().unwrap_or_else(|| DEFAULT_EVAL_SET.to_string()));

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

    let (result_path, mut strategy): (PathBuf, EvalStrategy) = match strategy_name.as_str() {
        "fulltext" => (
            PathBuf::from(BASELINE_RESULT),
            Box::new(move |query: &str| eval::fulltext_doc_ids(&corpus_dir, query)),
        ),
        "cosine" => {
            let index_dir = PathBuf::from(DEFAULT_INDEX_DIR);
            let store = match VectorStore::load(&index_dir) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("gs eval: {e}");
                    std::process::exit(1);
                }
            };
            let server_url = embed::DEFAULT_OLLAMA_URL.to_string();
            (
                PathBuf::from(COSINE_RESULT),
                Box::new(move |query: &str| {
                    let mut call = |inputs: &[String]| {
                        embed::ollama_embed(&server_url, embed::DEFAULT_EMBED_MODEL, inputs)
                    };
                    eval::cosine_doc_ids(&store, &mut call, query)
                }),
            )
        }
        other => {
            eprintln!("gs eval: unknown strategy {other:?} (fulltext|cosine)");
            std::process::exit(2);
        }
    };

    let mut result = match eval::run_eval(&strategy_name, &mut strategy, &set) {
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

    match eval::save_result(&result_path, &result) {
        Ok(()) => println!("saved {}", result_path.display()),
        Err(e) => {
            eprintln!("gs eval: cannot write {}: {e}", result_path.display());
            std::process::exit(1);
        }
    }

    if strategy_name != "fulltext" {
        match eval::EvalResult::load(Path::new(BASELINE_RESULT)) {
            Ok(base) => {
                let hash = base.corpus_hash.get(..8).unwrap_or(&base.corpus_hash);
                println!(
                    "vs fulltext baseline (corpus {hash}): recall@5 {:+.3}  recall@10 {:+.3}",
                    result.overall_recall_5 - base.overall_recall_5,
                    result.overall_recall_10 - base.overall_recall_10
                );
            }
            Err(e) => {
                eprintln!(
                    "gs eval: cannot compare to baseline {}: {e}",
                    BASELINE_RESULT
                );
            }
        }
    }
}

/// `gs chunk [CORPUS_DIR] [INDEX_DIR]` — chunk every corpus document into
/// deterministic heading-aware chunks and persist to `INDEX_DIR/chunks.json`.
/// Same manifest + corpus → identical chunks (bit-for-bit reproducible).
fn cmd_chunk(mut args: impl Iterator<Item = String>) {
    let corpus_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| DEFAULT_CORPUS_DIR.to_string()),
    );
    let index_dir = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_INDEX_DIR.to_string()));

    let manifest_path = corpus_dir.join("manifest.json");
    let manifest = match Manifest::load(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gs chunk: cannot load {}: {e}", manifest_path.display());
            std::process::exit(2);
        }
    };

    let mut chunks = Vec::new();
    for source in &manifest.sources {
        let path = corpus_dir.join(source.filename());
        let html = match std::fs::read_to_string(&path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("gs chunk: cannot read {}: {e}", path.display());
                std::process::exit(1);
            }
        };
        let doc_chunks = chunk::chunk_source(&html, source);
        println!("{:>6} chunks  {}", doc_chunks.len(), source.name);
        chunks.extend(doc_chunks);
    }

    let file = chunk::ChunksFile { version: 1, chunks };
    let out = index_dir.join("chunks.json");
    if let Err(e) = file.save(&out) {
        eprintln!("gs chunk: cannot write {}: {e}", out.display());
        std::process::exit(1);
    }
    println!("wrote {} ({} chunks)", out.display(), file.chunks.len());
}

/// `gs embed [INDEX_DIR] [OLLAMA_URL]` — embed every chunk via local Ollama
/// `nomic-embed-text` and persist to `INDEX_DIR/embeddings.json` for the
/// vector store (RAG-4) to consume. Fails with a clear error when Ollama is
/// down; never hangs (hard timeout).
fn cmd_embed(mut args: impl Iterator<Item = String>) {
    let index_dir = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_INDEX_DIR.to_string()));
    let server_url = args
        .next()
        .unwrap_or_else(|| embed::DEFAULT_OLLAMA_URL.to_string());

    let chunks_path = index_dir.join("chunks.json");
    let file = match chunk::ChunksFile::load(&chunks_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("gs embed: cannot load {}: {e}", chunks_path.display());
            std::process::exit(2);
        }
    };

    let mut call =
        |inputs: &[String]| embed::ollama_embed(&server_url, embed::DEFAULT_EMBED_MODEL, inputs);
    let embeddings = match embed::embed_chunks(
        &file.chunks,
        embed::DEFAULT_EMBED_MODEL,
        embed::DEFAULT_BATCH_SIZE,
        &mut call,
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("gs embed: {e}");
            std::process::exit(1);
        }
    };

    let out = index_dir.join("embeddings.json");
    if let Err(e) = embeddings.save(&out) {
        eprintln!("gs embed: cannot write {}: {e}", out.display());
        std::process::exit(1);
    }
    println!(
        "wrote {} ({} vectors, dim {})",
        out.display(),
        embeddings.vectors.len(),
        embeddings.dim
    );
}
