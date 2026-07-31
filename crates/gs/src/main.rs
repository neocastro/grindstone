//! gs — grindstone CLI.

use grindstone::{build_prompt, Issue};
use std::io::Read;

const VERSION: &str = env!("CARGO_PKG_VERSION");

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

    if sub.as_deref() != Some("build-prompt") {
        eprintln!("usage: gs build-prompt [REPO]   (issue JSON on stdin)");
        eprintln!("       gs --version");
        std::process::exit(2);
    }
    let repo = args.next().unwrap_or_else(|| "tlarc".to_string());

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("failed to read stdin");

    let issue: Issue = serde_json::from_str(input.trim())
        .expect("stdin must be issue JSON: {number, title, body}");

    print!("{}", build_prompt(&issue, &repo));
}
