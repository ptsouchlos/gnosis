use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::embed::{Embedder, TextEmbedder};
use crate::store::Store;
use crate::workspace::{Workspace, expand_tilde};

/// Semantic search over the indexed content.
#[derive(Debug, clap::Args)]
pub struct SearchArgs {
    /// The natural-language query.
    pub query: String,
    /// Restrict to these vector spaces (e.g. text,image). Defaults to all.
    #[arg(long, value_delimiter = ',')]
    pub r#in: Vec<String>,
    /// Restrict to documents from these vault roots (repeatable).
    #[arg(long)]
    pub from: Vec<PathBuf>,
    /// Maximum number of results.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,
    /// Print matching chunk text, not just paths.
    #[arg(long)]
    pub full: bool,
    /// Emit results as JSON.
    #[arg(long)]
    pub json: bool,
}

pub fn execute(ws: &Workspace, args: SearchArgs) -> Result<()> {
    if !ws.db_path.exists() {
        bail!(
            "no index found at {} — run `gnosis index`",
            ws.db_path.display()
        );
    }

    let store = Store::open(&ws.db_path)?;
    let mut embedder = TextEmbedder::new(&ws.config.embed.text.model)?;
    let query_vec = embedder
        .embed(&[args.query.clone()])?
        .into_iter()
        .next()
        .context("embedding produced no vector")?;

    // Resolve --from vault filters to canonical roots.
    let from: Vec<String> = args
        .from
        .iter()
        .filter_map(|p| std::fs::canonicalize(expand_tilde(p)).ok())
        .map(|c| c.to_string_lossy().to_string())
        .collect();
    let from_ref = (!from.is_empty()).then_some(from.as_slice());

    let hits = store.search_text(&query_vec, args.limit, from_ref)?;
    if hits.is_empty() {
        println!("No results.");
        return Ok(());
    }

    let show_root = ws.global || ws.config.vaults.len() > 1;
    for (i, hit) in hits.iter().enumerate() {
        let tag = if show_root {
            format!("[{}] ", root_label(&hit.source_root))
        } else {
            String::new()
        };
        println!(
            "{:>2}. [{:.3}] {tag}{}  ({})",
            i + 1,
            hit.score,
            hit.title,
            hit.path
        );
        if !hit.heading_path.is_empty() {
            println!("      § {}", hit.heading_path);
        }
        if args.full {
            let snippet: String = hit.text.chars().take(280).collect();
            println!("      {snippet}");
        }
    }
    Ok(())
}

/// Short label for a source vault: its final path component.
fn root_label(root: &str) -> &str {
    Path::new(root)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(root)
}
