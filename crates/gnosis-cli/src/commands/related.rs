use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::commands::search::root_label;
use crate::store::{SqliteStore, Store};
use crate::workspace::{Workspace, expand_tilde};

/// List notes related to a given file.
#[derive(Debug, clap::Args)]
pub struct RelatedArgs {
    /// The file to find related items for.
    pub file: PathBuf,
    /// Maximum number of results.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,
    /// Include items already linked from the source (Obsidian vaults).
    #[arg(long)]
    pub include_linked: bool,
    /// Emit results as JSON.
    #[arg(long)]
    pub json: bool,
}

pub fn execute(ws: &Workspace, args: RelatedArgs) -> Result<()> {
    if !ws.db_path.exists() {
        bail!(
            "no index found at {} — run `gnosis index`",
            ws.db_path.display()
        );
    }

    let canon = std::fs::canonicalize(expand_tilde(&args.file))
        .with_context(|| format!("resolving {}", args.file.display()))?;
    let path = canon.to_string_lossy().to_string();

    let store = SqliteStore::open(&ws.db_path)?;
    if store.document_hash(&path)?.is_none() {
        bail!("{} is not indexed — run `gnosis index`", canon.display());
    }

    let queries = store.text_chunk_vectors(&path)?;
    if queries.is_empty() {
        bail!("{} has no text chunks to compare", canon.display());
    }

    let mut exclude = vec![path.clone()];
    if !args.include_linked {
        let targets = store.linked_targets(&path)?;
        if !targets.is_empty() {
            let all_paths = store.all_paths()?;
            exclude.extend(resolve_link_targets(&targets, &all_paths));
        }
    }

    let hits = store.related_text(&queries, &exclude, args.limit)?;

    if args.json {
        // Full data regardless of terminal formatting — see search.rs's
        // --json for the same reasoning (programmatic consumers, no
        // "No results." special-casing).
        println!("{}", serde_json::to_string(&hits)?);
        return Ok(());
    }

    if hits.is_empty() {
        println!("No related notes found.");
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
    }
    Ok(())
}

/// Resolve raw wikilink target texts to indexed document paths by matching
/// against each candidate's filename stem, case-insensitively — Obsidian's
/// own default link-resolution behavior (by filename, not by any custom
/// title). Not full Obsidian-compatible resolution: no folder-path
/// disambiguation, no alias support beyond what `gnosis-parse` already
/// strips.
fn resolve_link_targets(targets: &[String], all_paths: &[String]) -> Vec<String> {
    all_paths
        .iter()
        .filter(|p| {
            let stem = Path::new(p).file_stem().and_then(|s| s.to_str());
            stem.is_some_and(|stem| {
                targets
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(stem))
            })
        })
        .cloned()
        .collect()
}
