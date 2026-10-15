use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::commands::search::{HIT_INDENT, resolve_spaces, root_label};
use crate::store::{SqliteStore, TextQuery};
use crate::workspace::{Workspace, expand_tilde};

/// List notes related to a given file.
#[derive(Debug, clap::Args)]
pub struct RelatedArgs {
    /// The file to find related items for.
    pub file: PathBuf,
    /// Restrict to these vector spaces (e.g. text,image). Defaults to all
    /// available — cross-modal by default.
    #[arg(long, value_delimiter = ',')]
    pub r#in: Vec<String>,
    /// Maximum number of results.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,
    /// Include items already linked from the source (Obsidian vaults).
    #[arg(long)]
    pub include_linked: bool,
    /// Restrict to documents having any of these tags (repeatable; matches
    /// any, not all).
    #[arg(long)]
    pub tag: Vec<String>,
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
    let spaces = resolve_spaces(ws, &args.r#in)?;

    let tags_ref = (!args.tag.is_empty()).then_some(args.tag.as_slice());
    let filter = TextQuery {
        from: None,
        tags: tags_ref,
    };

    let hits = query::run_related(
        &store,
        &path,
        &spaces,
        args.limit,
        args.include_linked,
        &filter,
        parse::extract_aliases,
    )?;

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
            println!("{HIT_INDENT}§ {}", hit.heading_path);
        }
    }
    Ok(())
}
