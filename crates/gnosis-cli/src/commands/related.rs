use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::commands::search::{resolve_spaces, root_label};
use crate::store::{SqliteStore, Store, TextQuery};
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
    if store.document_hash(&path)?.is_none() {
        bail!("{} is not indexed — run `gnosis index`", canon.display());
    }

    let spaces = resolve_spaces(ws, &args.r#in)?;

    let mut exclude = vec![path.clone()];
    if !args.include_linked {
        let targets = store.linked_targets(&path)?;
        if !targets.is_empty() {
            let all_meta = store.all_document_meta()?;
            exclude.extend(resolve_link_targets(&targets, &all_meta));
        }
    }

    let tags_ref = (!args.tag.is_empty()).then_some(args.tag.as_slice());
    let filter = TextQuery {
        from: None,
        tags: tags_ref,
    };

    let mut per_space: Vec<Vec<search::Hit>> = Vec::with_capacity(spaces.len());
    for space in &spaces {
        let queries = store.chunk_vectors(&path, space)?;
        if queries.is_empty() {
            // This file has no vectors in this space (e.g. an image queried
            // with `--in text`) — contributes nothing to the merge, not an
            // error, unless every requested space is empty (checked below).
            continue;
        }
        per_space.push(store.related_space(space, &queries, &exclude, args.limit, &filter)?);
    }
    if per_space.is_empty() {
        bail!(
            "{} has no chunks in any of [{}] to compare",
            canon.display(),
            spaces.join(", ")
        );
    }
    let hits = search::merge_normalized(per_space, args.limit);

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
/// against each candidate's filename stem (Obsidian's own default
/// link-resolution behavior — by filename, not any custom title) or its
/// frontmatter `aliases:`, case-insensitively. Not full Obsidian-compatible
/// resolution: no folder-path disambiguation.
fn resolve_link_targets(targets: &[String], all_meta: &[(String, Option<String>)]) -> Vec<String> {
    all_meta
        .iter()
        .filter(|(path, frontmatter)| {
            let stem = Path::new(path).file_stem().and_then(|s| s.to_str());
            let stem_matches =
                stem.is_some_and(|stem| targets.iter().any(|t| t.eq_ignore_ascii_case(stem)));
            let alias_matches = frontmatter.as_deref().is_some_and(|fm| {
                parse::extract_aliases(fm)
                    .iter()
                    .any(|a| targets.iter().any(|t| t.eq_ignore_ascii_case(a)))
            });
            stem_matches || alias_matches
        })
        .map(|(path, _)| path.clone())
        .collect()
}
