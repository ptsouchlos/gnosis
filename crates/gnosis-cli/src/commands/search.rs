use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::embedder::{build_clip_text_embedder, build_text_embedder};
use crate::store::{SqliteStore, Store, TextQuery};
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
    /// Restrict to documents having any of these tags (repeatable; matches
    /// any, not all).
    #[arg(long)]
    pub tag: Vec<String>,
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

/// Vector spaces gnosis can index. `--in` is validated against this rather
/// than silently ignored, so requesting an unsupported space fails clearly.
pub(crate) const SUPPORTED_SPACES: &[&str] = &["text", "image"];

/// Resolve `--in`: explicit spaces are validated against `SUPPORTED_SPACES`;
/// an empty `--in` defaults to every space with content available — `text`
/// always, `image` only when `[embed.image] enabled = true`.
pub(crate) fn resolve_spaces(ws: &Workspace, requested: &[String]) -> Result<Vec<String>> {
    if requested.is_empty() {
        let mut spaces = vec!["text".to_string()];
        if ws.config.embed.image.enabled {
            spaces.push("image".to_string());
        }
        return Ok(spaces);
    }
    if let Some(unsupported) = requested.iter().find(|s| !SUPPORTED_SPACES.contains(&s.as_str())) {
        bail!(
            "unsupported space '{unsupported}' — only {} are indexed",
            SUPPORTED_SPACES.join(", ")
        );
    }
    Ok(requested.to_vec())
}

pub fn execute(ws: &Workspace, args: SearchArgs) -> Result<()> {
    if !ws.db_path.exists() {
        bail!(
            "no index found at {} — run `gnosis index`",
            ws.db_path.display()
        );
    }

    let spaces = resolve_spaces(ws, &args.r#in)?;
    let store = SqliteStore::open(&ws.db_path)?;

    // Resolve --from vault filters to canonical roots.
    let from: Vec<String> = args
        .from
        .iter()
        .filter_map(|p| std::fs::canonicalize(expand_tilde(p)).ok())
        .map(|c| c.to_string_lossy().to_string())
        .collect();
    let from_ref = (!from.is_empty()).then_some(from.as_slice());
    let tags_ref = (!args.tag.is_empty()).then_some(args.tag.as_slice());
    let filter = TextQuery {
        from: from_ref,
        tags: tags_ref,
    };

    let mut per_space: Vec<Vec<search::Hit>> = Vec::with_capacity(spaces.len());
    for space in &spaces {
        let query_vec = match space.as_str() {
            "text" => {
                let mut embedder = build_text_embedder(&ws.config.embed.text.model)?;
                embedder
                    .embed(&[args.query.clone()])?
                    .into_iter()
                    .next()
                    .context("embedding produced no vector")?
            }
            "image" => {
                let mut embedder = build_clip_text_embedder(&ws.config.embed.image.model)?;
                embedder
                    .embed(&[args.query.clone()])?
                    .into_iter()
                    .next()
                    .context("embedding produced no vector")?
            }
            other => bail!("unsupported space '{other}'"),
        };
        per_space.push(store.search_space(space, &query_vec, args.limit, &filter)?);
    }
    let hits = search::merge_normalized(per_space, args.limit);

    if args.json {
        // Full data regardless of --full: JSON output is for programmatic
        // consumers, not terminal readability, and a script piping --json
        // output shouldn't have to special-case an empty non-JSON message.
        println!("{}", serde_json::to_string(&hits)?);
        return Ok(());
    }

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
            match (hit.width, hit.height) {
                (Some(w), Some(h)) => println!("      {w}x{h}"),
                _ => {
                    let snippet: String = hit.text.chars().take(280).collect();
                    if !snippet.is_empty() {
                        println!("      {snippet}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Short label for a source vault: its final path component.
pub(crate) fn root_label(root: &str) -> &str {
    Path::new(root)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(root)
}
