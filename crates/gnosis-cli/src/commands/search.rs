use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::embedder::{build_clip_text_embedder, build_text_embedder};
use crate::store::{Space, SqliteStore, Store, TextQuery};
use crate::workspace::{Workspace, expand_tilde};

/// How far past `limit` to reach for lexical candidates. A document worth
/// surfacing after fusion can sit well down the lexical ranking, so the
/// candidate set has to be wider than the result set.
const LEXICAL_OVERFETCH: usize = 5;

/// Left margin used for every indented detail line under a hit (heading
/// path, image dimensions, text snippet) — shared so the columns line up
/// regardless of which branch prints.
pub(crate) const HIT_INDENT: &str = "      ";

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

/// Resolve `--in`: explicit spaces are parsed and validated as `Space` (an
/// unsupported name fails clearly via `Space::from_str`'s error); an empty
/// `--in` defaults to every space with content available — `text` always,
/// `image` only when `[embed.image] enabled = true`.
pub(crate) fn resolve_spaces(ws: &Workspace, requested: &[String]) -> Result<Vec<Space>> {
    if requested.is_empty() {
        let mut spaces = vec![Space::Text];
        if ws.config.embed.image.enabled {
            spaces.push(Space::Image);
        }
        return Ok(spaces);
    }
    requested.iter().map(|s| s.parse()).collect()
}

/// Query-side embedders, built on first use and reused afterwards.
///
/// `search` runs a single query so this is immaterial there, but `eval` runs
/// a whole qrels file through the same path, and constructing a fastembed
/// model per query would dominate its runtime.
pub(crate) struct QueryEmbedders<'a> {
    ws: &'a Workspace,
    text: Option<Box<dyn embed::Embedder>>,
    clip_text: Option<Box<dyn embed::Embedder>>,
}

impl<'a> QueryEmbedders<'a> {
    pub(crate) fn new(ws: &'a Workspace) -> Self {
        Self {
            ws,
            text: None,
            clip_text: None,
        }
    }

    /// Embed `query` into `space`'s vector space, loading that space's model
    /// on first use.
    pub(crate) fn embed(&mut self, space: Space, query: &str) -> Result<Vec<f32>> {
        let embedder = match space {
            Space::Text => {
                if self.text.is_none() {
                    self.text = Some(build_text_embedder(&self.ws.config.embed.text.model, self.ws.config.embed.text.batch_size)?);
                }
                self.text.as_mut().expect("just built")
            }
            Space::Image => {
                if self.clip_text.is_none() {
                    self.clip_text =
                        Some(build_clip_text_embedder(&self.ws.config.embed.image.model)?);
                }
                self.clip_text.as_mut().expect("just built")
            }
        };
        embedder
            .embed(std::slice::from_ref(&query.to_string()))?
            .into_iter()
            .next()
            .context("embedding produced no vector")
    }
}

/// Run one query across `spaces` and merge the per-space results.
///
/// The single retrieval path shared by `search` and `eval`. Keeping it in one
/// place is the point: an evaluation harness that reimplemented retrieval
/// could not catch a retrieval bug, which is exactly how the image-search
/// defect survived its own tests.
pub(crate) fn run_query(
    store: &SqliteStore,
    embedders: &mut QueryEmbedders,
    query: &str,
    spaces: &[Space],
    limit: usize,
    filter: &TextQuery,
) -> Result<Vec<search::Hit>> {
    let cfg = &embedders.ws.config.search;

    let mut per_space: Vec<Vec<search::Hit>> = Vec::with_capacity(spaces.len());
    for space in spaces.iter().copied() {
        let query_vec = embedders.embed(space, query)?;
        let dense = store.search_space(space, &query_vec, limit, filter)?;

        // Lexical matching applies to the text space only: image chunks carry
        // no text, and title proxies exist for `related` to traverse.
        if space == Space::Text {
            // The title bonus is independent of the lexical channel, so it is
            // applied either way — `fuse_text_space` does it as part of fusing,
            // and the dense-only path does it here.
            let ranked = if cfg.lexical {
                fuse_text_space(store, query, dense, limit, filter, cfg)?
            } else {
                let terms = search::bm25::tokenize(query);
                search::boost_and_rank(dense, &terms, cfg.title_boost, limit)
            };
            per_space.push(ranked);
        } else {
            per_space.push(dense);
        }
    }
    Ok(search::merge_normalized(per_space, limit))
}

/// Score the lexical channel for `query` and fuse it with the dense ranking.
///
/// Both channels are normalized against their own theoretical ceiling before
/// fusing — dense against 1.0, since chunk vectors are unit-norm so cosine
/// cannot exceed it, and lexical against BM25+'s per-query maximum. Neither is
/// min-max normalized, so a weak result set stays weak.
fn fuse_text_space(
    store: &SqliteStore,
    query: &str,
    dense: Vec<search::Hit>,
    limit: usize,
    filter: &TextQuery,
    cfg: &search::bm25::SearchConfig,
) -> Result<Vec<search::Hit>> {
    let terms = search::bm25::tokenize(query);
    if terms.is_empty() {
        return Ok(dense);
    }

    // Over-fetch the lexical side: a document fusion should surface may rank
    // well below `limit` on lexical score alone.
    let matches = store.lexical_candidates(&terms, limit * LEXICAL_OVERFETCH, filter)?;
    let query_terms: Vec<search::bm25::QueryTerm> = matches
        .doc_freqs
        .iter()
        .map(|&doc_freq| search::bm25::QueryTerm { doc_freq })
        .collect();
    let ceiling = search::bm25::max_score(&query_terms, matches.stats, cfg.bm25);

    let lexical: Vec<search::Hit> = matches
        .candidates
        .into_iter()
        .map(|c| {
            // Score over heading plus body, matching what the index contains
            // and how the corpus average length is measured. A term appearing
            // only in a chunk's heading is exactly the lookup-by-note-name
            // case a lexical channel exists for.
            let scored_text = format!("{} {}", c.heading_path, c.text);
            let m = search::bm25::LexicalMatch {
                // Characters, matching how the corpus average is measured.
                chunk_len: scored_text.chars().count() as u32,
                term_freqs: search::bm25::term_freqs(&scored_text, &terms),
            };
            let raw = search::bm25::score(&m, &query_terms, matches.stats, cfg.bm25);
            search::Hit {
                path: c.path,
                title: c.title,
                source_root: c.source_root,
                heading_path: c.heading_path,
                text: c.text,
                score: search::bm25::normalize(raw, ceiling),
                width: c.width,
                height: c.height,
            }
        })
        .collect();

    Ok(search::fuse_hits(dense, lexical, &terms, cfg, limit))
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

    let mut embedders = QueryEmbedders::new(ws);
    let hits = run_query(
        &store,
        &mut embedders,
        &args.query,
        &spaces,
        args.limit,
        &filter,
    )?;

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
            println!("{HIT_INDENT}§ {}", hit.heading_path);
        }
        if args.full {
            match (hit.width, hit.height) {
                (Some(w), Some(h)) => println!("{HIT_INDENT}{w}x{h}"),
                _ => {
                    let snippet: String = hit.text.chars().take(280).collect();
                    if !snippet.is_empty() {
                        println!("{HIT_INDENT}{snippet}");
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
