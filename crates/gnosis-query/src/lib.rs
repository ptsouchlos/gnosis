//! The retrieval pipeline: embed a query, retrieve both channels, fuse them,
//! merge across spaces.
//!
//! This is the one path every frontend takes. It lived in `gnosis-cli` next to
//! the `search` command until the Obsidian plugin needed it: a wasm client that
//! called `gnosis-search`'s primitives directly would have had to reimplement
//! what is in this file — the over-fetch, the two normalizations, the fusion,
//! the title bonus, the per-space merge — and every one of those is a measured
//! decision rather than an obvious one. A reimplementation would drift, and
//! once it drifts none of this project's measurements describe the client any
//! more.
//!
//! Kept free of `rusqlite`, `fastembed` and `std::fs` so it compiles to
//! `wasm32`: it takes a `&dyn Store` and a [`QueryEmbedder`], and both have
//! implementations that reach the network, the filesystem or neither.

use anyhow::Result;
use search::bm25::SearchConfig;
use store::{Space, Store, TextQuery};

/// How far past `limit` to reach for lexical candidates. A document worth
/// surfacing after fusion can sit well down the lexical ranking, so the
/// candidate set has to be wider than the result set.
///
/// Travels with the pipeline rather than with a frontend: a client that
/// over-fetched differently would fuse over a different candidate set and get
/// different results from the same index.
pub const LEXICAL_OVERFETCH: usize = 5;

/// Turns a query string into a vector in one space.
///
/// A trait rather than a concrete embedder because the implementations have
/// nothing in common below this line: the CLI loads ONNX models through
/// fastembed, and the plugin calls transformers.js across `wasm-bindgen`. What
/// they must agree on is the *model*, which the index records and the artifact
/// carries — a query embedded by a different model than built the index is not
/// a worse query, it is a meaningless one.
pub trait QueryEmbedder {
    fn embed_query(&mut self, space: Space, query: &str) -> Result<Vec<f32>>;
}

/// Run one query across `spaces` and merge the per-space results.
///
/// Keeping this in one place is the point: an evaluation harness that
/// reimplemented retrieval could not catch a retrieval bug, which is exactly
/// how the image-search defect survived its own tests.
pub fn run_query(
    store: &dyn Store,
    embedder: &mut dyn QueryEmbedder,
    query: &str,
    spaces: &[Space],
    limit: usize,
    filter: &TextQuery,
    cfg: &SearchConfig,
) -> Result<Vec<search::Hit>> {
    let mut per_space: Vec<Vec<search::Hit>> = Vec::with_capacity(spaces.len());
    for space in spaces.iter().copied() {
        let query_vec = embedder.embed_query(space, query)?;
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
pub fn fuse_text_space(
    store: &dyn Store,
    query: &str,
    dense: Vec<search::Hit>,
    limit: usize,
    filter: &TextQuery,
    cfg: &SearchConfig,
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
