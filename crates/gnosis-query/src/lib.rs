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

#[cfg(test)]
mod tests {
    use super::*;
    use search::Candidate;
    use store::{DocWrite, LexicalMatches, Stats};

    /// A store that answers only what the pipeline asks of it, so the
    /// orchestration can be tested without SQLite — and, once the plugin
    /// exists, in wasm. Everything else is left unimplemented on purpose: a
    /// method the pipeline starts calling should show up as a failing test
    /// rather than as a silently empty result.
    struct FakeStore {
        dense: Vec<search::Hit>,
        lexical: LexicalMatches,
    }

    fn hit(path: &str, score: f32) -> search::Hit {
        search::Hit {
            path: path.to_string(),
            title: path.trim_end_matches(".md").to_string(),
            source_root: "/vault".to_string(),
            heading_path: String::new(),
            text: String::new(),
            score,
            width: None,
            height: None,
        }
    }

    fn candidate(path: &str, text: &str) -> Candidate {
        Candidate {
            path: path.to_string(),
            title: path.trim_end_matches(".md").to_string(),
            source_root: "/vault".to_string(),
            heading_path: String::new(),
            text: text.to_string(),
            vector: vec![1.0, 0.0],
            width: None,
            height: None,
            modality: "text".to_string(),
        }
    }

    impl Store for FakeStore {
        fn search_space(
            &self,
            _space: Space,
            _query: &[f32],
            limit: usize,
            _filter: &TextQuery,
        ) -> anyhow::Result<Vec<search::Hit>> {
            Ok(self.dense.iter().take(limit).cloned().collect())
        }

        fn lexical_candidates(
            &self,
            _terms: &[String],
            _limit: usize,
            _filter: &TextQuery,
        ) -> anyhow::Result<LexicalMatches> {
            Ok(self.lexical.clone())
        }

        fn set_meta(&self, _k: &str, _v: &str) -> anyhow::Result<()> {
            unimplemented!("the pipeline does not write")
        }
        fn get_meta(&self, _k: &str) -> anyhow::Result<Option<String>> {
            unimplemented!("the pipeline does not read meta")
        }
        fn stats(&self) -> anyhow::Result<Stats> {
            unimplemented!()
        }
        fn document_hash(&self, _p: &str) -> anyhow::Result<Option<Vec<u8>>> {
            unimplemented!()
        }
        fn paths_for_roots(&self, _r: &[String]) -> anyhow::Result<Vec<String>> {
            unimplemented!()
        }
        fn counts_by_root(&self) -> anyhow::Result<Vec<(String, i64)>> {
            unimplemented!()
        }
        fn delete_by_root(&self, _r: &str) -> anyhow::Result<usize> {
            unimplemented!()
        }
        fn replace_document(&mut self, _d: &DocWrite<'_>) -> anyhow::Result<()> {
            unimplemented!()
        }
        fn delete_document(&self, _p: &str) -> anyhow::Result<()> {
            unimplemented!()
        }
        fn chunk_vectors(&self, _p: &str, _s: Space) -> anyhow::Result<Vec<Vec<f32>>> {
            unimplemented!()
        }
        fn linked_targets(&self, _p: &str) -> anyhow::Result<Vec<String>> {
            unimplemented!()
        }
        fn all_document_meta(&self) -> anyhow::Result<Vec<(String, Option<String>)>> {
            unimplemented!()
        }
        fn related_space(
            &self,
            _s: Space,
            _q: &[Vec<f32>],
            _x: &[String],
            _l: usize,
            _f: &TextQuery,
        ) -> anyhow::Result<Vec<search::Hit>> {
            unimplemented!()
        }
    }

    fn store(dense: Vec<search::Hit>, candidates: Vec<Candidate>) -> FakeStore {
        FakeStore {
            dense,
            lexical: LexicalMatches {
                stats: search::bm25::CorpusStats {
                    total_chunks: 100,
                    avg_chunk_len: 40.0,
                },
                doc_freqs: vec![3],
                candidates,
            },
        }
    }

    struct FixedVector(Vec<f32>);

    impl QueryEmbedder for FixedVector {
        fn embed_query(&mut self, _space: Space, _query: &str) -> anyhow::Result<Vec<f32>> {
            Ok(self.0.clone())
        }
    }

    fn config(lexical: bool) -> SearchConfig {
        SearchConfig {
            lexical,
            ..SearchConfig::default()
        }
    }

    /// A document only the lexical channel found must still reach the results.
    /// That is the whole point of fusing, and it is the behavior a client
    /// reimplementing the pipeline would most plausibly lose.
    #[test]
    fn fusion_surfaces_a_document_the_dense_channel_missed() {
        let s = store(
            vec![hit("dense.md", 0.9)],
            vec![candidate("lexical.md", "hnsw hnsw hnsw graph")],
        );
        let hits = run_query(
            &s,
            &mut FixedVector(vec![1.0, 0.0]),
            "hnsw",
            &[Space::Text],
            10,
            &TextQuery::default(),
            &config(true),
        )
        .unwrap();
        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert!(paths.contains(&"lexical.md"), "got {paths:?}");
        assert!(paths.contains(&"dense.md"), "got {paths:?}");
    }

    /// With the lexical channel off, the store must not be asked for lexical
    /// candidates at all — `FakeStore` would answer, but a real one would do
    /// the work. The title bonus still applies, which is the part that is easy
    /// to lose when removing a channel.
    #[test]
    fn lexical_off_still_applies_the_title_bonus() {
        let s = store(vec![hit("alpha.md", 0.5), hit("beta.md", 0.5)], Vec::new());
        let hits = run_query(
            &s,
            &mut FixedVector(vec![1.0, 0.0]),
            "beta",
            &[Space::Text],
            10,
            &TextQuery::default(),
            &config(false),
        )
        .unwrap();
        assert_eq!(
            hits[0].path, "beta.md",
            "the title match must outrank an equal-scoring non-match"
        );
    }

    /// The image space has no lexical channel: image chunks carry no text and
    /// its title proxies exist for `related` to traverse. Asking for one would
    /// panic on this store, which is how the test knows.
    #[test]
    fn the_image_space_skips_the_lexical_channel() {
        let s = FakeStore {
            dense: vec![hit("photo.png", 0.4)],
            lexical: LexicalMatches {
                stats: search::bm25::CorpusStats {
                    total_chunks: 0,
                    avg_chunk_len: 0.0,
                },
                doc_freqs: Vec::new(),
                candidates: Vec::new(),
            },
        };
        let hits = run_query(
            &s,
            &mut FixedVector(vec![1.0, 0.0]),
            "a fish",
            &[Space::Image],
            10,
            &TextQuery::default(),
            &config(true),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "photo.png");
    }

    /// A query that tokenizes to nothing (punctuation, or a stop-free string of
    /// symbols) must return the dense ranking rather than an empty one.
    #[test]
    fn a_query_with_no_lexical_terms_falls_back_to_dense() {
        let s = store(vec![hit("dense.md", 0.9)], Vec::new());
        let hits = run_query(
            &s,
            &mut FixedVector(vec![1.0, 0.0]),
            "!!! ???",
            &[Space::Text],
            10,
            &TextQuery::default(),
            &config(true),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "dense.md");
    }
}
