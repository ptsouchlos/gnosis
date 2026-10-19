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

use std::path::Path;

use anyhow::{Result, bail};
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
        let mut dense = store.search_space(space, &query_vec, limit, filter)?;

        // Lexical matching applies to the text space only: image chunks carry
        // no text, and title proxies exist for `related` to traverse.
        if space == Space::Text {
            if cfg.title_boost != 0.0 {
                dense = with_named(store, &query_vec, query, dense, limit, filter)?;
            }
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

/// Add the notes whose file name carries every query term to the dense
/// candidates, scored by their own best chunk, so the title bonus can reach
/// them.
///
/// The bonus only re-ranks what a channel retrieved, and a note can be named
/// by a query without either channel finding it: "Who Not How" is all
/// stopword-like terms, and a book note's chunks rarely repeat its name.
/// Every term rather than some, because one shared word names half a vault;
/// half the terms was measured too and was never better.
fn with_named(
    store: &dyn Store,
    query_vec: &[f32],
    query: &str,
    mut dense: Vec<search::Hit>,
    limit: usize,
    filter: &TextQuery,
) -> Result<Vec<search::Hit>> {
    let terms = search::bm25::tokenize(query);
    if terms.is_empty() {
        return Ok(dense);
    }
    let mut paths: Vec<String> = store
        .all_document_meta()?
        .into_iter()
        .map(|(path, _)| path)
        .filter(|path| search::bm25::title_match(&terms, search::file_name_title(path)) >= 1.0)
        .collect();
    if paths.is_empty() {
        return Ok(dense);
    }
    // In path order, so which ones survive the cap does not depend on how the
    // store happened to list them.
    paths.sort();
    paths.truncate(limit);
    let scoped = TextQuery {
        from: filter.from,
        tags: filter.tags,
        paths: Some(&paths),
    };
    for hit in store.search_space(Space::Text, query_vec, limit, &scoped)? {
        match dense.iter_mut().find(|d| d.path == hit.path) {
            Some(existing) if existing.score >= hit.score => {}
            Some(existing) => *existing = hit,
            None => dense.push(hit),
        }
    }
    Ok(dense)
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
                parts: Default::default(),
            }
        })
        .collect();

    Ok(search::fuse_hits(dense, lexical, &terms, cfg, limit))
}

/// How a frontend reads `aliases:` out of a document's raw frontmatter.
///
/// A function rather than a direct call into `gnosis-parse`: that crate pulls
/// in a PDF text extractor, which has no business in a wasm query client whose
/// artifact carries no frontmatter at all. A client without a YAML parser
/// passes [`no_aliases`] and gets filename-stem matching, which is Obsidian's
/// own default.
pub type AliasExtractor = fn(&str) -> Vec<String>;

/// An [`AliasExtractor`] for a frontend that cannot parse frontmatter — or,
/// as with a portable artifact, has none to parse.
pub fn no_aliases(_frontmatter: &str) -> Vec<String> {
    Vec::new()
}

/// Rank the documents most similar to `path`'s own chunks, across `spaces`.
///
/// Here rather than in a frontend for the same reason [`run_query`] is: the
/// link exclusion, the per-space skip for a document with no chunks in a
/// space, and the normalized merge are all decisions a second implementation
/// would make differently.
pub fn run_related(
    store: &dyn Store,
    path: &str,
    spaces: &[Space],
    limit: usize,
    include_linked: bool,
    filter: &TextQuery,
    aliases: AliasExtractor,
) -> Result<Vec<search::Hit>> {
    if store.document_hash(path)?.is_none() {
        bail!("{path} is not indexed — run `gnosis index`");
    }

    let mut exclude = vec![path.to_string()];
    if !include_linked {
        let targets = store.linked_targets(path)?;
        if !targets.is_empty() {
            let all_meta = store.all_document_meta()?;
            exclude.extend(resolve_link_targets(&targets, &all_meta, aliases));
        }
    }

    let mut per_space: Vec<Vec<search::Hit>> = Vec::with_capacity(spaces.len());
    for space in spaces.iter().copied() {
        let queries = store.chunk_vectors(path, space)?;
        if queries.is_empty() {
            // This document has no vectors in this space (e.g. an image asked
            // about with `--in text`) — it contributes nothing to the merge,
            // which is not an error unless *every* requested space is empty.
            continue;
        }
        per_space.push(store.related_space(space, &queries, &exclude, limit, filter)?);
    }
    if per_space.is_empty() {
        bail!(
            "{path} has no chunks in any of [{}] to compare",
            spaces
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(search::merge_normalized(per_space, limit))
}

/// Resolve raw wikilink target texts to indexed document paths by matching
/// against each candidate's filename stem (Obsidian's own default
/// link-resolution behavior — by filename, not any custom title) or its
/// frontmatter `aliases:`, case-insensitively. Not full Obsidian-compatible
/// resolution: no folder-path disambiguation.
fn resolve_link_targets(
    targets: &[String],
    all_meta: &[(String, Option<String>)],
    aliases: AliasExtractor,
) -> Vec<String> {
    all_meta
        .iter()
        .filter(|(path, frontmatter)| {
            let stem = Path::new(path).file_stem().and_then(|s| s.to_str());
            let stem_matches =
                stem.is_some_and(|stem| targets.iter().any(|t| t.eq_ignore_ascii_case(stem)));
            let alias_matches = frontmatter.as_deref().is_some_and(|fm| {
                aliases(fm)
                    .iter()
                    .any(|a| targets.iter().any(|t| t.eq_ignore_ascii_case(a)))
            });
            stem_matches || alias_matches
        })
        .map(|(path, _)| path.clone())
        .collect()
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
        /// `related` fixtures. Kept separate from the query ones so that a
        /// `run_query` test still fails loudly if the pipeline ever starts
        /// asking for them.
        related: Option<Related>,
    }

    /// What `run_related` asks a store for, plus what it passed back out. The
    /// exclusion list is the part worth asserting on, and the store is the
    /// only place it is observable.
    #[derive(Default)]
    struct Related {
        hash: Option<Vec<u8>>,
        links: Vec<String>,
        meta: Vec<(String, Option<String>)>,
        vectors: Vec<(Space, Vec<Vec<f32>>)>,
        hits: Vec<search::Hit>,
        excluded: std::cell::RefCell<Vec<String>>,
    }

    impl FakeStore {
        /// Panics rather than defaulting: a `run_query` test that reaches a
        /// `related` method is a test whose store does not describe what the
        /// code under test actually does.
        fn related(&self) -> &Related {
            self.related
                .as_ref()
                .expect("this store has no `related` fixtures")
        }
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
            parts: Default::default(),
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
        /// `dense` is the whole corpus in rank order; `limit` cuts it, as an
        /// ANN search would, unless the filter names documents.
        fn search_space(
            &self,
            _space: Space,
            _query: &[f32],
            limit: usize,
            filter: &TextQuery,
        ) -> anyhow::Result<Vec<search::Hit>> {
            Ok(self
                .dense
                .iter()
                .filter(|h| filter.allows_path(&h.path))
                .take(limit)
                .cloned()
                .collect())
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
            Ok(self.related().hash.clone())
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
        fn chunk_vectors(&self, _p: &str, space: Space) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(self
                .related()
                .vectors
                .iter()
                .find(|(s, _)| *s == space)
                .map(|(_, v)| v.clone())
                .unwrap_or_default())
        }
        fn linked_targets(&self, _p: &str) -> anyhow::Result<Vec<String>> {
            Ok(self.related().links.clone())
        }
        fn all_document_meta(&self) -> anyhow::Result<Vec<(String, Option<String>)>> {
            Ok(match &self.related {
                Some(related) => related.meta.clone(),
                None => self.dense.iter().map(|h| (h.path.clone(), None)).collect(),
            })
        }
        fn related_space(
            &self,
            _s: Space,
            _q: &[Vec<f32>],
            exclude: &[String],
            limit: usize,
            _f: &TextQuery,
        ) -> anyhow::Result<Vec<search::Hit>> {
            let related = self.related();
            *related.excluded.borrow_mut() = exclude.to_vec();
            Ok(related
                .hits
                .iter()
                .filter(|h| !exclude.contains(&h.path))
                .take(limit)
                .cloned()
                .collect())
        }
    }

    fn store(dense: Vec<search::Hit>, candidates: Vec<Candidate>) -> FakeStore {
        FakeStore {
            dense,
            related: None,
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

    /// Four notes, the one named by the query ranked last by meaning, and a
    /// result set of three: the dense cut alone would drop it.
    fn named_store() -> FakeStore {
        store(
            vec![
                hit("Notes/Deep Work.md", 0.9),
                hit("Notes/Atomic Habits.md", 0.85),
                hit("Notes/Make Time.md", 0.8),
                hit("Notes/Who Not How.md", 0.75),
            ],
            Vec::new(),
        )
    }

    fn search_named(s: &FakeStore, query: &str, cfg: &SearchConfig) -> Vec<String> {
        run_query(
            s,
            &mut FixedVector(vec![1.0, 0.0]),
            query,
            &[Space::Text],
            3,
            &TextQuery::default(),
            cfg,
        )
        .unwrap()
        .into_iter()
        .map(|h| h.path)
        .collect()
    }

    /// A note whose file name carries the whole query is a candidate on that
    /// alone, scored by its own chunks, so the title bonus can reach it.
    #[test]
    fn a_note_named_by_the_query_is_a_candidate_past_the_dense_cut() {
        let paths = search_named(&named_store(), "who not how", &config(false));
        assert!(paths.contains(&"Notes/Who Not How.md".to_string()), "got {paths:?}");
    }

    /// With the title bonus off, a name means nothing, here as in ranking.
    #[test]
    fn no_title_bonus_adds_no_named_candidates() {
        let cfg = SearchConfig {
            title_boost: 0.0,
            ..config(false)
        };
        let paths = search_named(&named_store(), "who not how", &cfg);
        assert!(!paths.contains(&"Notes/Who Not How.md".to_string()), "got {paths:?}");
    }

    /// Only a file name carrying every query term: one shared word is how
    /// half the vault gets named by a two-word query.
    #[test]
    fn a_partly_matching_file_name_adds_no_candidate() {
        let paths = search_named(&named_store(), "who else", &config(false));
        assert!(!paths.contains(&"Notes/Who Not How.md".to_string()), "got {paths:?}");
    }

    /// The image space has no lexical channel: image chunks carry no text and
    /// its title proxies exist for `related` to traverse. Asking for one would
    /// panic on this store, which is how the test knows.
    #[test]
    fn the_image_space_skips_the_lexical_channel() {
        let s = FakeStore {
            dense: vec![hit("photo.png", 0.4)],
            related: None,
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

    fn related_store(related: Related) -> FakeStore {
        FakeStore {
            dense: Vec::new(),
            related: Some(related),
            lexical: LexicalMatches {
                stats: search::bm25::CorpusStats {
                    total_chunks: 0,
                    avg_chunk_len: 0.0,
                },
                doc_freqs: Vec::new(),
                candidates: Vec::new(),
            },
        }
    }

    /// The source note and anything it already links to are excluded by
    /// default: `related` exists to surface *unlinked* mentions, so a
    /// frontend that lost this would mostly return links the reader already
    /// has. Link targets are raw wikilink text, resolved here against
    /// filename stems.
    #[test]
    fn related_excludes_the_source_and_the_notes_it_links_to() {
        let s = related_store(Related {
            hash: Some(vec![1]),
            links: vec!["Already Linked".to_string()],
            meta: vec![
                ("/vault/notes/Already Linked.md".to_string(), None),
                ("/vault/notes/Unlinked.md".to_string(), None),
            ],
            vectors: vec![(Space::Text, vec![vec![1.0, 0.0]])],
            hits: vec![
                hit("/vault/notes/Already Linked.md", 0.9),
                hit("/vault/notes/Unlinked.md", 0.5),
            ],
            ..Related::default()
        });

        let hits = run_related(
            &s,
            "/vault/notes/Source.md",
            &[Space::Text],
            10,
            false,
            &TextQuery::default(),
            no_aliases,
        )
        .unwrap();

        let paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert_eq!(paths, vec!["/vault/notes/Unlinked.md"], "got {paths:?}");
        let excluded = s.related().excluded.borrow().clone();
        assert!(excluded.contains(&"/vault/notes/Source.md".to_string()));
        assert!(excluded.contains(&"/vault/notes/Already Linked.md".to_string()));
    }

    /// `--include-linked` turns the exclusion off, leaving only the source
    /// itself excluded — a note is never related to itself.
    #[test]
    fn including_linked_notes_still_excludes_the_source() {
        let s = related_store(Related {
            hash: Some(vec![1]),
            links: vec!["Already Linked".to_string()],
            meta: vec![("/vault/notes/Already Linked.md".to_string(), None)],
            vectors: vec![(Space::Text, vec![vec![1.0, 0.0]])],
            hits: vec![hit("/vault/notes/Already Linked.md", 0.9)],
            ..Related::default()
        });

        let hits = run_related(
            &s,
            "/vault/notes/Source.md",
            &[Space::Text],
            10,
            true,
            &TextQuery::default(),
            no_aliases,
        )
        .unwrap();

        assert_eq!(hits.len(), 1, "the linked note must come back");
        assert_eq!(
            s.related().excluded.borrow().clone(),
            vec!["/vault/notes/Source.md".to_string()]
        );
    }

    /// A frontend that can read frontmatter resolves `[[Alias]]` to the note
    /// that declares it; one that cannot ([`no_aliases`]) matches filename
    /// stems only. Both behaviors are correct for their frontend, and the
    /// difference is visible exactly here.
    #[test]
    fn an_alias_extractor_resolves_a_link_a_stem_match_would_miss() {
        let fixture = || Related {
            hash: Some(vec![1]),
            links: vec!["Nickname".to_string()],
            meta: vec![(
                "/vault/notes/Formal Title.md".to_string(),
                Some("aliases: [Nickname]".to_string()),
            )],
            vectors: vec![(Space::Text, vec![vec![1.0, 0.0]])],
            hits: vec![hit("/vault/notes/Formal Title.md", 0.9)],
            ..Related::default()
        };
        let aliases: AliasExtractor = |fm| {
            fm.trim_start_matches("aliases: [")
                .trim_end_matches(']')
                .split(',')
                .map(|a| a.trim().to_string())
                .collect()
        };

        let run = |extractor| {
            run_related(
                &related_store(fixture()),
                "/vault/notes/Source.md",
                &[Space::Text],
                10,
                false,
                &TextQuery::default(),
                extractor,
            )
            .unwrap()
        };

        assert!(
            run(aliases).is_empty(),
            "an alias-resolved link must be excluded"
        );
        assert_eq!(
            run(no_aliases).len(),
            1,
            "without an alias extractor the link is not resolved, so the note stays"
        );
    }

    /// A document with no chunks in one requested space contributes nothing
    /// rather than failing — asking an image about `text,image` is ordinary.
    /// Every requested space being empty is the error case, because then
    /// there was nothing to compare at all and an empty result would read as
    /// "nothing is related".
    #[test]
    fn a_space_the_document_has_no_chunks_in_is_skipped_not_fatal() {
        let present = Related {
            hash: Some(vec![1]),
            vectors: vec![(Space::Image, vec![vec![1.0, 0.0]])],
            hits: vec![hit("/vault/photos/other.png", 0.7)],
            ..Related::default()
        };
        let hits = run_related(
            &related_store(present),
            "/vault/photos/cat.png",
            &[Space::Text, Space::Image],
            10,
            false,
            &TextQuery::default(),
            no_aliases,
        )
        .unwrap();
        assert_eq!(hits.len(), 1, "the image space still answered");

        let absent = Related {
            hash: Some(vec![1]),
            ..Related::default()
        };
        let err = run_related(
            &related_store(absent),
            "/vault/photos/cat.png",
            &[Space::Text],
            10,
            false,
            &TextQuery::default(),
            no_aliases,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no chunks in any of [text]"), "got {err}");
    }

    /// An unindexed path is reported as such rather than returning nothing:
    /// "no related notes" and "this file was never indexed" are different
    /// answers, and only one of them is the user's to fix.
    #[test]
    fn an_unindexed_path_is_an_error_not_an_empty_result() {
        let err = run_related(
            &related_store(Related::default()),
            "/vault/notes/Never Indexed.md",
            &[Space::Text],
            10,
            false,
            &TextQuery::default(),
            no_aliases,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("is not indexed"), "got {err}");
    }
}
