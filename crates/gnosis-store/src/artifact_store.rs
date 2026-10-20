//! A read-only [`Store`] over a portable artifact.
//!
//! This is what lets a client — the Obsidian plugin, or anything else without
//! SQLite — run gnosis's own retrieval pipeline unchanged. `query::run_query`
//! takes a `&dyn Store`; this is the second implementation of that trait, and
//! the first one that reaches `wasm32`.
//!
//! What it is not: a database. There is no ANN index (S2 measured i8 brute
//! force at ~1 ms over 20k chunks, 0.983 recall@10, so there is nothing to
//! accelerate) and no FTS index — the artifact deliberately ships neither,
//! because both are rebuildable and SQLite-specific. The lexical channel is
//! rebuilt here instead, at `open`, with the same tokenizer the corpus
//! statistics were gathered with — frequencies and chunk lengths included, so
//! that a query scores its candidates without re-reading their text. Measured
//! in wasm over a 6,255-chunk artifact, that is the difference between an
//! 11 ms hybrid query and a 155 ms one.
//!
//! Three honest differences from `SqliteStore`, all of them consequences of
//! what the artifact carries rather than of this code:
//!
//! - **Writes fail.** An artifact is derived and query-only; a client that
//!   could half-write one would produce an index nothing can regenerate.
//! - **Paths are vault-relative and roots are labels**, so `TextQuery::from`
//!   filters on a root's label (`"notes"`), not on a machine path.
//! - **No frontmatter**, so [`Store::all_document_meta`] returns `None` for
//!   every document and `related`'s alias matching falls back to filename
//!   stems. A host with its own metadata index — Obsidian has one — should
//!   resolve aliases there rather than expect this to.

use std::collections::HashMap;

use anyhow::{Result, bail};
use search::Hit;

use crate::artifact::{Artifact, ArtifactChunk, DocumentEntry};
use crate::{DocWrite, LexicalMatches, Space, Stats, Store, TextQuery, search_modality};

/// An in-memory inverted index over the artifact's text chunks.
///
/// Built once at open rather than per query: the document frequencies it holds
/// are corpus-wide, so every query needs all of them anyway.
///
/// Term frequencies are stored alongside the postings, and a chunk's length
/// alongside them. They cost nothing to keep — the build already tokenizes
/// every chunk to find out which terms it contains, so counting them is the
/// same pass — and they are what lets a query score its candidate pool
/// without re-reading a single chunk's text.
///
/// That matters because the pool is not small. It has to be scored *before*
/// truncation (see [`Store::lexical_candidates`]), and it grows with the
/// query: re-tokenizing it per query cost 142 ms of a 155 ms hybrid query on
/// a 6,255-chunk artifact, measured in wasm.
struct Lexical {
    /// Per term, every text chunk containing it, in corpus order, with how
    /// many times it occurs there.
    postings: HashMap<String, Vec<Posting>>,
    /// Per chunk index, the length `CorpusStats::avg_chunk_len` averages —
    /// characters of heading plus body. Zero for chunks outside the text
    /// space, which the lexical channel never scores.
    lengths: Vec<u32>,
    stats: search::bm25::CorpusStats,
}

/// One term's occurrence in one chunk.
#[derive(Debug, Clone, Copy)]
struct Posting {
    chunk: u32,
    term_freq: u32,
}

impl Lexical {
    /// Tokenizes with `search::bm25::tokenize` — the same function the Rust
    /// scorer was tuned against, which is what makes the client's term
    /// statistics agree with it by construction rather than by care.
    ///
    /// One divergence from SQLite is inherent and must be known rather than
    /// hidden: FTS5's `unicode61` folds diacritics and this does not, so on an
    /// accented corpus the two can disagree about whether a chunk contains a
    /// term, and therefore about its document frequency.
    fn build(chunks: &[ArtifactChunk]) -> Self {
        let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
        let mut lengths = vec![0u32; chunks.len()];
        let mut total_len: u64 = 0;
        let mut counted: u64 = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            if chunk.space != Space::Text {
                continue;
            }
            // Heading plus body, matching what the scorer measures and what
            // SQLite averages over — in characters, which is the unit
            // `CorpusStats::avg_chunk_len` is in.
            let scored = scored_text(chunk);
            let len = scored.chars().count();
            lengths[i] = len as u32;
            total_len += len as u64;
            counted += 1;
            // Counted the way `search::bm25::term_freqs` counts — the same
            // tokenizer over the same text — so a stored frequency is the
            // number the scorer would have arrived at itself.
            let mut freqs: HashMap<String, u32> = HashMap::new();
            for token in search::bm25::tokenize(&scored) {
                *freqs.entry(token).or_insert(0) += 1;
            }
            for (term, term_freq) in freqs {
                postings.entry(term).or_default().push(Posting {
                    chunk: i as u32,
                    term_freq,
                });
            }
        }
        // `or_default().push(...)` builds each list in chunk order already,
        // except that a `HashMap`'s iteration order decides which *term* is
        // appended when — not the order within one term's list. Sorting is
        // therefore unnecessary, and asserted rather than assumed by
        // `postings_are_in_chunk_order`.
        let avg = if counted == 0 {
            0.0
        } else {
            total_len as f32 / counted as f32
        };
        Self {
            postings,
            lengths,
            stats: search::bm25::CorpusStats {
                total_chunks: counted,
                avg_chunk_len: avg,
            },
        }
    }
}

/// What the lexical channel scores over: heading path, a separator, then body.
fn scored_text(chunk: &ArtifactChunk) -> String {
    format!("{} {}", chunk.heading_path, chunk.text)
}

/// A portable artifact, answering the same `Store` trait the SQLite database
/// does.
pub struct ArtifactStore {
    artifact: Artifact,
    lexical: Lexical,
}

impl ArtifactStore {
    pub fn new(artifact: Artifact) -> Self {
        let lexical = Lexical::build(artifact.chunks());
        Self { artifact, lexical }
    }

    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    fn document(&self, chunk: &ArtifactChunk) -> Option<&DocumentEntry> {
        self.artifact.documents().get(chunk.doc_index)
    }

    /// A document's root *label*, since that is all an artifact carries.
    fn root_label(&self, doc: &DocumentEntry) -> &str {
        self.artifact
            .manifest()
            .roots
            .get(doc.root)
            .map(String::as_str)
            .unwrap_or_default()
    }

    /// The `--from` / `--tag` filter, applied the way `SqliteStore` applies it:
    /// AND between the two, OR within each.
    fn passes(&self, doc: &DocumentEntry, filter: &TextQuery) -> bool {
        if let Some(roots) = filter.from.filter(|r| !r.is_empty())
            && !roots.iter().any(|r| r == self.root_label(doc))
        {
            return false;
        }
        if let Some(tags) = filter.tags.filter(|t| !t.is_empty())
            && !tags.iter().any(|t| doc.tags.contains(t))
        {
            return false;
        }
        filter.allows_path(&doc.path)
    }

    fn candidate(&self, chunk: &ArtifactChunk) -> Option<search::Candidate> {
        let doc = self.document(chunk)?;
        Some(search::Candidate {
            path: doc.path.clone(),
            title: doc.title.clone(),
            source_root: self.root_label(doc).to_string(),
            heading_path: chunk.heading_path.clone(),
            text: chunk.text.clone(),
            vector: chunk.vector.clone(),
            // The artifact does not carry pixel dimensions: they are display
            // detail, and a host that has the image file has them already.
            width: None,
            height: None,
            modality: chunk.modality.clone(),
        })
    }

    /// How related each of `paths` is to each other, in `space`: row `i`,
    /// column `j` is the score `related(paths[i])` ranks `paths[j]` by in that
    /// space, from [`search::related_pair_score`] (before `related` merges
    /// spaces, which normalizes).
    ///
    /// For a client placing related notes relative to each other. `related`
    /// alone cannot say that without one call per note, each over every chunk
    /// in the artifact; this reads only the listed documents' chunks. The
    /// diagonal is `None` (a note is never related to itself), and so is a
    /// pair `related` would never list (a document of only embeds and links,
    /// or one with no chunks in `space`). Rows need not equal columns: a
    /// source of only embeds still compares from them.
    pub fn related_similarities(
        &self,
        paths: &[String],
        space: Space,
    ) -> Result<Vec<Vec<Option<f32>>>> {
        let mut chunks: Vec<Vec<&ArtifactChunk>> = vec![Vec::new(); paths.len()];
        let mut found = vec![false; paths.len()];
        for (doc_index, doc) in self.artifact.documents().iter().enumerate() {
            for (i, path) in paths.iter().enumerate() {
                if &doc.path == path {
                    found[i] = true;
                    chunks[i].extend(
                        self.artifact
                            .chunks()
                            .iter()
                            .filter(|c| c.doc_index == doc_index && c.space == space),
                    );
                }
            }
        }
        if let Some(missing) = found.iter().position(|f| !f) {
            bail!("{} is not indexed — run `gnosis index`", paths[missing]);
        }

        let text_space = space == Space::Text;
        let sources: Vec<Vec<Vec<f32>>> = chunks
            .iter()
            .map(|doc| {
                search::related_sources(
                    doc.iter()
                        .map(|c| (c.vector.clone(), c.text.clone()))
                        .collect(),
                    text_space,
                )
            })
            .collect();
        Ok((0..paths.len())
            .map(|i| {
                (0..paths.len())
                    .map(|j| {
                        if i == j || sources[i].is_empty() {
                            return None;
                        }
                        search::related_pair_score(
                            &sources[i],
                            chunks[j]
                                .iter()
                                .map(|c| (c.vector.as_slice(), c.text.as_str())),
                            text_space,
                        )
                    })
                    .collect()
            })
            .collect())
    }

    fn candidates(
        &self,
        space: Space,
        modality: Option<&str>,
        filter: &TextQuery,
    ) -> Vec<search::Candidate> {
        self.artifact
            .chunks()
            .iter()
            .filter(|c| c.space == space)
            .filter(|c| modality.is_none_or(|m| c.modality == m))
            .filter(|c| self.document(c).is_some_and(|d| self.passes(d, filter)))
            .filter_map(|c| self.candidate(c))
            .collect()
    }
}

/// An artifact is derived and query-only. Writing to one is a programming
/// error rather than a runtime condition, but it is reported rather than
/// panicked on, so a client that tries it gets a message instead of an abort
/// in the middle of someone's vault.
fn read_only<T>(what: &str) -> Result<T> {
    bail!("a portable artifact is read-only — cannot {what}; re-export with `gnosis export`")
}

impl Store for ArtifactStore {
    fn set_meta(&self, _key: &str, _value: &str) -> Result<()> {
        read_only("write index metadata")
    }

    /// Serves the handful of keys the artifact actually knows, from its
    /// manifest. Everything else is absent rather than invented — a caller
    /// reading, say, `chunk.max_tokens` off an artifact would be reading a
    /// value nobody recorded.
    fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let manifest = self.artifact.manifest();
        let text = manifest.space(Space::Text);
        let image = manifest.space(Space::Image);
        Ok(match key {
            "model.text" => text.map(|m| m.id.clone()),
            "dim.text" => text.map(|m| m.dim.to_string()),
            "model.image_vision" => image.map(|m| m.id.clone()),
            "model.image_text" => image.and_then(|m| m.query_id.clone()),
            "dim.image" => image.map(|m| m.dim.to_string()),
            _ => None,
        })
    }

    fn stats(&self) -> Result<Stats> {
        let count = |space: Space| {
            self.artifact
                .chunks()
                .iter()
                .filter(|c| c.space == space)
                .count() as i64
        };
        Ok(Stats {
            documents: self.artifact.documents().len() as i64,
            chunks_text: count(Space::Text),
            chunks_image: count(Space::Image),
            // When the artifact was built, not when the vault was indexed. The
            // distinction matters for staleness: an artifact is never fresher
            // than the index it came from.
            indexed_at: Some(self.artifact.manifest().created_unix),
        })
    }

    fn document_hash(&self, path: &str) -> Result<Option<Vec<u8>>> {
        Ok(self
            .artifact
            .documents()
            .iter()
            .find(|d| d.path == path)
            .and_then(|d| hex_to_bytes(&d.content_hash)))
    }

    fn paths_for_roots(&self, roots: &[String]) -> Result<Vec<String>> {
        Ok(self
            .artifact
            .documents()
            .iter()
            .filter(|d| roots.iter().any(|r| r == self.root_label(d)))
            .map(|d| d.path.clone())
            .collect())
    }

    fn counts_by_root(&self) -> Result<Vec<(String, i64)>> {
        let mut counts: Vec<(String, i64)> = Vec::new();
        for doc in self.artifact.documents() {
            let label = self.root_label(doc).to_string();
            match counts.iter_mut().find(|(r, _)| *r == label) {
                Some((_, n)) => *n += 1,
                None => counts.push((label, 1)),
            }
        }
        Ok(counts)
    }

    fn delete_by_root(&self, _root: &str) -> Result<usize> {
        read_only("delete a vault")
    }

    fn replace_document(&mut self, _doc: &DocWrite<'_>) -> Result<()> {
        read_only("write a document")
    }

    fn delete_document(&self, _path: &str) -> Result<()> {
        read_only("delete a document")
    }

    fn lexical_candidates(
        &self,
        terms: &[String],
        limit: usize,
        filter: &TextQuery,
    ) -> Result<LexicalMatches> {
        // Ordering the candidate pool needs the scorer, and the scorer needs
        // parameters. The pipeline passes its own when it scores for real;
        // these only decide which candidates survive truncation, so the
        // defaults are the right choice — they are what the parameters are
        // unless a vault has retuned them.
        let params = search::bm25::Bm25Params::default();
        let stats = self.lexical.stats;
        if terms.is_empty() || stats.total_chunks == 0 {
            return Ok(LexicalMatches {
                stats,
                doc_freqs: vec![0; terms.len()],
                candidates: Vec::new(),
            });
        }

        // Document frequency is corpus-wide, deliberately *before* the
        // root/tag filter — IDF describes the corpus, and recomputing it per
        // filter would make a term's weight depend on which vault was
        // selected. SQLite's `term_doc_freq` counts the same way.
        let doc_freqs: Vec<u64> = terms
            .iter()
            .map(|t| {
                self.lexical
                    .postings
                    .get(t)
                    .map(|p| p.len() as u64)
                    .unwrap_or(0)
            })
            .collect();

        // Any term matching is enough to be a candidate; how many and how
        // often is what BM25+ then weighs. The set is truncated to `limit`, so
        // *which* candidates survive matters: SQLite orders by FTS5's own
        // `bm25()` before its `LIMIT` so that a truncated set keeps the most
        // promising rows, and taking an arbitrary slice here instead would
        // fuse over a different pool and rank differently. Measured on
        // scifact, slicing by posting order agreed with the database on only
        // 41 of 300 queries; scoring first is what closes that.
        //
        // This orders by BM25+ itself — the same scorer the pipeline then
        // applies — rather than by a proxy, so the pool it keeps is the pool
        // that scores best.
        let query_terms: Vec<search::bm25::QueryTerm> = doc_freqs
            .iter()
            .map(|&doc_freq| search::bm25::QueryTerm { doc_freq })
            .collect();
        // Walk the postings, accumulating each matched chunk's per-term
        // frequencies as they are encountered. This is the whole reason the
        // frequencies are stored: the pool is assembled and scored without
        // touching a chunk's text, where re-tokenizing it was costing more
        // than everything else a query does put together.
        //
        // `slot` maps a chunk index to its row in `freqs`, so a chunk matched
        // by several query terms accumulates into one row. A full-length
        // vector rather than a hash map because it is indexed once per
        // posting and the artifact's chunk count is known.
        let mut slot: Vec<u32> = vec![u32::MAX; self.artifact.chunks().len()];
        let mut matched: Vec<usize> = Vec::new();
        let mut freqs: Vec<Vec<u32>> = Vec::new();
        for (t, term) in terms.iter().enumerate() {
            let Some(postings) = self.lexical.postings.get(term) else {
                continue;
            };
            for posting in postings {
                let i = posting.chunk as usize;
                let row = match slot.get(i).copied() {
                    Some(u32::MAX) => {
                        let chunk = &self.artifact.chunks()[i];
                        if !self.document(chunk).is_some_and(|d| self.passes(d, filter)) {
                            // Excluded by the filter. Marked with its own
                            // sentinel so the next term that matches this
                            // chunk does not re-check the document.
                            slot[i] = u32::MAX - 1;
                            continue;
                        }
                        let row = freqs.len() as u32;
                        slot[i] = row;
                        matched.push(i);
                        freqs.push(vec![0; terms.len()]);
                        row
                    }
                    Some(row) if row == u32::MAX - 1 => continue,
                    Some(row) => row,
                    None => continue,
                };
                freqs[row as usize][t] = posting.term_freq;
            }
        }
        let mut scored: Vec<(f32, usize)> = matched
            .iter()
            .copied()
            .zip(freqs)
            .map(|(i, term_freqs)| {
                let m = search::bm25::LexicalMatch {
                    chunk_len: self.lexical.lengths[i],
                    term_freqs,
                };
                (search::bm25::score(&m, &query_terms, stats, params), i)
            })
            .collect();
        // Descending by score, then by position, so a tie resolves the same
        // way every run rather than by whatever order the postings happened to
        // produce.
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.truncate(limit);
        let candidates: Vec<search::Candidate> = scored
            .into_iter()
            .filter_map(|(_, i)| self.candidate(&self.artifact.chunks()[i]))
            .collect();

        Ok(LexicalMatches {
            stats,
            doc_freqs,
            candidates,
        })
    }

    fn search_space(
        &self,
        space: Space,
        query: &[f32],
        limit: usize,
        filter: &TextQuery,
    ) -> Result<Vec<Hit>> {
        let modality = search_modality(space);
        let candidates = self.candidates(space, Some(modality), filter);
        Ok(search::rank(query, candidates, limit))
    }

    fn chunk_vectors(&self, path: &str, space: Space) -> Result<Vec<Vec<f32>>> {
        let chunks = self
            .artifact
            .chunks()
            .iter()
            .filter(|c| c.space == space)
            .filter(|c| self.document(c).is_some_and(|d| d.path == path))
            .map(|c| (c.vector.clone(), c.text.clone()))
            .collect();
        Ok(search::related_sources(chunks, space == Space::Text))
    }

    fn linked_targets(&self, path: &str) -> Result<Vec<String>> {
        Ok(self
            .artifact
            .documents()
            .iter()
            .find(|d| d.path == path)
            .map(|d| d.links.clone())
            .unwrap_or_default())
    }

    /// Paths with no frontmatter, since the artifact carries none. `related`
    /// still resolves a link by filename stem; an alias declared only in
    /// frontmatter will not resolve here.
    fn all_document_meta(&self) -> Result<Vec<(String, Option<String>)>> {
        Ok(self
            .artifact
            .documents()
            .iter()
            .map(|d| (d.path.clone(), None))
            .collect())
    }

    fn document_dates(&self) -> Result<std::collections::HashMap<String, i64>> {
        Ok(self
            .artifact
            .documents()
            .iter()
            .filter_map(|d| Some((d.path.clone(), d.created_unix?)))
            .collect())
    }

    fn related_space(
        &self,
        space: Space,
        query_vectors: &[Vec<f32>],
        exclude_paths: &[String],
        limit: usize,
        filter: &TextQuery,
    ) -> Result<Vec<Hit>> {
        // Every modality, unlike `search_space`: an image's title proxy is how
        // `related` reaches the notes that mention it.
        let candidates = self.candidates(space, None, filter);
        Ok(search::rank_related(
            query_vectors,
            candidates,
            exclude_paths,
            limit,
            space == Space::Text,
        ))
    }
}

fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(space: Space, heading: &str, text: &str) -> ArtifactChunk {
        ArtifactChunk {
            doc_index: 0,
            ord: 0,
            space,
            modality: "text".into(),
            heading_path: heading.into(),
            text: text.into(),
            vector: vec![1.0, 0.0],
        }
    }

    fn corpus() -> Vec<ArtifactChunk> {
        vec![
            chunk(
                Space::Text,
                "The cat",
                "The cat sat on the mat, and the cat stayed.",
            ),
            chunk(Space::Text, "Dogs", "A dog barked at the cat once."),
            // In the image space, so the lexical channel must not see it at
            // all — its words would otherwise inflate every term's document
            // frequency, and IDF is corpus-wide.
            chunk(Space::Image, "cat.png", ""),
        ]
    }

    /// The frequencies kept beside the postings must be the numbers
    /// `search::bm25::term_freqs` would have produced from the chunk's own
    /// text. This is the whole correctness condition of storing them: the
    /// scorer used to count them itself, and a query's score is unchanged
    /// only if these agree exactly.
    #[test]
    fn stored_term_frequencies_are_what_the_scorer_would_count() {
        let chunks = corpus();
        let lexical = Lexical::build(&chunks);
        assert!(!lexical.postings.is_empty());

        for (term, postings) in &lexical.postings {
            for posting in postings {
                let c = &chunks[posting.chunk as usize];
                let expected =
                    search::bm25::term_freqs(&scored_text(c), std::slice::from_ref(term))[0];
                assert_eq!(
                    posting.term_freq, expected,
                    "term '{term}' in chunk {}: stored {}, the scorer counts {expected}",
                    posting.chunk, posting.term_freq
                );
                assert!(expected > 0, "a posting must mean the term occurs");
            }
        }

        // "the" appears four times in the first chunk, counting its heading.
        let the = &lexical.postings["the"];
        assert_eq!(the[0].chunk, 0);
        assert_eq!(the[0].term_freq, 4);
    }

    /// A chunk's stored length must be the one the corpus average is taken
    /// over, or BM25's length normalization compares a chunk against an
    /// average measured differently.
    #[test]
    fn stored_lengths_match_what_the_average_is_taken_over() {
        let chunks = corpus();
        let lexical = Lexical::build(&chunks);

        let text_lengths: Vec<u32> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.space == Space::Text)
            .map(|(i, _)| lexical.lengths[i])
            .collect();
        for (i, c) in chunks.iter().enumerate() {
            let expected = if c.space == Space::Text {
                scored_text(c).chars().count() as u32
            } else {
                0
            };
            assert_eq!(lexical.lengths[i], expected, "chunk {i}");
        }

        let mean = text_lengths.iter().map(|&l| l as f64).sum::<f64>() / 2.0;
        assert_eq!(
            lexical.stats.total_chunks, 2,
            "the image chunk is not scored"
        );
        assert!(
            (lexical.stats.avg_chunk_len as f64 - mean).abs() < 1e-3,
            "avg {} vs {mean}",
            lexical.stats.avg_chunk_len
        );
    }

    /// Candidate selection relies on each posting list being in chunk order:
    /// ties in the score are broken by chunk index, and a list out of order
    /// would make that tie-break depend on how the index happened to be
    /// built. `build` produces this order incidentally, so it is asserted
    /// rather than assumed.
    #[test]
    fn postings_are_in_chunk_order() {
        let lexical = Lexical::build(&corpus());
        for (term, postings) in &lexical.postings {
            let chunks: Vec<u32> = postings.iter().map(|p| p.chunk).collect();
            let mut sorted = chunks.clone();
            sorted.sort_unstable();
            assert_eq!(chunks, sorted, "term '{term}' is out of order");
        }
    }
}
