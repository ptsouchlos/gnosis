//! Pure vector scoring/ranking, independent of how candidates are stored.
//! Kept storage-agnostic so it can back both the CLI's SQLite-backed search
//! and other front-ends (e.g. a WASM build indexing over IndexedDB).
pub mod bm25;

use std::collections::HashMap;

/// One embedded chunk to score against a query, before dedup/ranking.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub path: String,
    pub title: String,
    pub source_root: String,
    pub heading_path: String,
    pub text: String,
    pub vector: Vec<f32>,
    /// Image pixel dimensions, when this candidate's document is an image
    /// (or `None` for text documents / unknown dimensions).
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub modality: String,
}

/// One ranked search result (best chunk per document).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hit {
    pub path: String,
    pub title: String,
    pub source_root: String,
    pub heading_path: String,
    pub text: String,
    pub score: f32,
    pub width: Option<i64>,
    pub height: Option<i64>,
    /// What `score` is made of, within this hit's own space.
    pub parts: ScoreParts,
}

/// A hit's score, by where it came from: the three add up to `score`.
///
/// Contributions, not raw channel scores: `meaning` is the dense score times
/// its fusion weight, `keywords` the lexical score times its own, and `title`
/// the bonus as added. So a reader can see why a note ranks where it does, in
/// the same units as the score. A merge across spaces (`merge_normalized`)
/// rescales `score` and leaves these as they were in their space.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct ScoreParts {
    pub meaning: f32,
    pub keywords: f32,
    pub title: f32,
}

/// Score `candidates` against `query` by cosine similarity (vectors are
/// assumed L2-normalized, so a dot product suffices), keep only the
/// best-scoring chunk per document path, and return the top `limit` sorted
/// descending by score.
pub fn rank(query: &[f32], candidates: impl IntoIterator<Item = Candidate>, limit: usize) -> Vec<Hit> {
    rank_by(|v| dot(query, v), candidates, &[], limit)
}

/// How many of a document pair's best chunk matches [`rank_related`] averages.
///
/// The best single match alone lets one coincidence decide a pair: on a
/// templated vault, two notes' identical boilerplate sections. Averaging the
/// best three asks that the pair agree in more than one place. Measured by
/// link prediction on Obsidian's help vault (178 notes, 875 links): 3 beat
/// 1, 2 and 5 (MRR 0.760 → 0.788 with boilerplate kept; see
/// docs/gnosis/plugin-settings-redesign.md in the workspace).
pub const RELATED_TOP_K: usize = 3;

/// Characters of its own a chunk needs, once embeds, links and headings are
/// stripped, to count as content for [`rank_related`].
const MIN_CONTENT_CHARS: usize = 20;

/// Whether a chunk carries nothing of its own: only embeds (`![[a.pdf]]`),
/// wikilinks, markdown links and headings, with fewer than
/// [`MIN_CONTENT_CHARS`] word characters left over.
///
/// Such a chunk is nearly the same in every note that has one — a template's
/// `## Mind Map` / `![[<title>.pdf]]` section, say — so for `related` it is
/// evidence of a shared template, not of a shared subject. On a vault of 221
/// templated book notes it decided 98% of every note's related results.
pub fn carries_no_content(text: &str) -> bool {
    let mut own = 0usize;
    for line in text.lines() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let mut rest = line;
        while !rest.is_empty() {
            // `[[...]]` and `![[...]]`, then `[...](...)` and `![...](...)`.
            let skip = if let Some(r) = rest.strip_prefix("![[").or_else(|| rest.strip_prefix("[["))
            {
                r.find("]]").map(|end| rest.len() - r.len() + end + 2)
            } else if let Some(r) = rest.strip_prefix("![").or_else(|| rest.strip_prefix('[')) {
                r.find("](").and_then(|mid| {
                    r[mid..]
                        .find(')')
                        .map(|end| rest.len() - r.len() + mid + end + 1)
                })
            } else {
                None
            };
            match skip {
                Some(n) => rest = &rest[n..],
                None => {
                    let c = rest.chars().next().expect("rest is not empty");
                    if c.is_alphanumeric() || c == '_' {
                        own += 1;
                        if own >= MIN_CONTENT_CHARS {
                            return false;
                        }
                    }
                    rest = &rest[c.len_utf8()..];
                }
            }
        }
    }
    true
}

/// The chunks of a source document `related` compares from: its content
/// chunks, or every chunk when it has nothing but embeds and links — such a
/// note still deserves an answer, even if a weaker one.
///
/// `text_space` is false for the image space, whose chunks have no text by
/// design and are all kept.
pub fn related_sources(chunks: Vec<(Vec<f32>, String)>, text_space: bool) -> Vec<Vec<f32>> {
    if text_space && chunks.iter().any(|(_, text)| !carries_no_content(text)) {
        chunks
            .into_iter()
            .filter(|(_, text)| !carries_no_content(text))
            .map(|(v, _)| v)
            .collect()
    } else {
        chunks.into_iter().map(|(v, _)| v).collect()
    }
}

/// Rank documents by how alike they are to a source document's chunks
/// (`queries`), for `related`.
///
/// A pair of documents scores the mean of its [`RELATED_TOP_K`] best
/// chunk-to-chunk similarities. In the text space (`text_space`), candidate
/// chunks that carry no content of their own are skipped, so a document
/// that is nothing but embeds is never offered as related. Each hit shows
/// its best-matching chunk. `exclude_paths` is filtered before truncation to
/// `limit`, so an excluded document never displaces an eligible one.
pub fn rank_related(
    queries: &[Vec<f32>],
    candidates: impl IntoIterator<Item = Candidate>,
    exclude_paths: &[String],
    limit: usize,
    text_space: bool,
) -> Vec<Hit> {
    // Per document: its hit (best chunk), and its best pair scores so far.
    let mut docs: HashMap<String, (Hit, BestMatches)> = HashMap::new();
    for c in candidates {
        if exclude_paths.iter().any(|p| p == &c.path) || (text_space && carries_no_content(&c.text))
        {
            continue;
        }
        let (hit, best) = docs.entry(c.path.clone()).or_insert_with(|| {
            (
                Hit {
                    path: c.path.clone(),
                    title: c.title.clone(),
                    source_root: c.source_root.clone(),
                    heading_path: String::new(),
                    text: String::new(),
                    score: f32::NEG_INFINITY,
                    width: None,
                    height: None,
                    parts: Default::default(),
                },
                BestMatches::default(),
            )
        });
        let chunk_best = best.add_chunk(queries, &c.vector);
        if chunk_best > hit.score {
            hit.score = chunk_best;
            hit.heading_path = c.heading_path;
            hit.text = c.text;
            hit.width = c.width;
            hit.height = c.height;
        }
    }

    let mut hits: Vec<Hit> = docs
        .into_values()
        .map(|(mut hit, best)| {
            hit.score = best.score();
            hit
        })
        .collect();
    hits.sort_by(best_first);
    hits.truncate(limit);
    hits
}

/// One document pair's score, exactly as [`rank_related`] would give the
/// candidate document when asked from the document `queries` came from.
///
/// `candidate` is that document's chunks, as `(vector, text)`. `None` when
/// none of them would be considered (in the text space, a document of only
/// embeds and links), so it would never appear in `related` from here at all.
///
/// This is what lets a client place related notes relative to *each other*
/// without re-deriving the aggregation: a pair scored here and the same pair
/// in a `related` list are the same number.
pub fn related_pair_score<'a>(
    queries: &[Vec<f32>],
    candidate: impl IntoIterator<Item = (&'a [f32], &'a str)>,
    text_space: bool,
) -> Option<f32> {
    let mut best = BestMatches::default();
    for (vector, text) in candidate {
        if text_space && carries_no_content(text) {
            continue;
        }
        best.add_chunk(queries, vector);
    }
    (!best.0.is_empty()).then(|| best.score())
}

/// A document pair's [`RELATED_TOP_K`] best chunk-to-chunk similarities,
/// descending, which [`rank_related`] and [`related_pair_score`] both score.
#[derive(Default)]
struct BestMatches(Vec<f32>);

impl BestMatches {
    /// Score one candidate chunk against every query chunk, returning that
    /// chunk's own best similarity.
    fn add_chunk(&mut self, queries: &[Vec<f32>], vector: &[f32]) -> f32 {
        let mut chunk_best = f32::NEG_INFINITY;
        for q in queries {
            let score = dot(q, vector);
            chunk_best = chunk_best.max(score);
            // Keep the top K, descending; K is tiny, so insertion is cheapest.
            let at = self.0.partition_point(|&b| b >= score);
            if at < RELATED_TOP_K {
                self.0.insert(at, score);
                self.0.truncate(RELATED_TOP_K);
            }
        }
        chunk_best
    }

    /// The mean of the best matches kept, fewer than K when the pair has
    /// fewer chunk pairs than that.
    fn score(&self) -> f32 {
        self.0.iter().sum::<f32>() / self.0.len() as f32
    }
}

/// Shared aggregation for [`rank`]: score each candidate with
/// `score_fn`, keep the best-scoring chunk per document path (excluding
/// `exclude_paths`), and return the top `limit` sorted descending by score.
fn rank_by(
    score_fn: impl Fn(&[f32]) -> f32,
    candidates: impl IntoIterator<Item = Candidate>,
    exclude_paths: &[String],
    limit: usize,
) -> Vec<Hit> {
    let mut best: HashMap<String, Hit> = HashMap::new();
    for c in candidates {
        if exclude_paths.iter().any(|p| p == &c.path) {
            continue;
        }
        let score = score_fn(&c.vector);
        let entry = best.entry(c.path.clone()).or_insert_with(|| Hit {
            path: c.path,
            title: c.title,
            source_root: c.source_root,
            heading_path: String::new(),
            text: String::new(),
            score: f32::NEG_INFINITY,
            width: None,
            height: None,
            parts: Default::default(),
        });
        if score > entry.score {
            entry.score = score;
            // All of it is meaning until fusion, if any, divides it.
            entry.parts = ScoreParts {
                meaning: score,
                ..ScoreParts::default()
            };
            entry.heading_path = c.heading_path;
            entry.text = c.text;
            entry.width = c.width;
            entry.height = c.height;
        }
    }

    let mut hits: Vec<Hit> = best.into_values().collect();
    hits.sort_by(best_first);
    hits.truncate(limit);
    hits
}

/// Merge per-space search results into one ranked list. A single space
/// passes through with its raw scores unchanged (today's exact behavior,
/// preserved for callers that never touch image search). Two or more spaces
/// get per-space min-max score normalization first — plain cosine scores
/// aren't comparable across different embedding models — then are merged.
/// A document appearing in more than one space's results (e.g. a note has
/// both a `text`-space chunk and an `image`-space title-proxy, so `related`
/// can legitimately score it from both sides) is kept once, at its best
/// score — a caller-visible duplicate listing would look like a bug, not a
/// feature. Truncated to `limit` after dedup.
/// Fuse a dense and a lexical ranking into one, by document path.
///
/// Both inputs carry **already-normalized** scores in `Hit::score`, each
/// against its own channel's theoretical ceiling. A document present in only
/// one channel scores 0 in the other rather than being dropped — that is the
/// point of fusing: a literal match the embedder missed, or a semantic match
/// containing none of the query's words, should still surface.
///
/// Metadata comes from the dense hit when a document appears in both, so a
/// fused result looks exactly like a dense one to callers (same snippet, same
/// heading path).
pub fn fuse_hits(
    dense: Vec<Hit>,
    lexical: Vec<Hit>,
    query_terms: &[String],
    cfg: &bm25::SearchConfig,
    limit: usize,
) -> Vec<Hit> {
    let mut by_path: HashMap<String, (Hit, f32, f32)> = HashMap::new();

    for hit in dense {
        let dense_score = hit.score;
        by_path
            .entry(hit.path.clone())
            .and_modify(|slot| slot.1 = slot.1.max(dense_score))
            .or_insert((hit, dense_score, 0.0));
    }
    for hit in lexical {
        let lexical_score = hit.score;
        match by_path.get_mut(&hit.path) {
            Some(slot) => slot.2 = slot.2.max(lexical_score),
            None => {
                by_path.insert(hit.path.clone(), (hit, 0.0, lexical_score));
            }
        }
    }

    let fused: Vec<Hit> = by_path
        .into_values()
        .map(|(mut hit, dense_score, lexical_score)| {
            hit.score = bm25::fuse(dense_score, lexical_score, cfg.dense_weight);
            let alpha = cfg.dense_weight.clamp(0.0, 1.0);
            hit.parts = ScoreParts {
                meaning: alpha * dense_score,
                keywords: (1.0 - alpha) * lexical_score,
                title: 0.0,
            };
            hit
        })
        .collect();

    boost_and_rank(fused, query_terms, cfg.title_boost, limit)
}

/// Apply the title bonus, order, and truncate.
///
/// Shared by the fused and the dense-only paths so the bonus behaves the same
/// either way — it is independent of whether a lexical channel is running, and
/// quietly doing nothing when lexical retrieval is off would be a trap.
///
/// The bonus is additive and applied *before* truncation, so a titled document
/// can climb into the result set rather than only move within it. Applying it
/// after `limit` would make it useless for exactly the navigational case it
/// exists to serve.
pub fn boost_and_rank(
    mut hits: Vec<Hit>,
    query_terms: &[String],
    title_boost: f32,
    limit: usize,
) -> Vec<Hit> {
    if title_boost != 0.0 {
        for hit in &mut hits {
            // A note has two names: the title gnosis derived (frontmatter, then
            // the first H1) and its file name, which is what Obsidian shows
            // and what its reader searches for. Templated notes share an H1
            // ("# Notes") and lose the derived title, while a corpus named by
            // id keeps its title in the H1, so the better match of the two
            // counts.
            let matched = bm25::title_match(query_terms, &hit.title)
                .max(bm25::title_match(query_terms, file_name_title(&hit.path)));
            hit.score += title_boost * matched;
            hit.parts.title = title_boost * matched;
        }
    }
    hits.sort_by(best_first);
    hits.truncate(limit);
    hits
}

/// A document's file name without its folders or extension: `Book Notes/Atomic Habits.md`
/// is "Atomic Habits".
pub fn file_name_title(path: &str) -> &str {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    name.rsplit_once('.').map_or(name, |(stem, _)| stem)
}

pub fn merge_normalized(per_space: Vec<Vec<Hit>>, limit: usize) -> Vec<Hit> {
    if per_space.len() <= 1 {
        let mut hits = per_space.into_iter().next().unwrap_or_default();
        hits.truncate(limit);
        return hits;
    }

    let mut best: HashMap<String, Hit> = HashMap::new();
    for mut hits in per_space {
        if hits.is_empty() {
            continue;
        }
        let max = hits.iter().map(|h| h.score).fold(f32::NEG_INFINITY, f32::max);
        let min = hits.iter().map(|h| h.score).fold(f32::INFINITY, f32::min);
        let range = max - min;
        for hit in &mut hits {
            hit.score = if range > f32::EPSILON { (hit.score - min) / range } else { 1.0 };
        }
        for hit in hits {
            match best.get(&hit.path) {
                Some(existing) if existing.score >= hit.score => {}
                _ => {
                    best.insert(hit.path.clone(), hit);
                }
            }
        }
    }
    let mut merged: Vec<Hit> = best.into_values().collect();
    merged.sort_by(best_first);
    merged.truncate(limit);
    merged
}

/// Highest score first, ties broken by path, so a given index always ranks
/// the same way. Every ranking here collects from a `HashMap`, whose order is
/// random per process, and a vault with duplicate notes ties exactly: without
/// the path, the same query could rank them differently on each run, which
/// makes evaluation irreproducible and parity checks flaky.
fn best_first(a: &Hit, b: &Hit) -> std::cmp::Ordering {
    b.score.total_cmp(&a.score).then_with(|| a.path.cmp(&b.path))
}

/// Dot product of two equal-length vectors (0.0 on length mismatch).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Encode an f32 vector as little-endian bytes for compact BLOB storage.
pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

/// Decode a little-endian f32 BLOB back into a vector.
pub fn blob_to_vec(bytes: &[u8]) -> Vec<f32> {
    // `as_chunks` yields `&[u8; 4]` directly, so no per-byte indexing is
    // needed to build the array `from_le_bytes` wants. Any trailing bytes that
    // don't form a whole f32 are ignored, same as `chunks_exact`.
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(path: &str, heading: &str, text: &str, vector: Vec<f32>) -> Candidate {
        Candidate {
            path: path.to_string(),
            title: "t".to_string(),
            source_root: "/root".to_string(),
            heading_path: heading.to_string(),
            text: text.to_string(),
            vector,
            width: None,
            height: None,
            modality: "text".to_string(),
        }
    }

    #[test]
    fn embeds_links_and_headings_carry_no_content() {
        assert!(carries_no_content("![[Tiny Habits.pdf]]"));
        assert!(carries_no_content("## Mind Map\n![[Tiny Habits.pdf]]"));
        assert!(carries_no_content(
            "- [[Atomic Habits]]\n- [[The Power of Habit|Power]]"
        ));
        assert!(carries_no_content(
            "![cover](Media/cover.jpg) [site](https://example.com/a)"
        ));
        assert!(carries_no_content(""));
        assert!(!carries_no_content(
            "Habits are formed through a loop of cue, routine and reward."
        ));
        // Word characters in any script count, not just ASCII.
        assert!(!carries_no_content(
            "Ο Σωκράτης έλεγε ότι η γνώση είναι αρετή"
        ));
        // An embed beside a sentence of its own is content.
        assert!(!carries_no_content(
            "![[diagram.png]] The loop closes when the reward is felt."
        ));
    }

    #[test]
    fn a_source_with_nothing_but_embeds_still_compares_from_them() {
        let content = (
            vec![1.0],
            "A paragraph about the habit loop and its reward.".to_string(),
        );
        let embed = (vec![0.5], "![[x.pdf]]".to_string());
        assert_eq!(
            related_sources(vec![content.clone(), embed.clone()], true),
            vec![vec![1.0]]
        );
        assert_eq!(
            related_sources(vec![embed.clone()], true),
            vec![vec![0.5]],
            "falls back to all"
        );
        assert_eq!(
            related_sources(vec![content, embed], false).len(),
            2,
            "the image space keeps all"
        );
    }

    /// The case `related` was changed for: one coincidental match used to
    /// decide a pair; now it takes agreement in several places.
    #[test]
    fn related_prefers_agreement_in_several_places_over_one_perfect_match() {
        let queries = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ],
        ];
        let one_perfect = vec![
            candidate(
                "one.md",
                "",
                "A chunk that matches one source chunk exactly.",
                vec![1.0, 0.0],
            ),
            candidate(
                "one.md",
                "",
                "A chunk that matches nothing in the source.",
                vec![-1.0, 0.0],
            ),
        ];
        let broadly_alike = vec![
            candidate(
                "broad.md",
                "",
                "A chunk that is close to all of the source.",
                vec![0.8, 0.6],
            ),
            candidate(
                "broad.md",
                "",
                "Another chunk close to all of the source.",
                vec![0.6, 0.8],
            ),
        ];
        let hits = rank_related(
            &queries,
            one_perfect.into_iter().chain(broadly_alike),
            &[],
            10,
            true,
        );
        assert_eq!(hits[0].path, "broad.md", "{hits:?}");
        // Today's single best pair would have put `one.md` first (1.0 vs 0.99).
        assert!(hits[0].score > hits[1].score);
    }

    /// A pair scored on its own is the number `related` lists for it, so a
    /// client placing related notes by pair scores shows what `related` says.
    #[test]
    fn a_pair_scores_what_related_lists_for_it() {
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let docs = [
            candidate(
                "a.md",
                "",
                "A passage close to the first source chunk.",
                vec![0.9, 0.1],
            ),
            candidate(
                "a.md",
                "",
                "A passage close to the second source chunk.",
                vec![0.2, 0.9],
            ),
            candidate("a.md", "Mind Map", "![[a.pdf]]", vec![1.0, 0.0]),
            candidate(
                "b.md",
                "",
                "A single passage between the two of them.",
                vec![0.6, 0.8],
            ),
        ];
        let listed = rank_related(&queries, docs.iter().cloned(), &[], 10, true);
        for hit in &listed {
            let chunks = docs
                .iter()
                .filter(|c| c.path == hit.path)
                .map(|c| (c.vector.as_slice(), c.text.as_str()));
            assert_eq!(
                related_pair_score(&queries, chunks, true),
                Some(hit.score),
                "{}",
                hit.path
            );
        }
        assert_eq!(listed.len(), 2);
    }

    #[test]
    fn a_pair_with_only_boilerplate_on_the_candidate_side_has_no_score() {
        let queries = vec![vec![1.0, 0.0]];
        let template = [(vec![1.0, 0.0], "![[template.pdf]]".to_string())];
        let chunks = || template.iter().map(|(v, t)| (v.as_slice(), t.as_str()));
        assert_eq!(related_pair_score(&queries, chunks(), true), None);
        assert_eq!(
            related_pair_score(&queries, chunks(), false),
            Some(1.0),
            "image chunks are all kept"
        );
    }

    #[test]
    fn related_skips_boilerplate_candidates_in_the_text_space_only() {
        let queries = vec![vec![1.0, 0.0]];
        let make = || {
            vec![
                candidate(
                    "template.md",
                    "Mind Map",
                    "![[template.pdf]]",
                    vec![1.0, 0.0],
                ),
                candidate(
                    "real.md",
                    "",
                    "A passage about the same subject as the source.",
                    vec![0.6, 0.8],
                ),
            ]
        };
        let text = rank_related(&queries, make(), &[], 10, true);
        assert_eq!(
            text.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(),
            ["real.md"]
        );
        let image = rank_related(&queries, make(), &[], 10, false);
        assert_eq!(
            image[0].path, "template.md",
            "image chunks have no text by design"
        );
    }

    #[test]
    fn related_excludes_before_truncating_and_shows_the_best_chunk() {
        let queries = vec![vec![1.0, 0.0]];
        let candidates = vec![
            candidate(
                "source.md",
                "",
                "The source note itself, which is never related.",
                vec![1.0, 0.0],
            ),
            candidate(
                "a.md",
                "Weak",
                "A weaker passage of the related note here.",
                vec![0.6, 0.8],
            ),
            candidate(
                "a.md",
                "Strong",
                "The strongest passage of the related note here.",
                vec![0.9, 0.436],
            ),
            candidate(
                "b.md",
                "",
                "Some other note that is barely related at all.",
                vec![0.1, 0.995],
            ),
        ];
        let hits = rank_related(&queries, candidates, &["source.md".to_string()], 1, true);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "a.md");
        assert_eq!(hits[0].heading_path, "Strong");
        // The mean of a.md's two pairs, not its best one.
        assert!((hits[0].score - 0.75).abs() < 1e-6, "{}", hits[0].score);
    }

    #[test]
    fn blob_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.0];
        assert_eq!(blob_to_vec(&vec_to_blob(&v)), v);
    }

    #[test]
    fn rank_keeps_best_chunk_per_document() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            candidate("a.md", "H1", "weak", vec![0.1, 0.9]),
            candidate("a.md", "H2", "strong", vec![0.9, 0.1]),
            candidate("b.md", "H1", "mid", vec![0.5, 0.5]),
        ];

        let hits = rank(&query, candidates, 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, "a.md");
        assert_eq!(hits[0].text, "strong");
        assert_eq!(hits[1].path, "b.md");
    }

    #[test]
    fn rank_truncates_to_limit() {
        let query = vec![1.0];
        let candidates = (0..5).map(|i| candidate(&format!("{i}.md"), "", "x", vec![i as f32]));
        assert_eq!(rank(&query, candidates, 2).len(), 2);
    }

    /// Duplicate notes tie exactly. They must rank the same way on every run,
    /// whatever order the `HashMap` they were collected in hands them back:
    /// with ten tied documents, an order left to chance is almost never this.
    #[test]
    fn exact_ties_rank_by_path_every_time() {
        let paths: Vec<String> = (0..10).map(|i| format!("{i}.md")).collect();
        let tied = || {
            paths
                .iter()
                .rev()
                .map(|p| candidate(p, "", "A passage every one of these notes shares.", vec![1.0]))
        };
        let ranked = |hits: Vec<Hit>| hits.into_iter().map(|h| h.path).collect::<Vec<_>>();

        assert_eq!(ranked(rank(&[1.0], tied(), 10)), paths);
        assert_eq!(ranked(rank_related(&[vec![1.0]], tied(), &[], 10, true)), paths);
        let text = rank(&[1.0], tied(), 10);
        let image = vec![Hit { path: "x.png".into(), score: 0.5, ..text[0].clone() }];
        let merged = ranked(merge_normalized(vec![text, image], 11));
        assert_eq!(merged[..10], paths[..]);
    }

    #[test]
    fn rank_carries_width_height_from_best_chunk() {
        let query = vec![1.0, 0.0];
        let mut c = candidate("img.png", "", "", vec![0.9, 0.1]);
        c.width = Some(800);
        c.height = Some(600);
        let hits = rank(&query, vec![c], 10);
        assert_eq!(hits[0].width, Some(800));
        assert_eq!(hits[0].height, Some(600));
    }

    #[test]
    fn merge_normalized_passes_through_single_space_unnormalized() {
        // Backward compatibility: today's text-only callers must see raw
        // cosine scores, not normalized ones, when only one space is queried.
        let hits = vec![
            Hit { path: "a.md".into(), title: "a".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.42,
                  width: None, height: None, parts: Default::default() },
            Hit { path: "b.md".into(), title: "b".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.10,
                  width: None, height: None, parts: Default::default() },
        ];
        let merged = merge_normalized(vec![hits], 10);
        assert_eq!(merged[0].score, 0.42);
        assert_eq!(merged[1].score, 0.10);
    }

    #[test]
    fn merge_normalized_normalizes_per_space_before_merging() {
        let text_hits = vec![
            Hit { path: "a.md".into(), title: "a".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.80,
                  width: None, height: None, parts: Default::default() },
            Hit { path: "b.md".into(), title: "b".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.40,
                  width: None, height: None, parts: Default::default() },
        ];
        let image_hits = vec![
            Hit { path: "c.png".into(), title: "c".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.30,
                  width: Some(10), height: Some(10), parts: Default::default() },
            Hit { path: "d.png".into(), title: "d".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.10,
                  width: Some(20), height: Some(20), parts: Default::default() },
        ];
        let merged = merge_normalized(vec![text_hits, image_hits], 10);
        // Each space's best hit normalizes to 1.0, worst to 0.0 — so the top
        // two results are the best-in-space hits from each space, tied.
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0].score, 1.0);
        assert_eq!(merged[1].score, 1.0);
        assert_eq!(merged[2].score, 0.0);
        assert_eq!(merged[3].score, 0.0);
    }

    #[test]
    fn merge_normalized_truncates_to_limit() {
        let hits = (0..5)
            .map(|i| Hit { path: format!("{i}.md"), title: "t".into(), source_root: "/r".into(),
                           heading_path: String::new(), text: String::new(), score: i as f32,
                           width: None, height: None, parts: Default::default() })
            .collect();
        let other = vec![Hit { path: "x.png".into(), title: "x".into(), source_root: "/r".into(),
                                heading_path: String::new(), text: String::new(), score: 1.0,
                                width: None, height: None, parts: Default::default() }];
        assert_eq!(merge_normalized(vec![hits, other], 2).len(), 2);
    }

    #[test]
    fn merge_normalized_dedups_a_document_scored_in_multiple_spaces() {
        // A document with both a text-space chunk and an image-space
        // title-proxy (any markdown note, once image mode is on) can
        // legitimately score in both spaces' results for `related` — it
        // must appear once, at its best normalized score, not twice.
        let text_hits = vec![
            Hit { path: "note.md".into(), title: "note".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.90,
                  width: None, height: None, parts: Default::default() },
            Hit { path: "other.md".into(), title: "other".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.10,
                  width: None, height: None, parts: Default::default() },
        ];
        let image_hits = vec![
            Hit { path: "note.md".into(), title: "note".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.20,
                  width: None, height: None, parts: Default::default() },
            Hit { path: "img.png".into(), title: "img".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.05,
                  width: Some(1), height: Some(1), parts: Default::default() },
        ];
        let merged = merge_normalized(vec![text_hits, image_hits], 10);
        let note_hits: Vec<&Hit> = merged.iter().filter(|h| h.path == "note.md").collect();
        assert_eq!(note_hits.len(), 1, "note.md must appear exactly once");
        assert_eq!(note_hits[0].score, 1.0, "kept at its best (text-space) normalized score");
        assert_eq!(merged.len(), 3);
    }

    // ---- fuse_hits -------------------------------------------------------

    /// Fusion config with only the dense weight set — the title boost is
    /// exercised by its own tests below.
    fn fcfg(dense_weight: f32) -> bm25::SearchConfig {
        bm25::SearchConfig {
            dense_weight,
            title_boost: 0.0,
            ..bm25::SearchConfig::default()
        }
    }

    fn h(path: &str, score: f32) -> Hit {
        Hit {
            path: path.to_string(),
            title: path.to_string(),
            source_root: "/v".to_string(),
            heading_path: String::new(),
            text: format!("text of {path}"),
            score,
            width: None,
            height: None,
            parts: Default::default(),
        }
    }

    #[test]
    fn fuse_hits_combines_both_channels_for_a_shared_document() {
        let got = fuse_hits(vec![h("a.md", 1.0)], vec![h("a.md", 0.0)], &[], &fcfg(0.5), 10);
        assert_eq!(got.len(), 1);
        assert!((got[0].score - 0.5).abs() < 1e-6, "got {}", got[0].score);
    }

    /// A literal match the embedder missed must still surface — that is the
    /// whole reason for a lexical channel.
    #[test]
    fn fuse_hits_keeps_a_document_only_the_lexical_channel_found() {
        let got = fuse_hits(vec![h("dense.md", 0.9)], vec![h("lexonly.md", 1.0)], &[], &fcfg(0.5), 10);
        let paths: Vec<&str> = got.iter().map(|x| x.path.as_str()).collect();
        assert!(paths.contains(&"lexonly.md"), "got {paths:?}");
    }

    #[test]
    fn fuse_hits_keeps_a_document_only_the_dense_channel_found() {
        let got = fuse_hits(vec![h("denseonly.md", 1.0)], vec![], &[], &fcfg(0.5), 10);
        assert_eq!(got.len(), 1);
        assert!((got[0].score - 0.5).abs() < 1e-6, "missing channel scores 0");
    }

    #[test]
    fn fuse_hits_at_weight_one_ranks_exactly_like_dense_alone() {
        let dense = vec![h("a.md", 0.9), h("b.md", 0.4)];
        let lexical = vec![h("b.md", 1.0), h("c.md", 1.0)];
        let got = fuse_hits(dense, lexical, &[], &fcfg(1.0), 10);
        assert_eq!(got[0].path, "a.md");
        assert_eq!(got[0].score, 0.9);
        assert_eq!(got[1].path, "b.md");
        assert_eq!(got[1].score, 0.4);
        assert_eq!(got[2].score, 0.0, "a lexical-only hit contributes nothing at weight 1");
    }

    #[test]
    fn fuse_hits_at_weight_zero_ranks_by_lexical_alone() {
        let got = fuse_hits(vec![h("a.md", 1.0)], vec![h("b.md", 0.8)], &[], &fcfg(0.0), 10);
        assert_eq!(got[0].path, "b.md");
    }

    #[test]
    fn fuse_hits_prefers_dense_metadata_for_a_shared_document() {
        let mut dense_hit = h("a.md", 0.5);
        dense_hit.heading_path = "Design > Storage".to_string();
        let mut lexical_hit = h("a.md", 0.5);
        lexical_hit.heading_path = "somewhere else".to_string();
        let got = fuse_hits(vec![dense_hit], vec![lexical_hit], &[], &fcfg(0.5), 10);
        assert_eq!(got[0].heading_path, "Design > Storage");
    }

    #[test]
    fn fuse_hits_keeps_the_best_score_per_channel_across_chunks() {
        // Two chunks of one document in each channel.
        let dense = vec![h("a.md", 0.2), h("a.md", 0.9)];
        let lexical = vec![h("a.md", 0.1), h("a.md", 0.7)];
        let got = fuse_hits(dense, lexical, &[], &fcfg(0.5), 10);
        assert_eq!(got.len(), 1, "a document must appear once");
        assert!((got[0].score - 0.8).abs() < 1e-6, "got {}", got[0].score);
    }

    #[test]
    fn fuse_hits_respects_the_limit() {
        let dense: Vec<Hit> = (0..10).map(|i| h(&format!("{i}.md"), 0.5)).collect();
        assert_eq!(fuse_hits(dense, vec![], &[], &fcfg(0.5), 3).len(), 3);
    }

    #[test]
    fn fuse_hits_is_deterministic_for_tied_scores() {
        let dense = vec![h("b.md", 0.5), h("a.md", 0.5), h("c.md", 0.5)];
        let first = fuse_hits(dense.clone(), vec![], &[], &fcfg(1.0), 10);
        let second = fuse_hits(dense, vec![], &[], &fcfg(1.0), 10);
        let paths: Vec<&str> = first.iter().map(|x| x.path.as_str()).collect();
        assert_eq!(paths, vec!["a.md", "b.md", "c.md"], "ties break by path");
        assert_eq!(
            paths,
            second.iter().map(|x| x.path.as_str()).collect::<Vec<_>>(),
            "repeat runs must agree, or evaluation is irreproducible"
        );
    }

    #[test]
    fn fusing_nothing_yields_nothing() {
        assert!(fuse_hits(vec![], vec![], &[], &fcfg(0.5), 10).is_empty());
    }

    // ---- title boost -----------------------------------------------------

    fn tcfg(dense_weight: f32, title_boost: f32) -> bm25::SearchConfig {
        bm25::SearchConfig {
            dense_weight,
            title_boost,
            ..bm25::SearchConfig::default()
        }
    }

    fn titled(path: &str, title: &str, score: f32) -> Hit {
        let mut hit = h(path, score);
        hit.title = title.to_string();
        hit
    }

    #[test]
    fn a_title_boost_of_zero_changes_nothing() {
        let hits = vec![titled("a.md", "HNSW", 0.5), titled("b.md", "Cake", 0.6)];
        let got = fuse_hits(hits, vec![], &bm25::tokenize("hnsw"), &tcfg(1.0, 0.0), 10);
        assert_eq!(got[0].path, "b.md", "without a boost the stronger score wins");
    }

    #[test]
    fn a_title_match_can_overtake_a_slightly_better_body_match() {
        let hits = vec![titled("a.md", "HNSW", 0.5), titled("b.md", "Cake", 0.6)];
        let got = fuse_hits(hits, vec![], &bm25::tokenize("hnsw"), &tcfg(1.0, 0.2), 10);
        assert_eq!(got[0].path, "a.md", "the titled note should now lead");
    }

    /// Bounded, so a title match nudges near-ties rather than overriding a
    /// substantially better match.
    #[test]
    fn a_title_match_cannot_overtake_a_far_better_body_match() {
        let hits = vec![titled("a.md", "HNSW", 0.2), titled("b.md", "Cake", 0.95)];
        let got = fuse_hits(hits, vec![], &bm25::tokenize("hnsw"), &tcfg(1.0, 0.2), 10);
        assert_eq!(got[0].path, "b.md", "a 0.2 bonus must not close a 0.75 gap");
    }

    #[test]
    fn a_partial_title_match_gets_a_proportional_share_of_the_boost() {
        let full = fuse_hits(
            vec![titled("a.md", "hnsw graph", 0.5)],
            vec![],
            &bm25::tokenize("hnsw graph"),
            &tcfg(1.0, 0.4),
            10,
        );
        let half = fuse_hits(
            vec![titled("a.md", "hnsw cake", 0.5)],
            vec![],
            &bm25::tokenize("hnsw graph"),
            &tcfg(1.0, 0.4),
            10,
        );
        assert!((full[0].score - 0.9).abs() < 1e-6, "got {}", full[0].score);
        assert!((half[0].score - 0.7).abs() < 1e-6, "got {}", half[0].score);
    }

    /// The boost is applied before truncation, so a titled document can enter
    /// the result set rather than only move within it. Applying it after
    /// `limit` would silently make the boost useless for exactly the
    /// navigational case it exists to serve.
    #[test]
    fn a_title_boost_can_lift_a_document_into_the_result_set() {
        let mut hits: Vec<Hit> = (0..5).map(|i| titled(&format!("{i}.md"), "unrelated", 0.6)).collect();
        hits.push(titled("target.md", "HNSW", 0.45));
        let got = fuse_hits(hits, vec![], &bm25::tokenize("hnsw"), &tcfg(1.0, 0.3), 2);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].path, "target.md", "a boosted document must be able to enter the top-k");
    }

    #[test]
    fn a_title_boost_applies_to_a_lexical_only_hit_too() {
        let got = fuse_hits(
            vec![],
            vec![titled("a.md", "HNSW", 0.5)],
            &bm25::tokenize("hnsw"),
            &tcfg(0.0, 0.2),
            10,
        );
        assert!((got[0].score - 0.7).abs() < 1e-6, "got {}", got[0].score);
    }

    #[test]
    fn an_empty_query_gets_no_title_boost() {
        let got = fuse_hits(vec![titled("a.md", "HNSW", 0.5)], vec![], &[], &tcfg(1.0, 0.5), 10);
        assert!((got[0].score - 0.5).abs() < 1e-6);
    }

    /// The parts say why a hit scored what it did, in the score's own units,
    /// so they must add up to it.
    #[test]
    fn a_fused_hit_s_parts_add_up_to_its_score() {
        let got = fuse_hits(
            vec![titled("a.md", "HNSW", 0.8)],
            vec![titled("a.md", "HNSW", 0.5)],
            &bm25::tokenize("hnsw"),
            &tcfg(0.9, 0.1),
            10,
        );
        let parts = got[0].parts;
        assert!((parts.meaning - 0.72).abs() < 1e-6, "{parts:?}");
        assert!((parts.keywords - 0.05).abs() < 1e-6, "{parts:?}");
        assert!((parts.title - 0.1).abs() < 1e-6, "{parts:?}");
        assert!((parts.meaning + parts.keywords + parts.title - got[0].score).abs() < 1e-6);
    }

    /// With no lexical channel, the whole score before the bonus is meaning.
    #[test]
    fn a_dense_only_hit_s_score_is_all_meaning() {
        let candidates = vec![Candidate {
            path: "a.md".into(),
            title: "a".into(),
            source_root: "/r".into(),
            heading_path: String::new(),
            text: String::new(),
            vector: vec![0.6, 0.8],
            width: None,
            height: None,
            modality: "text".into(),
        }];
        let ranked = rank(&[1.0, 0.0], candidates, 10);
        let got = boost_and_rank(ranked, &bm25::tokenize("a"), 0.1, 10);
        assert!((got[0].parts.meaning - 0.6).abs() < 1e-6, "{:?}", got[0].parts);
        assert_eq!(got[0].parts.keywords, 0.0);
        assert!((got[0].parts.title - 0.1).abs() < 1e-6, "the file name a.md matches");
        assert!((got[0].parts.meaning + got[0].parts.title - got[0].score).abs() < 1e-6);
    }

    /// A vault made from a template: every note opens with the same H1, so
    /// the derived title says nothing, and the file name is the name.
    #[test]
    fn the_file_name_counts_when_the_derived_title_is_a_template_s() {
        let hits = vec![
            titled("Book Notes/Atomic Habits.md", "Notes", 0.5),
            titled("Book Notes/Deep Work.md", "Notes", 0.6),
        ];
        let got = fuse_hits(hits, vec![], &bm25::tokenize("atomic habits"), &tcfg(1.0, 0.2), 10);
        assert_eq!(got[0].path, "Book Notes/Atomic Habits.md");
    }

    /// A corpus named by id keeps its title in the H1, which still counts.
    #[test]
    fn the_derived_title_counts_when_the_file_name_is_an_id() {
        let got = fuse_hits(
            vec![titled("4983.md", "HNSW graphs", 0.5)],
            vec![],
            &bm25::tokenize("hnsw"),
            &tcfg(1.0, 0.2),
            10,
        );
        assert!((got[0].score - 0.7).abs() < 1e-6, "got {}", got[0].score);
    }

    /// The better of the two names, not their sum: a note whose H1 and file
    /// name agree is not boosted twice.
    #[test]
    fn a_title_and_file_name_that_agree_are_counted_once() {
        let got = fuse_hits(
            vec![titled("notes/HNSW.md", "HNSW", 0.5)],
            vec![],
            &bm25::tokenize("hnsw"),
            &tcfg(1.0, 0.2),
            10,
        );
        assert!((got[0].score - 0.7).abs() < 1e-6, "got {}", got[0].score);
    }

    #[test]
    fn a_file_name_title_drops_folders_and_extension() {
        assert_eq!(file_name_title("Book Notes/Atomic Habits.md"), "Atomic Habits");
        assert_eq!(file_name_title("a\\b\\v1.2 notes.md"), "v1.2 notes");
        assert_eq!(file_name_title("README"), "README");
    }
}
