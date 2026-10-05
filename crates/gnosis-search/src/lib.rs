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
}

/// Score `candidates` against `query` by cosine similarity (vectors are
/// assumed L2-normalized, so a dot product suffices), keep only the
/// best-scoring chunk per document path, and return the top `limit` sorted
/// descending by score.
pub fn rank(query: &[f32], candidates: impl IntoIterator<Item = Candidate>, limit: usize) -> Vec<Hit> {
    rank_by(|v| dot(query, v), candidates, &[], limit)
}

/// Like [`rank`], but scores each candidate against the best (max) cosine
/// similarity across several query vectors instead of one — e.g. a
/// document's own chunks, for `related`. `exclude_paths` is filtered before
/// truncation to `limit`, not after, so an excluded document (e.g. the
/// source itself) never displaces a genuinely eligible one.
pub fn rank_multi(
    queries: &[Vec<f32>],
    candidates: impl IntoIterator<Item = Candidate>,
    exclude_paths: &[String],
    limit: usize,
) -> Vec<Hit> {
    rank_by(
        |v| {
            queries
                .iter()
                .map(|q| dot(q, v))
                .fold(f32::NEG_INFINITY, f32::max)
        },
        candidates,
        exclude_paths,
        limit,
    )
}

/// Shared aggregation for [`rank`]/[`rank_multi`]: score each candidate with
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
        });
        if score > entry.score {
            entry.score = score;
            entry.heading_path = c.heading_path;
            entry.text = c.text;
            entry.width = c.width;
            entry.height = c.height;
        }
    }

    let mut hits: Vec<Hit> = best.into_values().collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
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
            hit.score += title_boost * bm25::title_match(query_terms, &hit.title);
        }
    }
    // Ties broken by path so a given index always ranks the same way; an
    // arbitrary order would make evaluation runs irreproducible.
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    hits.truncate(limit);
    hits
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
    merged.sort_by(|a, b| b.score.total_cmp(&a.score));
    merged.truncate(limit);
    merged
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
                  width: None, height: None },
            Hit { path: "b.md".into(), title: "b".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.10,
                  width: None, height: None },
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
                  width: None, height: None },
            Hit { path: "b.md".into(), title: "b".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.40,
                  width: None, height: None },
        ];
        let image_hits = vec![
            Hit { path: "c.png".into(), title: "c".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.30,
                  width: Some(10), height: Some(10) },
            Hit { path: "d.png".into(), title: "d".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.10,
                  width: Some(20), height: Some(20) },
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
                           width: None, height: None })
            .collect();
        let other = vec![Hit { path: "x.png".into(), title: "x".into(), source_root: "/r".into(),
                                heading_path: String::new(), text: String::new(), score: 1.0,
                                width: None, height: None }];
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
                  width: None, height: None },
            Hit { path: "other.md".into(), title: "other".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.10,
                  width: None, height: None },
        ];
        let image_hits = vec![
            Hit { path: "note.md".into(), title: "note".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.20,
                  width: None, height: None },
            Hit { path: "img.png".into(), title: "img".into(), source_root: "/r".into(),
                  heading_path: String::new(), text: String::new(), score: 0.05,
                  width: Some(1), height: Some(1) },
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
}
