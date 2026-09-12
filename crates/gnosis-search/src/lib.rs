//! Pure vector scoring/ranking, independent of how candidates are stored.
//! Kept storage-agnostic so it can back both the CLI's SQLite-backed search
//! and other front-ends (e.g. a WASM build indexing over IndexedDB).
use std::collections::HashMap;

/// One embedded chunk to score against a query, before dedup/ranking.
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
}

/// One ranked search result (best chunk per document).
#[derive(Debug, serde::Serialize)]
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
/// aren't comparable across different embedding models — then are merged
/// and truncated to `limit`.
pub fn merge_normalized(per_space: Vec<Vec<Hit>>, limit: usize) -> Vec<Hit> {
    if per_space.len() <= 1 {
        let mut hits = per_space.into_iter().next().unwrap_or_default();
        hits.truncate(limit);
        return hits;
    }

    let mut merged: Vec<Hit> = Vec::new();
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
        merged.extend(hits);
    }
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
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
}
