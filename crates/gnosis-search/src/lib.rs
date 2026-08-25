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
}

/// One ranked search result (best chunk per document).
#[derive(Debug)]
pub struct Hit {
    pub path: String,
    pub title: String,
    pub source_root: String,
    pub heading_path: String,
    pub text: String,
    pub score: f32,
}

/// Score `candidates` against `query` by cosine similarity (vectors are
/// assumed L2-normalized, so a dot product suffices), keep only the
/// best-scoring chunk per document path, and return the top `limit` sorted
/// descending by score.
pub fn rank(query: &[f32], candidates: impl IntoIterator<Item = Candidate>, limit: usize) -> Vec<Hit> {
    let mut best: HashMap<String, Hit> = HashMap::new();
    for c in candidates {
        let score = dot(query, &c.vector);
        let entry = best.entry(c.path.clone()).or_insert_with(|| Hit {
            path: c.path,
            title: c.title,
            source_root: c.source_root,
            heading_path: String::new(),
            text: String::new(),
            score: f32::NEG_INFINITY,
        });
        if score > entry.score {
            entry.score = score;
            entry.heading_path = c.heading_path;
            entry.text = c.text;
        }
    }

    let mut hits: Vec<Hit> = best.into_values().collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    hits.truncate(limit);
    hits
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
}
