//! Parsing for the qrels (query-relevance) file that drives `gnosis eval`.
//!
//! The file is hand-edited, so the parser is deliberately forgiving about
//! what can be left out (grade defaults to "relevant", space defaults to
//! "search everywhere") and strict about what cannot be guessed (the vault
//! path and each query's text).

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::store::Space;

/// A parsed qrels file: which vault it judges, and the graded queries.
#[derive(Debug, Clone, Deserialize)]
pub struct Qrels {
    /// Vault the judged paths are relative to. `~` is expanded by the caller.
    pub vault: PathBuf,
    /// Queries to score. `query` in TOML (`[[query]]` blocks) reads more
    /// naturally per-block than a plural would.
    #[serde(default, rename = "query")]
    pub queries: Vec<QrelsQuery>,
}

/// One judged query.
#[derive(Debug, Clone, Deserialize)]
pub struct QrelsQuery {
    /// The natural-language query, as a user would type it.
    pub text: String,
    /// Which space to search: `text`, `image`, or `all` (the default).
    #[serde(default)]
    pub space: Option<String>,
    /// Documents judged for this query. Unlisted documents are treated as
    /// irrelevant, and counted as unjudged.
    #[serde(default)]
    pub relevant: Vec<QrelsJudgement>,
}

/// One judged document. `grade` defaults to 1 so the common "this is
/// relevant" case needs no ceremony; 2 marks an ideal hit and 0 records an
/// explicit "not relevant", which is distinct from leaving it out.
#[derive(Debug, Clone, Deserialize)]
pub struct QrelsJudgement {
    pub path: String,
    #[serde(default = "default_grade")]
    pub grade: u8,
}

fn default_grade() -> u8 {
    1
}

impl Qrels {
    /// Parse a qrels file's TOML text.
    pub fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).context("parsing qrels file")
    }

    /// Load and parse a qrels file from disk.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading qrels file {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("in {}", path.display()))
    }
}

impl QrelsQuery {
    /// Spaces this query should search, given which ones the workspace has
    /// available. `None`/`"all"` means every available space; naming one
    /// restricts to it.
    pub fn spaces(&self, available: &[Space]) -> Result<Vec<Space>> {
        match self.space.as_deref() {
            None | Some("all") => Ok(available.to_vec()),
            Some(name) => Ok(vec![name.parse::<Space>()?]),
        }
    }

    /// Judgements in the form the metrics crate wants, with paths resolved
    /// against `vault` so they can be compared to indexed absolute paths.
    pub fn judged(&self, vault: &std::path::Path) -> Vec<eval::Judged> {
        self.relevant
            .iter()
            .map(|j| {
                let full = vault.join(&j.path);
                eval::Judged::new(full.to_string_lossy().to_string(), j.grade)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
vault = "/vault"

[[query]]
text = "how hnsw graph construction works"
space = "text"
relevant = [
  { path = "20 Notes/HNSW.md", grade = 2 },
  { path = "20 Notes/Vector search.md", grade = 1 },
]

[[query]]
text = "a photo of a person"
space = "image"
relevant = [{ path = "99 Meta/figure.png" }]

[[query]]
text = "anything at all"
relevant = []
"#;

    #[test]
    fn parses_vault_and_queries() {
        let q = Qrels::parse(SAMPLE).expect("valid qrels");
        assert_eq!(q.vault, PathBuf::from("/vault"));
        assert_eq!(q.queries.len(), 3);
        assert_eq!(q.queries[0].text, "how hnsw graph construction works");
    }

    #[test]
    fn parses_graded_judgements() {
        let q = Qrels::parse(SAMPLE).unwrap();
        let rel = &q.queries[0].relevant;
        assert_eq!(rel.len(), 2);
        assert_eq!(rel[0].path, "20 Notes/HNSW.md");
        assert_eq!(rel[0].grade, 2);
        assert_eq!(rel[1].grade, 1);
    }

    #[test]
    fn grade_defaults_to_relevant_when_omitted() {
        let q = Qrels::parse(SAMPLE).unwrap();
        assert_eq!(
            q.queries[1].relevant[0].grade, 1,
            "a judgement with no explicit grade means 'relevant'"
        );
    }

    #[test]
    fn omitted_space_means_every_available_space() {
        let q = Qrels::parse(SAMPLE).unwrap();
        let available = [Space::Text, Space::Image];
        assert_eq!(
            q.queries[2].spaces(&available).unwrap(),
            vec![Space::Text, Space::Image]
        );
    }

    #[test]
    fn named_space_restricts_to_it() {
        let q = Qrels::parse(SAMPLE).unwrap();
        let available = [Space::Text, Space::Image];
        assert_eq!(q.queries[0].spaces(&available).unwrap(), vec![Space::Text]);
        assert_eq!(q.queries[1].spaces(&available).unwrap(), vec![Space::Image]);
    }

    #[test]
    fn all_is_accepted_explicitly() {
        let q = Qrels::parse("vault = \"/v\"\n[[query]]\ntext = \"x\"\nspace = \"all\"\n").unwrap();
        let available = [Space::Text, Space::Image];
        assert_eq!(
            q.queries[0].spaces(&available).unwrap(),
            vec![Space::Text, Space::Image]
        );
    }

    #[test]
    fn an_unknown_space_is_an_error() {
        let q = Qrels::parse("vault = \"/v\"\n[[query]]\ntext = \"x\"\nspace = \"audio\"\n").unwrap();
        assert!(
            q.queries[0].spaces(&[Space::Text]).is_err(),
            "an unsupported space name must fail loudly, not be silently ignored"
        );
    }

    #[test]
    fn judged_paths_resolve_against_the_vault() {
        let q = Qrels::parse(SAMPLE).unwrap();
        let judged = q.queries[0].judged(std::path::Path::new("/vault"));
        assert_eq!(judged[0].path, "/vault/20 Notes/HNSW.md");
        assert_eq!(judged[0].grade, 2);
    }

    #[test]
    fn a_missing_vault_key_is_an_error() {
        assert!(
            Qrels::parse("[[query]]\ntext = \"x\"\n").is_err(),
            "vault cannot be guessed, so its absence must fail"
        );
    }

    #[test]
    fn a_query_without_text_is_an_error() {
        assert!(
            Qrels::parse("vault = \"/v\"\n[[query]]\nspace = \"text\"\n").is_err(),
            "a query with no text is meaningless and must fail"
        );
    }

    #[test]
    fn a_file_with_no_queries_parses_to_an_empty_run() {
        let q = Qrels::parse("vault = \"/v\"\n").expect("valid, if useless");
        assert!(q.queries.is_empty());
    }
}
