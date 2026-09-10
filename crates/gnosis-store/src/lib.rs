//! Interface for gnosis's durable document/chunk storage. Kept free of any
//! concrete backend's dependencies (e.g. `rusqlite`) so it can be depended on
//! by non-native targets (e.g. a future wasm build) that need only the shape
//! of the store, not a SQLite implementation.
use anyhow::Result;
use search::Hit;

/// A document about to be written, with its parsed chunks.
pub struct DocWrite<'a> {
    pub path: &'a str,
    pub kind: &'a str,
    /// Canonical vault root this document was discovered under.
    pub source_root: &'a str,
    pub content_hash: &'a [u8],
    pub mtime: i64,
    pub title: &'a str,
    pub frontmatter: Option<&'a str>,
    pub indexed_at: i64,
    pub chunks: &'a [ChunkWrite],
    pub links: &'a [String],
    pub tags: &'a [String],
}

/// A single chunk to persist, including its embedding.
pub struct ChunkWrite {
    pub ord: usize,
    pub space: String,
    pub modality: String,
    pub text: Option<String>,
    pub heading_path: String,
    pub vector: Vec<f32>,
}

/// Summary counts for the `status` command.
#[derive(Debug, Default)]
pub struct Stats {
    pub documents: i64,
    pub chunks_text: i64,
    pub chunks_image: i64,
    pub indexed_at: Option<i64>,
}

/// Filters applied when ranking text-space queries (`search_text`/
/// `related_text`). Bundled since both are optional, independent filter
/// dimensions applied together (AND between fields, OR within a field's
/// list) — mirrors how `IndexerArgs` bundles trait objects.
#[derive(Default)]
pub struct TextQuery<'a> {
    /// Restrict to these vault roots. Empty/absent = no restriction.
    pub from: Option<&'a [String]>,
    /// Restrict to documents having any of these tags. Empty/absent = no
    /// restriction.
    pub tags: Option<&'a [String]>,
}

/// Durable storage for gnosis's indexed documents, chunks, and links.
pub trait Store {
    /// Insert or update a meta key/value pair.
    fn set_meta(&self, key: &str, value: &str) -> Result<()>;

    /// Fetch a meta value by key, if present.
    fn get_meta(&self, key: &str) -> Result<Option<String>>;

    /// Compute summary statistics for `status`.
    fn stats(&self) -> Result<Stats>;

    /// Existing content hash for `path`, if the document is already indexed.
    fn document_hash(&self, path: &str) -> Result<Option<Vec<u8>>>;

    /// Document paths whose `source_root` is among `roots` (used to scope
    /// deletion detection to the vaults walked in a run). Empty `roots` matches
    /// nothing.
    fn paths_for_roots(&self, roots: &[String]) -> Result<Vec<String>>;

    /// Document counts grouped by source vault, for `status`.
    fn counts_by_root(&self) -> Result<Vec<(String, i64)>>;

    /// Delete every document (chunks/links cascade) belonging to `root`.
    /// Returns the number of documents removed.
    fn delete_by_root(&self, root: &str) -> Result<usize>;

    /// Insert or replace a document and all its chunks/links in one transaction.
    fn replace_document(&mut self, doc: &DocWrite<'_>) -> Result<()>;

    /// Delete a document (chunks/links cascade) by path.
    fn delete_document(&self, path: &str) -> Result<()>;

    /// Brute-force cosine search over the text space. Vectors are stored
    /// normalized, so a dot product is the cosine similarity. Returns the
    /// best chunk per document, ranked descending, capped at `limit`,
    /// restricted per `filter`.
    fn search_text(&self, query: &[f32], limit: usize, filter: &TextQuery) -> Result<Vec<Hit>>;

    /// A document's own text-space chunk vectors — the query set for `related`.
    fn text_chunk_vectors(&self, path: &str) -> Result<Vec<Vec<f32>>>;

    /// Raw wikilink target texts this document links out to (unresolved —
    /// `[[Some Note]]` yields `"Some Note"`, not a document path).
    fn linked_targets(&self, path: &str) -> Result<Vec<String>>;

    /// Every indexed document's path and raw frontmatter, for resolving
    /// link targets to paths (by filename stem or frontmatter alias).
    fn all_document_meta(&self) -> Result<Vec<(String, Option<String>)>>;

    /// Rank other documents by best chunk-to-chunk cosine similarity against
    /// `query_vectors` (the max across all of them per candidate), excluding
    /// `exclude_paths` before truncation to `limit`, restricted per `filter`.
    fn related_text(
        &self,
        query_vectors: &[Vec<f32>],
        exclude_paths: &[String],
        limit: usize,
        filter: &TextQuery,
    ) -> Result<Vec<Hit>>;
}
