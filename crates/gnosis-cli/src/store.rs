//! SQLite-backed [`Store`] implementation. Native (depends on `rusqlite`), so
//! it lives in the CLI binary crate rather than the `store` interface crate —
//! mirrors how `embedder.rs`'s `TextEmbedder` (native, fastembed/ort-backed)
//! stays out of the `embed` interface crate.
//!
// TODO: In the future consider using something like [sqlite-vec](https://github.com/asg017/sqlite-vec)
// for faster vector search/retrieval.
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::Connection;
use search::{Candidate, Hit};
use store::{DocWrite, Stats};
pub use store::{Store, TextQuery};
use usearch::{Index, IndexOptions, Key, MetricKind, ScalarKind};

/// Bumped whenever the schema changes in a backward-incompatible way.
pub const SCHEMA_VERSION: i64 = 1;

/// SQLite-backed [`Store`]; gnosis's durable source of truth.
pub struct SqliteStore {
    conn: Connection,
    /// Directory holding one on-disk ANN index per space
    /// (`<db_dir>/index/<space>.usearch`), sibling to the database file.
    /// Always reconstructable from SQLite via `rebuild_index` — a missing
    /// or stale file just means queries fall back to the brute-force path,
    /// never data loss.
    index_dir: PathBuf,
}

impl SqliteStore {
    /// Open (creating if needed) the database at `path`, ensuring the parent
    /// directory exists and the schema is initialized.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating db dir {}", parent.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening database {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", true)?;

        let index_dir = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("index");

        let store = SqliteStore { conn, index_dir };
        store.init_schema()?;
        Ok(store)
    }

    /// Path to one space's on-disk ANN index file.
    fn index_path(&self, space: &str) -> PathBuf {
        self.index_dir.join(format!("{space}.usearch"))
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id           INTEGER PRIMARY KEY,
                path         TEXT NOT NULL UNIQUE,
                kind         TEXT NOT NULL,           -- markdown | image | pdf
                source_root  TEXT NOT NULL,           -- canonical vault root
                content_hash BLOB NOT NULL,
                mtime        INTEGER NOT NULL,
                title        TEXT,
                frontmatter  TEXT,                    -- JSON
                indexed_at   INTEGER NOT NULL,
                width        INTEGER,                 -- image documents only
                height       INTEGER                  -- image documents only
            );
            CREATE INDEX IF NOT EXISTS idx_documents_root ON documents(source_root);

            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY,
                doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ord          INTEGER NOT NULL,
                space        TEXT NOT NULL,           -- text | image
                modality     TEXT NOT NULL,
                text         TEXT,                    -- NULL for image chunks
                heading_path TEXT,
                vector       BLOB                     -- f32[dim], populated at embed time
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc   ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_space ON chunks(space);

            CREATE TABLE IF NOT EXISTS links (
                src_doc  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                dst_path TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_links_src ON links(src_doc);

            CREATE TABLE IF NOT EXISTS tags (
                doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                tag    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tags_doc ON tags(doc_id);
            CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;

        self.ensure_column("documents", "width", "width INTEGER")?;
        self.ensure_column("documents", "height", "height INTEGER")?;

        self.set_meta("schema_version", &SCHEMA_VERSION.to_string())?;
        Ok(())
    }

    /// Add `column` to `table` if it isn't already present. `CREATE TABLE IF
    /// NOT EXISTS` only helps brand-new databases; existing ones need an
    /// explicit, idempotent `ALTER TABLE` — SQLite has no `ADD COLUMN IF NOT
    /// EXISTS`.
    fn ensure_column(&self, table: &str, column: &str, ddl: &str) -> Result<()> {
        let exists = self
            .conn
            .prepare(&format!("PRAGMA table_info({table})"))?
            .query_map([], |r| r.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .iter()
            .any(|c| c == column);
        if !exists {
            self.conn
                .execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {ddl}"))?;
        }
        Ok(())
    }

    /// Every document path having any of `tags`, for Rust-side filtering
    /// (mirrors how `from`/root filtering avoids mixing heterogeneous SQL
    /// param types in `candidates_by_ids`).
    fn paths_with_any_tag(&self, tags: &[String]) -> Result<HashSet<String>> {
        let sql = format!(
            "SELECT DISTINCT d.path FROM tags t JOIN documents d ON d.id = t.doc_id
             WHERE t.tag IN ({})",
            in_placeholders(tags.len())
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let paths = stmt
            .query_map(rusqlite::params_from_iter(tags), |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<HashSet<_>>>()?;
        Ok(paths)
    }

    /// Every chunk in `space` as a scoring candidate, scoped per `filter`.
    /// Shared by `search_space`/`related_space`. Params are kept as a single
    /// homogeneous `Vec<String>` (space, then roots, then tags) bound via
    /// `params_from_iter` against fully anonymous `?` placeholders — mirrors
    /// exactly how the pre-existing root/tag filtering already worked here,
    /// just with `space` folded into the same list instead of being a fixed
    /// SQL literal.
    fn space_candidates(&self, space: &str, filter: &TextQuery) -> Result<Vec<Candidate>> {
        let mut sql = String::from(
            "SELECT d.path, d.title, d.source_root, c.heading_path, c.text, c.vector,
                    d.width, d.height, c.modality
             FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.space = ? AND c.vector IS NOT NULL",
        );
        let roots = filter.from.filter(|f| !f.is_empty());
        if let Some(roots) = roots {
            sql.push_str(&format!(
                " AND d.source_root IN ({})",
                in_placeholders(roots.len())
            ));
        }
        let tags = filter.tags.filter(|t| !t.is_empty());
        if let Some(tags) = tags {
            sql.push_str(&format!(
                " AND d.id IN (SELECT doc_id FROM tags WHERE tag IN ({}))",
                in_placeholders(tags.len())
            ));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let map_row = space_candidate_row_mapper();
        let mut params: Vec<String> = vec![space.to_string()];
        params.extend(roots.into_iter().flatten().cloned());
        params.extend(tags.into_iter().flatten().cloned());
        let candidates: Vec<Candidate> = stmt
            .query_map(rusqlite::params_from_iter(params), map_row)?
            .collect::<rusqlite::Result<_>>()?;
        Ok(candidates)
    }

    /// Fetch full candidate metadata for a set of chunk ids (e.g. from an
    /// ANN search), filtered in Rust per `filter` (simpler than mixing
    /// heterogeneous SQL param types for what's already a small,
    /// over-fetched set).
    fn candidates_by_ids(&self, ids: &[u64], filter: &TextQuery) -> Result<Vec<Candidate>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "SELECT d.path, d.title, d.source_root, c.heading_path, c.text, c.vector,
                    d.width, d.height, c.modality
             FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.id IN ({})",
            in_placeholders(ids.len())
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let id_params: Vec<i64> = ids.iter().map(|id| *id as i64).collect();
        let map_row = space_candidate_row_mapper();
        let mut candidates: Vec<Candidate> = stmt
            .query_map(rusqlite::params_from_iter(&id_params), map_row)?
            .collect::<rusqlite::Result<_>>()?;

        if let Some(roots) = filter.from.filter(|f| !f.is_empty()) {
            candidates.retain(|c| roots.contains(&c.source_root));
        }
        if let Some(tags) = filter.tags.filter(|t| !t.is_empty()) {
            let allowed = self.paths_with_any_tag(tags)?;
            candidates.retain(|c| allowed.contains(&c.path));
        }
        Ok(candidates)
    }

    /// Prefilter candidates via one space's on-disk ANN index for one or
    /// more query vectors (unioned across queries). `None` when no index
    /// exists yet, or an existing one fails to load (treated as "not built"
    /// rather than a hard error — reconstructable via `rebuild_index`, so a
    /// stale/corrupt file should degrade to the brute-force path, not break
    /// search). `k` is how many neighbors to request per query; callers
    /// pick the over-fetch factor since ANN can't filter by vault root,
    /// tag, or modality itself.
    fn ann_candidates(
        &self,
        space: &str,
        queries: &[Vec<f32>],
        k: usize,
        filter: &TextQuery,
    ) -> Result<Option<Vec<Candidate>>> {
        let index_path = self.index_path(space);
        if !index_path.exists() {
            return Ok(None);
        }
        let Some(path_str) = index_path.to_str() else {
            return Ok(None);
        };
        let index = match Index::restore(path_str) {
            Ok(index) => index,
            Err(_) => return Ok(None),
        };

        let mut ids: HashSet<u64> = HashSet::new();
        for q in queries {
            let matches = index.search(q, k)?;
            ids.extend(matches.keys);
        }
        let ids: Vec<u64> = ids.into_iter().collect();

        Ok(Some(self.candidates_by_ids(&ids, filter)?))
    }

    /// Rebuild the on-disk ANN index for `space` from the current SQLite
    /// contents. Always safe to call — the index is fully derived from
    /// SQLite, so a stale or missing index is a (re)build, never data loss.
    /// No-ops (removing any existing index file) when there are zero chunks
    /// in this space, so queries correctly fall back to the brute-force
    /// path rather than querying an empty/stale index.
    pub fn rebuild_index(&mut self, space: &str) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, vector FROM chunks WHERE space = ?1 AND vector IS NOT NULL")?;
        let rows: Vec<(i64, Vec<f32>)> = stmt
            .query_map([space], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(|(id, blob)| (id, search::blob_to_vec(&blob)))
            .collect();

        let index_path = self.index_path(space);
        if rows.is_empty() {
            let _ = std::fs::remove_file(&index_path);
            return Ok(());
        }

        let dimensions = rows[0].1.len();
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options)?;
        index.reserve(rows.len())?;
        for (id, vector) in &rows {
            index.add(*id as Key, vector)?;
        }

        if let Some(parent) = index_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let path_str = index_path
            .to_str()
            .context("index path is not valid UTF-8")?;
        index.save(path_str)?;
        Ok(())
    }
}

/// Shared row → `Candidate` mapper for the two candidate-fetching queries,
/// both of which select the same nine columns in the same order.
fn space_candidate_row_mapper() -> impl Fn(&rusqlite::Row<'_>) -> rusqlite::Result<Candidate> {
    |r: &rusqlite::Row<'_>| {
        Ok(Candidate {
            path: r.get::<_, String>(0)?,
            title: r.get::<_, String>(1)?,
            source_root: r.get::<_, String>(2)?,
            heading_path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            text: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
            vector: search::blob_to_vec(&r.get::<_, Vec<u8>>(5)?),
            width: r.get::<_, Option<i64>>(6)?,
            height: r.get::<_, Option<i64>>(7)?,
            modality: r.get::<_, String>(8)?,
        })
    }
}

impl Store for SqliteStore {
    fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )?;
        Ok(())
    }

    fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let value = self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |row| {
                row.get::<_, String>(0)
            })
            .ok();
        Ok(value)
    }

    fn stats(&self) -> Result<Stats> {
        let documents = self
            .conn
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
        let chunks_text = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE space = 'text'",
            [],
            |r| r.get(0),
        )?;
        let chunks_image = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE space = 'image'",
            [],
            |r| r.get(0),
        )?;
        let indexed_at = self
            .conn
            .query_row("SELECT MAX(indexed_at) FROM documents", [], |r| {
                r.get::<_, Option<i64>>(0)
            })?;

        Ok(Stats {
            documents,
            chunks_text,
            chunks_image,
            indexed_at,
        })
    }

    fn document_hash(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let hash = self
            .conn
            .query_row(
                "SELECT content_hash FROM documents WHERE path = ?1",
                [path],
                |r| r.get::<_, Vec<u8>>(0),
            )
            .ok();
        Ok(hash)
    }

    fn paths_for_roots(&self, roots: &[String]) -> Result<Vec<String>> {
        if roots.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = in_placeholders(roots.len());
        let sql = format!("SELECT path FROM documents WHERE source_root IN ({placeholders})");
        let mut stmt = self.conn.prepare(&sql)?;
        let paths = stmt
            .query_map(rusqlite::params_from_iter(roots), |r| {
                r.get::<_, String>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(paths)
    }

    fn counts_by_root(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT source_root, COUNT(*) FROM documents
             GROUP BY source_root ORDER BY source_root",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    fn delete_by_root(&self, root: &str) -> Result<usize> {
        let n = self
            .conn
            .execute("DELETE FROM documents WHERE source_root = ?1", [root])?;
        Ok(n)
    }

    fn replace_document(&mut self, doc: &DocWrite<'_>) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO documents
                (path, kind, source_root, content_hash, mtime, title, frontmatter, indexed_at, width, height)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(path) DO UPDATE SET
                kind = excluded.kind,
                source_root = excluded.source_root,
                content_hash = excluded.content_hash,
                mtime = excluded.mtime,
                title = excluded.title,
                frontmatter = excluded.frontmatter,
                indexed_at = excluded.indexed_at,
                width = excluded.width,
                height = excluded.height",
            rusqlite::params![
                doc.path,
                doc.kind,
                doc.source_root,
                doc.content_hash,
                doc.mtime,
                doc.title,
                doc.frontmatter,
                doc.indexed_at,
                doc.width,
                doc.height,
            ],
        )?;

        let doc_id: i64 =
            tx.query_row("SELECT id FROM documents WHERE path = ?1", [doc.path], |r| {
                r.get(0)
            })?;

        tx.execute("DELETE FROM chunks WHERE doc_id = ?1", [doc_id])?;
        tx.execute("DELETE FROM links WHERE src_doc = ?1", [doc_id])?;

        for c in doc.chunks {
            tx.execute(
                "INSERT INTO chunks
                    (doc_id, ord, space, modality, text, heading_path, vector)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    doc_id,
                    c.ord as i64,
                    c.space,
                    c.modality,
                    c.text,
                    c.heading_path,
                    search::vec_to_blob(&c.vector),
                ],
            )?;
        }

        for link in doc.links {
            tx.execute(
                "INSERT INTO links (src_doc, dst_path) VALUES (?1, ?2)",
                rusqlite::params![doc_id, link],
            )?;
        }

        tx.execute("DELETE FROM tags WHERE doc_id = ?1", [doc_id])?;
        for tag in doc.tags {
            tx.execute(
                "INSERT INTO tags (doc_id, tag) VALUES (?1, ?2)",
                rusqlite::params![doc_id, tag],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    fn delete_document(&self, path: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM documents WHERE path = ?1", [path])?;
        Ok(())
    }

    fn search_space(&self, space: &str, query: &[f32], limit: usize, filter: &TextQuery) -> Result<Vec<Hit>> {
        // Over-fetch when filtering by root/tag (ANN can't apply either),
        // or when searching the image space (the modality post-filter below
        // can drop a meaningful fraction of the ANN-returned candidates).
        let k = if filter.from.is_some() || filter.tags.is_some() || space == "image" {
            limit * 5
        } else {
            limit
        };
        let queries = [query.to_vec()];
        let mut candidates = if let Some(candidates) = self.ann_candidates(space, &queries, k, filter)? {
            candidates
        } else {
            self.space_candidates(space, filter)?
        };
        if space == "image" {
            candidates.retain(|c| c.modality == "image");
        }
        Ok(search::rank(query, candidates, limit))
    }

    fn chunk_vectors(&self, path: &str, space: &str) -> Result<Vec<Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.vector FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.space = ?1 AND c.vector IS NOT NULL AND d.path = ?2",
        )?;
        let vectors = stmt
            .query_map(rusqlite::params![space, path], |r| r.get::<_, Vec<u8>>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(|b| search::blob_to_vec(&b))
            .collect();
        Ok(vectors)
    }

    fn linked_targets(&self, path: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT l.dst_path FROM links l JOIN documents d ON d.id = l.src_doc
             WHERE d.path = ?1",
        )?;
        let targets = stmt
            .query_map([path], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(targets)
    }

    fn all_document_meta(&self) -> Result<Vec<(String, Option<String>)>> {
        let mut stmt = self.conn.prepare("SELECT path, frontmatter FROM documents")?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    fn related_space(
        &self,
        space: &str,
        query_vectors: &[Vec<f32>],
        exclude_paths: &[String],
        limit: usize,
        filter: &TextQuery,
    ) -> Result<Vec<Hit>> {
        // Always over-fetch: exclude_paths includes the source document
        // itself, and its own chunks are guaranteed to be the top ANN
        // matches for their own queries, so a tight k would leave too few
        // results after exclusion.
        let k = limit * 5;
        if let Some(candidates) = self.ann_candidates(space, query_vectors, k, filter)? {
            return Ok(search::rank_multi(
                query_vectors,
                candidates,
                exclude_paths,
                limit,
            ));
        }
        let candidates = self.space_candidates(space, filter)?;
        Ok(search::rank_multi(
            query_vectors,
            candidates,
            exclude_paths,
            limit,
        ))
    }
}

/// Build `?,?,...` placeholders for an SQL `IN` clause of length `n`.
fn in_placeholders(n: usize) -> String {
    std::iter::repeat_n("?", n).collect::<Vec<_>>().join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use store::ChunkWrite;

    fn write_doc(store: &mut SqliteStore, path: &str, root: &str) {
        store
            .replace_document(&DocWrite {
                path,
                kind: "markdown",
                source_root: root,
                content_hash: b"hash",
                mtime: 0,
                title: "t",
                frontmatter: None,
                indexed_at: 0,
                chunks: &[],
                links: &[],
                tags: &[],
                width: None,
                height: None,
            })
            .unwrap();
    }

    /// Each test gets its own directory (not just its own `.db` filename) —
    /// `SqliteStore` derives its ANN index directory from the db path's
    /// *parent*, so tests sharing a parent directory would silently share
    /// (and corrupt each other's) `index/text.usearch`/`index/image.usearch`
    /// files. Returns the directory so callers can `remove_dir_all` it.
    fn temp_store() -> (SqliteStore, PathBuf) {
        let dir = std::env::temp_dir()
            .join(format!("gnosis-store-space-test-{}-{}", std::process::id(), rand_suffix()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gnosis.db");
        (SqliteStore::open(&path).unwrap(), dir)
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() as u64
    }

    fn write_full_doc(
        store: &mut SqliteStore,
        path: &str,
        kind: &str,
        chunks: &[ChunkWrite],
        width: Option<i64>,
        height: Option<i64>,
    ) {
        store
            .replace_document(&DocWrite {
                path,
                kind,
                source_root: "/vault",
                content_hash: path.as_bytes(),
                mtime: 0,
                title: path,
                frontmatter: None,
                indexed_at: 0,
                chunks,
                links: &[],
                tags: &[],
                width,
                height,
            })
            .unwrap();
    }

    fn chunk(space: &str, modality: &str, text: Option<&str>, vector: Vec<f32>) -> ChunkWrite {
        ChunkWrite {
            ord: 0,
            space: space.to_string(),
            modality: modality.to_string(),
            text: text.map(str::to_string),
            heading_path: String::new(),
            vector,
        }
    }

    #[test]
    fn search_space_excludes_title_proxy_rows() {
        let (mut store, path) = temp_store();
        write_full_doc(
            &mut store,
            "/vault/photo.png",
            "image",
            &[chunk("image", "image", None, vec![1.0, 0.0])],
            Some(800),
            Some(600),
        );
        write_full_doc(
            &mut store,
            "/vault/note.md",
            "markdown",
            &[chunk("image", "text_title", Some("note"), vec![1.0, 0.0])],
            None,
            None,
        );

        let hits = store
            .search_space("image", &[1.0, 0.0], 10, &TextQuery::default())
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/vault/photo.png");
        assert_eq!(hits[0].width, Some(800));
        assert_eq!(hits[0].height, Some(600));

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn related_space_includes_title_proxy_rows() {
        let (mut store, path) = temp_store();
        write_full_doc(
            &mut store,
            "/vault/photo.png",
            "image",
            &[chunk("image", "image", None, vec![1.0, 0.0])],
            Some(800),
            Some(600),
        );
        write_full_doc(
            &mut store,
            "/vault/note.md",
            "markdown",
            &[chunk("image", "text_title", Some("note"), vec![1.0, 0.0])],
            None,
            None,
        );

        let hits = store
            .related_space("image", &[vec![1.0, 0.0]], &[], 10, &TextQuery::default())
            .unwrap();

        let mut paths: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        paths.sort();
        assert_eq!(paths, vec!["/vault/note.md", "/vault/photo.png"]);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn chunk_vectors_scoped_to_space() {
        let (mut store, path) = temp_store();
        write_full_doc(
            &mut store,
            "/vault/note.md",
            "markdown",
            &[
                chunk("text", "text", Some("body"), vec![0.5, 0.5]),
                chunk("image", "text_title", Some("note"), vec![1.0, 0.0]),
            ],
            None,
            None,
        );

        assert_eq!(store.chunk_vectors("/vault/note.md", "text").unwrap().len(), 1);
        assert_eq!(store.chunk_vectors("/vault/note.md", "image").unwrap().len(), 1);
        assert_eq!(
            store.chunk_vectors("/vault/photo.png", "image").unwrap().len(),
            0
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn rebuild_index_is_per_space() {
        let (mut store, path) = temp_store();
        write_full_doc(
            &mut store,
            "/vault/photo.png",
            "image",
            &[chunk("image", "image", None, vec![1.0, 0.0])],
            Some(1),
            Some(1),
        );
        store.rebuild_index("image").unwrap();
        store.rebuild_index("text").unwrap(); // no text chunks — must no-op, not error

        let hits = store
            .search_space("image", &[1.0, 0.0], 10, &TextQuery::default())
            .unwrap();
        assert_eq!(hits.len(), 1);

        let _ = std::fs::remove_dir_all(&path);
    }

    /// `paths_for_roots` and `delete_by_root` must stay scoped to one vault, so
    /// indexing/forgetting one vault never touches another in a shared DB.
    #[test]
    fn root_scoped_queries() {
        let path = std::env::temp_dir().join(format!("gnosis-store-test-{}.db", std::process::id()));
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }

        let mut store = SqliteStore::open(&path).unwrap();
        write_doc(&mut store, "/a/1.md", "/a");
        write_doc(&mut store, "/a/2.md", "/a");
        write_doc(&mut store, "/b/1.md", "/b");

        let mut a = store.paths_for_roots(&["/a".to_string()]).unwrap();
        a.sort();
        assert_eq!(a, vec!["/a/1.md".to_string(), "/a/2.md".to_string()]);
        assert!(store.paths_for_roots(&[]).unwrap().is_empty());

        assert_eq!(store.delete_by_root("/a").unwrap(), 2);
        assert!(store.paths_for_roots(&["/a".to_string()]).unwrap().is_empty());
        assert_eq!(
            store.paths_for_roots(&["/b".to_string()]).unwrap(),
            vec!["/b/1.md".to_string()]
        );

        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
}
