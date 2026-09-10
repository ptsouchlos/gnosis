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
    /// On-disk ANN index for the text space, sibling to the database file
    /// (`<db_dir>/index/text.usearch`). Always reconstructable from SQLite
    /// via `rebuild_text_index` — a missing or stale file just means
    /// queries fall back to the brute-force path, never data loss.
    text_index_path: PathBuf,
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

        let text_index_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("index")
            .join("text.usearch");

        let store = SqliteStore {
            conn,
            text_index_path,
        };
        store.init_schema()?;
        Ok(store)
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
                indexed_at   INTEGER NOT NULL
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

        self.set_meta("schema_version", &SCHEMA_VERSION.to_string())?;
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

    /// Fetch every text-space chunk as a scoring candidate, scoped per
    /// `filter`. Shared by `search_text`/`related_text`.
    fn text_candidates(&self, filter: &TextQuery) -> Result<Vec<Candidate>> {
        let mut sql = String::from(
            "SELECT d.path, d.title, d.source_root, c.heading_path, c.text, c.vector
             FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.space = 'text' AND c.vector IS NOT NULL",
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
        let map_row = |r: &rusqlite::Row<'_>| {
            Ok(Candidate {
                path: r.get::<_, String>(0)?,
                title: r.get::<_, String>(1)?,
                source_root: r.get::<_, String>(2)?,
                heading_path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                text: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                vector: search::blob_to_vec(&r.get::<_, Vec<u8>>(5)?),
            })
        };
        let params: Vec<&String> = roots.into_iter().flatten().chain(tags.into_iter().flatten()).collect();
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
            "SELECT d.path, d.title, d.source_root, c.heading_path, c.text, c.vector
             FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.id IN ({})",
            in_placeholders(ids.len())
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let id_params: Vec<i64> = ids.iter().map(|id| *id as i64).collect();
        let map_row = |r: &rusqlite::Row<'_>| {
            Ok(Candidate {
                path: r.get::<_, String>(0)?,
                title: r.get::<_, String>(1)?,
                source_root: r.get::<_, String>(2)?,
                heading_path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                text: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                vector: search::blob_to_vec(&r.get::<_, Vec<u8>>(5)?),
            })
        };
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

    /// Prefilter candidates via the on-disk ANN index for one or more query
    /// vectors (unioned across queries). `None` when no index exists yet,
    /// or an existing one fails to load (treated as "not built" rather
    /// than a hard error — reconstructable via `rebuild_text_index`, so a
    /// stale/corrupt file should degrade to the brute-force path, not
    /// break search). `k` is how many neighbors to request per query;
    /// callers pick the over-fetch factor since ANN can't filter by vault
    /// root, tag, or excluded path itself.
    fn ann_candidates(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        filter: &TextQuery,
    ) -> Result<Option<Vec<Candidate>>> {
        if !self.text_index_path.exists() {
            return Ok(None);
        }
        let Some(path_str) = self.text_index_path.to_str() else {
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

    /// Rebuild the on-disk ANN index for the text space from the current
    /// SQLite contents. Always safe to call — the index is fully derived
    /// from SQLite, so a stale or missing index is a (re)build, never data
    /// loss. No-ops (removing any existing index file) when there are zero
    /// text chunks, so queries correctly fall back to the brute-force path
    /// rather than querying an empty/stale index.
    pub fn rebuild_text_index(&mut self) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, vector FROM chunks WHERE space = 'text' AND vector IS NOT NULL")?;
        let rows: Vec<(i64, Vec<f32>)> = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(|(id, blob)| (id, search::blob_to_vec(&blob)))
            .collect();

        if rows.is_empty() {
            let _ = std::fs::remove_file(&self.text_index_path);
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

        if let Some(parent) = self.text_index_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let path_str = self
            .text_index_path
            .to_str()
            .context("index path is not valid UTF-8")?;
        index.save(path_str)?;
        Ok(())
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
                (path, kind, source_root, content_hash, mtime, title, frontmatter, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(path) DO UPDATE SET
                kind = excluded.kind,
                source_root = excluded.source_root,
                content_hash = excluded.content_hash,
                mtime = excluded.mtime,
                title = excluded.title,
                frontmatter = excluded.frontmatter,
                indexed_at = excluded.indexed_at",
            rusqlite::params![
                doc.path,
                doc.kind,
                doc.source_root,
                doc.content_hash,
                doc.mtime,
                doc.title,
                doc.frontmatter,
                doc.indexed_at,
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

    fn search_text(&self, query: &[f32], limit: usize, filter: &TextQuery) -> Result<Vec<Hit>> {
        // Over-fetch when filtering by root or tag, since ANN can't apply
        // either filter itself.
        let k = if filter.from.is_some() || filter.tags.is_some() {
            limit * 5
        } else {
            limit
        };
        let queries = [query.to_vec()];
        if let Some(candidates) = self.ann_candidates(&queries, k, filter)? {
            return Ok(search::rank(query, candidates, limit));
        }
        let candidates = self.text_candidates(filter)?;
        Ok(search::rank(query, candidates, limit))
    }

    fn text_chunk_vectors(&self, path: &str) -> Result<Vec<Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.vector FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.space = 'text' AND c.vector IS NOT NULL AND d.path = ?1",
        )?;
        let vectors = stmt
            .query_map([path], |r| r.get::<_, Vec<u8>>(0))?
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

    fn related_text(
        &self,
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
        if let Some(candidates) = self.ann_candidates(query_vectors, k, filter)? {
            return Ok(search::rank_multi(
                query_vectors,
                candidates,
                exclude_paths,
                limit,
            ));
        }
        let candidates = self.text_candidates(filter)?;
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
            })
            .unwrap();
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
