use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::Connection;

/// Bumped whenever the schema changes in a backward-incompatible way.
pub const SCHEMA_VERSION: i64 = 1;

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

/// One ranked search result (best chunk per document).
#[derive(Debug)]
pub struct SearchHit {
    pub path: String,
    pub title: String,
    pub source_root: String,
    pub heading_path: String,
    pub text: String,
    pub score: f32,
}

/// Encode an f32 vector as little-endian bytes for BLOB storage.
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

/// Decode a little-endian f32 BLOB back into a vector.
fn blob_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Wraps the SQLite connection that is gnosis's durable source of truth.
pub struct Store {
    conn: Connection,
}

/// Summary counts for the `status` command.
#[derive(Debug, Default)]
pub struct Stats {
    pub documents: i64,
    pub chunks_text: i64,
    pub chunks_image: i64,
    pub indexed_at: Option<i64>,
}

impl Store {
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

        let store = Store { conn };
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

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;

        self.set_meta("schema_version", &SCHEMA_VERSION.to_string())?;
        Ok(())
    }

    /// Insert or update a meta key/value pair.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )?;
        Ok(())
    }

    /// Fetch a meta value by key, if present.
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let value = self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |row| {
                row.get::<_, String>(0)
            })
            .ok();
        Ok(value)
    }

    /// Compute summary statistics for `status`.
    pub fn stats(&self) -> Result<Stats> {
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

    /// Existing content hash for `path`, if the document is already indexed.
    pub fn document_hash(&self, path: &str) -> Result<Option<Vec<u8>>> {
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

    /// Document paths whose `source_root` is among `roots` (used to scope
    /// deletion detection to the vaults walked in a run). Empty `roots` matches
    /// nothing.
    pub fn paths_for_roots(&self, roots: &[String]) -> Result<Vec<String>> {
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

    /// Document counts grouped by source vault, for `status`.
    pub fn counts_by_root(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT source_root, COUNT(*) FROM documents
             GROUP BY source_root ORDER BY source_root",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Delete every document (chunks/links cascade) belonging to `root`.
    /// Returns the number of documents removed.
    pub fn delete_by_root(&self, root: &str) -> Result<usize> {
        let n = self
            .conn
            .execute("DELETE FROM documents WHERE source_root = ?1", [root])?;
        Ok(n)
    }

    /// Insert or replace a document and all its chunks/links in one transaction.
    pub fn replace_document(&mut self, doc: &DocWrite<'_>) -> Result<()> {
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
                    vec_to_blob(&c.vector),
                ],
            )?;
        }

        for link in doc.links {
            tx.execute(
                "INSERT INTO links (src_doc, dst_path) VALUES (?1, ?2)",
                rusqlite::params![doc_id, link],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Delete a document (chunks/links cascade) by path.
    pub fn delete_document(&self, path: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM documents WHERE path = ?1", [path])?;
        Ok(())
    }

    /// Brute-force cosine search over the text space. Vectors are stored
    /// normalized, so a dot product is the cosine similarity. Returns the best
    /// chunk per document, ranked descending, capped at `limit`. When `from` is
    /// given, results are restricted to those source vault roots.
    pub fn search_text(
        &self,
        query: &[f32],
        limit: usize,
        from: Option<&[String]>,
    ) -> Result<Vec<SearchHit>> {
        let mut sql = String::from(
            "SELECT d.path, d.title, d.source_root, c.heading_path, c.text, c.vector
             FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.space = 'text' AND c.vector IS NOT NULL",
        );
        let filter = from.filter(|f| !f.is_empty());
        if let Some(roots) = filter {
            sql.push_str(&format!(
                " AND d.source_root IN ({})",
                in_placeholders(roots.len())
            ));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let map_row = |r: &rusqlite::Row<'_>| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Vec<u8>>(5)?,
            ))
        };
        let rows = match filter {
            Some(roots) => stmt.query_map(rusqlite::params_from_iter(roots), map_row)?,
            None => stmt.query_map([], map_row)?,
        };

        let mut best: HashMap<String, SearchHit> = HashMap::new();
        for row in rows {
            let (path, title, source_root, heading_path, text, blob) = row?;
            let score = dot(query, &blob_to_vec(&blob));
            let entry = best.entry(path.clone()).or_insert_with(|| SearchHit {
                path,
                title,
                source_root,
                heading_path: String::new(),
                text: String::new(),
                score: f32::NEG_INFINITY,
            });
            if score > entry.score {
                entry.score = score;
                entry.heading_path = heading_path.unwrap_or_default();
                entry.text = text.unwrap_or_default();
            }
        }

        let mut hits: Vec<SearchHit> = best.into_values().collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(limit);
        Ok(hits)
    }
}

/// Build `?,?,...` placeholders for an SQL `IN` clause of length `n`.
fn in_placeholders(n: usize) -> String {
    std::iter::repeat_n("?", n).collect::<Vec<_>>().join(",")
}

/// Dot product of two equal-length vectors (0.0 on length mismatch).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_doc(store: &mut Store, path: &str, root: &str) {
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

        let mut store = Store::open(&path).unwrap();
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
