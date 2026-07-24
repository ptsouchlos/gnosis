use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::Connection;

/// Bumped whenever the schema changes in a backward-incompatible way.
pub const SCHEMA_VERSION: i64 = 1;

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
                content_hash BLOB NOT NULL,
                mtime        INTEGER NOT NULL,
                title        TEXT,
                frontmatter  TEXT,                    -- JSON
                indexed_at   INTEGER NOT NULL
            );

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
}
