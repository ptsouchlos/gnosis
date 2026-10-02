//! SQLite-backed [`Store`] implementation. Native (depends on `rusqlite`), so
//! it lives in the CLI binary crate rather than the `store` interface crate —
//! mirrors how `embedder.rs`'s `TextEmbedder` (native, fastembed/ort-backed)
//! stays out of the `embed` interface crate.
//!
// TODO: In the future consider using something like [sqlite-vec](https://github.com/asg017/sqlite-vec)
// for faster vector search/retrieval.
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rusqlite::Connection;
use search::{Candidate, Hit};
use store::{DocWrite, Stats};
pub use store::{Space, Store, TextQuery};
use usearch::{Index, IndexOptions, Key, MetricKind, ScalarKind};

/// Bumped whenever the schema changes in a backward-incompatible way.
pub const SCHEMA_VERSION: i64 = 1;

/// Bumped whenever the lexical index's contents would differ — a changed
/// tokenizer, or a change to which chunks are indexed — to force a rebuild on
/// next open.
const LEXICAL_INDEX_VERSION: i64 = 1;
const LEXICAL_INDEX_VERSION_KEY: &str = "lexical_index_version";

/// Name of the directory (sibling to the database file) holding one on-disk
/// ANN index file per `(space, modality)` pair.
const INDEX_DIR_NAME: &str = "index";

/// SQLite-backed [`Store`]; gnosis's durable source of truth.
pub struct SqliteStore {
    conn: Connection,
    /// Directory holding one on-disk ANN index per `(space, modality)` pair
    /// (`<db_dir>/index/<space>.<modality>.usearch`), sibling to the database
    /// file. Always reconstructable from SQLite via `rebuild_index` — a
    /// missing or stale file just means queries fall back to the brute-force
    /// path, never data loss.
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
            .join(INDEX_DIR_NAME);

        let store = SqliteStore { conn, index_dir };
        store.init_schema()?;
        Ok(store)
    }

    /// Path to one `(space, modality)` pair's on-disk ANN index file.
    ///
    /// Granularity is the pair, not the space, because an ANN index is only
    /// meaningful over vectors that are mutually rankable. The `image` space
    /// holds CLIP vision vectors *and* CLIP text title proxies, and CLIP's
    /// modality gap puts those in disjoint similarity ranges (measured on a
    /// real vault: text->image peaks around 0.36, text->text_title averages
    /// 0.76). Indexing them together meant every nearest neighbour of a text
    /// query was a title proxy, so image search returned nothing once a vault
    /// had more than a handful of notes.
    fn index_path(&self, space: Space, modality: &str) -> PathBuf {
        self.index_dir.join(format!("{space}.{modality}.usearch"))
    }

    /// Every modality present in `space` with at least one vector, sorted for
    /// deterministic rebuild order.
    fn modalities_in_space(&self, space: Space) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT modality FROM chunks
             WHERE space = ?1 AND vector IS NOT NULL
             ORDER BY modality",
        )?;
        let modalities = stmt
            .query_map([space.as_str()], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(modalities)
    }

    /// Delete every ANN index file belonging to `space`, including the
    /// pre-split `{space}.usearch` layout, so a rebuild can never leave a
    /// stale file behind for a modality that no longer has rows.
    fn remove_space_indexes(&self, space: Space) -> Result<()> {
        let prefix = format!("{space}.");
        let entries = match std::fs::read_dir(&self.index_dir) {
            Ok(entries) => entries,
            Err(_) => return Ok(()), // no index dir yet: nothing to clean
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && name.ends_with(".usearch") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        Ok(())
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

            -- Full-text index over text-space chunks, for the lexical channel.
            -- `content='chunks'` makes this an external-content index: FTS5
            -- stores only the inverted index and reads the text back from
            -- `chunks`, so the text is not duplicated. Sync is therefore the
            -- triggers' job, not FTS5's.
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            );

            -- Only text-space chunks with text are indexed: image chunks carry
            -- no text, and title proxies exist for `related` to traverse rather
            -- than to be matched lexically.
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks
            WHEN new.space = 'text' AND new.text IS NOT NULL BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks
            WHEN old.space = 'text' AND old.text IS NOT NULL BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END;


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

        self.ensure_lexical_index()?;

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

    /// Build the full-text index when it is missing or was built by an older
    /// scheme.
    ///
    /// The sync triggers only fire on new writes, so a database indexed before
    /// this index existed would otherwise have a silently dead lexical channel:
    /// every query simply matches nothing, with no error to notice.
    ///
    /// Gated on a `meta` marker rather than on whether the index looks empty.
    /// There is no cheap, honest emptiness test for an external-content FTS5
    /// table — `COUNT(*) FROM chunks_fts` reports rows in the *content* table,
    /// so it is non-zero even when the index holds nothing, which is exactly
    /// the trap an earlier version of this function fell into. A version marker
    /// also means a future change to the tokenizer or the filter below can force
    /// a rebuild by bumping the constant.
    ///
    /// The repopulation is a filtered `INSERT ... SELECT`, not FTS5's own
    /// `'rebuild'`: `rebuild` reads every row of the content table, which would
    /// pull in the image chunks and title proxies the triggers exclude, and a
    /// title proxy in the lexical index would make filenames match as prose.
    fn ensure_lexical_index(&self) -> Result<()> {
        let current = LEXICAL_INDEX_VERSION.to_string();
        if self.get_meta(LEXICAL_INDEX_VERSION_KEY)?.as_deref() == Some(current.as_str()) {
            return Ok(());
        }
        self.conn.execute_batch(
            "INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all');
             INSERT INTO chunks_fts(rowid, text)
             SELECT id, text FROM chunks WHERE space = 'text' AND text IS NOT NULL;",
        )?;
        self.set_meta(LEXICAL_INDEX_VERSION_KEY, &current)?;
        Ok(())
    }

    /// Corpus statistics for the lexical channel.
    ///
    /// Length is measured in **characters**, not tokens, for both the per-chunk
    /// length and the average. BM25's length normalization uses the ratio
    /// `dl / avgdl`, so any consistent unit works, and characters come straight
    /// from SQL (`LENGTH` over TEXT counts characters) without tokenizing the
    /// whole corpus at query time or storing a second length column.
    fn lexical_corpus_stats(&self) -> Result<search::bm25::CorpusStats> {
        let (count, avg): (i64, Option<f64>) = self.conn.query_row(
            "SELECT COUNT(*), AVG(LENGTH(text)) FROM chunks
             WHERE space = 'text' AND text IS NOT NULL",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        Ok(search::bm25::CorpusStats {
            total_chunks: count.max(0) as u64,
            avg_chunk_len: avg.unwrap_or(0.0) as f32,
        })
    }

    /// How many text chunks contain `term`.
    fn term_doc_freq(&self, term: &str) -> Result<u64> {
        // Quoted so FTS5 reads it as a bare term rather than as syntax.
        let expr = format!("\"{}\"", term.replace('"', ""));
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?1",
            [expr],
            |r| r.get(0),
        )?;
        Ok(count.max(0) as u64)
    }

    /// Every chunk in `space` as a scoring candidate, scoped per `filter`.
    /// Shared by `search_space`/`related_space`. Params are kept as a single
    /// homogeneous `Vec<String>` (space, then roots, then tags) bound via
    /// `params_from_iter` against fully anonymous `?` placeholders — mirrors
    /// exactly how the pre-existing root/tag filtering already worked here,
    /// just with `space` folded into the same list instead of being a fixed
    /// SQL literal.
    /// `modality` restricts to a single modality within the space (what
    /// `search_space` wants); `None` spans every modality (what
    /// `related_space` wants, since it traverses title proxies too).
    fn space_candidates(
        &self,
        space: Space,
        modality: Option<&str>,
        filter: &TextQuery,
    ) -> Result<Vec<Candidate>> {
        let mut sql = String::from(
            "SELECT d.path, d.title, d.source_root, c.heading_path, c.text, c.vector,
                    d.width, d.height, c.modality
             FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.space = ? AND c.vector IS NOT NULL",
        );
        if modality.is_some() {
            sql.push_str(" AND c.modality = ?");
        }
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
        let mut params: Vec<String> = vec![space.as_str().to_string()];
        params.extend(modality.map(str::to_string));
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

    /// Chunk ids from one `(space, modality)` ANN index for one or more query
    /// vectors (unioned across queries). `None` when no index exists yet, or
    /// an existing one fails to load (treated as "not built" rather than a
    /// hard error — reconstructable via `rebuild_index`, so a stale/corrupt
    /// file should degrade to the brute-force path, not break search). `k` is
    /// how many neighbors to request per query; callers pick the over-fetch
    /// factor since ANN can't filter by vault root or tag.
    ///
    /// Modality is no longer something callers over-fetch to compensate for:
    /// the index itself is per-modality, so what comes back needs no
    /// modality post-filter.
    fn ann_ids(
        &self,
        space: Space,
        modality: &str,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Option<HashSet<u64>>> {
        let index_path = self.index_path(space, modality);
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
        Ok(Some(ids))
    }

    /// Rebuild every on-disk ANN index belonging to `space` from the current
    /// SQLite contents — one per modality present. Always safe to call: the
    /// indexes are fully derived from SQLite, so a stale or missing index is a
    /// (re)build, never data loss.
    ///
    /// Every existing index file for the space is removed first, so a modality
    /// that no longer has rows leaves nothing stale behind, and the pre-split
    /// `{space}.usearch` layout is cleaned up on first rebuild after upgrade.
    pub fn rebuild_index(&mut self, space: Space, quantization: ScalarKind) -> Result<()> {
        self.remove_space_indexes(space)?;
        for modality in self.modalities_in_space(space)? {
            self.rebuild_modality_index(space, &modality, quantization)?;
        }
        Ok(())
    }

    /// Build one `(space, modality)` ANN index. No-ops when the pair has no
    /// vectors, so queries fall back to the brute-force path rather than
    /// reading an empty index.
    fn rebuild_modality_index(
        &self,
        space: Space,
        modality: &str,
        quantization: ScalarKind,
    ) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "SELECT id, vector FROM chunks
             WHERE space = ?1 AND modality = ?2 AND vector IS NOT NULL",
        )?;
        let rows: Vec<(i64, Vec<f32>)> = stmt
            .query_map([space.as_str(), modality], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .map(|(id, blob)| (id, search::blob_to_vec(&blob)))
            .collect();

        let index_path = self.index_path(space, modality);
        if rows.is_empty() {
            let _ = std::fs::remove_file(&index_path);
            return Ok(());
        }

        let dimensions = rows[0].1.len();
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::IP,
            quantization,
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

/// The one modality `search` retrieves from within a space.
///
/// A space can hold several modalities, but only one of them is what a user
/// means by searching it. The `image` space additionally stores CLIP-text
/// title proxies, which exist so `related` can traverse from an image to a
/// note and back; they are not themselves search results. `related` therefore
/// spans every modality while `search` targets exactly this one.
fn search_modality(space: Space) -> &'static str {
    match space {
        Space::Text => "text",
        Space::Image => "image",
    }
}

/// Map a config quantization name to a `usearch` scalar kind.
pub fn resolve_quantization(name: &str) -> Result<ScalarKind> {
    let kind = match name {
        "f32" => ScalarKind::F32,
        "f16" => ScalarKind::F16,
        "i8" => ScalarKind::I8,
        other => bail!("unknown ann quantization '{other}' (try: f32, f16, i8)"),
    };
    Ok(kind)
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

    fn lexical_candidates(
        &self,
        terms: &[String],
        limit: usize,
        filter: &TextQuery,
    ) -> Result<store::LexicalMatches> {
        let stats = self.lexical_corpus_stats()?;
        if terms.is_empty() || stats.total_chunks == 0 {
            return Ok(store::LexicalMatches {
                stats,
                doc_freqs: vec![0; terms.len()],
                candidates: Vec::new(),
            });
        }

        let doc_freqs = terms
            .iter()
            .map(|t| self.term_doc_freq(t))
            .collect::<Result<Vec<u64>>>()?;

        // Any term matching is enough to be a candidate; how many and how
        // often is what BM25+ then weighs.
        let match_expr = terms
            .iter()
            .map(|t| format!("\"{}\"", t.replace('"', "")))
            .collect::<Vec<_>>()
            .join(" OR ");

        let mut sql = String::from(
            "SELECT d.path, d.title, d.source_root, c.heading_path, c.text, c.vector,
                    d.width, d.height, c.modality
             FROM chunks_fts f
             JOIN chunks c ON c.id = f.rowid
             JOIN documents d ON d.id = c.doc_id
             WHERE chunks_fts MATCH ?",
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
        // FTS5's own bm25() only orders the candidate set here; BM25+ does the
        // real scoring. Ordering by it means a truncated set keeps the most
        // promising rows rather than an arbitrary slice.
        sql.push_str(" ORDER BY bm25(chunks_fts) LIMIT ?");

        let mut params: Vec<String> = vec![match_expr];
        params.extend(roots.into_iter().flatten().cloned());
        params.extend(tags.into_iter().flatten().cloned());
        params.push(limit.to_string());

        let mut stmt = self.conn.prepare(&sql)?;
        let map_row = space_candidate_row_mapper();
        let candidates: Vec<Candidate> = stmt
            .query_map(rusqlite::params_from_iter(params), map_row)?
            .collect::<rusqlite::Result<_>>()?;

        Ok(store::LexicalMatches {
            stats,
            doc_freqs,
            candidates,
        })
    }

    fn search_space(&self, space: Space, query: &[f32], limit: usize, filter: &TextQuery) -> Result<Vec<Hit>> {
        // Over-fetch only when filtering by root/tag, which ANN can't apply.
        // Modality needs no over-fetch: the index is per-modality, so nothing
        // is discarded after retrieval.
        let k = if filter.from.is_some() || filter.tags.is_some() {
            limit * 5
        } else {
            limit
        };
        let modality = search_modality(space);
        let queries = [query.to_vec()];
        let candidates = match self.ann_ids(space, modality, &queries, k)? {
            Some(ids) => {
                let ids: Vec<u64> = ids.into_iter().collect();
                self.candidates_by_ids(&ids, filter)?
            }
            None => self.space_candidates(space, Some(modality), filter)?,
        };
        Ok(search::rank(query, candidates, limit))
    }

    fn chunk_vectors(&self, path: &str, space: Space) -> Result<Vec<Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.vector FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.space = ?1 AND c.vector IS NOT NULL AND d.path = ?2",
        )?;
        let vectors = stmt
            .query_map(rusqlite::params![space.as_str(), path], |r| r.get::<_, Vec<u8>>(0))?
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
        space: Space,
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

        // `related` is cross-modal by design (an image's title proxy is how it
        // reaches related notes), so it queries every modality index in the
        // space and unions the hits. If any one of them is missing, fall back
        // to brute force over the whole space rather than silently returning a
        // partial candidate set.
        let modalities = self.modalities_in_space(space)?;
        let mut ids: HashSet<u64> = HashSet::new();
        let mut fully_indexed = !modalities.is_empty();
        for modality in &modalities {
            match self.ann_ids(space, modality, query_vectors, k)? {
                Some(found) => ids.extend(found),
                None => {
                    fully_indexed = false;
                    break;
                }
            }
        }

        let candidates = if fully_indexed {
            let ids: Vec<u64> = ids.into_iter().collect();
            self.candidates_by_ids(&ids, filter)?
        } else {
            self.space_candidates(space, None, filter)?
        };
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
            .search_space(Space::Image, &[1.0, 0.0], 10, &TextQuery::default())
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
            .related_space(Space::Image, &[vec![1.0, 0.0]], &[], 10, &TextQuery::default())
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

        assert_eq!(store.chunk_vectors("/vault/note.md", Space::Text).unwrap().len(), 1);
        assert_eq!(store.chunk_vectors("/vault/note.md", Space::Image).unwrap().len(), 1);
        assert_eq!(
            store.chunk_vectors("/vault/photo.png", Space::Image).unwrap().len(),
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
        store.rebuild_index(Space::Image, ScalarKind::F32).unwrap();
        store.rebuild_index(Space::Text, ScalarKind::F32).unwrap(); // no text chunks — must no-op, not error

        let hits = store
            .search_space(Space::Image, &[1.0, 0.0], 10, &TextQuery::default())
            .unwrap();
        assert_eq!(hits.len(), 1);

        let _ = std::fs::remove_dir_all(&path);
    }

    /// A `rebuild_index` call at a lossy quantization must still round-trip
    /// through `Index::save`/`Index::restore` and preserve ranking — the
    /// exact score can shift slightly from rounding, but the nearer vector
    /// must still come out on top.
    #[test]
    fn rebuild_index_supports_i8_quantization() {
        let (mut store, path) = temp_store();
        write_full_doc(
            &mut store,
            "/vault/near.png",
            "image",
            &[chunk("image", "image", None, vec![1.0, 0.0])],
            Some(1),
            Some(1),
        );
        write_full_doc(
            &mut store,
            "/vault/far.png",
            "image",
            &[chunk("image", "image", None, vec![0.0, 1.0])],
            Some(1),
            Some(1),
        );
        store.rebuild_index(Space::Image, ScalarKind::I8).unwrap();

        let hits = store
            .search_space(Space::Image, &[1.0, 0.0], 10, &TextQuery::default())
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, "/vault/near.png");

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

    /// Regression test for the P0 bug: image search returned nothing on any
    /// real-sized vault.
    ///
    /// The `image` space holds two modalities — CLIP vision vectors
    /// (`modality = "image"`) and CLIP text title proxies
    /// (`modality = "text_title"`) — and CLIP's modality gap means a text
    /// query scores far closer to other text than to any image. So when both
    /// live in one ANN index, every one of the `limit * 5` nearest neighbours
    /// is a title proxy and the modality filter drops the entire candidate
    /// set.
    ///
    /// The fixture must therefore exceed the over-fetch window: with fewer
    /// than `limit * 5` title proxies the ANN returns the whole index and the
    /// bug cannot appear, which is exactly why the original unit test (1
    /// image + 1 note) and the documented manual walkthrough both missed it.
    #[test]
    fn image_search_is_not_starved_by_title_proxy_rows() {
        let (mut store, path) = temp_store();

        // 60 title proxies (> limit * 5 == 50), each a perfect match for the
        // query, standing in for a vault's markdown notes.
        for i in 0..60 {
            write_full_doc(
                &mut store,
                &format!("/vault/note{i}.md"),
                "markdown",
                &[chunk("image", "text_title", Some("note"), vec![1.0, 0.0])],
                None,
                None,
            );
        }
        // 3 images, genuinely less similar to the query than any title proxy —
        // mirroring the real measured gap (text->image peaks ~0.36 while
        // text->text_title averages ~0.76).
        for i in 0..3 {
            write_full_doc(
                &mut store,
                &format!("/vault/photo{i}.png"),
                "image",
                &[chunk("image", "image", None, vec![0.6, 0.8])],
                Some(800),
                Some(600),
            );
        }

        store.rebuild_index(Space::Image, ScalarKind::F32).unwrap();

        let hits = store
            .search_space(Space::Image, &[1.0, 0.0], 10, &TextQuery::default())
            .unwrap();

        assert!(
            !hits.is_empty(),
            "image search returned nothing despite 3 indexed images; \
             title proxies starved the ANN candidate set"
        );
        assert!(
            hits.iter().all(|h| h.path.ends_with(".png")),
            "image-space search must return only real images, got {:?}",
            hits.iter().map(|h| &h.path).collect::<Vec<_>>()
        );
        assert_eq!(hits.len(), 3, "all three images should be found");

        let _ = std::fs::remove_dir_all(&path);
    }

    /// The ANN path and the brute-force path must agree. `ann_candidates`
    /// falls back to brute force only when the index file is missing, so a
    /// divergence between the two is invisible until someone happens to have
    /// an index on disk — which is how the P0 bug shipped.
    #[test]
    fn ann_and_brute_force_agree_on_image_search() {
        let (mut store, path) = temp_store();

        for i in 0..60 {
            write_full_doc(
                &mut store,
                &format!("/vault/note{i}.md"),
                "markdown",
                &[chunk("image", "text_title", Some("note"), vec![1.0, 0.0])],
                None,
                None,
            );
        }
        for (i, v) in [vec![0.6, 0.8], vec![0.8, 0.6], vec![0.0, 1.0]].iter().enumerate() {
            write_full_doc(
                &mut store,
                &format!("/vault/photo{i}.png"),
                "image",
                &[chunk("image", "image", None, v.clone())],
                Some(1),
                Some(1),
            );
        }

        // Brute force: no index files on disk yet.
        let brute: Vec<String> = store
            .search_space(Space::Image, &[1.0, 0.0], 5, &TextQuery::default())
            .unwrap()
            .into_iter()
            .map(|h| h.path)
            .collect();

        store.rebuild_index(Space::Image, ScalarKind::F32).unwrap();

        let ann: Vec<String> = store
            .search_space(Space::Image, &[1.0, 0.0], 5, &TextQuery::default())
            .unwrap()
            .into_iter()
            .map(|h| h.path)
            .collect();

        assert_eq!(brute, ann, "ANN and brute-force rankings diverged");
        assert!(!brute.is_empty(), "brute force found nothing to compare");

        let _ = std::fs::remove_dir_all(&path);
    }

    /// One ANN index file per `(space, modality)` pair, not per space — the
    /// structural fix for the starvation above. A pair with no rows writes no
    /// file, so queries for it fall back to brute force rather than reading a
    /// stale one.
    #[test]
    fn rebuild_index_writes_one_file_per_space_and_modality() {
        let (mut store, path) = temp_store();
        write_full_doc(
            &mut store,
            "/vault/note.md",
            "markdown",
            &[
                chunk("text", "text", Some("body"), vec![1.0, 0.0]),
                chunk("image", "text_title", Some("note"), vec![1.0, 0.0]),
            ],
            None,
            None,
        );
        write_full_doc(
            &mut store,
            "/vault/photo.png",
            "image",
            &[chunk("image", "image", None, vec![0.0, 1.0])],
            Some(1),
            Some(1),
        );

        store.rebuild_index(Space::Text, ScalarKind::F32).unwrap();
        store.rebuild_index(Space::Image, ScalarKind::F32).unwrap();

        let index_dir = path.join("index");
        let mut files: Vec<String> = std::fs::read_dir(&index_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();

        assert_eq!(
            files,
            vec![
                "image.image.usearch".to_string(),
                "image.text_title.usearch".to_string(),
                "text.text.usearch".to_string(),
            ],
            "expected one index file per (space, modality) pair"
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    // ---- lexical channel -------------------------------------------------

    fn write_text_doc_with(store: &mut SqliteStore, path: &str, texts: &[&str]) {
        let chunks: Vec<ChunkWrite> = texts
            .iter()
            .map(|t| chunk("text", "text", Some(t), vec![1.0, 0.0]))
            .collect();
        write_full_doc(store, path, "markdown", &chunks, None, None);
    }

    fn lex(store: &SqliteStore, query: &str) -> store::LexicalMatches {
        let terms = search::bm25::tokenize(query);
        store
            .lexical_candidates(&terms, 50, &TextQuery::default())
            .unwrap()
    }

    #[test]
    fn lexical_candidates_finds_a_matching_chunk() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["hnsw graph construction"]);
        write_text_doc_with(&mut store, "/vault/b.md", &["a recipe for chocolate cake"]);

        let got = lex(&store, "hnsw");
        assert_eq!(got.candidates.len(), 1);
        assert_eq!(got.candidates[0].path, "/vault/a.md");

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn lexical_candidates_matches_any_term_not_all() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["hnsw graph"]);
        write_text_doc_with(&mut store, "/vault/b.md", &["chocolate cake"]);

        let got = lex(&store, "hnsw chocolate");
        let mut paths: Vec<&str> = got.candidates.iter().map(|c| c.path.as_str()).collect();
        paths.sort();
        assert_eq!(paths, vec!["/vault/a.md", "/vault/b.md"]);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn lexical_candidates_reports_document_frequency_per_term() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["vector search"]);
        write_text_doc_with(&mut store, "/vault/b.md", &["vector indexes"]);
        write_text_doc_with(&mut store, "/vault/c.md", &["unrelated prose"]);

        let terms = search::bm25::tokenize("vector missingterm");
        let got = store
            .lexical_candidates(&terms, 50, &TextQuery::default())
            .unwrap();
        assert_eq!(got.doc_freqs, vec![2, 0], "df must align with the terms given");

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn lexical_corpus_stats_count_only_text_chunks() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["aaaa", "bb"]);
        // An image chunk (no text) and a title proxy must not be counted.
        write_full_doc(
            &mut store,
            "/vault/photo.png",
            "image",
            &[
                chunk("image", "image", None, vec![1.0, 0.0]),
                chunk("image", "text_title", Some("photo"), vec![1.0, 0.0]),
            ],
            Some(1),
            Some(1),
        );

        let got = lex(&store, "aaaa");
        assert_eq!(got.stats.total_chunks, 2, "only the two text chunks count");
        assert!(
            (got.stats.avg_chunk_len - 3.0).abs() < 1e-6,
            "avg of 4 and 2 characters should be 3, got {}",
            got.stats.avg_chunk_len
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn image_and_title_chunks_are_not_lexically_searchable() {
        let (mut store, path) = temp_store();
        write_full_doc(
            &mut store,
            "/vault/photo.png",
            "image",
            &[chunk("image", "text_title", Some("sunset"), vec![1.0, 0.0])],
            Some(1),
            Some(1),
        );

        let got = lex(&store, "sunset");
        assert!(
            got.candidates.is_empty(),
            "a title proxy exists for `related`, not for lexical search, got {:?}",
            got.candidates.iter().map(|c| &c.path).collect::<Vec<_>>()
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    /// Document frequency is read from `chunks_fts` alone, with no join back to
    /// `chunks`, so a stale FTS row inflates it even though the orphan would be
    /// filtered out of the candidate list. This is what the delete trigger is
    /// actually for: without it the candidates still look right while every
    /// IDF silently drifts.
    #[test]
    fn deleting_a_document_lowers_the_document_frequency_of_its_terms() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["sharedterm alpha"]);
        write_text_doc_with(&mut store, "/vault/b.md", &["sharedterm beta"]);
        assert_eq!(lex(&store, "sharedterm").doc_freqs, vec![2]);

        store.delete_document("/vault/a.md").unwrap();
        assert_eq!(
            lex(&store, "sharedterm").doc_freqs,
            vec![1],
            "a deleted chunk must stop counting toward document frequency"
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    /// Same hazard on the reindex path: replacing a document's chunks deletes
    /// the old rows, and their terms must stop counting.
    #[test]
    fn replacing_a_document_lowers_the_document_frequency_of_dropped_terms() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["originalterm here"]);
        assert_eq!(lex(&store, "originalterm").doc_freqs, vec![1]);

        write_text_doc_with(&mut store, "/vault/a.md", &["replacementterm here"]);
        assert_eq!(
            lex(&store, "originalterm").doc_freqs,
            vec![0],
            "the replaced chunk's terms must leave the index"
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    /// The FTS index is kept in sync by triggers, so a reindex that replaces a
    /// document's chunks must not leave the old text matchable.
    #[test]
    fn replacing_a_document_updates_the_lexical_index() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["originaltext here"]);
        assert_eq!(lex(&store, "originaltext").candidates.len(), 1);

        write_text_doc_with(&mut store, "/vault/a.md", &["replacementtext here"]);
        assert!(
            lex(&store, "originaltext").candidates.is_empty(),
            "stale text must not remain matchable after a reindex"
        );
        assert_eq!(lex(&store, "replacementtext").candidates.len(), 1);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn deleting_a_document_removes_it_from_the_lexical_index() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["deletabletext here"]);
        assert_eq!(lex(&store, "deletabletext").candidates.len(), 1);

        store.delete_document("/vault/a.md").unwrap();
        assert!(lex(&store, "deletabletext").candidates.is_empty());

        let _ = std::fs::remove_dir_all(&path);
    }

    /// A database indexed before the full-text index existed must get one built
    /// on open. The sync triggers only fire on new writes, so without this the
    /// lexical channel is silently dead — every query simply matches nothing.
    ///
    /// Asserted through search results, never through `COUNT(*) FROM
    /// chunks_fts`: this is an external-content table, so that count reports
    /// rows in `chunks` rather than index entries and is non-zero even when the
    /// index is completely empty.
    #[test]
    fn opening_a_store_builds_a_missing_lexical_index() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["cholesterol and statins"]);
        assert_eq!(lex(&store, "cholesterol").candidates.len(), 1);

        // Simulate a pre-FTS database: wipe the index and the marker that says
        // it was built, leaving the chunks in place.
        store
            .conn
            .execute_batch("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all');")
            .unwrap();
        store.set_meta("lexical_index_version", "0").unwrap();
        assert!(
            lex(&store, "cholesterol").candidates.is_empty(),
            "precondition: the index should now be empty"
        );
        drop(store);

        let reopened = SqliteStore::open(&path.join("gnosis.db")).unwrap();
        assert_eq!(
            lex(&reopened, "cholesterol").candidates.len(),
            1,
            "reopening must rebuild the missing lexical index"
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn lexical_candidates_respects_the_root_filter() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["sharedterm one"]);
        store
            .replace_document(&DocWrite {
                path: "/other/b.md",
                kind: "markdown",
                source_root: "/other",
                content_hash: b"h",
                mtime: 0,
                title: "b",
                frontmatter: None,
                indexed_at: 0,
                chunks: &[chunk("text", "text", Some("sharedterm two"), vec![1.0, 0.0])],
                links: &[],
                tags: &[],
                width: None,
                height: None,
            })
            .unwrap();

        let roots = vec!["/vault".to_string()];
        let terms = search::bm25::tokenize("sharedterm");
        let got = store
            .lexical_candidates(
                &terms,
                50,
                &TextQuery {
                    from: Some(&roots),
                    tags: None,
                },
            )
            .unwrap();
        assert_eq!(got.candidates.len(), 1);
        assert_eq!(got.candidates[0].path, "/vault/a.md");

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn an_empty_query_returns_nothing_without_erroring() {
        let (mut store, path) = temp_store();
        write_text_doc_with(&mut store, "/vault/a.md", &["some prose"]);
        let got = store
            .lexical_candidates(&[], 50, &TextQuery::default())
            .unwrap();
        assert!(got.candidates.is_empty());
        assert!(got.doc_freqs.is_empty());
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn a_query_against_an_empty_corpus_returns_nothing() {
        let (store, path) = temp_store();
        let got = lex(&store, "anything");
        assert_eq!(got.stats.total_chunks, 0);
        assert!(got.candidates.is_empty());
        let _ = std::fs::remove_dir_all(&path);
    }
}
