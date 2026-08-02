use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

use crate::embed::{Embedder, TextEmbedder};
use crate::parse::parse_markdown;
use crate::store::{ChunkWrite, DocWrite, Store};
use crate::walk::{self, DocKind};
use crate::workspace::Workspace;
use chunker;

/// Outcome of an indexing run.
#[derive(Debug, Default)]
pub struct IndexReport {
    pub scanned: usize,
    pub indexed: usize,
    pub skipped: usize,
    pub deleted: usize,
    pub chunks: usize,
}

/// Walk each root in `roots`, (re)embed changed documents, and prune deleted
/// ones. Pruning is scoped to the source vaults walked in this run, so indexing
/// one vault never removes another's documents from a shared database.
pub fn run(ws: &Workspace, roots: &[PathBuf], force: bool) -> Result<IndexReport> {
    let cfg = &ws.config;
    let mut store = Store::open(&ws.db_path)?;

    let mut embedder = TextEmbedder::new(&cfg.embed.text.model)?;
    guard_model(&store, &embedder, force)?;
    store.set_meta("model.text", embedder.model_id())?;
    store.set_meta("dim.text", &embedder.dim().to_string())?;

    let mut report = IndexReport::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut walked_roots: Vec<String> = Vec::new();

    for root in roots {
        let root_canon = match std::fs::canonicalize(root) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("warning: skipping vault {} ({e})", root.display());
                continue;
            }
        };
        let root_str = root_canon.to_string_lossy().to_string();
        walked_roots.push(root_str.clone());

        let found = walk::discover(&root_canon, &cfg.ignore.globs)?;
        report.scanned += found.len();

        for file in &found {
            let path = std::fs::canonicalize(&file.path)
                .with_context(|| format!("resolving {}", file.path.display()))?;
            let path_str = path.to_string_lossy().to_string();

            // Dedupe across overlapping roots; first root wins.
            if !seen.insert(path_str.clone()) {
                continue;
            }

            let bytes = std::fs::read(&path).with_context(|| format!("reading {path_str}"))?;
            let hash = blake3::hash(&bytes);

            if !force
                && let Some(existing) = store.document_hash(&path_str)?
                && existing.as_slice() == hash.as_bytes()
            {
                report.skipped += 1;
                continue;
            }

            let n = index_file(
                &mut store,
                &mut embedder,
                &path_str,
                &root_str,
                file.kind,
                &bytes,
                &cfg.chunk,
            )?;
            report.indexed += 1;
            report.chunks += n;
        }
    }

    // Prune only documents belonging to the roots walked in this run.
    for path in store.paths_for_roots(&walked_roots)? {
        if !seen.contains(&path) {
            store.delete_document(&path)?;
            report.deleted += 1;
        }
    }

    Ok(report)
}

/// Parse, chunk, embed, and persist a single markdown file. Returns chunk count.
fn index_file(
    store: &mut Store,
    embedder: &mut TextEmbedder,
    path_str: &str,
    source_root: &str,
    kind: DocKind,
    bytes: &[u8],
    chunk_cfg: &crate::config::ChunkConfig,
) -> Result<usize> {
    let content = String::from_utf8_lossy(bytes);
    let parsed = parse_markdown(std::path::Path::new(path_str), &content);
    let chunks = chunker::chunk_markdown(&parsed.body, chunk_cfg.max_tokens, chunk_cfg.overlap);

    // Prepend the heading trail so chunks carry their structural context.
    let texts: Vec<String> = chunks
        .iter()
        .map(|c| {
            if c.heading_path.is_empty() {
                c.text.clone()
            } else {
                format!("{}\n{}", c.heading_path, c.text)
            }
        })
        .collect();

    let vectors = embedder.embed(&texts)?;

    let chunk_writes: Vec<ChunkWrite> = chunks
        .iter()
        .zip(vectors)
        .map(|(c, vector)| ChunkWrite {
            ord: c.ord,
            space: "text".to_string(),
            modality: "text".to_string(),
            text: Some(c.text.clone()),
            heading_path: c.heading_path.clone(),
            vector,
        })
        .collect();

    let hash = blake3::hash(bytes);
    let mtime = file_mtime(path_str);
    let indexed_at = now_unix();

    store.replace_document(&DocWrite {
        path: path_str,
        kind: kind.as_str(),
        source_root,
        content_hash: hash.as_bytes(),
        mtime,
        title: &parsed.title,
        frontmatter: parsed.frontmatter.as_deref(),
        indexed_at,
        chunks: &chunk_writes,
        links: &parsed.links,
    })?;

    Ok(chunk_writes.len())
}

/// Refuse to mix vectors from a different model into an existing index.
fn guard_model(store: &Store, embedder: &TextEmbedder, force: bool) -> Result<()> {
    if force {
        return Ok(());
    }
    if let Some(existing) = store.get_meta("model.text")?
        && existing != embedder.model_id()
    {
        bail!(
            "index was built with text model '{existing}' but config now specifies \
             '{}'; run `gnosis rebuild` to re-embed",
            embedder.model_id()
        );
    }
    Ok(())
}

fn file_mtime(path: &str) -> i64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
