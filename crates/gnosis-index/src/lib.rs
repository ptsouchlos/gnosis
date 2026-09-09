//! Indexing pipeline orchestration: walk, parse, chunk, embed, and persist.
//! Generic over [`Store`]/[`Walker`]/[`FileReader`]/[`Embedder`] so it carries
//! no native dependencies of its own — native-ness comes only from whichever
//! concrete implementations the caller supplies.
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use embed::Embedder;
use fs::FileReader;
use progress::Progress;
use store::{ChunkWrite, DocWrite, Store};
use walk::{DocKind, Walker};

/// Outcome of an indexing run.
#[derive(Debug, Default)]
pub struct IndexReport {
    pub scanned: usize,
    pub indexed: usize,
    pub skipped: usize,
    pub deleted: usize,
    pub chunks: usize,
}

/// The trait-object backends an indexing run needs. Bundled so `run` doesn't
/// take four separate `dyn` parameters alongside its plain-data ones.
pub struct IndexerArgs<'a> {
    pub store: &'a mut dyn Store,
    pub walker: &'a dyn Walker,
    pub fs_reader: &'a dyn FileReader,
    pub embedder: &'a mut dyn Embedder,
    pub progress: &'a dyn Progress,
}

/// Walk each root in `roots`, (re)embed changed documents, and prune deleted
/// ones. Pruning is scoped to the source vaults walked in this run, so indexing
/// one vault never removes another's documents from a shared database.
pub fn run(
    args: &mut IndexerArgs,
    roots: &[PathBuf],
    ignore_globs: &[String],
    chunk_cfg: &chunker::ChunkConfig,
    force: bool,
) -> Result<IndexReport> {
    guard_model(&*args.store, &*args.embedder, force)?;
    args.store
        .set_meta("model.text", args.embedder.model_id())?;
    args.store
        .set_meta("dim.text", &args.embedder.dim().to_string())?;

    let mut report = IndexReport::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut walked_roots: Vec<String> = Vec::new();

    for root in roots {
        let root_canon = match args.fs_reader.canonicalize(root) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("warning: skipping vault {} ({e})", root.display());
                continue;
            }
        };
        let root_str = root_canon.to_string_lossy().to_string();
        walked_roots.push(root_str.clone());

        let found = args.walker.discover(&root_canon, ignore_globs)?;
        report.scanned += found.len();
        args.progress.inc_total(found.len() as u64);

        for file in &found {
            let path = args.fs_reader.canonicalize(&file.path)?;
            let path_str = path.to_string_lossy().to_string();

            // Dedupe across overlapping roots; first root wins.
            if !seen.insert(path_str.clone()) {
                args.progress.inc(1);
                continue;
            }

            let bytes = args.fs_reader.read(&path)?;
            let hash = blake3::hash(&bytes);

            if !force
                && let Some(existing) = args.store.document_hash(&path_str)?
                && existing.as_slice() == hash.as_bytes()
            {
                report.skipped += 1;
                args.progress.inc(1);
                continue;
            }

            let n = index_file(args, &path_str, &root_str, file.kind, &bytes, chunk_cfg)?;
            report.indexed += 1;
            report.chunks += n;
            args.progress.inc(1);
        }
    }
    // Prune only documents belonging to the roots walked in this run.
    for path in args.store.paths_for_roots(&walked_roots)? {
        if !seen.contains(&path) {
            args.store.delete_document(&path)?;
            report.deleted += 1;
        }
    }
    args.progress.finish();

    Ok(report)
}

/// Parse, chunk, embed, and persist a single markdown file. Returns chunk count.
fn index_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    kind: DocKind,
    bytes: &[u8],
    chunk_cfg: &chunker::ChunkConfig,
) -> Result<usize> {
    let content = String::from_utf8_lossy(bytes);
    let parsed = parse::parse_markdown(Path::new(path_str), &content);
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

    let vectors = args.embedder.embed(&texts)?;

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
    let mtime = args.fs_reader.mtime(Path::new(path_str));
    let indexed_at = now_unix();

    args.store.replace_document(&DocWrite {
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
        tags: &parsed.tags,
    })?;

    Ok(chunk_writes.len())
}

/// Refuse to mix vectors from a different model into an existing index.
fn guard_model(store: &dyn Store, embedder: &dyn Embedder, force: bool) -> Result<()> {
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

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
