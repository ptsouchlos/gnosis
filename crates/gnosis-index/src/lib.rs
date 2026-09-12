//! Indexing pipeline orchestration: walk, parse, chunk, embed, and persist.
//! Generic over [`Store`]/[`Walker`]/[`FileReader`]/[`Embedder`] so it carries
//! no native dependencies of its own — native-ness comes only from whichever
//! concrete implementations the caller supplies.
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
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

/// The embedders an indexing run needs, one per space. `image`/`image_text`
/// are `None` when `[embed.image] enabled = false` — a text-only vault
/// indexes exactly as it did before this field existed.
pub struct EmbedderSet<'a> {
    pub text: &'a mut dyn Embedder,
    pub image: Option<&'a mut dyn Embedder>,
    pub image_text: Option<&'a mut dyn Embedder>,
}

/// The trait-object backends an indexing run needs. Bundled so `run` doesn't
/// take four separate `dyn` parameters alongside its plain-data ones.
pub struct IndexerArgs<'a> {
    pub store: &'a mut dyn Store,
    pub walker: &'a dyn Walker,
    pub fs_reader: &'a dyn FileReader,
    pub embedders: EmbedderSet<'a>,
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
    guard_model(&*args.store, &args.embedders, force)?;
    args.store
        .set_meta("model.text", args.embedders.text.model_id())?;
    args.store
        .set_meta("dim.text", &args.embedders.text.dim().to_string())?;
    if let Some(image) = &args.embedders.image {
        args.store.set_meta("model.image_vision", image.model_id())?;
        args.store.set_meta("dim.image", &image.dim().to_string())?;
    }
    if let Some(image_text) = &args.embedders.image_text {
        args.store
            .set_meta("model.image_text", image_text.model_id())?;
    }

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

/// Parse/embed/persist a single discovered file, dispatching on its kind.
/// Returns chunk count.
fn index_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    kind: DocKind,
    bytes: &[u8],
    chunk_cfg: &chunker::ChunkConfig,
) -> Result<usize> {
    match kind {
        DocKind::Markdown => index_markdown_file(args, path_str, source_root, bytes, chunk_cfg),
        DocKind::Image => index_image_file(args, path_str, source_root, bytes),
    }
}

/// Parse, chunk, embed, and persist a single markdown file.
fn index_markdown_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
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

    let vectors = args.embedders.text.embed(&texts)?;

    let mut chunk_writes: Vec<ChunkWrite> = chunks
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

    if let Some(image_text) = &mut args.embedders.image_text {
        chunk_writes.push(title_proxy_chunk(image_text, chunk_writes.len(), &parsed.title)?);
    }

    let hash = blake3::hash(bytes);
    let mtime = args.fs_reader.mtime(Path::new(path_str));
    let indexed_at = now_unix();

    args.store.replace_document(&DocWrite {
        path: path_str,
        kind: DocKind::Markdown.as_str(),
        source_root,
        content_hash: hash.as_bytes(),
        mtime,
        title: &parsed.title,
        frontmatter: parsed.frontmatter.as_deref(),
        indexed_at,
        chunks: &chunk_writes,
        links: &parsed.links,
        tags: &parsed.tags,
        width: None,
        height: None,
    })?;

    Ok(chunk_writes.len())
}

/// Embed and persist a single image file: one `image`-space chunk from its
/// pixel content, plus (when image mode is on) a title-proxy chunk from its
/// filename. No parsing/chunking — images have no frontmatter, links, or
/// tags to extract.
fn index_image_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    bytes: &[u8],
) -> Result<usize> {
    let title = Path::new(path_str)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path_str)
        .to_string();

    let image_embedder = args
        .embedders
        .image
        .as_mut()
        .context("an image file was discovered but [embed.image].enabled is false")?;
    let vector = image_embedder
        .embed(&[path_str.to_string()])?
        .into_iter()
        .next()
        .context("image embedding produced no vector")?;

    let mut chunk_writes = vec![ChunkWrite {
        ord: 0,
        space: "image".to_string(),
        modality: "image".to_string(),
        text: None,
        heading_path: String::new(),
        vector,
    }];

    if let Some(image_text) = &mut args.embedders.image_text {
        chunk_writes.push(title_proxy_chunk(image_text, 1, &title)?);
    }

    let (width, height) = args
        .fs_reader
        .image_dimensions(Path::new(path_str))
        .map(|(w, h)| (Some(w as i64), Some(h as i64)))
        .unwrap_or((None, None));

    let hash = blake3::hash(bytes);
    let mtime = args.fs_reader.mtime(Path::new(path_str));
    let indexed_at = now_unix();

    args.store.replace_document(&DocWrite {
        path: path_str,
        kind: DocKind::Image.as_str(),
        source_root,
        content_hash: hash.as_bytes(),
        mtime,
        title: &title,
        frontmatter: None,
        indexed_at,
        chunks: &chunk_writes,
        links: &[],
        tags: &[],
        width,
        height,
    })?;

    Ok(chunk_writes.len())
}

/// Embed `title` via the CLIP text encoder into a title-proxy chunk — the
/// mechanism that lets `related` traverse from an image to a text document
/// (and vice versa) without ever comparing incompatible vector spaces
/// directly. See `docs/gnosis/image-support.md`.
fn title_proxy_chunk(image_text: &mut &mut dyn Embedder, ord: usize, title: &str) -> Result<ChunkWrite> {
    let vector = image_text
        .embed(&[title.to_string()])?
        .into_iter()
        .next()
        .context("title embedding produced no vector")?;
    Ok(ChunkWrite {
        ord,
        space: "image".to_string(),
        modality: "text_title".to_string(),
        text: Some(title.to_string()),
        heading_path: String::new(),
        vector,
    })
}

/// Refuse to mix vectors from a different model into an existing index, for
/// any embedder currently in use.
fn guard_model(store: &dyn Store, embedders: &EmbedderSet, force: bool) -> Result<()> {
    if force {
        return Ok(());
    }
    guard_one(store, "model.text", embedders.text.model_id())?;
    if let Some(image) = &embedders.image {
        guard_one(store, "model.image_vision", image.model_id())?;
    }
    if let Some(image_text) = &embedders.image_text {
        guard_one(store, "model.image_text", image_text.model_id())?;
    }
    Ok(())
}

fn guard_one(store: &dyn Store, key: &str, model_id: &str) -> Result<()> {
    if let Some(existing) = store.get_meta(key)?
        && existing != model_id
    {
        bail!(
            "index was built with model '{existing}' ({key}) but config now specifies \
             '{model_id}'; run `gnosis rebuild` to re-embed"
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
