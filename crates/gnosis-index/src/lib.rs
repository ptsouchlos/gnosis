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
    pub errors: Vec<IndexError>,
}

/// A file that failed to index. Currently only possible for images — a
/// corrupt or unsupported file fails to open/decode; markdown has no
/// equivalent failure mode (`chunk_markdown` is infallible over any UTF-8
/// text). Indexing continues past these by default; `fail_fast` makes `run`
/// return the error immediately instead of collecting it here.
#[derive(Debug)]
pub struct IndexError {
    pub path: String,
    pub message: String,
}

/// The embedders an indexing run needs, one per space. `image`/`image_text`
/// are `None` when `[embed.image] enabled = false` — a text-only vault
/// indexes exactly as it did before this field existed. `image`/`image_text`
/// are owned (not borrowed like `text`) so the caller can move freshly-built
/// `Box<dyn Embedder>`s straight in — borrowing from separate sibling
/// `Option<Box<dyn Embedder>>` locals here instead runs into a real rustc
/// dropck limitation (conservative drop-order analysis across sibling
/// trait-object-holding locals sharing a borrowing struct).
pub struct EmbedderSet<'a> {
    pub text: &'a mut dyn Embedder,
    pub image: Option<Box<dyn Embedder>>,
    pub image_text: Option<Box<dyn Embedder>>,
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
///
/// A file that fails to read or index is, by default, skipped and recorded
/// in the returned report's `errors` rather than aborting the whole run —
/// today this only happens for images (a corrupt or unsupported file fails
/// to open/decode). `fail_fast` makes `run` return that error immediately
/// instead. Markdown files always propagate their errors immediately
/// regardless of `fail_fast` — there's no equivalent soft failure mode to
/// skip past for them.
///
/// Images are not embedded inline as they're discovered: each one that needs
/// (re)indexing is staged into an in-memory buffer, which is flushed — one
/// `Embedder::embed` call across every buffered path — once it reaches
/// `image_batch_size`, and once more for the remainder at the end of the
/// walk. Batching a vision-model call over many files is far more efficient
/// than one call per file. If a batch's `embed` call fails, the batch is
/// retried one file at a time to isolate which one is actually bad, so a
/// single corrupt image still costs only itself, not its batch-mates.
pub fn run(
    args: &mut IndexerArgs,
    roots: &[PathBuf],
    ignore_globs: &[String],
    chunk_cfg: &chunker::ChunkConfig,
    force: bool,
    fail_fast: bool,
    image_batch_size: usize,
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
    let mut pending_images: Vec<PendingImage> = Vec::new();
    let mut pending_titles: Vec<PendingTitleDoc> = Vec::new();
    let image_batch_size = image_batch_size.max(1);
    let batching_titles = args.embedders.image_text.is_some();

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

            match file.kind {
                DocKind::Markdown if batching_titles => {
                    match stage_markdown_file(args, &path_str, &root_str, chunk_cfg, force) {
                        Ok(Some(pending)) => {
                            pending_titles.push(pending);
                            if pending_titles.len() >= image_batch_size {
                                flush_title_batch(args, &mut pending_titles, &mut report, fail_fast)?;
                            }
                        }
                        Ok(None) => report.skipped += 1,
                        Err(e) => return Err(e),
                    }
                }
                DocKind::Markdown => {
                    match process_markdown_file(args, &path_str, &root_str, chunk_cfg, force) {
                        Ok(Some(n)) => {
                            report.indexed += 1;
                            report.chunks += n;
                        }
                        Ok(None) => report.skipped += 1,
                        Err(e) => return Err(e),
                    }
                }
                DocKind::Image => match stage_image_file(args, &path_str, &root_str, force) {
                    Ok(Some(pending)) => {
                        pending_images.push(pending);
                        if pending_images.len() >= image_batch_size {
                            flush_image_batch(
                                args,
                                &mut pending_images,
                                &mut pending_titles,
                                image_batch_size,
                                &mut report,
                                fail_fast,
                            )?;
                        }
                    }
                    Ok(None) => report.skipped += 1,
                    Err(e) if !fail_fast => {
                        report.errors.push(IndexError {
                            path: path_str.clone(),
                            message: e.to_string(),
                        });
                    }
                    Err(e) => return Err(e),
                },
            }
            args.progress.inc(1);
        }
    }
    flush_image_batch(
        args,
        &mut pending_images,
        &mut pending_titles,
        image_batch_size,
        &mut report,
        fail_fast,
    )?;
    flush_title_batch(args, &mut pending_titles, &mut report, fail_fast)?;

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

/// Read, hash, skip-check, parse, chunk, embed, and persist one discovered
/// markdown file. Returns `Ok(Some(chunk_count))` if (re)indexed, `Ok(None)`
/// if skipped (unchanged since the last index), or `Err` if reading or
/// indexing failed — markdown has no soft-failure mode, so the caller always
/// propagates.
fn process_markdown_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    chunk_cfg: &chunker::ChunkConfig,
    force: bool,
) -> Result<Option<usize>> {
    let bytes = args.fs_reader.read(Path::new(path_str))?;
    let hash = blake3::hash(&bytes);

    if !force
        && let Some(existing) = args.store.document_hash(path_str)?
        && existing.as_slice() == hash.as_bytes()
    {
        return Ok(None);
    }

    let n = index_markdown_file(args, path_str, source_root, &bytes, chunk_cfg)?;
    Ok(Some(n))
}

/// Parse, chunk, embed, and persist a single markdown file. Only used when
/// no title-proxy chunk is needed (`image_text` disabled) — otherwise
/// `stage_markdown_file` defers the write so the title embeds in a batch.
fn index_markdown_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    bytes: &[u8],
    chunk_cfg: &chunker::ChunkConfig,
) -> Result<usize> {
    let (parsed, chunk_writes) = embed_markdown_chunks(args, path_str, bytes, chunk_cfg)?;
    let hash = blake3::hash(bytes);
    let mtime = args.fs_reader.mtime(Path::new(path_str));
    write_markdown_doc(
        args,
        path_str,
        source_root,
        hash.as_bytes(),
        mtime,
        &parsed.title,
        parsed.frontmatter.as_deref(),
        &parsed.links,
        &parsed.tags,
        chunk_writes,
    )
}

/// Parse, chunk markdown `bytes`, and text-embed the chunks (shared by the
/// immediate-write and staged/title-batched paths).
fn embed_markdown_chunks(
    args: &mut IndexerArgs,
    path_str: &str,
    bytes: &[u8],
    chunk_cfg: &chunker::ChunkConfig,
) -> Result<(parse::ParsedDoc, Vec<ChunkWrite>)> {
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

    Ok((parsed, chunk_writes))
}

/// Persist a markdown document from its already-embedded chunks. Updates
/// `report` counters are the caller's responsibility (the two call sites —
/// immediate and title-batch-flush — account differently).
#[allow(clippy::too_many_arguments)]
fn write_markdown_doc(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    content_hash: &[u8],
    mtime: i64,
    title: &str,
    frontmatter: Option<&str>,
    links: &[String],
    tags: &[String],
    chunk_writes: Vec<ChunkWrite>,
) -> Result<usize> {
    let indexed_at = now_unix();
    args.store.replace_document(&DocWrite {
        path: path_str,
        kind: DocKind::Markdown.as_str(),
        source_root,
        content_hash,
        mtime,
        title,
        frontmatter,
        indexed_at,
        chunks: &chunk_writes,
        links,
        tags,
        width: None,
        height: None,
    })?;
    Ok(chunk_writes.len())
}

/// One image file that has been read/hashed/skip-checked and is waiting for
/// its vision embedding — everything `flush_image_batch` needs to finish
/// indexing it once a vector arrives, without re-reading the file.
struct PendingImage {
    path: String,
    source_root: String,
    title: String,
    content_hash: Vec<u8>,
    mtime: i64,
    width: Option<i64>,
    height: Option<i64>,
}

/// Read, hash, and skip-check a single discovered image file. Returns
/// `Ok(Some(pending))` if it needs (re)indexing, `Ok(None)` if skipped
/// (unchanged since the last index), or `Err` if reading failed — the
/// caller decides whether that's fatal or worth skipping past. Does not
/// embed — that happens later, batched, in `flush_image_batch`.
fn stage_image_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    force: bool,
) -> Result<Option<PendingImage>> {
    let bytes = args.fs_reader.read(Path::new(path_str))?;
    let hash = blake3::hash(&bytes);

    if !force
        && let Some(existing) = args.store.document_hash(path_str)?
        && existing.as_slice() == hash.as_bytes()
    {
        return Ok(None);
    }

    let title = Path::new(path_str)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path_str)
        .to_string();
    let (width, height) = args
        .fs_reader
        .image_dimensions(Path::new(path_str))
        .map(|(w, h)| (Some(w as i64), Some(h as i64)))
        .unwrap_or((None, None));

    Ok(Some(PendingImage {
        path: path_str.to_string(),
        source_root: source_root.to_string(),
        title,
        content_hash: hash.as_bytes().to_vec(),
        mtime: args.fs_reader.mtime(Path::new(path_str)),
        width,
        height,
    }))
}

/// Read, hash, skip-check, parse, chunk, and text-embed one discovered
/// markdown file, but defer its title-proxy embed and its write — those
/// happen later, batched, in `flush_title_batch`. Returns `Ok(Some(pending))`
/// if it needs (re)indexing, `Ok(None)` if skipped, or `Err` if reading or
/// text-embedding failed — markdown has no soft-failure mode, so the caller
/// always propagates.
fn stage_markdown_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    chunk_cfg: &chunker::ChunkConfig,
    force: bool,
) -> Result<Option<PendingTitleDoc>> {
    let bytes = args.fs_reader.read(Path::new(path_str))?;
    let hash = blake3::hash(&bytes);

    if !force
        && let Some(existing) = args.store.document_hash(path_str)?
        && existing.as_slice() == hash.as_bytes()
    {
        return Ok(None);
    }

    let (parsed, chunk_writes) = embed_markdown_chunks(args, path_str, &bytes, chunk_cfg)?;
    let mtime = args.fs_reader.mtime(Path::new(path_str));

    Ok(Some(PendingTitleDoc::Markdown {
        path: path_str.to_string(),
        source_root: source_root.to_string(),
        content_hash: hash.as_bytes().to_vec(),
        mtime,
        title: parsed.title,
        frontmatter: parsed.frontmatter,
        links: parsed.links,
        tags: parsed.tags,
        chunk_writes,
    }))
}

/// Embed every buffered `PendingImage` in one `Embedder::embed` call and
/// finish each one — written immediately when no title-proxy is needed, or
/// staged into `pending_titles` (flushed here too, at `image_batch_size`)
/// when one is. On a batch failure, retries the batch one file at a time to
/// isolate which one is bad — a single corrupt image then costs only itself
/// (recorded in `report.errors`, or propagated immediately under
/// `fail_fast`), not its batch-mates. No-ops on an empty buffer, so calling
/// this at both the per-batch threshold and the end of the walk is always
/// safe. Always drains `pending`, even on error paths, so a caller never
/// re-flushes the same items twice.
#[allow(clippy::too_many_arguments)]
fn flush_image_batch(
    args: &mut IndexerArgs,
    pending: &mut Vec<PendingImage>,
    pending_titles: &mut Vec<PendingTitleDoc>,
    image_batch_size: usize,
    report: &mut IndexReport,
    fail_fast: bool,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(pending);
    let paths: Vec<String> = batch.iter().map(|p| p.path.clone()).collect();

    let image_embedder = args
        .embedders
        .image
        .as_mut()
        .context("an image file was discovered but [embed.image].enabled is false")?;

    match image_embedder.embed(&paths) {
        Ok(vectors) => {
            for (item, vector) in batch.into_iter().zip(vectors) {
                finish_image_doc(args, item, vector, pending_titles, report)?;
                if pending_titles.len() >= image_batch_size {
                    flush_title_batch(args, pending_titles, report, fail_fast)?;
                }
            }
            Ok(())
        }
        Err(_) => {
            for item in batch {
                let image_embedder = args
                    .embedders
                    .image
                    .as_mut()
                    .context("an image file was discovered but [embed.image].enabled is false")?;
                match image_embedder.embed(std::slice::from_ref(&item.path)) {
                    Ok(vectors) => {
                        let vector = vectors
                            .into_iter()
                            .next()
                            .context("image embedding produced no vector")?;
                        finish_image_doc(args, item, vector, pending_titles, report)?;
                        if pending_titles.len() >= image_batch_size {
                            flush_title_batch(args, pending_titles, report, fail_fast)?;
                        }
                    }
                    Err(e) if !fail_fast => {
                        report.errors.push(IndexError {
                            path: item.path.clone(),
                            message: e.to_string(),
                        });
                    }
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }
}

/// An embedded image chunk, one image ahead of being written: either
/// written immediately (no title-proxy configured) or staged for
/// `flush_title_batch` to finish once image mode wants a title-proxy chunk
/// alongside it too.
fn finish_image_doc(
    args: &mut IndexerArgs,
    item: PendingImage,
    vector: Vec<f32>,
    pending_titles: &mut Vec<PendingTitleDoc>,
    report: &mut IndexReport,
) -> Result<()> {
    let image_chunk = ChunkWrite {
        ord: 0,
        space: "image".to_string(),
        modality: "image".to_string(),
        text: None,
        heading_path: String::new(),
        vector,
    };

    if args.embedders.image_text.is_some() {
        pending_titles.push(PendingTitleDoc::Image { item, image_chunk });
        return Ok(());
    }

    write_image_doc(args, item, vec![image_chunk], report)
}

/// Persist an image document from its already-embedded chunk(s). Updates
/// `report.indexed`/`chunks`.
fn write_image_doc(
    args: &mut IndexerArgs,
    item: PendingImage,
    chunk_writes: Vec<ChunkWrite>,
    report: &mut IndexReport,
) -> Result<()> {
    let indexed_at = now_unix();

    args.store.replace_document(&DocWrite {
        path: &item.path,
        kind: DocKind::Image.as_str(),
        source_root: &item.source_root,
        content_hash: &item.content_hash,
        mtime: item.mtime,
        title: &item.title,
        frontmatter: None,
        indexed_at,
        chunks: &chunk_writes,
        links: &[],
        tags: &[],
        width: item.width,
        height: item.height,
    })?;

    report.indexed += 1;
    report.chunks += chunk_writes.len();
    Ok(())
}

/// One document (markdown or image) whose non-title chunk(s) are already
/// embedded and is now waiting only for its title-proxy vector —
/// everything `flush_title_batch` needs to finish writing it once that
/// vector arrives. The mechanism that lets `related` traverse from an
/// image to a text document (and vice versa) without ever comparing
/// incompatible vector spaces directly. See `docs/gnosis/image-support.md`.
enum PendingTitleDoc {
    Markdown {
        path: String,
        source_root: String,
        content_hash: Vec<u8>,
        mtime: i64,
        title: String,
        frontmatter: Option<String>,
        links: Vec<String>,
        tags: Vec<String>,
        chunk_writes: Vec<ChunkWrite>,
    },
    Image {
        item: PendingImage,
        image_chunk: ChunkWrite,
    },
}

impl PendingTitleDoc {
    fn title(&self) -> &str {
        match self {
            PendingTitleDoc::Markdown { title, .. } => title,
            PendingTitleDoc::Image { item, .. } => &item.title,
        }
    }

    fn path(&self) -> &str {
        match self {
            PendingTitleDoc::Markdown { path, .. } => path,
            PendingTitleDoc::Image { item, .. } => &item.path,
        }
    }

    /// Whether a title-embed failure for this document must propagate
    /// immediately regardless of `fail_fast` — matches the existing
    /// per-kind error-isolation contract (markdown always propagates,
    /// image soft-fails).
    fn must_propagate(&self) -> bool {
        matches!(self, PendingTitleDoc::Markdown { .. })
    }
}

/// Embed every buffered `PendingTitleDoc`'s title in one `Embedder::embed`
/// call (via the CLIP text encoder) and persist each as its own document.
/// On a batch failure, retries the batch one file at a time to isolate
/// which one is bad, same as `flush_image_batch` — except a markdown-origin
/// failure always propagates (per `must_propagate`), matching markdown's
/// existing no-soft-failure contract. No-ops on an empty buffer.
fn flush_title_batch(
    args: &mut IndexerArgs,
    pending: &mut Vec<PendingTitleDoc>,
    report: &mut IndexReport,
    fail_fast: bool,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let batch = std::mem::take(pending);
    let titles: Vec<String> = batch.iter().map(|p| p.title().to_string()).collect();

    let image_text = args
        .embedders
        .image_text
        .as_mut()
        .context("a title-proxy chunk was staged but no image-text embedder is configured")?;

    match image_text.embed(&titles) {
        Ok(vectors) => {
            for (item, vector) in batch.into_iter().zip(vectors) {
                write_title_doc(args, item, vector, report)?;
            }
            Ok(())
        }
        Err(_) => {
            for item in batch {
                let path = item.path().to_string();
                let must_propagate = item.must_propagate();
                let title = item.title().to_string();
                let image_text = args.embedders.image_text.as_mut().context(
                    "a title-proxy chunk was staged but no image-text embedder is configured",
                )?;
                match image_text.embed(&[title]) {
                    Ok(vectors) => {
                        let vector = vectors
                            .into_iter()
                            .next()
                            .context("title embedding produced no vector")?;
                        write_title_doc(args, item, vector, report)?;
                    }
                    Err(e) if !must_propagate && !fail_fast => {
                        report.errors.push(IndexError { path, message: e.to_string() });
                    }
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }
}

/// Build the title-proxy chunk from its embedded `vector` and persist the
/// document (markdown or image) it belongs to.
fn write_title_doc(
    args: &mut IndexerArgs,
    item: PendingTitleDoc,
    vector: Vec<f32>,
    report: &mut IndexReport,
) -> Result<()> {
    match item {
        PendingTitleDoc::Markdown {
            path,
            source_root,
            content_hash,
            mtime,
            title,
            frontmatter,
            links,
            tags,
            mut chunk_writes,
        } => {
            let ord = chunk_writes.len();
            chunk_writes.push(ChunkWrite {
                ord,
                space: "image".to_string(),
                modality: "text_title".to_string(),
                text: Some(title.clone()),
                heading_path: String::new(),
                vector,
            });
            let n = write_markdown_doc(
                args,
                &path,
                &source_root,
                &content_hash,
                mtime,
                &title,
                frontmatter.as_deref(),
                &links,
                &tags,
                chunk_writes,
            )?;
            report.indexed += 1;
            report.chunks += n;
        }
        PendingTitleDoc::Image { item, image_chunk } => {
            let title_chunk = ChunkWrite {
                ord: 1,
                space: "image".to_string(),
                modality: "text_title".to_string(),
                text: Some(item.title.clone()),
                heading_path: String::new(),
                vector,
            };
            write_image_doc(args, item, vec![image_chunk, title_chunk], report)?;
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use store::{Space, Stats, TextQuery};
    use walk::Found;

    /// In-memory `Store` fake — just enough of the trait for `run` to
    /// exercise (`document_hash`/`replace_document`/`get_meta`/`set_meta`/
    /// `paths_for_roots`). Everything else is unreachable from `run` and
    /// stubbed trivially.
    #[derive(Default)]
    struct FakeStore {
        written: Vec<String>,
    }

    impl Store for FakeStore {
        fn set_meta(&self, _key: &str, _value: &str) -> Result<()> {
            Ok(())
        }
        fn get_meta(&self, _key: &str) -> Result<Option<String>> {
            Ok(None)
        }
        fn stats(&self) -> Result<Stats> {
            Ok(Stats::default())
        }
        fn document_hash(&self, _path: &str) -> Result<Option<Vec<u8>>> {
            Ok(None)
        }
        fn paths_for_roots(&self, _roots: &[String]) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn counts_by_root(&self) -> Result<Vec<(String, i64)>> {
            Ok(Vec::new())
        }
        fn delete_by_root(&self, _root: &str) -> Result<usize> {
            Ok(0)
        }
        fn replace_document(&mut self, doc: &DocWrite<'_>) -> Result<()> {
            self.written.push(doc.path.to_string());
            Ok(())
        }
        fn delete_document(&self, _path: &str) -> Result<()> {
            Ok(())
        }
        fn search_space(&self, _: Space, _: &[f32], _: usize, _: &TextQuery) -> Result<Vec<search::Hit>> {
            Ok(Vec::new())
        }
        fn chunk_vectors(&self, _: &str, _: Space) -> Result<Vec<Vec<f32>>> {
            Ok(Vec::new())
        }
        fn linked_targets(&self, _: &str) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn all_document_meta(&self) -> Result<Vec<(String, Option<String>)>> {
            Ok(Vec::new())
        }
        fn related_space(
            &self,
            _: Space,
            _: &[Vec<f32>],
            _: &[String],
            _: usize,
            _: &TextQuery,
        ) -> Result<Vec<search::Hit>> {
            Ok(Vec::new())
        }
    }

    struct FakeWalker {
        files: Vec<(PathBuf, DocKind)>,
    }

    impl Walker for FakeWalker {
        fn discover(&self, _root: &Path, _ignore_globs: &[String]) -> Result<Vec<Found>> {
            Ok(self
                .files
                .iter()
                .map(|(path, kind)| Found { path: path.clone(), kind: *kind })
                .collect())
        }
    }

    struct FakeFileReader;

    impl FileReader for FakeFileReader {
        fn canonicalize(&self, path: &Path) -> Result<PathBuf> {
            Ok(path.to_path_buf())
        }
        fn read(&self, _path: &Path) -> Result<Vec<u8>> {
            Ok(b"# Heading\n\nbody text".to_vec())
        }
        fn mtime(&self, _path: &Path) -> i64 {
            0
        }
        fn image_dimensions(&self, _path: &Path) -> Option<(u32, u32)> {
            None
        }
    }

    struct FakeTextEmbedder;

    impl Embedder for FakeTextEmbedder {
        fn space(&self) -> &str {
            "text"
        }
        fn dim(&self) -> usize {
            2
        }
        fn model_id(&self) -> &str {
            "fake-text"
        }
        fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(inputs.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    /// Simulates a corrupt/unsupported image: always fails to "decode".
    struct FailingImageEmbedder;

    impl Embedder for FailingImageEmbedder {
        fn space(&self) -> &str {
            "image"
        }
        fn dim(&self) -> usize {
            2
        }
        fn model_id(&self) -> &str {
            "fake-image"
        }
        fn embed(&mut self, _inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            anyhow::bail!("simulated decode failure")
        }
    }

    /// Records every `embed()` call's inputs, so tests can assert on batching
    /// (how many calls, how many inputs per call) instead of just outputs.
    /// Shares its call log via `Rc<RefCell<_>>` so the test can inspect it
    /// after `run` has consumed the boxed embedder.
    #[derive(Clone, Default)]
    struct RecordingEmbedder {
        calls: std::rc::Rc<std::cell::RefCell<Vec<Vec<String>>>>,
    }

    impl Embedder for RecordingEmbedder {
        fn space(&self) -> &str {
            "image"
        }
        fn dim(&self) -> usize {
            2
        }
        fn model_id(&self) -> &str {
            "fake-image"
        }
        fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.borrow_mut().push(inputs.to_vec());
            Ok(inputs.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    /// One markdown, one (failing) image, one more markdown — proves
    /// processing continues past the failure to the next file.
    fn three_file_walker() -> FakeWalker {
        FakeWalker {
            files: vec![
                (PathBuf::from("/vault/a.md"), DocKind::Markdown),
                (PathBuf::from("/vault/bad.png"), DocKind::Image),
                (PathBuf::from("/vault/b.md"), DocKind::Markdown),
            ],
        }
    }

    #[test]
    fn run_skips_failing_images_by_default_and_reports_them() {
        let mut store = FakeStore::default();
        let walker = three_file_walker();
        let fs_reader = FakeFileReader;
        let mut text_embedder = FakeTextEmbedder;
        let progress = progress::NoopProgress;

        let mut args = IndexerArgs {
            store: &mut store,
            walker: &walker,
            fs_reader: &fs_reader,
            embedders: EmbedderSet {
                text: &mut text_embedder,
                image: Some(Box::new(FailingImageEmbedder)),
                image_text: None,
            },
            progress: &progress,
        };

        let report = run(
            &mut args,
            &[PathBuf::from("/vault")],
            &[],
            &chunker::ChunkConfig::default(),
            false,
            false,
            32,
        )
        .expect("a failing image must not abort the run by default");

        assert_eq!(report.scanned, 3);
        assert_eq!(report.indexed, 2, "both markdown files indexed despite the image failing");
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].path, "/vault/bad.png");
        assert!(report.errors[0].message.contains("simulated decode failure"));
    }

    #[test]
    fn run_fail_fast_aborts_immediately_on_first_error() {
        let mut store = FakeStore::default();
        let walker = three_file_walker();
        let fs_reader = FakeFileReader;
        let mut text_embedder = FakeTextEmbedder;
        let progress = progress::NoopProgress;

        let mut args = IndexerArgs {
            store: &mut store,
            walker: &walker,
            fs_reader: &fs_reader,
            embedders: EmbedderSet {
                text: &mut text_embedder,
                image: Some(Box::new(FailingImageEmbedder)),
                image_text: None,
            },
            progress: &progress,
        };

        let result = run(
            &mut args,
            &[PathBuf::from("/vault")],
            &[],
            &chunker::ChunkConfig::default(),
            false,
            true,
            32,
        );

        assert!(result.is_err(), "fail_fast must propagate the image error instead of collecting it");
    }

    #[test]
    fn run_batches_image_embeds_into_one_call() {
        let mut store = FakeStore::default();
        let walker = FakeWalker {
            files: vec![
                (PathBuf::from("/vault/a.png"), DocKind::Image),
                (PathBuf::from("/vault/b.png"), DocKind::Image),
                (PathBuf::from("/vault/c.png"), DocKind::Image),
            ],
        };
        let fs_reader = FakeFileReader;
        let mut text_embedder = FakeTextEmbedder;
        let image_embedder = RecordingEmbedder::default();
        let progress = progress::NoopProgress;

        let mut args = IndexerArgs {
            store: &mut store,
            walker: &walker,
            fs_reader: &fs_reader,
            embedders: EmbedderSet {
                text: &mut text_embedder,
                image: Some(Box::new(image_embedder.clone())),
                image_text: None,
            },
            progress: &progress,
        };

        let report = run(
            &mut args,
            &[PathBuf::from("/vault")],
            &[],
            &chunker::ChunkConfig::default(),
            false,
            false,
            32,
        )
        .expect("all three images embed successfully");

        assert_eq!(report.indexed, 3);
        let calls = image_embedder.calls.borrow();
        assert_eq!(calls.len(), 1, "three images under the default batch size must embed in a single call");
        assert_eq!(calls[0].len(), 3);
    }

    #[test]
    fn run_flushes_when_batch_size_reached_and_again_at_end() {
        let mut store = FakeStore::default();
        let walker = FakeWalker {
            files: vec![
                (PathBuf::from("/vault/a.png"), DocKind::Image),
                (PathBuf::from("/vault/b.png"), DocKind::Image),
                (PathBuf::from("/vault/c.png"), DocKind::Image),
            ],
        };
        let fs_reader = FakeFileReader;
        let mut text_embedder = FakeTextEmbedder;
        let image_embedder = RecordingEmbedder::default();
        let progress = progress::NoopProgress;

        let mut args = IndexerArgs {
            store: &mut store,
            walker: &walker,
            fs_reader: &fs_reader,
            embedders: EmbedderSet {
                text: &mut text_embedder,
                image: Some(Box::new(image_embedder.clone())),
                image_text: None,
            },
            progress: &progress,
        };

        let report = run(
            &mut args,
            &[PathBuf::from("/vault")],
            &[],
            &chunker::ChunkConfig::default(),
            false,
            false,
            2,
        )
        .expect("all three images embed successfully");

        assert_eq!(report.indexed, 3);
        let calls = image_embedder.calls.borrow();
        assert_eq!(calls.len(), 2, "batch_size=2 over 3 images must flush once at the threshold and once at the end");
        assert_eq!(calls[0].len(), 2);
        assert_eq!(calls[1].len(), 1);
    }

    /// Fails the batch call (2+ inputs) but succeeds on a single-item retry
    /// call, unless that single item is `bad.png`.
    struct PartiallyFailingImageEmbedder;

    impl Embedder for PartiallyFailingImageEmbedder {
        fn space(&self) -> &str {
            "image"
        }
        fn dim(&self) -> usize {
            2
        }
        fn model_id(&self) -> &str {
            "fake-image"
        }
        fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            if inputs.len() > 1 {
                anyhow::bail!("simulated batch failure");
            }
            if inputs[0].contains("bad.png") {
                anyhow::bail!("simulated decode failure");
            }
            Ok(inputs.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    #[test]
    fn run_retries_batch_one_by_one_on_batch_failure_and_isolates_the_bad_file() {
        let mut store = FakeStore::default();
        let walker = FakeWalker {
            files: vec![
                (PathBuf::from("/vault/a.png"), DocKind::Image),
                (PathBuf::from("/vault/bad.png"), DocKind::Image),
                (PathBuf::from("/vault/c.png"), DocKind::Image),
            ],
        };
        let fs_reader = FakeFileReader;
        let mut text_embedder = FakeTextEmbedder;
        let progress = progress::NoopProgress;

        let mut args = IndexerArgs {
            store: &mut store,
            walker: &walker,
            fs_reader: &fs_reader,
            embedders: EmbedderSet {
                text: &mut text_embedder,
                image: Some(Box::new(PartiallyFailingImageEmbedder)),
                image_text: None,
            },
            progress: &progress,
        };

        let report = run(
            &mut args,
            &[PathBuf::from("/vault")],
            &[],
            &chunker::ChunkConfig::default(),
            false,
            false,
            32,
        )
        .expect("a single bad file in a batch must not abort the run by default");

        assert_eq!(report.indexed, 2, "a.png and c.png survive the one-by-one retry");
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].path, "/vault/bad.png");
    }

    #[test]
    fn run_batches_title_proxy_embeds_across_markdown_and_image_files() {
        let mut store = FakeStore::default();
        let walker = FakeWalker {
            files: vec![
                (PathBuf::from("/vault/a.md"), DocKind::Markdown),
                (PathBuf::from("/vault/b.md"), DocKind::Markdown),
                (PathBuf::from("/vault/c.png"), DocKind::Image),
            ],
        };
        let fs_reader = FakeFileReader;
        let mut text_embedder = FakeTextEmbedder;
        let image_embedder = RecordingEmbedder::default();
        let title_embedder = RecordingEmbedder::default();
        let progress = progress::NoopProgress;

        let mut args = IndexerArgs {
            store: &mut store,
            walker: &walker,
            fs_reader: &fs_reader,
            embedders: EmbedderSet {
                text: &mut text_embedder,
                image: Some(Box::new(image_embedder.clone())),
                image_text: Some(Box::new(title_embedder.clone())),
            },
            progress: &progress,
        };

        let report = run(
            &mut args,
            &[PathBuf::from("/vault")],
            &[],
            &chunker::ChunkConfig::default(),
            false,
            false,
            32,
        )
        .expect("two markdown files and one image all index successfully");

        assert_eq!(report.indexed, 3);
        let calls = title_embedder.calls.borrow();
        assert_eq!(
            calls.len(),
            1,
            "titles from both markdown files and the image must embed in a single batched call"
        );
        assert_eq!(calls[0].len(), 3);
    }
}
