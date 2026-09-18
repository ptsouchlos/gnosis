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
pub fn run(
    args: &mut IndexerArgs,
    roots: &[PathBuf],
    ignore_globs: &[String],
    chunk_cfg: &chunker::ChunkConfig,
    force: bool,
    fail_fast: bool,
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

            match process_file(args, &path_str, &root_str, file.kind, chunk_cfg, force) {
                Ok(Some(n)) => {
                    report.indexed += 1;
                    report.chunks += n;
                }
                Ok(None) => report.skipped += 1,
                Err(e) if file.kind == DocKind::Image && !fail_fast => {
                    report.errors.push(IndexError {
                        path: path_str.clone(),
                        message: e.to_string(),
                    });
                }
                Err(e) => return Err(e),
            }
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

/// Read, hash, skip-check, and (if needed) index one discovered file.
/// Returns `Ok(Some(chunk_count))` if (re)indexed, `Ok(None)` if skipped
/// (unchanged since the last index), or `Err` if reading or indexing
/// failed — the caller decides whether that's fatal or worth skipping past.
fn process_file(
    args: &mut IndexerArgs,
    path_str: &str,
    source_root: &str,
    kind: DocKind,
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

    let n = index_file(args, path_str, source_root, kind, &bytes, chunk_cfg)?;
    Ok(Some(n))
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
        chunk_writes.push(title_proxy_chunk(image_text.as_mut(), chunk_writes.len(), &parsed.title)?);
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
        chunk_writes.push(title_proxy_chunk(image_text.as_mut(), 1, &title)?);
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
fn title_proxy_chunk(image_text: &mut dyn Embedder, ord: usize, title: &str) -> Result<ChunkWrite> {
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
        );

        assert!(result.is_err(), "fail_fast must propagate the image error instead of collecting it");
    }
}
