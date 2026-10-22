use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use store::artifact::{
    ARTIFACT_FORMAT_VERSION, ArtifactChunk, Compression, DocumentEntry, Manifest, ModelInfo,
    Quantization, ShardInfo, SpaceInfo, SpaceSpec, assign_shard, shard_hash, shard_name,
    write_shard,
};
use store::{Space, Store};

use crate::embedder::resolve_pooling;
use crate::store::SqliteStore;
use crate::workspace::{Workspace, expand_tilde};

const MANIFEST_NAME: &str = "manifest.json";

/// Default documents per shard.
///
/// Sharding exists so one edited note does not retransmit the artifact. At this
/// size a 20k-document vault comes to ~80 shards of roughly 100–150 KB of
/// vectors each — small enough that an edit costs little, few enough that a
/// directory stays legible. The number is a judgment about that tradeoff rather
/// than a measurement, and it is recorded in the manifest so it can change.
const DEFAULT_SHARD_DOCS: usize = 256;

/// A shard's file name and its bytes, as `build_artifact` hands them back for
/// the caller to write (or to skip, when unchanged).
type ShardBody = (String, Vec<u8>);

/// Write a portable, query-only copy of the index.
#[derive(Debug, clap::Args)]
pub struct ExportArgs {
    /// Directory to write the artifact into (created if missing).
    #[arg(long)]
    pub out: PathBuf,
    /// Vector precision in the artifact. `f32` is exact; `i8` is ~4x smaller,
    /// measured at 0.98 recall@10 against exact in the text space but 0.93 in
    /// the image space, where CLIP's scores sit in a much narrower band.
    #[arg(long, default_value = "f32")]
    pub quantize: String,
    /// Compression for the artifact's text. `deflate` roughly halves it;
    /// `none` leaves a shard's text greppable.
    #[arg(long, default_value = "deflate")]
    pub compress: String,
    /// Vector spaces to export, comma-separated. Defaults to every space the
    /// index holds; `--space text` drops the image space, whose title proxies
    /// cost one vector per document even in a vault with no images.
    #[arg(long, value_delimiter = ',')]
    pub space: Vec<String>,
    /// Target documents per shard.
    #[arg(long, default_value_t = DEFAULT_SHARD_DOCS)]
    pub shard_docs: usize,
    /// Rewrite every shard, even ones whose contents have not changed.
    #[arg(long)]
    pub force: bool,
}

pub fn execute(ws: &Workspace, args: ExportArgs) -> Result<()> {
    if !ws.db_path.exists() {
        bail!(
            "no index found at {} — run `gnosis index`",
            ws.db_path.display()
        );
    }
    let quantization = match args.quantize.as_str() {
        "f32" => Quantization::F32,
        "i8" => Quantization::I8,
        other => bail!("unknown --quantize '{other}' (try: f32, i8)"),
    };
    let compression = match args.compress.as_str() {
        "none" => Compression::None,
        "deflate" => Compression::Deflate,
        other => bail!("unknown --compress '{other}' (try: deflate, none)"),
    };
    let wanted: Vec<Space> = if args.space.is_empty() {
        Space::ALL.to_vec()
    } else {
        args.space
            .iter()
            .map(|s| s.parse::<Space>())
            .collect::<Result<_>>()?
    };
    if args.shard_docs == 0 {
        bail!("--shard-docs must be at least 1");
    }

    let out = expand_tilde(&args.out);
    let manifest_path = out.join(MANIFEST_NAME);

    let store = SqliteStore::open(&ws.db_path)?;
    let (manifest, shards) = build_artifact(
        &store,
        &ws.config.embed.text.model,
        quantization,
        compression,
        &wanted,
        args.shard_docs,
    )?;

    // Re-exporting into an existing artifact is the normal case, not an error:
    // only shards whose contents changed need rewriting, which is the point of
    // sharding at all.
    let previous = read_previous_manifest(&manifest_path);
    let reusable: HashMap<String, String> = match &previous {
        Some(old) if !args.force && carries_over(old, &manifest) => old
            .shards
            .iter()
            .map(|s| (s.name.clone(), s.hash.clone()))
            .collect(),
        _ => HashMap::new(),
    };

    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;

    let mut written = 0usize;
    let mut unchanged = 0usize;
    let mut bytes_written = 0usize;
    for (name, body) in &shards {
        let path = out.join(name);
        let hash = shard_hash(body);
        // The hash has to agree with the file actually on disk, not just with
        // the old manifest: a shard deleted or truncated outside gnosis would
        // otherwise be skipped and leave the artifact unreadable.
        if reusable.get(name) == Some(&hash) && path.exists() {
            unchanged += 1;
            continue;
        }
        std::fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
        written += 1;
        bytes_written += body.len();
    }

    // A shrunken vault needs fewer shards; leaving the extras behind would mean
    // a directory of files no manifest refers to, growing forever.
    let mut removed = 0usize;
    if let Some(old) = &previous {
        let current: std::collections::HashSet<&str> =
            shards.iter().map(|(n, _)| n.as_str()).collect();
        for stale in old
            .shards
            .iter()
            .filter(|s| !current.contains(s.name.as_str()))
        {
            if std::fs::remove_file(out.join(&stale.name)).is_ok() {
                removed += 1;
            }
        }
    }

    // Manifest last: it is what names and checksums the shards, so a run
    // interrupted mid-write leaves the old manifest rather than one pointing at
    // a half-written shard.
    std::fs::write(&manifest_path, manifest.to_json()?)
        .with_context(|| format!("writing {}", manifest_path.display()))?;

    let chunks: usize = manifest.shards.iter().map(|s| s.chunks).sum();
    let documents: usize = manifest.shards.iter().map(|s| s.documents).sum();
    println!(
        "Exported {documents} documents, {chunks} chunks to {}",
        out.display()
    );
    for info in &manifest.spaces {
        let query = match &info.model.query_id {
            Some(id) => format!(", queried with {id}"),
            None => String::new(),
        };
        println!(
            "  {} space: {} ({} pooling, {}d){query}",
            info.space, info.model.id, info.model.pooling, info.model.dim
        );
    }
    println!("  vectors {}, text {}", args.quantize, args.compress);
    print!(
        "  {} shard(s): {written} written ({})",
        shards.len(),
        human_bytes(bytes_written)
    );
    if unchanged > 0 {
        print!(", {unchanged} unchanged");
    }
    if removed > 0 {
        print!(", {removed} removed");
    }
    println!();
    Ok(())
}

/// Whether an existing artifact's shards can be compared against the new ones
/// at all, rather than being rewritten wholesale.
///
/// Every field here re-encodes or reassigns the contents of *every* shard, so a
/// change to any of them makes the old hashes meaningless: a model or dimension
/// change replaces the vectors, adding or dropping a space changes every
/// shard's section table, quantization and compression change the encoding, and
/// `shard_docs` moves documents between shards. Missing one would
/// not fail loudly — it would leave stale shards in place and quietly serve an
/// artifact that mixes two encodings.
fn carries_over(old: &Manifest, new: &Manifest) -> bool {
    old.spaces == new.spaces
        && old.quantization == new.quantization
        && old.compression == new.compression
        && old.shard_docs == new.shard_docs
}

/// The manifest already in `path`, if there is a readable one.
///
/// A manifest that cannot be parsed — most likely written by an older gnosis
/// with a superseded format version — is treated as absent rather than fatal,
/// so a re-export silently upgrades it instead of demanding the user delete a
/// directory.
fn read_previous_manifest(path: &Path) -> Option<Manifest> {
    let json = std::fs::read_to_string(path).ok()?;
    Manifest::from_json(&json).ok()
}

/// Assemble a manifest and its shards from a store, without touching the
/// filesystem.
///
/// Separate from `execute` so the round-trip test can exercise it: the
/// acceptance gate for this format is that querying an artifact returns the
/// same ranking as querying the database it came from, and that is not
/// something to verify by eye over a directory of files.
pub(crate) fn build_artifact(
    store: &SqliteStore,
    configured_model: &str,
    quantization: Quantization,
    compression: Compression,
    wanted: &[Space],
    shard_docs: usize,
) -> Result<(Manifest, Vec<ShardBody>)> {
    // One `SpaceInfo` per space the index holds *and* the caller asked for, in
    // `Space::ALL` order so an artifact's section order does not depend on how
    // `--space` was spelled.
    let mut spaces: Vec<SpaceInfo> = Vec::new();
    for space in Space::ALL {
        if !wanted.contains(space) {
            continue;
        }
        let model = match space {
            Space::Text => Some(text_model_info(store, configured_model)?),
            Space::Image => image_model_info(store)?,
        };
        if let Some(model) = model {
            spaces.push(SpaceInfo {
                space: *space,
                model,
            });
        }
    }
    if !spaces.iter().any(|s| s.space == Space::Text) {
        bail!("the text space is not exportable from this index — run `gnosis index`");
    }

    let docs = store.export_documents()?;
    if docs.is_empty() {
        bail!("the index holds no documents — nothing to export");
    }

    // Root *labels*, not paths: an artifact that carried `/home/alex/notes`
    // would leak the author's layout and mean nothing on another machine.
    let mut roots: Vec<String> = Vec::new();
    let mut root_index: HashMap<&str, usize> = HashMap::new();
    for doc in &docs {
        if !root_index.contains_key(doc.root.as_str()) {
            root_index.insert(doc.root.as_str(), roots.len());
            roots.push(root_label(&doc.root).to_string());
        }
    }

    let shard_count = docs.len().div_ceil(shard_docs).max(1);

    // Which shard each document lands in, and its index within that shard.
    // Assignment hashes the path so that adding a note cannot move any other
    // note between shards.
    let mut shard_docs_out: Vec<Vec<DocumentEntry>> = vec![Vec::new(); shard_count];
    let mut placement: HashMap<i64, (usize, usize)> = HashMap::new();
    for doc in &docs {
        let root = root_index[doc.root.as_str()];
        let shard = assign_shard(root, &doc.rel_path, shard_count);
        let local = shard_docs_out[shard].len();
        shard_docs_out[shard].push(DocumentEntry {
            root,
            path: doc.rel_path.clone(),
            title: doc.title.clone(),
            kind: doc.kind.clone(),
            content_hash: hex(&doc.content_hash),
            mtime: doc.mtime,
            created_unix: doc.created_unix,
            links: doc.links.clone(),
            tags: doc.tags.clone(),
        });
        placement.insert(doc.id, (shard, local));
    }

    let mut shard_chunks: Vec<Vec<ArtifactChunk>> = vec![Vec::new(); shard_count];
    let mut text_chunks = 0usize;
    for info in &spaces {
        for chunk in store.export_chunks(info.space)? {
            let Some(&(shard, local)) = placement.get(&chunk.doc_id) else {
                // A chunk whose document vanished between the two queries. Not
                // expected inside one process, but skipping beats writing a
                // dangling index into the artifact.
                continue;
            };
            if chunk.vector.len() != info.model.dim {
                bail!(
                    "{} chunk of document {} has {} dimensions but the index records {} — \
                     run `gnosis rebuild`",
                    info.space,
                    shard_docs_out[shard][local].path,
                    chunk.vector.len(),
                    info.model.dim
                );
            }
            shard_chunks[shard].push(ArtifactChunk {
                doc_index: local,
                ord: chunk.ord,
                space: chunk.space,
                modality: chunk.modality,
                heading_path: chunk.heading_path,
                text: chunk.text,
                vector: chunk.vector,
            });
            if info.space == Space::Text {
                text_chunks += 1;
            }
        }
    }
    if text_chunks == 0 {
        bail!("the index holds no text vectors — nothing to export");
    }

    let specs: Vec<SpaceSpec> = spaces
        .iter()
        .map(|s| SpaceSpec {
            space: s.space,
            dim: s.model.dim,
        })
        .collect();
    let mut infos = Vec::with_capacity(shard_count);
    let mut bodies = Vec::with_capacity(shard_count);
    for i in 0..shard_count {
        let body = write_shard(
            &shard_docs_out[i],
            &shard_chunks[i],
            &specs,
            quantization,
            compression,
        )?;
        infos.push(ShardInfo {
            name: shard_name(i),
            hash: shard_hash(&body),
            documents: shard_docs_out[i].len(),
            chunks: shard_chunks[i].len(),
        });
        bodies.push((shard_name(i), body));
    }

    let manifest = Manifest {
        format_version: ARTIFACT_FORMAT_VERSION,
        created_unix: now_unix(),
        spaces,
        quantization,
        compression,
        roots,
        shard_docs,
        shards: infos,
    };
    Ok((manifest, bodies))
}

/// Which model built this index's text space, and how a client reproduces a
/// query vector in it.
///
/// The model id comes from `meta` rather than from config: config can be edited
/// after indexing, and what matters is what actually produced the stored
/// vectors.
fn text_model_info(store: &SqliteStore, configured: &str) -> Result<ModelInfo> {
    let id = store
        .get_meta("model.text")?
        .unwrap_or_else(|| configured.to_string());
    let dim: usize = store
        .get_meta("dim.text")?
        .and_then(|d| d.parse().ok())
        .context("the index does not record its vector dimensionality — run `gnosis rebuild`")?;
    let pooling = resolve_pooling(&id).with_context(|| {
        format!("the index was built with model '{id}', which this build does not know")
    })?;
    // Read from what indexing recorded, never re-derived from config:
    // `query_instruction` can be flipped after indexing, and what a client must
    // reproduce is what produced the stored vectors.
    let recorded = |key: &str| -> Result<String> {
        store.get_meta(key)?.with_context(|| {
            format!(
                "the index does not record which prefixes it embedded with ({key}) — \
                 run `gnosis index` once to record them, then export again"
            )
        })
    };
    Ok(ModelInfo {
        id,
        pooling: pooling.to_string(),
        dim,
        query_id: None,
        query_prefix: recorded("prefix.text.query")?,
        document_prefix: recorded("prefix.text.document")?,
    })
}

/// Which models built the image space, if this index has one.
///
/// Two models, not one. The stored vectors come from CLIP's vision tower
/// (`model.image_vision`) for real images and from CLIP's text tower
/// (`model.image_text`) for the title proxies; a client's *query* always goes
/// through the text tower, which is why it is recorded separately. `pooling`
/// describes that query path — fastembed pools mean over
/// `Qdrant/clip-ViT-B-32-text`, which is not the EOS-token convention a client
/// reaching for a stock CLIP text encoder would get, so it has to be stated.
///
/// Returns `None` when the index holds no image space, which is the default:
/// image embedding is opt-in.
fn image_model_info(store: &SqliteStore) -> Result<Option<ModelInfo>> {
    let Some(vision) = store.get_meta("model.image_vision")? else {
        return Ok(None);
    };
    let dim: usize = store
        .get_meta("dim.image")?
        .and_then(|d| d.parse().ok())
        .context("the index records an image model but not its dimensionality — rebuild")?;
    let query_id = store
        .get_meta("model.image_text")?
        .context("the index records an image model but not the text tower that queries it")?;
    Ok(Some(ModelInfo {
        id: vision,
        pooling: CLIP_TEXT_POOLING.to_string(),
        dim,
        query_id: Some(query_id),
        query_prefix: String::new(),
        document_prefix: String::new(),
    }))
}

/// How the CLIP text tower pools, as fastembed configures it. Hard-coded rather
/// than resolved through `resolve_pooling`, which knows only the text-space
/// models.
const CLIP_TEXT_POOLING: &str = "mean";

/// A vault's final path component — enough to tell two vaults apart in results
/// without recording where either lives.
fn root_label(root: &str) -> &str {
    Path::new(root)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(root)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn human_bytes(bytes: usize) -> String {
    const MB: usize = 1024 * 1024;
    const KB: usize = 1024;
    if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    }
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use store::artifact::Artifact;
    use store::artifact_store::ArtifactStore;

    use crate::commands::search::QueryEmbedders;
    use store::{ChunkWrite, DocWrite, TextQuery};

    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let dir = std::env::temp_dir().join(format!(
            "gnosis-export-{tag}-{}-{nanos}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// What `TextEmbedder` prepends to a bge query. Spelled out rather than
    /// imported so a change to it shows up as a failing test here.
    const BGE_QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";

    /// A store with three documents whose vectors are known, so the expected
    /// ranking is obvious rather than whatever the model happens to produce.
    fn seeded_store(dir: &std::path::Path) -> SqliteStore {
        let mut store = SqliteStore::open(&dir.join("gnosis.db")).unwrap();
        store.set_meta("model.text", "bge-small-en-v1.5").unwrap();
        store.set_meta("dim.text", "4").unwrap();
        store
            .set_meta("prefix.text.query", BGE_QUERY_INSTRUCTION)
            .unwrap();
        store.set_meta("prefix.text.document", "").unwrap();

        let docs: [(&str, &str, Vec<f32>, &str); 3] = [
            (
                "/vault/near.md",
                "Near",
                vec![1.0, 0.0, 0.0, 0.0],
                "alpha beta",
            ),
            (
                "/vault/mid.md",
                "Mid",
                vec![0.7, 0.7, 0.0, 0.0],
                "beta gamma",
            ),
            (
                "/vault/deep/far.md",
                "Far",
                vec![0.0, 1.0, 0.0, 0.0],
                "gamma delta",
            ),
        ];
        for (path, title, vector, text) in docs {
            store
                .replace_document(&DocWrite {
                    path,
                    kind: "markdown",
                    source_root: "/vault",
                    content_hash: b"hash",
                    mtime: 42,
                    title,
                    frontmatter: None,
                    indexed_at: 0,
                    chunks: &[ChunkWrite {
                        ord: 0,
                        space: "text".to_string(),
                        modality: "text".to_string(),
                        text: Some(text.to_string()),
                        heading_path: "Heading".to_string(),
                        vector,
                    }],
                    links: &["mid".to_string()],
                    tags: &["note".to_string()],
                    width: None,
                    height: None,
                })
                .unwrap();
        }
        store
    }

    /// The seeded store plus an image space: a photo with a CLIP vision vector,
    /// and a title proxy on a *note* — which is what the indexer does for every
    /// markdown file once image embedding is on. The proxy has to belong to a
    /// different document from the image, or best-chunk-per-document would hide
    /// a reader that forgot to filter by modality.
    ///
    /// At 3d, so a wrong stride shows up rather than happening to line up with
    /// the text space's 4.
    fn with_image_space(store: &mut SqliteStore) {
        store
            .set_meta("model.image_vision", "clip-vit-b-32")
            .unwrap();
        store.set_meta("model.image_text", "clip-vit-b-32").unwrap();
        store.set_meta("dim.image", "3").unwrap();
        store
            .replace_document(&DocWrite {
                path: "/vault/photo.png",
                kind: "image",
                source_root: "/vault",
                content_hash: b"img",
                mtime: 7,
                title: "Photo",
                frontmatter: None,
                indexed_at: 0,
                chunks: &[
                    ChunkWrite {
                        ord: 0,
                        space: "image".to_string(),
                        modality: "image".to_string(),
                        text: None,
                        heading_path: String::new(),
                        vector: vec![0.0, 0.0, 1.0],
                    },
                    ChunkWrite {
                        ord: 1,
                        space: "image".to_string(),
                        modality: "text_title".to_string(),
                        text: Some("Photo".to_string()),
                        heading_path: String::new(),
                        vector: vec![0.0, 1.0, 0.0],
                    },
                ],
                links: &[],
                tags: &[],
                width: Some(800),
                height: Some(600),
            })
            .unwrap();
        // near.md as the indexer would write it with image embedding on: its
        // text chunk, plus a CLIP-text proxy for its title.
        store
            .replace_document(&DocWrite {
                path: "/vault/near.md",
                kind: "markdown",
                source_root: "/vault",
                content_hash: b"hash",
                mtime: 42,
                title: "Near",
                frontmatter: None,
                indexed_at: 0,
                chunks: &[
                    ChunkWrite {
                        ord: 0,
                        space: "text".to_string(),
                        modality: "text".to_string(),
                        text: Some("alpha beta".to_string()),
                        heading_path: "Heading".to_string(),
                        vector: vec![1.0, 0.0, 0.0, 0.0],
                    },
                    ChunkWrite {
                        ord: 1,
                        space: "image".to_string(),
                        modality: "text_title".to_string(),
                        text: Some("Near".to_string()),
                        heading_path: String::new(),
                        vector: vec![0.0, 0.0, 1.0],
                    },
                ],
                links: &["mid".to_string()],
                tags: &["note".to_string()],
                width: None,
                height: None,
            })
            .unwrap();
    }

    /// The text space's model, which most tests are about.
    fn text_space(manifest: &Manifest) -> &ModelInfo {
        manifest.space(Space::Text).expect("a text space")
    }

    /// A query embedder that always answers the same vector, so a parity test
    /// compares two stores rather than two model runs.
    struct FixedVector(Vec<f32>);

    impl query::QueryEmbedder for FixedVector {
        fn embed_query(&mut self, _space: Space, _query: &str) -> Result<Vec<f32>> {
            Ok(self.0.clone())
        }
    }

    fn open_artifact(manifest: &Manifest, shards: Vec<ShardBody>) -> Artifact {
        Artifact::open(&manifest.to_json().unwrap(), &shards).expect("artifact should open")
    }

    /// One shard, so a test that does not care about sharding stays readable.
    fn build_one(
        store: &SqliteStore,
        quantization: Quantization,
        compression: Compression,
    ) -> (Manifest, Vec<ShardBody>) {
        build_artifact(
            store,
            "bge-small-en-v1.5",
            quantization,
            compression,
            Space::ALL,
            256,
        )
        .unwrap()
    }

    /// **The acceptance gate for this format.** Querying an artifact must
    /// return what querying its database returns — same documents, same order,
    /// same scores. Anything less means the plugin would rank differently from
    /// the CLI, which is the entire failure this design exists to avoid.
    #[test]
    fn artifact_search_matches_database_search_exactly() {
        let dir = temp_dir("roundtrip");
        let store = seeded_store(&dir);
        let query = [1.0, 0.0, 0.0, 0.0];

        let from_db = store
            .search_space(Space::Text, &query, 10, &TextQuery::default())
            .unwrap();

        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = open_artifact(&manifest, shards);
        let from_artifact = search::rank(&query, artifact.search_candidates(Space::Text), 10);

        assert_eq!(from_db.len(), 3);
        assert_eq!(from_artifact.len(), from_db.len());
        for (db, art) in from_db.iter().zip(&from_artifact) {
            // The artifact stores vault-relative paths on purpose, so compare
            // the database's path with its root stripped.
            assert_eq!(
                db.path.trim_start_matches("/vault/"),
                art.path,
                "ranking order diverged"
            );
            assert_eq!(db.score, art.score, "score diverged for {}", art.path);
            assert_eq!(db.title, art.title);
            assert_eq!(db.heading_path, art.heading_path);
            assert_eq!(db.text, art.text);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn f32_export_preserves_vectors_bit_for_bit() {
        let dir = temp_dir("vectors");
        let store = seeded_store(&dir);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = open_artifact(&manifest, shards);

        let mut vectors: Vec<Vec<f32>> =
            artifact.chunks().iter().map(|c| c.vector.clone()).collect();
        vectors.sort_by(|a, b| b[0].partial_cmp(&a[0]).unwrap());
        assert_eq!(vectors[0], vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(vectors[1], vec![0.7, 0.7, 0.0, 0.0]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The artifact must not carry machine paths: it would be unportable, and
    /// sharing one would disclose the author's directory layout.
    #[test]
    fn artifact_records_no_absolute_paths() {
        let dir = temp_dir("relative");
        let store = seeded_store(&dir);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        assert_eq!(
            manifest.roots,
            vec!["vault".to_string()],
            "roots are labels"
        );
        let json = manifest.to_json().unwrap();
        assert!(
            !json.contains("/vault"),
            "an absolute path leaked into the manifest"
        );

        // Documents live in the shards now, so check the bytes that ship.
        for (_, body) in &shards {
            assert!(
                !String::from_utf8_lossy(body).contains("/vault"),
                "an absolute path leaked into a shard"
            );
        }
        let artifact = open_artifact(&manifest, shards);
        for doc in artifact.documents() {
            assert!(!doc.path.starts_with('/'), "{} is absolute", doc.path);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn artifact_carries_the_models_pooling_and_each_documents_hash() {
        let dir = temp_dir("meta");
        let store = seeded_store(&dir);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);

        assert_eq!(text_space(&manifest).id, "bge-small-en-v1.5");
        assert_eq!(
            text_space(&manifest).pooling,
            "cls",
            "a client that guesses pooling produces plausible, wrong vectors"
        );
        assert_eq!(text_space(&manifest).dim, 4);
        let artifact = open_artifact(&manifest, shards);
        assert_eq!(artifact.documents().len(), 3);
        for doc in artifact.documents() {
            assert_eq!(doc.content_hash, "68617368", "hex of the source hash");
            assert_eq!(doc.mtime, 42, "needed to report staleness");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `related` excludes already-linked notes by default, and `--tag` filters.
    /// Without these the plugin's results would differ from the CLI's for
    /// reasons no one would think to look for.
    #[test]
    fn artifact_carries_links_and_tags() {
        let dir = temp_dir("links");
        let store = seeded_store(&dir);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = open_artifact(&manifest, shards);
        for doc in artifact.documents() {
            assert_eq!(doc.links, vec!["mid".to_string()]);
            assert_eq!(doc.tags, vec!["note".to_string()]);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The index records what actually produced its vectors; config can be
    /// edited afterwards. Exporting the configured model instead would write a
    /// manifest that quietly lies about the vector space.
    #[test]
    fn model_info_prefers_what_the_index_recorded_over_config() {
        let dir = temp_dir("model");
        let store = seeded_store(&dir);
        let model = text_model_info(&store, "all-MiniLM-L6-v2").unwrap();
        assert_eq!(model.id, "bge-small-en-v1.5");
        assert_eq!(model.pooling, "cls");
        assert_eq!(model.query_prefix, BGE_QUERY_INSTRUCTION);
        assert_eq!(model.document_prefix, "");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// An index from before prefixes were recorded cannot say what its
    /// vectors were embedded with, and guessing from the model id is wrong for
    /// any vault indexed with `query_instruction = false`. One `gnosis index`
    /// records them, so that is what the error says.
    #[test]
    fn export_asks_for_an_index_run_when_prefixes_were_never_recorded() {
        let dir = temp_dir("no-prefix");
        let store = SqliteStore::open(&dir.join("gnosis.db")).unwrap();
        store.set_meta("model.text", "bge-small-en-v1.5").unwrap();
        store.set_meta("dim.text", "4").unwrap();
        let err = text_model_info(&store, "bge-small-en-v1.5").expect_err("should refuse");
        assert!(format!("{err:#}").contains("gnosis index"), "{err:#}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn export_refuses_a_model_whose_pooling_is_unknown() {
        let dir = temp_dir("unknown");
        let store = seeded_store(&dir);
        store.set_meta("model.text", "some-future-model").unwrap();
        let err = build_artifact(
            &store,
            "bge-small-en-v1.5",
            Quantization::F32,
            Compression::Deflate,
            Space::ALL,
            256,
        )
        .expect_err("should refuse");
        assert!(err.to_string().contains("some-future-model"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn documents_are_split_across_shards_and_still_all_present() {
        let dir = temp_dir("shards");
        let store = seeded_store(&dir);
        // One document per shard, so the split is unambiguous.
        let (manifest, shards) = build_artifact(
            &store,
            "bge-small-en-v1.5",
            Quantization::F32,
            Compression::Deflate,
            Space::ALL,
            1,
        )
        .unwrap();

        assert_eq!(manifest.shards.len(), 3, "three documents, one per shard");
        assert_eq!(shards.len(), 3);
        assert_eq!(
            manifest.shards.iter().map(|s| s.documents).sum::<usize>(),
            3
        );
        assert_eq!(manifest.shards.iter().map(|s| s.chunks).sum::<usize>(), 3);

        // And the assembled artifact is indistinguishable from a single-shard
        // one, which is the property that matters to a reader.
        let artifact = open_artifact(&manifest, shards);
        let mut paths: Vec<&str> = artifact
            .documents()
            .iter()
            .map(|d| d.path.as_str())
            .collect();
        paths.sort();
        assert_eq!(paths, vec!["deep/far.md", "mid.md", "near.md"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Sharded or not, the ranking has to be identical — a client must not get
    /// different results because the exporter chose a different shard size.
    #[test]
    fn shard_size_does_not_change_the_ranking() {
        let dir = temp_dir("shardrank");
        let store = seeded_store(&dir);
        let query = [1.0, 0.0, 0.0, 0.0];

        let mut rankings = Vec::new();
        for shard_docs in [1, 2, 256] {
            let (manifest, shards) = build_artifact(
                &store,
                "bge-small-en-v1.5",
                Quantization::F32,
                Compression::Deflate,
                Space::ALL,
                shard_docs,
            )
            .unwrap();
            let artifact = open_artifact(&manifest, shards);
            let hits = search::rank(&query, artifact.search_candidates(Space::Text), 10);
            rankings.push(
                hits.iter()
                    .map(|h| (h.path.clone(), h.score))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(rankings[0], rankings[1]);
        assert_eq!(rankings[1], rankings[2]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// **Adding** a note must dirty only the shard it lands in.
    ///
    /// Exported at `--shard-docs 2`, so three documents and then four both come
    /// to two shards and the count is held fixed — otherwise every document
    /// would be legitimately reassigned and this would prove nothing. The added
    /// note sorts before every existing one, so a bug that let document
    /// *ordering* leak across shards would dirty both.
    ///
    /// Note what this does *not* test: that assignment ignores the rest of the
    /// corpus. `assign_shard` takes no document list, so that is enforced by its
    /// signature — a positional scheme cannot be expressed through it — and a
    /// test cannot distinguish the two. Golden values in `artifact.rs` pin the
    /// mapping itself.
    #[test]
    fn adding_a_note_dirties_only_the_shard_it_lands_in() {
        let dir = temp_dir("insert");
        let mut store = seeded_store(&dir);
        let (_, before) = build_artifact(
            &store,
            "bge-small-en-v1.5",
            Quantization::F32,
            Compression::Deflate,
            Space::ALL,
            2,
        )
        .unwrap();
        assert_eq!(before.len(), 2, "three documents at two per shard");

        store
            .replace_document(&DocWrite {
                // Sorts before every existing document on purpose.
                path: "/vault/aaa-added.md",
                kind: "markdown",
                source_root: "/vault",
                content_hash: b"hash",
                mtime: 42,
                title: "Added",
                frontmatter: None,
                indexed_at: 0,
                chunks: &[ChunkWrite {
                    ord: 0,
                    space: "text".to_string(),
                    modality: "text".to_string(),
                    text: Some("epsilon".to_string()),
                    heading_path: "Heading".to_string(),
                    vector: vec![0.0, 0.0, 1.0, 0.0],
                }],
                links: &[],
                tags: &[],
                width: None,
                height: None,
            })
            .unwrap();

        let (manifest, after) = build_artifact(
            &store,
            "bge-small-en-v1.5",
            Quantization::F32,
            Compression::Deflate,
            Space::ALL,
            2,
        )
        .unwrap();
        assert_eq!(
            after.len(),
            2,
            "four documents at two per shard is still two shards"
        );

        let changed = before
            .iter()
            .zip(&after)
            .filter(|((_, a), (_, b))| a != b)
            .count();
        assert_eq!(
            changed, 1,
            "adding one note should dirty exactly one of two shards"
        );
        assert_eq!(
            open_artifact(&manifest, after).documents().len(),
            4,
            "and nothing may be lost in the process"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// At a fixed shard count, editing one note must leave every other shard
    /// byte-identical. This is the property `--shard-docs` exists to deliver,
    /// and it holds only because assignment hashes the path.
    #[test]
    fn editing_one_note_changes_exactly_one_shard() {
        let dir = temp_dir("oneshard");
        let mut store = seeded_store(&dir);
        // Three shards for three documents, and keep it there by exporting at
        // the same shard_docs after the edit.
        let (_, before) = build_artifact(
            &store,
            "bge-small-en-v1.5",
            Quantization::F32,
            Compression::Deflate,
            Space::ALL,
            1,
        )
        .unwrap();

        store
            .replace_document(&DocWrite {
                path: "/vault/mid.md",
                kind: "markdown",
                source_root: "/vault",
                content_hash: b"changed",
                mtime: 99,
                title: "Mid",
                frontmatter: None,
                indexed_at: 0,
                chunks: &[ChunkWrite {
                    ord: 0,
                    space: "text".to_string(),
                    modality: "text".to_string(),
                    text: Some("beta gamma edited".to_string()),
                    heading_path: "Heading".to_string(),
                    vector: vec![0.7, 0.7, 0.0, 0.0],
                }],
                links: &["mid".to_string()],
                tags: &["note".to_string()],
                width: None,
                height: None,
            })
            .unwrap();

        let (_, after) = build_artifact(
            &store,
            "bge-small-en-v1.5",
            Quantization::F32,
            Compression::Deflate,
            Space::ALL,
            1,
        )
        .unwrap();
        assert_eq!(before.len(), after.len(), "shard count should not change");

        let changed = before
            .iter()
            .zip(&after)
            .filter(|((_, a), (_, b))| a != b)
            .count();
        assert_eq!(
            changed,
            1,
            "editing one note should rewrite exactly one of {} shards",
            before.len()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn i8_export_is_smaller_and_still_ranks_the_same_order() {
        let dir = temp_dir("i8");
        let store = seeded_store(&dir);
        let query = [1.0, 0.0, 0.0, 0.0];

        let (_, f32_shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let (manifest, i8_shards) = build_one(&store, Quantization::I8, Compression::Deflate);
        let i8_bytes: usize = i8_shards.iter().map(|(_, b)| b.len()).sum();
        let f32_bytes: usize = f32_shards.iter().map(|(_, b)| b.len()).sum();
        assert!(
            i8_bytes < f32_bytes,
            "i8 {i8_bytes} should be smaller than f32 {f32_bytes}"
        );

        let artifact = open_artifact(&manifest, i8_shards);
        let hits = search::rank(&query, artifact.search_candidates(Space::Text), 10);
        let order: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert_eq!(order, vec!["near.md", "mid.md", "deep/far.md"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Compression is lossless, so it must be invisible above the format: the
    /// same documents, the same text, the same ranking, the same scores. Only
    /// the bytes on disk differ.
    #[test]
    fn compression_changes_the_bytes_and_nothing_else() {
        let dir = temp_dir("compress");
        let store = seeded_store(&dir);
        let query = [1.0, 0.0, 0.0, 0.0];

        let (plain_manifest, plain_shards) = build_one(&store, Quantization::I8, Compression::None);
        let (packed_manifest, packed_shards) =
            build_one(&store, Quantization::I8, Compression::Deflate);
        assert_ne!(
            plain_shards[0].1, packed_shards[0].1,
            "a deflated shard should not be byte-identical to a plain one"
        );
        assert_eq!(plain_manifest.compression, Compression::None);
        assert_eq!(packed_manifest.compression, Compression::Deflate);

        let plain = open_artifact(&plain_manifest, plain_shards);
        let packed = open_artifact(&packed_manifest, packed_shards);
        assert_eq!(packed.documents(), plain.documents());
        let ranked = |a: &Artifact| -> Vec<(String, f32, String)> {
            search::rank(&query, a.search_candidates(Space::Text), 10)
                .into_iter()
                .map(|h| (h.path, h.score, h.text))
                .collect()
        };
        assert_eq!(ranked(&packed), ranked(&plain));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Re-exporting at a different setting must not keep a single old shard:
    /// each of these re-encodes or reassigns every one of them, and a surviving
    /// shard would leave the artifact a mixture of two encodings that still
    /// passes its hash check.
    #[test]
    fn changing_how_shards_are_encoded_reuses_none_of_them() {
        let dir = temp_dir("carryover");
        let store = seeded_store(&dir);
        let (base, _) = build_one(&store, Quantization::I8, Compression::Deflate);

        assert!(carries_over(&base, &base), "an unchanged export reuses");

        let mut other = base.clone();
        other.compression = Compression::None;
        assert!(!carries_over(&base, &other), "compression must invalidate");

        let mut other = base.clone();
        other.quantization = Quantization::F32;
        assert!(!carries_over(&base, &other), "quantization must invalidate");

        let mut other = base.clone();
        other.shard_docs = 2;
        assert!(!carries_over(&base, &other), "shard_docs must invalidate");

        let mut other = base.clone();
        other.spaces[0].model.id = "multilingual-e5-small".to_string();
        assert!(!carries_over(&base, &other), "the model must invalidate");

        let mut other = base.clone();
        other.spaces.clear();
        assert!(
            !carries_over(&base, &other),
            "dropping a space must invalidate"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The image space travels at its own dimensionality beside the text
    /// space's, which is what v4 is for, and a search of it returns the image
    /// rather than the title proxy.
    #[test]
    fn an_index_with_images_exports_both_spaces() {
        let dir = temp_dir("images");
        let mut store = seeded_store(&dir);
        with_image_space(&mut store);

        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        assert_eq!(text_space(&manifest).dim, 4);
        let image = manifest.space(Space::Image).expect("an image space");
        assert_eq!(image.dim, 3);
        assert_eq!(image.id, "clip-vit-b-32");
        assert_eq!(
            image.query_id.as_deref(),
            Some("clip-vit-b-32"),
            "a client queries the image space through CLIP's text tower"
        );

        let artifact = open_artifact(&manifest, shards);
        let query = [0.0, 0.0, 1.0];
        let hits = search::rank(&query, artifact.search_candidates(Space::Image), 10);
        assert_eq!(
            hits.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(),
            vec!["photo.png"],
            "a search of the image space must not return the title proxy"
        );
        // The acceptance gate, now for the image space: the artifact's ranking
        // is the database's ranking.
        let from_db = store
            .search_space(Space::Image, &query, 10, &TextQuery::default())
            .unwrap();
        assert_eq!(from_db.len(), hits.len());
        for (db, art) in from_db.iter().zip(&hits) {
            assert_eq!(db.path.trim_start_matches("/vault/"), art.path);
            assert_eq!(db.score, art.score, "score diverged for {}", art.path);
        }
        assert_eq!(
            artifact.related_candidates(Space::Image).len(),
            3,
            "related must still see both title proxies and the image"
        );
        // The text space is untouched by the image space riding along.
        let text = search::rank(
            &[1.0, 0.0, 0.0, 0.0],
            artifact.search_candidates(Space::Text),
            10,
        );
        assert_eq!(text.len(), 3);
        assert_eq!(text[0].path, "near.md");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The image space costs one vector per document even in a vault with no
    /// images, because every note gets a title proxy. `--space text` is how a
    /// user who does not want image search declines that.
    #[test]
    fn space_text_drops_the_image_space_from_the_artifact() {
        let dir = temp_dir("spacefilter");
        let mut store = seeded_store(&dir);
        with_image_space(&mut store);

        let (manifest, shards) = build_artifact(
            &store,
            "bge-small-en-v1.5",
            Quantization::F32,
            Compression::Deflate,
            &[Space::Text],
            256,
        )
        .unwrap();
        assert!(manifest.space(Space::Image).is_none());
        let artifact = open_artifact(&manifest, shards);
        assert!(artifact.related_candidates(Space::Image).is_empty());
        assert_eq!(
            artifact.documents().len(),
            4,
            "the image document still travels — its text-space neighbors link to it"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// **The acceptance gate for the plugin.** Both channels, fused, scored
    /// through the same pipeline a frontend uses: an artifact must return what
    /// its database returns, or every measurement in this project stops
    /// describing the client.
    ///
    /// PR #35's gate compared the dense channel alone. Most of gnosis's
    /// measured quality is BM25 — lexical-only scores 0.666 on scifact against
    /// the hybrid's 0.722 — so a dense-only gate would have passed an artifact
    /// whose lexical channel was missing or mis-normalized.
    #[test]
    fn the_hybrid_pipeline_agrees_between_the_artifact_and_the_database() {
        let dir = temp_dir("hybrid-parity");
        let mut store = seeded_store(&dir);
        with_image_space(&mut store);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = ArtifactStore::new(open_artifact(&manifest, shards));

        let cfg = search::bm25::SearchConfig::default();
        assert!(cfg.lexical, "this gate is pointless with the channel off");
        for query in ["beta gamma", "alpha", "near", "delta"] {
            let from_db = query::run_query(
                &store,
                &mut FixedVector(vec![1.0, 0.0, 0.0, 0.0]),
                query,
                &[Space::Text],
                10,
                &TextQuery::default(),
                &cfg,
            )
            .unwrap();
            let from_artifact = query::run_query(
                &artifact,
                &mut FixedVector(vec![1.0, 0.0, 0.0, 0.0]),
                query,
                &[Space::Text],
                10,
                &TextQuery::default(),
                &cfg,
            )
            .unwrap();

            assert!(!from_db.is_empty(), "'{query}' matched nothing at all");
            assert_eq!(
                from_db.len(),
                from_artifact.len(),
                "'{query}' returned {} from the database and {} from the artifact",
                from_db.len(),
                from_artifact.len()
            );
            for (db, art) in from_db.iter().zip(&from_artifact) {
                assert_eq!(
                    db.path.trim_start_matches("/vault/"),
                    art.path,
                    "'{query}' ranked differently"
                );
                assert_eq!(
                    db.score, art.score,
                    "'{query}' scored {} differently",
                    art.path
                );
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `related` is the feature that needs no model at all, so it is the first
    /// thing a plugin can ship — and it reads a different set of store methods
    /// than search does.
    #[test]
    fn related_agrees_between_the_artifact_and_the_database() {
        let dir = temp_dir("related-parity");
        let mut store = seeded_store(&dir);
        with_image_space(&mut store);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = ArtifactStore::new(open_artifact(&manifest, shards));

        let vectors = store.chunk_vectors("/vault/near.md", Space::Text).unwrap();
        assert_eq!(
            artifact.chunk_vectors("near.md", Space::Text).unwrap(),
            vectors,
            "the artifact must find the same document's vectors by relative path"
        );

        let exclude_db = vec!["/vault/near.md".to_string()];
        let exclude_art = vec!["near.md".to_string()];
        let from_db = store
            .related_space(
                Space::Text,
                &vectors,
                &exclude_db,
                10,
                &TextQuery::default(),
            )
            .unwrap();
        let from_artifact = artifact
            .related_space(
                Space::Text,
                &vectors,
                &exclude_art,
                10,
                &TextQuery::default(),
            )
            .unwrap();
        assert_eq!(from_db.len(), from_artifact.len());
        for (db, art) in from_db.iter().zip(&from_artifact) {
            assert_eq!(db.path.trim_start_matches("/vault/"), art.path);
            assert_eq!(db.score, art.score);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// An artifact is derived and query-only. A client that could half-write
    /// one would produce an index nothing can regenerate, so the write methods
    /// report rather than panic — and rather than silently succeed.
    #[test]
    fn an_artifact_store_refuses_every_write() {
        let dir = temp_dir("readonly");
        let store = seeded_store(&dir);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let mut artifact = ArtifactStore::new(open_artifact(&manifest, shards));

        let err = artifact.set_meta("model.text", "x").expect_err("read-only");
        assert!(err.to_string().contains("read-only"), "{err}");
        assert!(artifact.delete_document("near.md").is_err());
        assert!(artifact.delete_by_root("vault").is_err());
        assert!(
            artifact
                .replace_document(&DocWrite {
                    path: "new.md",
                    kind: "markdown",
                    source_root: "/vault",
                    content_hash: b"h",
                    mtime: 0,
                    title: "New",
                    frontmatter: None,
                    indexed_at: 0,
                    chunks: &[],
                    links: &[],
                    tags: &[],
                    width: None,
                    height: None,
                })
                .is_err()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Named paths restrict an artifact the way they restrict the database:
    /// exactly those documents, and none for an empty list.
    #[test]
    fn an_artifact_filters_by_named_paths() {
        let dir = temp_dir("paths");
        let store = seeded_store(&dir);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = ArtifactStore::new(open_artifact(&manifest, shards));
        let query = [1.0, 0.0, 0.0, 0.0];
        let all = artifact
            .search_space(Space::Text, &query, 10, &TextQuery::default())
            .unwrap();
        let last = vec![all.last().expect("a seeded document").path.clone()];

        let named = TextQuery {
            paths: Some(&last),
            ..TextQuery::default()
        };
        let hits = artifact
            .search_space(Space::Text, &query, 1, &named)
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.path.clone()).collect::<Vec<_>>(),
            last
        );

        let none: Vec<String> = Vec::new();
        let nothing = TextQuery {
            paths: Some(&none),
            ..TextQuery::default()
        };
        assert!(
            artifact
                .search_space(Space::Text, &query, 10, &nothing)
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The filter has to mean the same thing on both sides, and this is where
    /// the artifact genuinely differs: it holds a root *label*, not a machine
    /// path, so that is what `--from` matches against.
    #[test]
    fn an_artifact_filters_by_root_label_and_by_tag() {
        let dir = temp_dir("filters");
        let store = seeded_store(&dir);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = ArtifactStore::new(open_artifact(&manifest, shards));
        let query = [1.0, 0.0, 0.0, 0.0];

        let roots = vec!["vault".to_string()];
        let hits = artifact
            .search_space(
                Space::Text,
                &query,
                10,
                &TextQuery {
                    from: Some(&roots),
                    tags: None,
                    paths: None,
                },
            )
            .unwrap();
        assert_eq!(hits.len(), 3, "the label matches every document");

        let elsewhere = vec!["/vault".to_string()];
        let none = artifact
            .search_space(
                Space::Text,
                &query,
                10,
                &TextQuery {
                    from: Some(&elsewhere),
                    tags: None,
                    paths: None,
                },
            )
            .unwrap();
        assert!(none.is_empty(), "an absolute path is not a label");

        let tags = vec!["note".to_string()];
        let tagged = artifact
            .search_space(
                Space::Text,
                &query,
                10,
                &TextQuery {
                    from: None,
                    tags: Some(&tags),
                    paths: None,
                },
            )
            .unwrap();
        assert_eq!(tagged.len(), 3);
        let missing = vec!["absent".to_string()];
        assert!(
            artifact
                .search_space(
                    Space::Text,
                    &query,
                    10,
                    &TextQuery {
                        from: None,
                        tags: Some(&missing),
                        paths: None,
                    },
                )
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The one divergence that is inherent rather than fixable, pinned with
    /// evidence instead of asserted in a comment.
    ///
    /// SQLite's FTS5 `unicode61` tokenizer folds diacritics; `bm25::tokenize`
    /// does not, and says so. So on accented text the two backends disagree
    /// about whether a chunk contains a term — and therefore about its
    /// document frequency, which feeds IDF for every query. The artifact is
    /// the conservative side: it misses the accented chunk rather than
    /// inventing a match.
    #[test]
    fn the_artifacts_tokenizer_does_not_fold_diacritics_and_sqlite_does() {
        let dir = temp_dir("diacritics");
        let mut store = SqliteStore::open(&dir.join("gnosis.db")).unwrap();
        store.set_meta("model.text", "bge-small-en-v1.5").unwrap();
        store.set_meta("dim.text", "2").unwrap();
        store
            .set_meta("prefix.text.query", BGE_QUERY_INSTRUCTION)
            .unwrap();
        store.set_meta("prefix.text.document", "").unwrap();
        for (path, text) in [
            ("accented.md", "une pause café"),
            ("plain.md", "a cafe stop"),
        ] {
            store
                .replace_document(&DocWrite {
                    path: &format!("/vault/{path}"),
                    kind: "markdown",
                    source_root: "/vault",
                    content_hash: b"hash",
                    mtime: 0,
                    title: path,
                    frontmatter: None,
                    indexed_at: 0,
                    chunks: &[ChunkWrite {
                        ord: 0,
                        space: "text".to_string(),
                        modality: "text".to_string(),
                        text: Some(text.to_string()),
                        heading_path: String::new(),
                        vector: vec![1.0, 0.0],
                    }],
                    links: &[],
                    tags: &[],
                    width: None,
                    height: None,
                })
                .unwrap();
        }
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = ArtifactStore::new(open_artifact(&manifest, shards));

        let terms = vec!["cafe".to_string()];
        let from_db = store
            .lexical_candidates(&terms, 50, &TextQuery::default())
            .unwrap();
        let from_artifact = artifact
            .lexical_candidates(&terms, 50, &TextQuery::default())
            .unwrap();

        assert_eq!(
            from_db.doc_freqs,
            vec![2],
            "FTS5 folds the accent, so both chunks contain the term"
        );
        assert_eq!(
            from_artifact.doc_freqs,
            vec![1],
            "the artifact's tokenizer keeps the accent, so only the plain chunk matches"
        );
        assert_eq!(from_artifact.candidates.len(), 1);
        assert_eq!(from_artifact.candidates[0].path, "plain.md");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Fuzzy expansion lives in the shared pipeline over each store's own
    /// vocabulary, so a misspelled query must rank the same on both.
    #[test]
    fn both_stores_answer_a_misspelled_query_the_same_way() {
        let dir = temp_dir("fuzzy-agree");
        let mut store = SqliteStore::open(&dir.join("gnosis.db")).unwrap();
        store.set_meta("model.text", "bge-small-en-v1.5").unwrap();
        store.set_meta("dim.text", "2").unwrap();
        store
            .set_meta("prefix.text.query", BGE_QUERY_INSTRUCTION)
            .unwrap();
        store.set_meta("prefix.text.document", "").unwrap();
        for (path, text) in [
            ("a.md", "retrieval of graph indexes"),
            ("b.md", "graph retrieval retrieval retrieval"),
            ("c.md", "a recipe for chocolate cake"),
        ] {
            store
                .replace_document(&DocWrite {
                    path: &format!("/vault/{path}"),
                    kind: "markdown",
                    source_root: "/vault",
                    content_hash: b"hash",
                    mtime: 0,
                    title: path,
                    frontmatter: None,
                    indexed_at: 0,
                    chunks: &[ChunkWrite {
                        ord: 0,
                        space: "text".to_string(),
                        modality: "text".to_string(),
                        text: Some(text.to_string()),
                        heading_path: String::new(),
                        vector: vec![1.0, 0.0],
                    }],
                    links: &[],
                    tags: &[],
                    width: None,
                    height: None,
                })
                .unwrap();
        }
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = ArtifactStore::new(open_artifact(&manifest, shards));

        let cfg = search::bm25::SearchConfig {
            fuzzy: true,
            title_boost: 0.0,
            ..Default::default()
        };
        let rank = |s: &dyn Store| -> Vec<(String, f32)> {
            query::fuse_text_space(
                s,
                "retreival graph",
                Vec::new(),
                10,
                &TextQuery::default(),
                &cfg,
            )
            .unwrap()
            .into_iter()
            .map(|h| (h.path.rsplit('/').next().unwrap().to_string(), h.score))
            .collect()
        };
        let from_db = rank(&store);
        let from_artifact = rank(&artifact);

        assert_eq!(from_db.len(), 2, "both retrieval notes match: {from_db:?}");
        assert_eq!(from_db, from_artifact);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A deterministic stand-in for an embedding model: the same text always
    /// gives the same vector, and different text almost never does.
    ///
    /// Deterministic is the whole point. Both paths embed whatever strings
    /// they decide to embed, so if their chunk boundaries or their
    /// heading-trail joining differ by one character, the vectors differ and
    /// the shards stop matching. A real model would hide that behind noise;
    /// this turns it into a byte comparison.
    ///
    /// It declares a model's prefixes the way `TextEmbedder` does, so the
    /// trait's own `embed_document` applies them on the CLI's side and the
    /// builder has to apply the same ones on its side.
    struct HashEmbedder {
        model: &'static str,
        query_prefix: &'static str,
        document_prefix: &'static str,
    }

    const HASH_BGE: HashEmbedder = HashEmbedder {
        model: "bge-small-en-v1.5",
        query_prefix: BGE_QUERY_INSTRUCTION,
        document_prefix: "",
    };

    impl embed::Embedder for HashEmbedder {
        fn space(&self) -> &str {
            "text"
        }
        fn dim(&self) -> usize {
            4
        }
        fn model_id(&self) -> &str {
            self.model
        }
        fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(inputs.iter().map(|t| hash_vector(t)).collect())
        }
        fn query_prefix(&self) -> &str {
            self.query_prefix
        }
        fn document_prefix(&self) -> &str {
            self.document_prefix
        }
    }

    /// A builder spec matching `embedder`, at the CLI's defaults otherwise.
    fn spec_for(embedder: &HashEmbedder, pooling: &str, root: &str) -> index::builder::BuildSpec {
        let chunk_cfg = chunker::ChunkConfig::default();
        index::builder::BuildSpec {
            model: embedder.model.to_string(),
            pooling: pooling.to_string(),
            dim: 4,
            query_prefix: embedder.query_prefix.to_string(),
            document_prefix: embedder.document_prefix.to_string(),
            root: root.to_string(),
            quantization: Quantization::I8,
            compression: Compression::Deflate,
            shard_docs: 256,
            max_tokens: chunk_cfg.max_tokens,
            overlap: chunk_cfg.overlap,
            image: None,
        }
    }

    /// A unit vector derived from the text's blake3 hash.
    fn hash_vector(text: &str) -> Vec<f32> {
        let hash = blake3::hash(text.as_bytes());
        let bytes = hash.as_bytes();
        let raw: Vec<f32> = (0..4)
            .map(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as f32)
            .collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.into_iter().map(|x| x / norm).collect()
    }

    /// A small vault on disk, covering the shapes that make the two paths
    /// disagree if anything about them differs: several chunks from one note,
    /// a property-only note (whose only chunk is synthetic), frontmatter with
    /// links and tags, and a note with no body at all.
    fn parity_vault(dir: &std::path::Path) -> PathBuf {
        let vault = dir.join("vault");
        std::fs::create_dir_all(vault.join("notes")).unwrap();

        std::fs::write(
            vault.join("notes/sourdough.md"),
            format!(
                "---\ncreated: 2024-05-03\ntags: [baking, bread]\n---\n\n# Sourdough\n\n{}\n\n## Feeding\n\nRefresh \
                 at one to two to two and wait for peak. See [[Hydration]].\n",
                "A wild yeast starter and a long overnight proof. ".repeat(40)
            ),
        )
        .unwrap();
        std::fs::write(
            vault.join("notes/hydration.md"),
            "# Hydration\n\nBaker's percentages: water as a share of flour weight.\n",
        )
        .unwrap();
        // Everything in properties: its only chunk is the identity chunk.
        std::fs::write(
            vault.join("notes/blade-runner.md"),
            "---\ncategories:\n  - \"[[Movies]]\"\ndirector:\n  - \"[[Ridley Scott]]\"\nyear: 1982\n---\n",
        )
        .unwrap();
        // Nothing at all: still has to be findable by name.
        std::fs::write(vault.join("notes/stub.md"), "").unwrap();
        vault
    }

    /// The claim that makes plugin-side indexing worth having: an artifact
    /// built through [`index::builder`] from a vault's text is **the same
    /// artifact** `gnosis index` + `gnosis export` produce from the same
    /// vault, byte for byte.
    ///
    /// Not "equivalent" and not "close": identical shard bytes. That covers
    /// every decision the two paths could have made differently — chunk
    /// boundaries, the identity chunk, what gets embedded, title derivation,
    /// links and tags, content hashes, mtimes, document order, shard
    /// assignment, quantization and compression. Only the manifest's
    /// `created_unix` is allowed to differ, and the test pins that too by
    /// passing the CLI's own timestamp.
    #[test]
    fn the_builder_and_the_cli_produce_the_same_artifact() {
        assert_builder_matches_cli(HASH_BGE, "cls");
    }

    /// The same claim for E5, the one supported model that prefixes
    /// *documents*. bge's empty document prefix cannot tell a builder that
    /// applies it from one that ignores it; this can.
    #[test]
    fn the_builder_and_the_cli_agree_for_a_model_that_prefixes_documents() {
        assert_builder_matches_cli(
            HashEmbedder {
                model: "multilingual-e5-small",
                query_prefix: "query: ",
                document_prefix: "passage: ",
            },
            "mean",
        );
    }

    fn assert_builder_matches_cli(mut text: HashEmbedder, pooling: &str) {
        let dir = temp_dir("builder-parity");
        let vault = parity_vault(&dir);

        // The CLI's path: walk, parse, chunk, embed, store, export.
        let mut store = SqliteStore::open(&dir.join("gnosis.db")).unwrap();
        let progress = progress::NoopProgress;
        let chunk_cfg = chunker::ChunkConfig::default();
        let mut args = index::IndexerArgs {
            store: &mut store,
            walker: &crate::walk::FsWalker::new(false),
            fs_reader: &crate::fs::StdFs,
            embedders: index::EmbedderSet {
                text: &mut text,
                image: None,
                image_text: None,
            },
            progress: &progress,
        };
        let report = index::run(
            &mut args,
            std::slice::from_ref(&vault),
            &[],
            &chunk_cfg,
            false,
            true,
            32,
        )
        .expect("indexing the parity vault");
        assert_eq!(report.indexed, 4, "every note must index");
        // The one dated note travels into the artifact on both paths: the
        // shard comparison below fails if only one of them carries it.
        assert_eq!(store.document_dates().unwrap().len(), 1);

        let (cli_manifest, cli_shards) = build_artifact(
            &store,
            text.model,
            Quantization::I8,
            Compression::Deflate,
            &[Space::Text],
            256,
        )
        .expect("exporting");

        // The client's path: the same text, chunked and packed by the builder.
        let spec = spec_for(
            &text,
            pooling,
            &vault.file_name().unwrap().to_string_lossy(),
        );
        let mut builder = index::builder::ArtifactBuilder::new(spec);
        let mut files: Vec<PathBuf> = std::fs::read_dir(vault.join("notes"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        // Deliberately not sorted the way `finish` sorts: the builder must not
        // depend on the order documents arrive in.
        files.sort_by_key(|p| std::cmp::Reverse(p.to_string_lossy().to_string()));
        for path in &files {
            let rel = format!("notes/{}", path.file_name().unwrap().to_string_lossy());
            let text = std::fs::read_to_string(path).unwrap();
            let mtime = fs::FileReader::mtime(&crate::fs::StdFs, path);
            let added = builder.add_document(&rel, &text, mtime);
            assert!(!added.reused, "a fresh build reuses nothing");
            let vectors: Vec<f32> = added.texts.iter().flat_map(|t| hash_vector(t)).collect();
            builder
                .set_vectors(&rel, &vectors)
                .expect("setting vectors");
        }
        let built = builder
            .finish(cli_manifest.created_unix)
            .expect("building the artifact");

        assert_eq!(built.documents, 4);
        assert_eq!(
            built.chunks,
            cli_manifest.shards.iter().map(|s| s.chunks).sum::<usize>(),
            "the two paths must chunk the vault the same way"
        );
        assert_eq!(built.shards.len(), cli_shards.len(), "same shard count");
        for ((cli_name, cli_body), (name, body)) in cli_shards.iter().zip(&built.shards) {
            assert_eq!(cli_name, name);
            assert_eq!(
                body, cli_body,
                "shard '{name}' differs: the builder and the CLI disagree about this vault"
            );
        }
        assert_eq!(
            built.manifest, cli_manifest,
            "the manifests must match, timestamp included"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// An incremental rebuild re-embeds only what changed, and leaves the
    /// shards it did not have to touch byte-identical — which is what keeps a
    /// file-level syncer quiet.
    #[test]
    fn rebuilding_reuses_every_unchanged_document() {
        let dir = temp_dir("builder-reuse");
        let vault = parity_vault(&dir);

        let spec = || index::builder::BuildSpec {
            model: "bge-small-en-v1.5".to_string(),
            pooling: "cls".to_string(),
            dim: 4,
            query_prefix: BGE_QUERY_INSTRUCTION.to_string(),
            document_prefix: String::new(),
            root: "vault".to_string(),
            quantization: Quantization::F32,
            compression: Compression::Deflate,
            shard_docs: 256,
            max_tokens: 256,
            overlap: 43,
            image: None,
        };
        let read = |name: &str| std::fs::read_to_string(vault.join("notes").join(name)).unwrap();
        let names = ["sourdough.md", "hydration.md", "blade-runner.md", "stub.md"];

        let mut builder = index::builder::ArtifactBuilder::new(spec());
        for name in names {
            let rel = format!("notes/{name}");
            let added = builder.add_document(&rel, &read(name), 1_700_000_000);
            let vectors: Vec<f32> = added.texts.iter().flat_map(|t| hash_vector(t)).collect();
            builder.set_vectors(&rel, &vectors).unwrap();
        }
        let first = builder.finish(1_700_000_000).unwrap();
        let artifact = open_artifact(&first.manifest, first.shards.clone());

        // One note edited, the rest untouched.
        std::fs::write(
            vault.join("notes/hydration.md"),
            "# Hydration\n\nRewritten: water as a share of flour weight, by mass.\n",
        )
        .unwrap();

        let mut builder =
            index::builder::ArtifactBuilder::reusing(spec(), artifact).expect("reusing");
        let mut reused = 0;
        for name in names {
            let rel = format!("notes/{name}");
            let added = builder.add_document(&rel, &read(name), 1_700_000_000);
            if added.reused {
                reused += 1;
                assert!(added.texts.is_empty(), "a reused document embeds nothing");
                continue;
            }
            let vectors: Vec<f32> = added.texts.iter().flat_map(|t| hash_vector(t)).collect();
            builder.set_vectors(&rel, &vectors).unwrap();
        }
        assert_eq!(reused, 3, "only the edited note needs re-embedding");

        let second = builder.finish(1_700_000_001).unwrap();
        assert_eq!(
            second.manifest.shards.len(),
            1,
            "this vault fits in one shard, so that shard must be rewritten"
        );
        assert!(
            second.unchanged.is_empty(),
            "the only shard holds the edited note, so nothing is unchanged"
        );
        assert_ne!(
            second.manifest.shards[0].hash, first.manifest.shards[0].hash,
            "the edit must change the shard"
        );

        // And a rebuild with nothing edited rewrites nothing at all.
        let artifact = open_artifact(&second.manifest, second.shards.clone());
        let mut builder =
            index::builder::ArtifactBuilder::reusing(spec(), artifact).expect("reusing");
        for name in names {
            let rel = format!("notes/{name}");
            let added = builder.add_document(&rel, &read(name), 1_700_000_000);
            assert!(added.reused, "{name} is unchanged and must be reused");
        }
        let third = builder.finish(1_700_000_002).unwrap();
        assert_eq!(
            third.unchanged,
            vec!["shard-0000.bin".to_string()],
            "an unchanged vault must produce no shard writes"
        );
        assert!(third.shards.is_empty());

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A previous artifact from a different model cannot be reused: its
    /// vectors are in another space, and a mixture would rank confidently and
    /// wrongly, which is the one failure mode with no symptom.
    #[test]
    fn reusing_an_artifact_from_another_model_is_refused() {
        let dir = temp_dir("builder-model");
        let store = seeded_store(&dir);
        let (manifest, shards) = build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = open_artifact(&manifest, shards);

        let outcome = index::builder::ArtifactBuilder::reusing(
            index::builder::BuildSpec {
                model: "multilingual-e5-small".to_string(),
                pooling: "mean".to_string(),
                dim: 4,
                query_prefix: "query: ".to_string(),
                document_prefix: "passage: ".to_string(),
                root: "vault".to_string(),
                quantization: Quantization::F32,
                compression: Compression::Deflate,
                shard_docs: 256,
                max_tokens: 256,
                overlap: 43,
                image: None,
            },
            artifact,
        );
        let Err(err) = outcome else {
            panic!("an artifact from another model must not be reusable");
        };
        let message = format!("{err:#}");
        assert!(message.contains("bge-small-en-v1.5"), "got {message}");
        assert!(message.contains("multilingual-e5-small"), "got {message}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The same model with a different document prefix is a different vector
    /// convention — what `query_instruction = false` does to an E5 index — so
    /// reusing across it is refused just as across models.
    #[test]
    fn reusing_an_artifact_with_another_document_prefix_is_refused() {
        let dir = temp_dir("builder-prefix");
        let vault = parity_vault(&dir);
        let e5 = HashEmbedder {
            model: "multilingual-e5-small",
            query_prefix: "query: ",
            document_prefix: "passage: ",
        };
        let mut builder = index::builder::ArtifactBuilder::new(spec_for(&e5, "mean", "vault"));
        let text = std::fs::read_to_string(vault.join("notes/hydration.md")).unwrap();
        let added = builder.add_document("notes/hydration.md", &text, 1_700_000_000);
        assert!(
            added.texts.iter().all(|t| t.starts_with("passage: ")),
            "every embedding input carries the document prefix: {:?}",
            added.texts
        );
        let vectors: Vec<f32> = added.texts.iter().flat_map(|t| hash_vector(t)).collect();
        builder.set_vectors("notes/hydration.md", &vectors).unwrap();
        let built = builder.finish(1_700_000_000).unwrap();
        let artifact = open_artifact(&built.manifest, built.shards);

        let unprefixed = HashEmbedder {
            document_prefix: "",
            query_prefix: "",
            ..e5
        };
        let Err(err) = index::builder::ArtifactBuilder::reusing(
            spec_for(&unprefixed, "mean", "vault"),
            artifact,
        ) else {
            panic!("an artifact embedded with another document prefix must not be reusable");
        };
        assert!(format!("{err:#}").contains("passage: "), "got {err:#}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The same gate on a real corpus, with a real model and real queries.
    ///
    /// Ignored by default: it needs an indexed vault and will download the
    /// embedding model. Point it at one and run it:
    ///
    /// ```text
    /// GNOSIS_PARITY_VAULT=/path/to/vault \
    ///   cargo test --release -p gnosis-cli -- --ignored parity_on_a_real_corpus --nocapture
    /// ```
    ///
    /// The three-document fixture above cannot show what a corpus does: IDF is
    /// meaningless at that size, the over-fetch never binds, and no term is
    /// ambiguous. This runs the queries through both stores at scale and
    /// reports how far apart they land.
    ///
    /// **What this can and cannot assert.** When the lexical candidate pool is
    /// not truncated the two agree *exactly*: measured on scifact with
    /// `LEXICAL_OVERFETCH` raised so it never binds, 300/300 identical top-10s
    /// and a largest score difference of 0.000e0. At the shipped over-fetch
    /// they keep different pools — SQLite orders by FTS5's own `bm25()` before
    /// its `LIMIT`, this orders by BM25+ itself — so near-ties at the
    /// truncation boundary fall differently and agreement drops to 226/300.
    ///
    /// So the assertion is relative rather than absolute: the artifact must be
    /// no further from the database's exact ranking than the database's *own*
    /// ANN index is, which on the same corpus agrees on 137/300. An artifact
    /// that ranked worse than the approximation the CLI already ships with
    /// would not be fit to serve a client.
    #[test]
    #[ignore = "needs an indexed vault; downloads a model"]
    fn parity_on_a_real_corpus() {
        let Ok(vault) = std::env::var("GNOSIS_PARITY_VAULT") else {
            panic!("set GNOSIS_PARITY_VAULT to an indexed vault");
        };
        let vault = PathBuf::from(vault);
        let ws = crate::workspace::Workspace::resolve(false, Some(&vault.join("gnosis.toml")))
            .expect("resolving the vault's workspace");

        // Compare against the database's *exact* ranking. A database with an
        // ANN index answers approximately — f16 HNSW, by default since PR #34
        // — so comparing the artifact's brute force against it would measure
        // that index's recall rather than the artifact's fidelity. Copying the
        // `.db` alone into an empty directory leaves no `index/` beside it,
        // which is how `SqliteStore` falls back to brute force.
        let exact_dir = temp_dir("parity-exact");
        std::fs::copy(&ws.db_path, exact_dir.join("gnosis.db")).expect("copying the index");
        let store = SqliteStore::open(&exact_dir.join("gnosis.db")).expect("opening the copy");
        let approximate = SqliteStore::open(&ws.db_path).expect("opening the index");

        let (manifest, shards) = build_artifact(
            &store,
            &ws.config.embed.text.model,
            Quantization::F32,
            Compression::Deflate,
            Space::ALL,
            256,
        )
        .expect("building the artifact");
        let artifact = ArtifactStore::new(open_artifact(&manifest, shards));

        let queries: Vec<String> = std::env::var("GNOSIS_PARITY_QUERIES")
            .map(|path| {
                std::fs::read_to_string(path)
                    .expect("reading the query file")
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_else(|_| {
                vec![
                    "what is the effect of the treatment".to_string(),
                    "cell migration".to_string(),
                    "bone mineral density".to_string(),
                ]
            });

        let mut embedders = QueryEmbedders::new(&ws);
        let cfg = embedders.config().clone();
        let mut ask = |store: &dyn Store, q: &str| -> Vec<search::Hit> {
            query::run_query(
                store,
                &mut embedders,
                q,
                &[Space::Text],
                10,
                &TextQuery::default(),
                &cfg,
            )
            .expect("query")
        };

        let strip = |hits: &[search::Hit]| -> Vec<String> {
            let prefix = format!("{}/", vault.display());
            hits.iter()
                .map(|h| h.path.trim_start_matches(&prefix).to_string())
                .collect()
        };

        let mut agreed = 0usize;
        let mut ann_agreed = 0usize;
        let mut ann_overlap = 0usize;
        let mut worst_score_gap = 0f32;
        let mut disagreements: Vec<String> = Vec::new();
        for q in &queries {
            let exact = ask(&store, q);
            let from_artifact = ask(&artifact, q);
            let from_ann = ask(&approximate, q);

            let exact_paths = strip(&exact);
            let art_paths: Vec<String> = from_artifact.iter().map(|h| h.path.clone()).collect();
            let ann_paths = strip(&from_ann);
            for (a, b) in exact.iter().zip(&from_artifact) {
                worst_score_gap = worst_score_gap.max((a.score - b.score).abs());
            }
            if exact_paths == art_paths {
                agreed += 1;
            } else if disagreements.len() < 3 {
                disagreements.push(format!(
                    "  '{q}'\n    exact:    {exact_paths:?}\n    artifact: {art_paths:?}"
                ));
            }
            ann_agreed += usize::from(ann_paths == exact_paths);
            ann_overlap += ann_paths.iter().filter(|p| exact_paths.contains(p)).count();
        }
        let n = queries.len();
        println!(
            "artifact vs the database's exact ranking: identical top-10 {agreed}/{n}, \
             largest score gap {worst_score_gap:.3e}"
        );
        println!(
            "for reference, the database's own ANN index vs the same exact ranking: \
             identical top-10 {ann_agreed}/{n}, recall@10 {:.4}",
            ann_overlap as f64 / (10 * n) as f64
        );
        for d in &disagreements {
            println!("{d}");
        }
        let _ = std::fs::remove_dir_all(exact_dir);
        assert!(
            agreed >= ann_agreed,
            "the artifact agreed with the exact ranking on {agreed}/{n} queries, worse than the \
             database's own ANN index at {ann_agreed}/{n}"
        );
    }

    /// An index with image embedding switched off must not grow an empty image
    /// space, and asking for one explicitly must not invent it.
    #[test]
    fn an_index_without_images_exports_only_the_text_space() {
        let dir = temp_dir("noimages");
        let store = seeded_store(&dir);
        let (manifest, _) = build_one(&store, Quantization::F32, Compression::Deflate);
        assert_eq!(manifest.spaces.len(), 1);
        assert!(manifest.space(Space::Image).is_none());

        let (explicit, _) = build_artifact(
            &store,
            "bge-small-en-v1.5",
            Quantization::F32,
            Compression::Deflate,
            &[Space::Image, Space::Text],
            256,
        )
        .unwrap();
        assert!(explicit.space(Space::Image).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
