use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use store::artifact::{
    ARTIFACT_FORMAT_VERSION, ArtifactChunk, Compression, DocumentEntry, Manifest, ModelInfo,
    Quantization, ShardInfo, assign_shard, shard_hash, shard_name, write_shard,
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
    /// Vector precision in the artifact. `f32` is exact; `i8` is ~4x smaller.
    #[arg(long, default_value = "f32")]
    pub quantize: String,
    /// Compression for the artifact's text. `deflate` roughly halves it;
    /// `none` leaves a shard's text greppable.
    #[arg(long, default_value = "deflate")]
    pub compress: String,
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
        for stale in old.shards.iter().filter(|s| !current.contains(s.name.as_str())) {
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
    println!(
        "  model {} ({} pooling, {}d), vectors {}, text {}",
        manifest.model.id,
        manifest.model.pooling,
        manifest.model.dim,
        args.quantize,
        args.compress
    );
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
/// change replaces the vectors, quantization and compression change their
/// encoding, and `shard_docs` moves documents between shards. Missing one would
/// not fail loudly — it would leave stale shards in place and quietly serve an
/// artifact that mixes two encodings.
fn carries_over(old: &Manifest, new: &Manifest) -> bool {
    old.model == new.model
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
    shard_docs: usize,
) -> Result<(Manifest, Vec<ShardBody>)> {
    let model = model_info(store, configured_model)?;

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
            links: doc.links.clone(),
            tags: doc.tags.clone(),
        });
        placement.insert(doc.id, (shard, local));
    }

    // Text only so far. Image chunks are 512d CLIP vectors, so a mixed artifact
    // needs a per-space dim and that is its own step.
    let mut shard_chunks: Vec<Vec<ArtifactChunk>> = vec![Vec::new(); shard_count];
    let mut total_chunks = 0usize;
    for chunk in store.export_chunks(Space::Text)? {
        let Some(&(shard, local)) = placement.get(&chunk.doc_id) else {
            // A chunk whose document vanished between the two queries. Not
            // expected inside one process, but skipping beats writing a
            // dangling index into the artifact.
            continue;
        };
        if chunk.vector.len() != model.dim {
            bail!(
                "chunk of document {} has {} dimensions but the index records {} — \
                 run `gnosis rebuild`",
                shard_docs_out[shard][local].path,
                chunk.vector.len(),
                model.dim
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
        total_chunks += 1;
    }
    if total_chunks == 0 {
        bail!("the index holds no text vectors — nothing to export");
    }

    let mut infos = Vec::with_capacity(shard_count);
    let mut bodies = Vec::with_capacity(shard_count);
    for i in 0..shard_count {
        let body = write_shard(
            &shard_docs_out[i],
            &shard_chunks[i],
            model.dim,
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
        model,
        quantization,
        compression,
        roots,
        shard_docs,
        shards: infos,
    };
    Ok((manifest, bodies))
}

/// Which model built this index, and how a client reproduces a query vector in
/// its space.
///
/// The model id comes from `meta` rather than from config: config can be edited
/// after indexing, and what matters is what actually produced the stored
/// vectors.
fn model_info(store: &SqliteStore, configured: &str) -> Result<ModelInfo> {
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
    Ok(ModelInfo {
        id,
        pooling: pooling.to_string(),
        dim,
    })
}

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
    use store::{ChunkWrite, DocWrite, TextQuery};

    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
        let dir = std::env::temp_dir().join(format!(
            "gnosis-export-{tag}-{}-{nanos}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A store with three documents whose vectors are known, so the expected
    /// ranking is obvious rather than whatever the model happens to produce.
    fn seeded_store(dir: &std::path::Path) -> SqliteStore {
        let mut store = SqliteStore::open(&dir.join("gnosis.db")).unwrap();
        store.set_meta("model.text", "bge-small-en-v1.5").unwrap();
        store.set_meta("dim.text", "4").unwrap();

        let docs: [(&str, &str, Vec<f32>, &str); 3] = [
            ("/vault/near.md", "Near", vec![1.0, 0.0, 0.0, 0.0], "alpha beta"),
            ("/vault/mid.md", "Mid", vec![0.7, 0.7, 0.0, 0.0], "beta gamma"),
            ("/vault/deep/far.md", "Far", vec![0.0, 1.0, 0.0, 0.0], "gamma delta"),
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

    fn open_artifact(manifest: &Manifest, shards: Vec<ShardBody>) -> Artifact {
        Artifact::open(&manifest.to_json().unwrap(), &shards).expect("artifact should open")
    }

    /// One shard, so a test that does not care about sharding stays readable.
    fn build_one(
        store: &SqliteStore,
        quantization: Quantization,
        compression: Compression,
    ) -> (Manifest, Vec<ShardBody>) {
        build_artifact(store, "bge-small-en-v1.5", quantization, compression, 256).unwrap()
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

        let (manifest, shards) =
            build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = open_artifact(&manifest, shards);
        let from_artifact = search::rank(&query, artifact.candidates(Space::Text), 10);

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
        let (manifest, shards) =
            build_one(&store, Quantization::F32, Compression::Deflate);
        let artifact = open_artifact(&manifest, shards);

        let mut vectors: Vec<Vec<f32>> = artifact
            .chunks()
            .iter()
            .map(|c| c.vector.clone())
            .collect();
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
        assert_eq!(manifest.roots, vec!["vault".to_string()], "roots are labels");
        let json = manifest.to_json().unwrap();
        assert!(!json.contains("/vault"), "an absolute path leaked into the manifest");

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

        assert_eq!(manifest.model.id, "bge-small-en-v1.5");
        assert_eq!(
            manifest.model.pooling, "cls",
            "a client that guesses pooling produces plausible, wrong vectors"
        );
        assert_eq!(manifest.model.dim, 4);
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
        let model = model_info(&store, "all-MiniLM-L6-v2").unwrap();
        assert_eq!(model.id, "bge-small-en-v1.5");
        assert_eq!(model.pooling, "cls");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn export_refuses_a_model_whose_pooling_is_unknown() {
        let dir = temp_dir("unknown");
        let store = seeded_store(&dir);
        store.set_meta("model.text", "some-future-model").unwrap();
        let err = build_artifact(&store, "bge-small-en-v1.5", Quantization::F32, Compression::Deflate, 256)
            .expect_err("should refuse");
        assert!(err.to_string().contains("some-future-model"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn documents_are_split_across_shards_and_still_all_present() {
        let dir = temp_dir("shards");
        let store = seeded_store(&dir);
        // One document per shard, so the split is unambiguous.
        let (manifest, shards) =
            build_artifact(&store, "bge-small-en-v1.5", Quantization::F32, Compression::Deflate, 1).unwrap();

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
        let mut paths: Vec<&str> = artifact.documents().iter().map(|d| d.path.as_str()).collect();
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
            let (manifest, shards) =
                build_artifact(&store, "bge-small-en-v1.5", Quantization::F32, Compression::Deflate, shard_docs).unwrap();
            let artifact = open_artifact(&manifest, shards);
            let hits = search::rank(&query, artifact.candidates(Space::Text), 10);
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
        let (_, before) = build_artifact(&store, "bge-small-en-v1.5", Quantization::F32, Compression::Deflate, 2).unwrap();
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

        let (manifest, after) =
            build_artifact(&store, "bge-small-en-v1.5", Quantization::F32, Compression::Deflate, 2).unwrap();
        assert_eq!(after.len(), 2, "four documents at two per shard is still two shards");

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
        let (_, before) = build_artifact(&store, "bge-small-en-v1.5", Quantization::F32, Compression::Deflate, 1).unwrap();

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

        let (_, after) = build_artifact(&store, "bge-small-en-v1.5", Quantization::F32, Compression::Deflate, 1).unwrap();
        assert_eq!(before.len(), after.len(), "shard count should not change");

        let changed = before
            .iter()
            .zip(&after)
            .filter(|((_, a), (_, b))| a != b)
            .count();
        assert_eq!(
            changed, 1,
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

        let (_, f32_shards) =
            build_one(&store, Quantization::F32, Compression::Deflate);
        let (manifest, i8_shards) =
            build_one(&store, Quantization::I8, Compression::Deflate);
        let i8_bytes: usize = i8_shards.iter().map(|(_, b)| b.len()).sum();
        let f32_bytes: usize = f32_shards.iter().map(|(_, b)| b.len()).sum();
        assert!(
            i8_bytes < f32_bytes,
            "i8 {i8_bytes} should be smaller than f32 {f32_bytes}"
        );

        let artifact = open_artifact(&manifest, i8_shards);
        let hits = search::rank(&query, artifact.candidates(Space::Text), 10);
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
            search::rank(&query, a.candidates(Space::Text), 10)
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
        other.model.id = "multilingual-e5-small".to_string();
        assert!(!carries_over(&base, &other), "the model must invalidate");

        let _ = std::fs::remove_dir_all(dir);
    }
}
