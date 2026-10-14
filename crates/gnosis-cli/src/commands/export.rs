use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use store::artifact::{
    ARTIFACT_FORMAT_VERSION, ArtifactChunk, DocumentEntry, Manifest, ModelInfo, Quantization,
    ShardInfo, shard_hash, write_shard,
};
use store::{Space, Store};

use crate::embedder::resolve_pooling;
use crate::store::SqliteStore;
use crate::workspace::{Workspace, expand_tilde};

/// The shard file every v1 export writes. Sharding by document group is the
/// next step; the manifest already carries a list, so adding shards is not a
/// format change.
const SHARD_NAME: &str = "shard-0000.bin";

const MANIFEST_NAME: &str = "manifest.json";

/// Write a portable, query-only copy of the index.
#[derive(Debug, clap::Args)]
pub struct ExportArgs {
    /// Directory to write the artifact into (created if missing).
    #[arg(long)]
    pub out: PathBuf,
    /// Vector precision in the artifact. `f32` is exact; `i8` is ~4x smaller.
    #[arg(long, default_value = "f32")]
    pub quantize: String,
    /// Overwrite an existing artifact in `--out`.
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

    let out = expand_tilde(&args.out);
    let manifest_path = out.join(MANIFEST_NAME);
    if manifest_path.exists() && !args.force {
        bail!(
            "{} already holds an artifact; pass --force to overwrite",
            out.display()
        );
    }

    let store = SqliteStore::open(&ws.db_path)?;
    let (manifest, shard) = build_artifact(&store, &ws.config.embed.text.model, quantization)?;

    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let shard_path = out.join(SHARD_NAME);
    std::fs::write(&shard_path, &shard)
        .with_context(|| format!("writing {}", shard_path.display()))?;
    // Manifest last: it is what names and checksums the shards, so a run
    // interrupted mid-write leaves the old manifest rather than one pointing at
    // a half-written shard.
    std::fs::write(&manifest_path, manifest.to_json()?)
        .with_context(|| format!("writing {}", manifest_path.display()))?;

    let bytes = shard.len() + manifest_path.metadata().map(|m| m.len() as usize).unwrap_or(0);
    println!(
        "Exported {} documents, {} chunks ({}) to {}",
        manifest.documents.len(),
        manifest.shards.iter().map(|s| s.chunks).sum::<usize>(),
        human_bytes(bytes),
        out.display()
    );
    println!(
        "  model {} ({} pooling, {}d), vectors {}",
        manifest.model.id, manifest.model.pooling, manifest.model.dim, args.quantize
    );
    Ok(())
}

/// Assemble a manifest and its shard from a store, without touching the
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
) -> Result<(Manifest, Vec<u8>)> {
    let model = model_info(store, configured_model)?;

    let docs = store.export_documents()?;
    if docs.is_empty() {
        bail!("the index holds no documents — nothing to export");
    }
    // Document order fixes the indices chunks refer to, so it is decided once
    // here and never recomputed.
    let doc_index: HashMap<i64, usize> =
        docs.iter().enumerate().map(|(i, d)| (d.id, i)).collect();

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

    let documents: Vec<DocumentEntry> = docs
        .iter()
        .map(|doc| DocumentEntry {
            root: root_index[doc.root.as_str()],
            path: doc.rel_path.clone(),
            title: doc.title.clone(),
            kind: doc.kind.clone(),
            content_hash: hex(&doc.content_hash),
            mtime: doc.mtime,
            links: doc.links.clone(),
            tags: doc.tags.clone(),
        })
        .collect();

    // Text only in v1. Image chunks are 512d CLIP vectors, so a mixed artifact
    // needs a per-space dim and that is its own step.
    let mut chunks: Vec<ArtifactChunk> = Vec::new();
    for chunk in store.export_chunks(Space::Text)? {
        let Some(&index) = doc_index.get(&chunk.doc_id) else {
            // A chunk whose document vanished between the two queries. Not
            // expected inside one process, but skipping beats writing a
            // dangling index into the artifact.
            continue;
        };
        if chunk.vector.len() != model.dim {
            bail!(
                "chunk of document {} has {} dimensions but the index records {} — \
                 run `gnosis rebuild`",
                documents[index].path,
                chunk.vector.len(),
                model.dim
            );
        }
        chunks.push(ArtifactChunk {
            doc_index: index,
            ord: chunk.ord,
            space: chunk.space,
            modality: chunk.modality,
            heading_path: chunk.heading_path,
            text: chunk.text,
            vector: chunk.vector,
        });
    }
    if chunks.is_empty() {
        bail!("the index holds no text vectors — nothing to export");
    }

    let shard = write_shard(&chunks, model.dim, quantization)?;
    let manifest = Manifest {
        format_version: ARTIFACT_FORMAT_VERSION,
        created_unix: now_unix(),
        model,
        quantization,
        roots,
        documents,
        shards: vec![ShardInfo {
            name: SHARD_NAME.to_string(),
            hash: shard_hash(&shard),
            chunks: chunks.len(),
        }],
    };

    Ok((manifest, shard))
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

    fn open_artifact(manifest: &Manifest, shard: Vec<u8>) -> Artifact {
        Artifact::open(
            &manifest.to_json().unwrap(),
            &[(SHARD_NAME.to_string(), shard)],
        )
        .expect("artifact should open")
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

        let (manifest, shard) =
            build_artifact(&store, "bge-small-en-v1.5", Quantization::F32).unwrap();
        let artifact = open_artifact(&manifest, shard);
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
        let (manifest, shard) =
            build_artifact(&store, "bge-small-en-v1.5", Quantization::F32).unwrap();
        let artifact = open_artifact(&manifest, shard);

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
        let (manifest, _) = build_artifact(&store, "bge-small-en-v1.5", Quantization::F32).unwrap();

        assert_eq!(manifest.roots, vec!["vault".to_string()], "roots are labels");
        for doc in &manifest.documents {
            assert!(!doc.path.starts_with('/'), "{} is absolute", doc.path);
        }
        let json = manifest.to_json().unwrap();
        assert!(!json.contains("/vault"), "an absolute path leaked into the manifest");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn artifact_carries_the_models_pooling_and_each_documents_hash() {
        let dir = temp_dir("meta");
        let store = seeded_store(&dir);
        let (manifest, _) = build_artifact(&store, "bge-small-en-v1.5", Quantization::F32).unwrap();

        assert_eq!(manifest.model.id, "bge-small-en-v1.5");
        assert_eq!(
            manifest.model.pooling, "cls",
            "a client that guesses pooling produces plausible, wrong vectors"
        );
        assert_eq!(manifest.model.dim, 4);
        for doc in &manifest.documents {
            assert_eq!(doc.content_hash, "68617368", "blake3 hex of the source");
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
        let (manifest, _) = build_artifact(&store, "bge-small-en-v1.5", Quantization::F32).unwrap();
        for doc in &manifest.documents {
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
        let err = build_artifact(&store, "bge-small-en-v1.5", Quantization::F32)
            .expect_err("should refuse");
        assert!(err.to_string().contains("some-future-model"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn i8_export_is_smaller_and_still_ranks_the_same_order() {
        let dir = temp_dir("i8");
        let store = seeded_store(&dir);
        let query = [1.0, 0.0, 0.0, 0.0];

        let (_, f32_shard) =
            build_artifact(&store, "bge-small-en-v1.5", Quantization::F32).unwrap();
        let (manifest, i8_shard) =
            build_artifact(&store, "bge-small-en-v1.5", Quantization::I8).unwrap();
        assert!(
            i8_shard.len() < f32_shard.len(),
            "i8 {} should be smaller than f32 {}",
            i8_shard.len(),
            f32_shard.len()
        );

        let artifact = open_artifact(&manifest, i8_shard);
        let hits = search::rank(&query, artifact.candidates(Space::Text), 10);
        let order: Vec<&str> = hits.iter().map(|h| h.path.as_str()).collect();
        assert_eq!(order, vec!["near.md", "mid.md", "deep/far.md"]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
