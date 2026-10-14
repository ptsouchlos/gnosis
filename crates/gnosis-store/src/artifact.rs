//! The portable index artifact: a query-only, derived view of a `gnosis.db`.
//!
//! Written by `gnosis export`, read by a second machine or by the Obsidian
//! plugin. This module is deliberately free of native dependencies and of file
//! I/O — it turns bytes into chunks and chunks into bytes — so the same code
//! compiles to `wasm32` and backs the plugin's reader. The caller supplies the
//! bytes, because a plugin reads them through Obsidian's `Vault` API and the
//! CLI reads them with `std::fs`.
//!
//! Design and rationale: `docs/gnosis/portable-index-export.md` in the
//! workspace repo. The decisions worth knowing while reading this file:
//!
//! - **No absolute paths.** Documents carry a vault-relative path and a root
//!   *label*, never a machine path, or the artifact would not be portable and
//!   would leak the author's directory layout.
//! - **The model's pooling is recorded.** A client that guesses wrong produces
//!   plausible unit-norm vectors that are wrong by ~55% of the usable cosine
//!   range for `multilingual-e5-small`, with nothing to signal it.
//! - **No ANN index and no FTS index.** Both are rebuildable client-side, and a
//!   linear scan over i8 vectors answers a query in ~1 ms over 20k chunks.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::Space;

/// Bumped whenever the manifest or shard layout changes incompatibly. A reader
/// refuses an unknown version rather than mis-parsing one.
///
/// **v3** deflates the two text-bearing blocks of a shard — the document JSON
/// and the string blob. Measured on scifact, text was 70% of an i8 artifact and
/// vectors 21%, so compression halves the artifact where quantization could not:
/// 12.1 MB to 5.9 MB. Records and vectors stay raw, see [`write_shard`].
///
/// **v2** moved per-document metadata out of the manifest and into the shard
/// that holds the document, and made shard assignment a hash of the document's
/// path. v1 kept every document in the manifest, which measured 270 bytes each
/// — 5.4 MB at 20k documents — and rewrote in full on every export, so sharding
/// the chunks alone would have saved nothing. The artifact is derived and
/// regenerable, so the fix was a version bump rather than a migration.
pub const ARTIFACT_FORMAT_VERSION: u32 = 3;

/// Shard file magic: "GNSA" (gnosis artifact).
const SHARD_MAGIC: &[u8; 4] = b"GNSA";

/// Byte length of the shard header.
const SHARD_HEADER_LEN: usize = 24;

/// Byte length of one per-chunk record in a shard.
const RECORD_LEN: usize = 32;

/// How a shard's vectors are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// Full precision, 4 bytes per dimension. Exact, and the only form a
    /// second machine can re-export faithfully.
    F32,
    /// 8-bit, 1 byte per dimension plus a per-chunk scale. ~4x smaller.
    I8,
}

impl Quantization {
    fn code(self) -> u8 {
        match self {
            Quantization::F32 => 0,
            Quantization::I8 => 1,
        }
    }

    fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Quantization::F32),
            1 => Ok(Quantization::I8),
            other => bail!("unknown vector quantization code {other} in shard"),
        }
    }

    fn bytes_per_dim(self) -> usize {
        match self {
            Quantization::F32 => 4,
            Quantization::I8 => 1,
        }
    }
}

/// How a shard's text-bearing blocks are stored.
///
/// Raw deflate, not gzip or zlib: the framing buys nothing here — a shard
/// already carries its own magic, version and a blake3 in the manifest — and
/// raw deflate is what a client without this crate can hand to the browser's
/// `DecompressionStream("deflate-raw")`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Compression {
    /// Stored as written. Leaves a shard's text greppable, which is worth
    /// something while debugging and nothing in a synced vault.
    #[default]
    None,
    Deflate,
}

impl Compression {
    fn code(self) -> u16 {
        match self {
            Compression::None => 0,
            Compression::Deflate => 1,
        }
    }

    fn from_code(code: u16) -> Result<Self> {
        match code {
            0 => Ok(Compression::None),
            1 => Ok(Compression::Deflate),
            other => bail!("unknown compression code {other} in shard"),
        }
    }

    fn pack(self, block: Vec<u8>) -> Vec<u8> {
        match self {
            Compression::None => block,
            // Level 6. Level 9 measured 0.3% smaller for 1.7x the time, and
            // zstd-19 (which would mean a C dependency in wasm) only 8%.
            Compression::Deflate => miniz_oxide::deflate::compress_to_vec(&block, 6),
        }
    }

    fn unpack(self, block: &[u8], what: &str) -> Result<Vec<u8>> {
        match self {
            Compression::None => Ok(block.to_vec()),
            Compression::Deflate => miniz_oxide::inflate::decompress_to_vec(block)
                .map_err(|e| anyhow::anyhow!("inflating a shard's {what} block: {e:?}")),
        }
    }
}

/// Which model produced the vectors, and how to reproduce a query vector in
/// the same space.
///
/// `pooling` is not decoration. fastembed pools CLS for the whole BGE family
/// and mean for every multilingual-E5 size, so a client cannot infer it from
/// the dimensionality or guess one value for all models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub pooling: String,
    pub dim: usize,
}

/// One document in the artifact. Paths are vault-relative; `root` indexes
/// [`Manifest::roots`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentEntry {
    pub root: usize,
    pub path: String,
    pub title: String,
    pub kind: String,
    /// blake3 of the source file, hex — the authority for staleness. A client
    /// that cannot afford to hash the vault compares `mtime` and says so.
    pub content_hash: String,
    pub mtime: i64,
    /// Wikilink targets, so a client's `related` can exclude already-linked
    /// notes the way the CLI does by default.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

/// One shard file, named in the manifest with the hash of its contents.
///
/// Names are stable (`shard-0000.bin`) rather than content-addressed: a
/// renamed shard reaches a file-level syncer as an add plus a delete, which is
/// the churn pattern they handle worst and which leaks garbage when a delete is
/// lost. The hash here is what tells a client the shard changed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardInfo {
    pub name: String,
    pub hash: String,
    pub documents: usize,
    pub chunks: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub created_unix: i64,
    pub model: ModelInfo,
    pub quantization: Quantization,
    /// How the shards' text blocks are stored. Each shard also states this in
    /// its own header, which is what the reader parses by; the manifest records
    /// it so a client can see what it will need before fetching a shard, and so
    /// a re-export at a different setting knows every shard must be rewritten.
    ///
    /// Defaulted on the way in only so that a superseded manifest parses far
    /// enough to reach the format-version check and its actionable error.
    #[serde(default)]
    pub compression: Compression,
    /// Root labels — a vault's final path component, never its absolute path.
    pub roots: Vec<String>,
    /// Target documents per shard, as asked for at export time. Recorded
    /// because it determines `shards.len()`, and changing it reassigns every
    /// document and so rewrites the whole artifact.
    pub shard_docs: usize,
    /// Every shard, in name order. Documents live inside their shard, not here:
    /// a manifest holding them all would rewrite on any change, which is what
    /// sharding exists to avoid.
    pub shards: Vec<ShardInfo>,
}

impl Manifest {
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serializing artifact manifest")
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let manifest: Manifest =
            serde_json::from_str(json).context("parsing artifact manifest")?;
        if manifest.format_version != ARTIFACT_FORMAT_VERSION {
            bail!(
                "artifact format version {} is not supported (this build reads version {}); \
                 re-export with a matching gnosis",
                manifest.format_version,
                ARTIFACT_FORMAT_VERSION
            );
        }
        Ok(manifest)
    }
}

/// One chunk on its way into or out of a shard.
#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactChunk {
    /// Index into **this shard's** document list, not a global one. Shards are
    /// self-contained so a client can load one without the others.
    pub doc_index: usize,
    pub ord: u32,
    pub space: Space,
    pub modality: String,
    pub heading_path: String,
    pub text: String,
    pub vector: Vec<f32>,
}

/// Modality is stored as a small code rather than a string per chunk: it
/// repeats on every row and only three values exist. Unknown values are
/// rejected rather than guessed at, since a wrong modality would silently mix
/// vector spaces — the defect P0 was about.
fn modality_code(modality: &str) -> Result<u8> {
    match modality {
        "text" => Ok(0),
        "image" => Ok(1),
        "text_title" => Ok(2),
        other => bail!("unknown chunk modality '{other}'"),
    }
}

fn modality_name(code: u8) -> Result<&'static str> {
    match code {
        0 => Ok("text"),
        1 => Ok("image"),
        2 => Ok("text_title"),
        other => bail!("unknown chunk modality code {other} in shard"),
    }
}

fn space_code(space: Space) -> u8 {
    match space {
        Space::Text => 0,
        Space::Image => 1,
    }
}

fn space_from_code(code: u8) -> Result<Space> {
    match code {
        0 => Ok(Space::Text),
        1 => Ok(Space::Image),
        other => bail!("unknown space code {other} in shard"),
    }
}

/// Quantize to i8 with a per-chunk scale, symmetric around zero.
///
/// The scale is per chunk rather than per dimension: one extra f32 per vector
/// against 384 of them, and S2 measured 0.983 recall@10 against exact search
/// this way.
fn quantize_i8(vector: &[f32]) -> (Vec<i8>, f32) {
    let max = vector.iter().fold(0f32, |m, x| m.max(x.abs()));
    if max <= f32::EPSILON {
        return (vec![0; vector.len()], 0.0);
    }
    let scale = max / 127.0;
    let quantized = vector
        .iter()
        .map(|x| (x / max * 127.0).round() as i8)
        .collect();
    (quantized, scale)
}

/// Serialize `chunks` into one shard.
///
/// Layout, all little-endian: a header, then the document block, then a
/// fixed-size record per chunk, then the vector block, then the string blob.
/// Vectors are contiguous and fixed-stride so a client can scan them without
/// touching the strings, which is the whole shape of a brute-force query.
///
/// `compression` applies to the document block and the string blob — the two
/// that hold text — and never to the records or the vectors. Measured on
/// scifact: deflate takes the string blob 8.20 MB to 2.75 MB (3.0x) and the
/// document JSON 1.34 MB to 0.50 MB, while i8 vectors give up only 1.33x, worth
/// 0.6 MB of a 12 MB artifact. Paying that 0.6 MB keeps the vector block a
/// fixed-stride array a client can scan in place, which is the one access
/// pattern that has to stay cheap. Each block is deflated whole rather than per
/// chunk, so chunks share a window; a shard is small enough to inflate on load.
///
/// The header's `docs_json_len` is the *stored* length, so the layout is
/// unchanged by compression: only the reader differs.
pub fn write_shard(
    documents: &[DocumentEntry],
    chunks: &[ArtifactChunk],
    dim: usize,
    quantization: Quantization,
    compression: Compression,
) -> Result<Vec<u8>> {
    let mut records = Vec::with_capacity(chunks.len() * RECORD_LEN);
    let mut vectors = Vec::with_capacity(chunks.len() * dim * quantization.bytes_per_dim());
    let mut strings: Vec<u8> = Vec::new();

    for chunk in chunks {
        if chunk.doc_index >= documents.len() {
            bail!(
                "chunk refers to document {} but this shard holds {}",
                chunk.doc_index,
                documents.len()
            );
        }
        if chunk.vector.len() != dim {
            bail!(
                "chunk vector has {} dimensions but the artifact declares {dim}",
                chunk.vector.len()
            );
        }

        let heading_off = strings.len() as u32;
        strings.extend_from_slice(chunk.heading_path.as_bytes());
        let text_off = strings.len() as u32;
        strings.extend_from_slice(chunk.text.as_bytes());

        let scale = match quantization {
            Quantization::F32 => {
                for value in &chunk.vector {
                    vectors.extend_from_slice(&value.to_le_bytes());
                }
                1.0f32
            }
            Quantization::I8 => {
                let (quantized, scale) = quantize_i8(&chunk.vector);
                vectors.extend(quantized.iter().map(|x| *x as u8));
                scale
            }
        };

        records.extend_from_slice(&(chunk.doc_index as u32).to_le_bytes());
        records.extend_from_slice(&chunk.ord.to_le_bytes());
        records.push(space_code(chunk.space));
        records.push(modality_code(&chunk.modality)?);
        records.extend_from_slice(&0u16.to_le_bytes()); // reserved, keeps records 4-aligned
        records.extend_from_slice(&heading_off.to_le_bytes());
        records.extend_from_slice(&(chunk.heading_path.len() as u32).to_le_bytes());
        records.extend_from_slice(&text_off.to_le_bytes());
        records.extend_from_slice(&(chunk.text.len() as u32).to_le_bytes());
        records.extend_from_slice(&scale.to_le_bytes());
    }
    debug_assert_eq!(records.len(), chunks.len() * RECORD_LEN);

    // Documents ride inside the shard as JSON at a known offset: self-contained
    // in one file with one hash, still readable with `head -c`, and small enough
    // next to the vector block that a binary encoding would not pay for itself.
    let docs_json = compression.pack(
        serde_json::to_vec(documents).context("serializing shard documents")?,
    );
    let strings = compression.pack(strings);

    let mut out = Vec::with_capacity(
        SHARD_HEADER_LEN + docs_json.len() + records.len() + vectors.len() + strings.len(),
    );
    out.extend_from_slice(SHARD_MAGIC);
    out.extend_from_slice(&(ARTIFACT_FORMAT_VERSION as u16).to_le_bytes());
    out.extend_from_slice(&compression.code().to_le_bytes());
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.push(quantization.code());
    out.extend_from_slice(&[0u8; 3]); // pad to a 4-byte boundary
    out.extend_from_slice(&(docs_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&docs_json);
    out.extend_from_slice(&records);
    out.extend_from_slice(&vectors);
    out.extend_from_slice(&strings);
    Ok(out)
}

/// Which shard a document belongs in, from a hash of its path.
///
/// Hashed rather than positional on purpose. With documents assigned by
/// position, inserting one note shifts every document after it and so rewrites
/// every later shard — which is most of the artifact, for the commonest edit a
/// vault sees. Hashing makes a document's shard independent of what else exists,
/// so adding, editing or deleting a note touches exactly one shard.
///
/// That independence is enforced by this signature rather than by any test:
/// taking only `(root, rel_path, shard_count)`, there is no corpus to depend on.
///
/// The cost is that `shard_count` is baked in: changing it reassigns everything.
/// That is why it is recorded in the manifest.
pub fn assign_shard(root: usize, rel_path: &str, shard_count: usize) -> usize {
    if shard_count <= 1 {
        return 0;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(root as u64).to_le_bytes());
    hasher.update(rel_path.as_bytes());
    let digest = hasher.finalize();
    let bucket = u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("8 bytes"));
    (bucket % shard_count as u64) as usize
}

/// A shard's file name. Zero-padded so a directory listing sorts in shard
/// order.
pub fn shard_name(index: usize) -> String {
    format!("shard-{index:04}.bin")
}

/// blake3 of a shard's bytes, hex — what the manifest records so a client can
/// tell a changed shard from an unchanged one.
pub fn shard_hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Parse one shard back into its documents and chunks.
pub fn read_shard(bytes: &[u8]) -> Result<(Vec<DocumentEntry>, Vec<ArtifactChunk>)> {
    if bytes.len() < SHARD_HEADER_LEN {
        bail!("shard is truncated: {} bytes", bytes.len());
    }
    if &bytes[0..4] != SHARD_MAGIC {
        bail!("not a gnosis artifact shard (bad magic)");
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]) as u32;
    if version != ARTIFACT_FORMAT_VERSION {
        bail!(
            "shard format version {version} is not supported (this build reads version {})",
            ARTIFACT_FORMAT_VERSION
        );
    }
    let compression = Compression::from_code(u16::from_le_bytes([bytes[6], bytes[7]]))?;
    let count = u32_at(bytes, 8) as usize;
    let dim = u32_at(bytes, 12) as usize;
    let quantization = Quantization::from_code(bytes[16])?;
    let docs_json_len = u32_at(bytes, 20) as usize;

    let docs_at = SHARD_HEADER_LEN;
    let docs_end = docs_at
        .checked_add(docs_json_len)
        .filter(|end| *end <= bytes.len())
        .context("shard's document block runs past the end of the file")?;
    let docs_json = compression.unpack(&bytes[docs_at..docs_end], "document")?;
    let documents: Vec<DocumentEntry> =
        serde_json::from_slice(&docs_json).context("parsing a shard's document block")?;

    let records_at = docs_end;
    let vectors_at = records_at + count * RECORD_LEN;
    let strings_at = vectors_at + count * dim * quantization.bytes_per_dim();
    if bytes.len() < strings_at {
        bail!(
            "shard is truncated: {} bytes, expected at least {strings_at} for {count} chunks",
            bytes.len()
        );
    }
    let strings = compression.unpack(&bytes[strings_at..], "string")?;
    let strings = strings.as_slice();

    let mut chunks = Vec::with_capacity(count);
    for i in 0..count {
        let r = records_at + i * RECORD_LEN;
        let doc_index = u32_at(bytes, r) as usize;
        let ord = u32_at(bytes, r + 4);
        let space = space_from_code(bytes[r + 8])?;
        let modality = modality_name(bytes[r + 9])?;
        let heading_off = u32_at(bytes, r + 12) as usize;
        let heading_len = u32_at(bytes, r + 16) as usize;
        let text_off = u32_at(bytes, r + 20) as usize;
        let text_len = u32_at(bytes, r + 24) as usize;
        let scale = f32::from_le_bytes([
            bytes[r + 28],
            bytes[r + 29],
            bytes[r + 30],
            bytes[r + 31],
        ]);

        let heading_path = read_str(strings, heading_off, heading_len, "heading_path")?;
        let text = read_str(strings, text_off, text_len, "text")?;

        let v = vectors_at + i * dim * quantization.bytes_per_dim();
        let vector = match quantization {
            Quantization::F32 => (0..dim)
                .map(|d| f32::from_le_bytes([
                    bytes[v + d * 4],
                    bytes[v + d * 4 + 1],
                    bytes[v + d * 4 + 2],
                    bytes[v + d * 4 + 3],
                ]))
                .collect(),
            Quantization::I8 => (0..dim)
                .map(|d| (bytes[v + d] as i8) as f32 * scale)
                .collect(),
        };

        if doc_index >= documents.len() {
            bail!(
                "shard chunk refers to document {doc_index} but the shard holds {}",
                documents.len()
            );
        }
        chunks.push(ArtifactChunk {
            doc_index,
            ord,
            space,
            modality: modality.to_string(),
            heading_path,
            text,
            vector,
        });
    }
    Ok((documents, chunks))
}

fn read_str(strings: &[u8], offset: usize, len: usize, what: &str) -> Result<String> {
    let end = offset
        .checked_add(len)
        .filter(|end| *end <= strings.len())
        .with_context(|| format!("{what} runs past the end of the shard's string blob"))?;
    String::from_utf8(strings[offset..end].to_vec())
        .with_context(|| format!("{what} is not valid UTF-8"))
}

/// A manifest plus its shards' documents and chunks, ready to query.
#[derive(Debug)]
pub struct Artifact {
    manifest: Manifest,
    documents: Vec<DocumentEntry>,
    chunks: Vec<ArtifactChunk>,
}

impl Artifact {
    /// Assemble from bytes the caller has already fetched: the manifest as
    /// JSON, then each shard by name.
    ///
    /// Shards are verified against the manifest's hashes. A mismatch is an
    /// error rather than a warning: a half-synced artifact would otherwise
    /// answer queries from a mixture of two indexes.
    ///
    /// Each shard numbers its documents from zero; this rebases them onto one
    /// list, so `ArtifactChunk::doc_index` on an assembled `Artifact` indexes
    /// [`Artifact::documents`] rather than any one shard.
    pub fn open(manifest_json: &str, shards: &[(String, Vec<u8>)]) -> Result<Self> {
        let manifest = Manifest::from_json(manifest_json)?;
        let supplied: HashMap<&str, &Vec<u8>> =
            shards.iter().map(|(n, b)| (n.as_str(), b)).collect();

        let mut documents: Vec<DocumentEntry> = Vec::new();
        let mut chunks: Vec<ArtifactChunk> = Vec::new();
        for info in &manifest.shards {
            let bytes = supplied
                .get(info.name.as_str())
                .with_context(|| format!("artifact shard '{}' was not supplied", info.name))?;
            let actual = shard_hash(bytes);
            if actual != info.hash {
                bail!(
                    "artifact shard '{}' does not match the manifest (expected {}, got {actual}) \
                     — the artifact may be partially synced",
                    info.name,
                    info.hash
                );
            }
            let (shard_docs, shard_chunks) = read_shard(bytes)
                .with_context(|| format!("reading artifact shard '{}'", info.name))?;
            if shard_docs.len() != info.documents || shard_chunks.len() != info.chunks {
                bail!(
                    "artifact shard '{}' holds {} documents and {} chunks but the manifest says \
                     {} and {}",
                    info.name,
                    shard_docs.len(),
                    shard_chunks.len(),
                    info.documents,
                    info.chunks
                );
            }
            let base = documents.len();
            documents.extend(shard_docs);
            chunks.extend(shard_chunks.into_iter().map(|mut c| {
                c.doc_index += base;
                c
            }));
        }
        Ok(Self {
            manifest,
            documents,
            chunks,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Every document across every shard, in shard order.
    pub fn documents(&self) -> &[DocumentEntry] {
        &self.documents
    }

    pub fn chunks(&self) -> &[ArtifactChunk] {
        &self.chunks
    }

    /// Candidates for one space, ready for `search::rank` and the rest of the
    /// ranking code the CLI uses.
    ///
    /// This is the reuse surface the plugin exists on: scoring an artifact goes
    /// through the same `gnosis-search` functions as scoring the database, so
    /// the two cannot drift. `path` is **vault-relative** and `source_root` is
    /// a label, not a machine path — a client joins them to whatever the vault
    /// is called locally.
    pub fn candidates(&self, space: Space) -> Vec<search::Candidate> {
        self.chunks
            .iter()
            .filter(|c| c.space == space)
            .filter_map(|c| {
                let doc = self.documents.get(c.doc_index)?;
                let root = self.manifest.roots.get(doc.root).cloned().unwrap_or_default();
                Some(search::Candidate {
                    path: doc.path.clone(),
                    title: doc.title.clone(),
                    source_root: root,
                    heading_path: c.heading_path.clone(),
                    text: c.text.clone(),
                    vector: c.vector.clone(),
                    width: None,
                    height: None,
                    modality: c.modality.clone(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(doc_index: usize, heading: &str, text: &str, vector: Vec<f32>) -> ArtifactChunk {
        ArtifactChunk {
            doc_index,
            ord: 0,
            space: Space::Text,
            modality: "text".to_string(),
            heading_path: heading.to_string(),
            text: text.to_string(),
            vector,
        }
    }

    fn doc(path: &str, title: &str) -> DocumentEntry {
        DocumentEntry {
            root: 0,
            path: path.to_string(),
            title: title.to_string(),
            kind: "markdown".to_string(),
            content_hash: "00".to_string(),
            mtime: 0,
            links: Vec::new(),
            tags: Vec::new(),
        }
    }

    fn manifest(shards: Vec<ShardInfo>, quantization: Quantization) -> Manifest {
        Manifest {
            format_version: ARTIFACT_FORMAT_VERSION,
            created_unix: 0,
            model: ModelInfo {
                id: "bge-small-en-v1.5".to_string(),
                pooling: "cls".to_string(),
                dim: 2,
            },
            quantization,
            compression: Compression::None,
            roots: vec!["vault".to_string()],
            shard_docs: 256,
            shards,
        }
    }

    fn one_shard(documents: &[DocumentEntry], chunks: &[ArtifactChunk]) -> (Manifest, Vec<u8>) {
        let bytes = write_shard(documents, chunks, 2, Quantization::F32, Compression::None).expect("write");
        let m = manifest(
            vec![ShardInfo {
                name: shard_name(0),
                hash: shard_hash(&bytes),
                documents: documents.len(),
                chunks: chunks.len(),
            }],
            Quantization::F32,
        );
        (m, bytes)
    }

    #[test]
    fn f32_shard_round_trips_exactly() {
        let docs = vec![doc("a.md", "A"), doc("b.md", "B")];
        let chunks = vec![
            chunk(0, "Heading", "first body", vec![1.0, 0.0]),
            chunk(1, "", "second body", vec![0.0, 1.0]),
        ];
        let bytes = write_shard(&docs, &chunks, 2, Quantization::F32, Compression::None).expect("write");
        let (back_docs, back_chunks) = read_shard(&bytes).expect("read");
        assert_eq!(back_docs, docs);
        assert_eq!(
            back_chunks, chunks,
            "f32 is the exact form — a round trip must change nothing at all"
        );
    }

    #[test]
    fn i8_shard_round_trips_within_quantization_error() {
        let docs = vec![doc("a.md", "A")];
        let chunks = vec![chunk(0, "", "body", vec![0.6, -0.8])];
        let bytes = write_shard(&docs, &chunks, 2, Quantization::I8, Compression::None).expect("write");
        let (_, back) = read_shard(&bytes).expect("read");
        for (original, restored) in chunks[0].vector.iter().zip(&back[0].vector) {
            assert!(
                (original - restored).abs() < 0.01,
                "i8 round trip moved {original} to {restored}"
            );
        }
        assert_eq!(back[0].text, "body", "metadata must survive exactly");
    }

    #[test]
    fn i8_saves_exactly_three_bytes_per_dimension() {
        let docs = vec![doc("a.md", "A")];
        let chunks: Vec<ArtifactChunk> = (0..16)
            .map(|i| {
                let mut c = chunk(0, "", "x", vec![0.5; 384]);
                c.ord = i;
                c
            })
            .collect();
        let f32_len = write_shard(&docs, &chunks, 384, Quantization::F32, Compression::None).unwrap().len();
        let i8_len = write_shard(&docs, &chunks, 384, Quantization::I8, Compression::None).unwrap().len();
        assert_eq!(f32_len - i8_len, 16 * 384 * 3);
    }

    /// Prose, so the ratio means something: 24 chunks of repeated vocabulary is
    /// the shape a vault actually has.
    fn prose_shard(compression: Compression) -> Vec<u8> {
        let docs = vec![doc("a.md", "A")];
        let chunks: Vec<ArtifactChunk> = (0..24)
            .map(|i| {
                let mut c = chunk(
                    0,
                    "Notes > Retrieval",
                    "The lexical channel scores a chunk with BM25+ over the same tokens \
                     the corpus statistics were gathered with, so the client's scores \
                     agree with the indexer's by construction.",
                    vec![0.5; 384],
                );
                c.ord = i;
                c
            })
            .collect();
        write_shard(&docs, &chunks, 384, Quantization::I8, compression).expect("write")
    }

    #[test]
    fn deflate_shard_round_trips_exactly() {
        let docs = vec![doc("a.md", "A"), doc("nested/b.md", "B")];
        let chunks = vec![
            chunk(0, "Heading", "first body", vec![1.0, 0.0]),
            chunk(1, "", "second body", vec![0.0, 1.0]),
        ];
        let bytes = write_shard(&docs, &chunks, 2, Quantization::F32, Compression::Deflate)
            .expect("write");
        let (back_docs, back_chunks) = read_shard(&bytes).expect("read");
        assert_eq!(back_docs, docs);
        assert_eq!(
            back_chunks, chunks,
            "deflate is lossless — the only lossy step in the artifact is i8"
        );
    }

    /// The reason this step exists. Measured on scifact, deflate took the
    /// string blob 8.20 MB to 2.75 MB; if compression stopped paying off on
    /// prose, the default should change. The whole-shard ratio is not the thing
    /// to assert — it depends on how much text a vault has per vector, and this
    /// fixture has far less than a real one.
    #[test]
    fn deflate_shrinks_a_shards_string_blob() {
        let strings_len = |bytes: &[u8]| {
            let count = u32_at(bytes, 8) as usize;
            let dim = u32_at(bytes, 12) as usize;
            let at = SHARD_HEADER_LEN
                + u32_at(bytes, 20) as usize
                + count * RECORD_LEN
                + count * dim;
            bytes.len() - at
        };
        let plain = strings_len(&prose_shard(Compression::None));
        let packed = strings_len(&prose_shard(Compression::Deflate));
        assert!(
            (plain as f64) / (packed as f64) > 3.0,
            "deflate took {plain} bytes of text to {packed} — below the measured 3x"
        );
    }

    /// Compression must leave the vector block a fixed-stride array a client
    /// can scan in place. Comparing the two shards' vector regions byte for
    /// byte is what makes that a property rather than an intention.
    #[test]
    fn deflate_leaves_the_vector_block_byte_identical() {
        let vectors = |bytes: &[u8]| {
            let count = u32_at(bytes, 8) as usize;
            let dim = u32_at(bytes, 12) as usize;
            let at = SHARD_HEADER_LEN + u32_at(bytes, 20) as usize + count * RECORD_LEN;
            bytes[at..at + count * dim].to_vec()
        };
        let plain = prose_shard(Compression::None);
        let packed = prose_shard(Compression::Deflate);
        assert_eq!(vectors(&plain), vectors(&packed));
    }

    /// Incremental re-export compares shard hashes, so the encoder has to be
    /// deterministic: a codec that varied its output — by thread count, by a
    /// timestamp — would rewrite every shard on every export and quietly undo
    /// sharding.
    #[test]
    fn deflating_the_same_shard_twice_gives_identical_bytes() {
        assert_eq!(
            prose_shard(Compression::Deflate),
            prose_shard(Compression::Deflate)
        );
    }

    #[test]
    fn read_shard_rejects_an_unknown_compression_code() {
        let mut bytes = prose_shard(Compression::Deflate);
        bytes[6] = 9;
        let err = read_shard(&bytes).expect_err("should refuse");
        assert!(err.to_string().contains("compression code 9"), "{err}");
    }

    /// A deflated block cannot be range-checked the way a raw one can, so
    /// truncation has to surface as an inflate failure rather than be inflated
    /// into a short blob and read as missing text.
    #[test]
    fn read_shard_rejects_a_truncated_deflated_shard() {
        let bytes = prose_shard(Compression::Deflate);
        let err = read_shard(&bytes[..bytes.len() - 64]).expect_err("should refuse");
        assert!(err.to_string().contains("inflating"), "{err}");
    }

    #[test]
    fn utf8_text_survives_the_string_blob() {
        // Offsets and lengths are byte counts, not character counts — getting
        // that wrong would split a multi-byte character and fail only on
        // non-ASCII vaults.
        let docs = vec![doc("한국.md", "한국")];
        let chunks = vec![chunk(0, "제목", "대한민국의 수도는 서울이다", vec![1.0, 0.0])];
        let bytes = write_shard(&docs, &chunks, 2, Quantization::F32, Compression::None).expect("write");
        let (back_docs, back_chunks) = read_shard(&bytes).expect("read");
        assert_eq!(back_docs, docs);
        assert_eq!(back_chunks, chunks);
    }

    #[test]
    fn a_shard_with_no_chunks_is_valid() {
        let bytes = write_shard(&[], &[], 384, Quantization::F32, Compression::None).expect("write");
        let (docs, chunks) = read_shard(&bytes).expect("read");
        assert!(docs.is_empty());
        assert!(chunks.is_empty());
    }

    #[test]
    fn write_shard_rejects_a_vector_of_the_wrong_dimensionality() {
        let docs = vec![doc("a.md", "A")];
        let chunks = vec![chunk(0, "", "body", vec![1.0, 0.0, 0.0])];
        let err = write_shard(&docs, &chunks, 2, Quantization::F32, Compression::None).expect_err("should refuse");
        assert!(err.to_string().contains("dimensions"), "{err}");
    }

    /// A chunk pointing past its shard's document list would read as a
    /// different note's content, so it is refused on the way in and on the way
    /// out.
    #[test]
    fn write_shard_rejects_a_chunk_with_a_dangling_document() {
        let docs = vec![doc("a.md", "A")];
        let chunks = vec![chunk(3, "", "body", vec![1.0, 0.0])];
        let err = write_shard(&docs, &chunks, 2, Quantization::F32, Compression::None).expect_err("should refuse");
        assert!(err.to_string().contains("refers to document"), "{err}");
    }

    #[test]
    fn read_shard_rejects_foreign_bytes() {
        let err = read_shard(b"not a shard at all......").expect_err("should refuse");
        assert!(err.to_string().contains("bad magic"), "{err}");
    }

    #[test]
    fn read_shard_rejects_a_truncated_file() {
        let docs = vec![doc("a.md", "A")];
        let chunks = vec![chunk(0, "", "body", vec![1.0, 0.0])];
        let bytes = write_shard(&docs, &chunks, 2, Quantization::F32, Compression::None).expect("write");
        let err = read_shard(&bytes[..bytes.len() - 4]).expect_err("should refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("truncated") || msg.contains("past the end"),
            "{msg}"
        );
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let m = manifest(
            vec![ShardInfo {
                name: shard_name(0),
                hash: "deadbeef".to_string(),
                documents: 2,
                chunks: 2,
            }],
            Quantization::F32,
        );
        let back = Manifest::from_json(&m.to_json().expect("serialize")).expect("parse");
        assert_eq!(back, m);
    }

    /// The manifest must not grow with the vault: that is the whole reason
    /// documents moved into shards in v2. A v1 manifest measured 270 bytes per
    /// document, so 20k notes came to 5.4 MB rewritten on every export.
    #[test]
    fn manifest_size_is_independent_of_document_count() {
        let shards = vec![ShardInfo {
            name: shard_name(0),
            hash: "0".repeat(64),
            documents: 20_000,
            chunks: 20_048,
        }];
        let json = manifest(shards, Quantization::F32).to_json().unwrap();
        assert!(
            json.len() < 1024,
            "a manifest describing 20k documents should stay tiny, got {} bytes",
            json.len()
        );
    }

    /// A future format change must fail loudly here rather than be mis-parsed
    /// into plausible nonsense. This is also what let v1 be replaced outright.
    #[test]
    fn manifest_refuses_an_unknown_format_version() {
        let mut m = manifest(Vec::new(), Quantization::F32);
        m.format_version = ARTIFACT_FORMAT_VERSION + 1;
        let json = serde_json::to_string(&m).unwrap();
        let err = Manifest::from_json(&json).expect_err("should refuse");
        assert!(err.to_string().contains("not supported"), "{err}");
    }

    #[test]
    fn manifest_refuses_the_superseded_v1() {
        let mut m = manifest(Vec::new(), Quantization::F32);
        m.format_version = 1;
        let json = serde_json::to_string(&m).unwrap();
        let err = Manifest::from_json(&json).expect_err("should refuse");
        assert!(err.to_string().contains("re-export"), "{err}");
    }

    /// A manifest written before compression existed has no `compression` field
    /// at all. It must still reach the version check and its actionable error,
    /// rather than fail as a parse error about a missing field.
    #[test]
    fn manifest_refuses_a_v2_manifest_that_predates_compression() {
        let json = r#"{
            "format_version": 2,
            "created_unix": 0,
            "model": {"id": "bge-small-en-v1.5", "pooling": "cls", "dim": 384},
            "quantization": "i8",
            "roots": ["vault"],
            "shard_docs": 256,
            "shards": []
        }"#;
        let err = Manifest::from_json(json).expect_err("should refuse");
        assert!(err.to_string().contains("re-export"), "{err}");
    }

    #[test]
    fn open_assembles_candidates_with_relative_paths() {
        let docs = vec![doc("a.md", "A"), doc("nested/b.md", "B")];
        let chunks = vec![
            chunk(0, "Heading", "first", vec![1.0, 0.0]),
            chunk(1, "", "second", vec![0.0, 1.0]),
        ];
        let (m, bytes) = one_shard(&docs, &chunks);
        let artifact =
            Artifact::open(&m.to_json().unwrap(), &[(shard_name(0), bytes)]).expect("open");

        let candidates = artifact.candidates(Space::Text);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].path, "a.md", "paths stay vault-relative");
        assert_eq!(
            candidates[0].source_root, "vault",
            "the root is a label, never a machine path"
        );
        assert!(artifact.candidates(Space::Image).is_empty());
    }

    /// Each shard numbers its documents from zero, so assembling several has to
    /// rebase them. Getting this wrong would attribute one note's chunks to
    /// another — and it would only show up with more than one shard, which is
    /// why this test exists at two.
    #[test]
    fn open_rebases_document_indices_across_shards() {
        let docs_a = vec![doc("a.md", "A")];
        let chunks_a = vec![chunk(0, "", "alpha", vec![1.0, 0.0])];
        let docs_b = vec![doc("b.md", "B"), doc("c.md", "C")];
        let chunks_b = vec![
            chunk(0, "", "beta", vec![0.0, 1.0]),
            chunk(1, "", "gamma", vec![0.5, 0.5]),
        ];

        let bytes_a = write_shard(&docs_a, &chunks_a, 2, Quantization::F32, Compression::None).unwrap();
        let bytes_b = write_shard(&docs_b, &chunks_b, 2, Quantization::F32, Compression::None).unwrap();
        let m = manifest(
            vec![
                ShardInfo {
                    name: shard_name(0),
                    hash: shard_hash(&bytes_a),
                    documents: 1,
                    chunks: 1,
                },
                ShardInfo {
                    name: shard_name(1),
                    hash: shard_hash(&bytes_b),
                    documents: 2,
                    chunks: 2,
                },
            ],
            Quantization::F32,
        );
        let artifact = Artifact::open(
            &m.to_json().unwrap(),
            &[(shard_name(0), bytes_a), (shard_name(1), bytes_b)],
        )
        .expect("open");

        assert_eq!(artifact.documents().len(), 3);
        let by_text: Vec<(&str, &str)> = artifact
            .chunks()
            .iter()
            .map(|c| (c.text.as_str(), artifact.documents()[c.doc_index].path.as_str()))
            .collect();
        assert_eq!(
            by_text,
            vec![("alpha", "a.md"), ("beta", "b.md"), ("gamma", "c.md")],
            "every chunk must still point at its own document"
        );
    }

    /// A half-synced artifact must not answer queries from a mixture of two
    /// indexes, so a hash mismatch is fatal rather than a warning.
    #[test]
    fn open_rejects_a_shard_that_does_not_match_the_manifest() {
        let docs = vec![doc("a.md", "A")];
        let bytes = write_shard(&docs, &[chunk(0, "", "x", vec![1.0, 0.0])], 2, Quantization::F32, Compression::None)
            .unwrap();
        let m = manifest(
            vec![ShardInfo {
                name: shard_name(0),
                hash: "0".repeat(64),
                documents: 1,
                chunks: 1,
            }],
            Quantization::F32,
        );
        let err = Artifact::open(&m.to_json().unwrap(), &[(shard_name(0), bytes)])
            .expect_err("should refuse");
        assert!(err.to_string().contains("partially synced"), "{err}");
    }

    #[test]
    fn open_reports_a_missing_shard_by_name() {
        let m = manifest(
            vec![ShardInfo {
                name: shard_name(0),
                hash: "0".repeat(64),
                documents: 1,
                chunks: 1,
            }],
            Quantization::F32,
        );
        let err = Artifact::open(&m.to_json().unwrap(), &[]).expect_err("should refuse");
        assert!(err.to_string().contains("shard-0000.bin"), "{err}");
    }

    /// Golden values. `assign_shard` is a pure function of `(root, path,
    /// count)` by signature, so stability under insertion is structural — what
    /// needs locking down is the mapping itself, because changing the hash
    /// silently reassigns every document in every existing artifact and so
    /// rewrites all of it on the next export. Change these only with a format
    /// version bump.
    #[test]
    fn shard_assignment_is_pinned_to_known_buckets() {
        assert_eq!(assign_shard(0, "a.md", 8), 1);
        assert_eq!(assign_shard(0, "m.md", 8), 5);
        assert_eq!(assign_shard(0, "z.md", 8), 4);
        assert_eq!(assign_shard(0, "deep/nested/note.md", 8), 0);
    }

    #[test]
    fn shard_assignment_spreads_documents_across_shards() {
        let mut counts = [0usize; 8];
        for i in 0..800 {
            counts[assign_shard(0, &format!("note{i}.md"), 8)] += 1;
        }
        for (shard, count) in counts.iter().enumerate() {
            assert!(
                *count > 40,
                "shard {shard} got only {count} of 800 documents — assignment is lopsided"
            );
        }
    }

    #[test]
    fn shard_assignment_separates_identical_paths_in_different_roots() {
        assert_ne!(
            assign_shard(0, "notes/a.md", 64),
            assign_shard(1, "notes/a.md", 64),
            "two vaults with the same relative path must not collide by construction"
        );
    }

    #[test]
    fn a_single_shard_takes_every_document() {
        for path in ["a.md", "deep/b.md", "z.md"] {
            assert_eq!(assign_shard(0, path, 1), 0);
        }
    }

    #[test]
    fn shard_names_sort_in_shard_order() {
        assert_eq!(shard_name(0), "shard-0000.bin");
        assert_eq!(shard_name(12), "shard-0012.bin");
        let mut names = vec![shard_name(10), shard_name(2), shard_name(1)];
        names.sort();
        assert_eq!(names, vec![shard_name(1), shard_name(2), shard_name(10)]);
    }
}
