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
pub const ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Shard file magic: "GNSA" (gnosis artifact).
const SHARD_MAGIC: &[u8; 4] = b"GNSA";

/// Byte length of the shard header.
const SHARD_HEADER_LEN: usize = 20;

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
    pub chunks: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub created_unix: i64,
    pub model: ModelInfo,
    pub quantization: Quantization,
    /// Root labels — a vault's final path component, never its absolute path.
    pub roots: Vec<String>,
    pub documents: Vec<DocumentEntry>,
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
    /// Index into [`Manifest::documents`].
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
/// Layout, all little-endian: a header, then a fixed-size record per chunk,
/// then the vector block, then the string blob. Vectors are contiguous and
/// fixed-stride so a client can scan them without touching the strings, which
/// is the whole shape of a brute-force query.
pub fn write_shard(
    chunks: &[ArtifactChunk],
    dim: usize,
    quantization: Quantization,
) -> Result<Vec<u8>> {
    let mut records = Vec::with_capacity(chunks.len() * RECORD_LEN);
    let mut vectors = Vec::with_capacity(chunks.len() * dim * quantization.bytes_per_dim());
    let mut strings: Vec<u8> = Vec::new();

    for chunk in chunks {
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

    let mut out = Vec::with_capacity(SHARD_HEADER_LEN + records.len() + vectors.len() + strings.len());
    out.extend_from_slice(SHARD_MAGIC);
    out.extend_from_slice(&(ARTIFACT_FORMAT_VERSION as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // flags, unused in v1
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.push(quantization.code());
    out.extend_from_slice(&[0u8; 3]); // pad to a 4-byte boundary
    out.extend_from_slice(&records);
    out.extend_from_slice(&vectors);
    out.extend_from_slice(&strings);
    Ok(out)
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

/// Parse one shard back into chunks.
pub fn read_shard(bytes: &[u8]) -> Result<Vec<ArtifactChunk>> {
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
    let count = u32_at(bytes, 8) as usize;
    let dim = u32_at(bytes, 12) as usize;
    let quantization = Quantization::from_code(bytes[16])?;

    let records_at = SHARD_HEADER_LEN;
    let vectors_at = records_at + count * RECORD_LEN;
    let strings_at = vectors_at + count * dim * quantization.bytes_per_dim();
    if bytes.len() < strings_at {
        bail!(
            "shard is truncated: {} bytes, expected at least {strings_at} for {count} chunks",
            bytes.len()
        );
    }
    let strings = &bytes[strings_at..];

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
    Ok(chunks)
}

fn read_str(strings: &[u8], offset: usize, len: usize, what: &str) -> Result<String> {
    let end = offset
        .checked_add(len)
        .filter(|end| *end <= strings.len())
        .with_context(|| format!("{what} runs past the end of the shard's string blob"))?;
    String::from_utf8(strings[offset..end].to_vec())
        .with_context(|| format!("{what} is not valid UTF-8"))
}

/// A manifest plus its shards' chunks, ready to query.
#[derive(Debug)]
pub struct Artifact {
    manifest: Manifest,
    chunks: Vec<ArtifactChunk>,
}

impl Artifact {
    /// Assemble from bytes the caller has already fetched: the manifest as
    /// JSON, then each shard by name.
    ///
    /// Shards are verified against the manifest's hashes. A mismatch is an
    /// error rather than a warning: a half-synced artifact would otherwise
    /// answer queries from a mixture of two indexes.
    pub fn open(manifest_json: &str, shards: &[(String, Vec<u8>)]) -> Result<Self> {
        let manifest = Manifest::from_json(manifest_json)?;
        let supplied: HashMap<&str, &Vec<u8>> =
            shards.iter().map(|(n, b)| (n.as_str(), b)).collect();

        let mut chunks = Vec::new();
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
            let shard = read_shard(bytes)
                .with_context(|| format!("reading artifact shard '{}'", info.name))?;
            if shard.len() != info.chunks {
                bail!(
                    "artifact shard '{}' holds {} chunks but the manifest says {}",
                    info.name,
                    shard.len(),
                    info.chunks
                );
            }
            chunks.extend(shard);
        }
        Ok(Self { manifest, chunks })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
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
                let doc = self.manifest.documents.get(c.doc_index)?;
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
            roots: vec!["vault".to_string()],
            documents: vec![
                DocumentEntry {
                    root: 0,
                    path: "a.md".to_string(),
                    title: "A".to_string(),
                    kind: "markdown".to_string(),
                    content_hash: "00".to_string(),
                    mtime: 0,
                    links: vec!["b".to_string()],
                    tags: vec!["t".to_string()],
                },
                DocumentEntry {
                    root: 0,
                    path: "nested/b.md".to_string(),
                    title: "B".to_string(),
                    kind: "markdown".to_string(),
                    content_hash: "01".to_string(),
                    mtime: 0,
                    links: Vec::new(),
                    tags: Vec::new(),
                },
            ],
            shards,
        }
    }

    #[test]
    fn f32_shard_round_trips_exactly() {
        let chunks = vec![
            chunk(0, "Heading", "first body", vec![1.0, 0.0]),
            chunk(1, "", "second body", vec![0.0, 1.0]),
        ];
        let bytes = write_shard(&chunks, 2, Quantization::F32).expect("write");
        let back = read_shard(&bytes).expect("read");
        assert_eq!(
            back, chunks,
            "f32 is the exact form — a round trip must change nothing at all"
        );
    }

    #[test]
    fn i8_shard_round_trips_within_quantization_error() {
        let chunks = vec![chunk(0, "", "body", vec![0.6, -0.8])];
        let bytes = write_shard(&chunks, 2, Quantization::I8).expect("write");
        let back = read_shard(&bytes).expect("read");
        for (original, restored) in chunks[0].vector.iter().zip(&back[0].vector) {
            assert!(
                (original - restored).abs() < 0.01,
                "i8 round trip moved {original} to {restored}"
            );
        }
        assert_eq!(back[0].text, "body", "metadata must survive exactly");
    }

    #[test]
    fn i8_is_four_times_smaller_than_f32() {
        let chunks: Vec<ArtifactChunk> = (0..16)
            .map(|i| chunk(0, "", "x", vec![0.5; 384]).tap(i))
            .collect();
        let f32_len = write_shard(&chunks, 384, Quantization::F32).unwrap().len();
        let i8_len = write_shard(&chunks, 384, Quantization::I8).unwrap().len();
        let vectors_f32 = 16 * 384 * 4;
        let vectors_i8 = 16 * 384;
        assert_eq!(f32_len - i8_len, vectors_f32 - vectors_i8);
    }

    /// Helper so the size test can build distinct chunks without another fn.
    trait Tap {
        fn tap(self, ord: u32) -> Self;
    }
    impl Tap for ArtifactChunk {
        fn tap(mut self, ord: u32) -> Self {
            self.ord = ord;
            self
        }
    }

    #[test]
    fn utf8_text_survives_the_string_blob() {
        // Offsets and lengths are byte counts, not character counts — getting
        // that wrong would split a multi-byte character and fail only on
        // non-ASCII vaults.
        let chunks = vec![chunk(0, "제목", "대한민국의 수도는 서울이다", vec![1.0, 0.0])];
        let bytes = write_shard(&chunks, 2, Quantization::F32).expect("write");
        let back = read_shard(&bytes).expect("read");
        assert_eq!(back, chunks);
    }

    #[test]
    fn a_shard_with_no_chunks_is_valid() {
        let bytes = write_shard(&[], 384, Quantization::F32).expect("write");
        assert!(read_shard(&bytes).expect("read").is_empty());
    }

    #[test]
    fn write_shard_rejects_a_vector_of_the_wrong_dimensionality() {
        let chunks = vec![chunk(0, "", "body", vec![1.0, 0.0, 0.0])];
        let err = write_shard(&chunks, 2, Quantization::F32).expect_err("should refuse");
        assert!(err.to_string().contains("dimensions"), "{err}");
    }

    #[test]
    fn read_shard_rejects_foreign_bytes() {
        let err = read_shard(b"not a shard at all......").expect_err("should refuse");
        assert!(err.to_string().contains("bad magic"), "{err}");
    }

    #[test]
    fn read_shard_rejects_a_truncated_file() {
        let chunks = vec![chunk(0, "", "body", vec![1.0, 0.0])];
        let bytes = write_shard(&chunks, 2, Quantization::F32).expect("write");
        let err = read_shard(&bytes[..bytes.len() - 4]).expect_err("should refuse");
        assert!(err.to_string().contains("truncated") || err.to_string().contains("past the end"), "{err}");
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let m = manifest(
            vec![ShardInfo {
                name: "shard-0000.bin".to_string(),
                hash: "deadbeef".to_string(),
                chunks: 2,
            }],
            Quantization::F32,
        );
        let back = Manifest::from_json(&m.to_json().expect("serialize")).expect("parse");
        assert_eq!(back, m);
    }

    /// A future format change must fail loudly here rather than be mis-parsed
    /// into plausible nonsense.
    #[test]
    fn manifest_refuses_an_unknown_format_version() {
        let mut m = manifest(Vec::new(), Quantization::F32);
        m.format_version = ARTIFACT_FORMAT_VERSION + 1;
        let json = serde_json::to_string(&m).unwrap();
        let err = Manifest::from_json(&json).expect_err("should refuse");
        assert!(err.to_string().contains("not supported"), "{err}");
    }

    #[test]
    fn open_assembles_candidates_with_relative_paths() {
        let chunks = vec![
            chunk(0, "Heading", "first", vec![1.0, 0.0]),
            chunk(1, "", "second", vec![0.0, 1.0]),
        ];
        let bytes = write_shard(&chunks, 2, Quantization::F32).unwrap();
        let m = manifest(
            vec![ShardInfo {
                name: "shard-0000.bin".to_string(),
                hash: shard_hash(&bytes),
                chunks: 2,
            }],
            Quantization::F32,
        );
        let artifact =
            Artifact::open(&m.to_json().unwrap(), &[("shard-0000.bin".to_string(), bytes)])
                .expect("open");

        let candidates = artifact.candidates(Space::Text);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].path, "a.md", "paths stay vault-relative");
        assert_eq!(
            candidates[0].source_root, "vault",
            "the root is a label, never a machine path"
        );
        assert!(artifact.candidates(Space::Image).is_empty());
    }

    /// A half-synced artifact must not answer queries from a mixture of two
    /// indexes, so a hash mismatch is fatal rather than a warning.
    #[test]
    fn open_rejects_a_shard_that_does_not_match_the_manifest() {
        let bytes = write_shard(&[chunk(0, "", "x", vec![1.0, 0.0])], 2, Quantization::F32).unwrap();
        let m = manifest(
            vec![ShardInfo {
                name: "shard-0000.bin".to_string(),
                hash: "0".repeat(64),
                chunks: 1,
            }],
            Quantization::F32,
        );
        let err = Artifact::open(&m.to_json().unwrap(), &[("shard-0000.bin".to_string(), bytes)])
            .expect_err("should refuse");
        assert!(err.to_string().contains("partially synced"), "{err}");
    }

    #[test]
    fn open_reports_a_missing_shard_by_name() {
        let m = manifest(
            vec![ShardInfo {
                name: "shard-0000.bin".to_string(),
                hash: "0".repeat(64),
                chunks: 1,
            }],
            Quantization::F32,
        );
        let err = Artifact::open(&m.to_json().unwrap(), &[]).expect_err("should refuse");
        assert!(err.to_string().contains("shard-0000.bin"), "{err}");
    }
}
