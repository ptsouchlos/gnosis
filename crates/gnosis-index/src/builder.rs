//! Building a portable artifact without a database.
//!
//! The native indexer (`crate::run`) writes to a [`store::Store`] and
//! `gnosis export` then packs that into an artifact. A client — the Obsidian
//! plugin — has no database and no filesystem, so it needs the other shape:
//! hand over a document's text, get back the strings to embed, hand back the
//! vectors, get an artifact.
//!
//! Everything that decides *what the index contains* happens here, in the same
//! code the CLI runs:
//!
//! - [`parse::parse_markdown`] finds the title, frontmatter, wikilinks, tags.
//! - [`chunker::chunk_document`] decides the chunk boundaries, including the
//!   identity chunk a property-only note would otherwise go unindexed without.
//! - `blake3` hashes content, so an unchanged document is skipped by the same
//!   measure the CLI skips it by.
//! - [`store::artifact::write_shard`] packs the result, so the output is the
//!   artifact format rather than something like it.
//!
//! A reimplementation of any of those in JavaScript would produce a
//! *different* index from the same vault — different chunk boundaries above
//! all, which every retrieval number in this project is measured against.
//!
//! What the caller must supply that this cannot: the vectors, and the clock.
//! `wasm32-unknown-unknown` has no clock, so [`ArtifactBuilder::finish`] takes
//! the timestamp rather than inventing one.

use std::collections::HashMap;

use anyhow::{Result, bail};
use serde::Deserialize;
use store::Space;
use store::artifact::{
    ARTIFACT_FORMAT_VERSION, Artifact, ArtifactChunk, Compression, DocumentEntry, Manifest,
    ModelInfo, Quantization, ShardInfo, SpaceInfo, SpaceSpec, assign_shard, shard_hash, shard_name,
    write_shard,
};

/// What the caller must say before anything can be built.
///
/// The model id and its pooling are recorded, not guessed: an index whose
/// manifest misreports either is worse than no index, because every query
/// against it looks healthy and ranks by vectors from a different space.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildSpec {
    /// The embedding model's id, as the manifest will record it.
    pub model: String,
    /// How token vectors are pooled into one — `cls` or `mean`.
    pub pooling: String,
    pub dim: usize,
    /// Prepended to a query, recorded so a client querying this artifact
    /// later embeds the way the CLI would. Required, like `pooling`: an empty
    /// string is a statement that the model wants none, not a default.
    pub query_prefix: String,
    /// Prepended to every text `add_document` hands back for embedding —
    /// `"passage: "` for E5. Applied here rather than by the caller so the
    /// strings returned are exactly what `gnosis index` embeds.
    pub document_prefix: String,
    /// The vault root's label. One root: an artifact spanning several vaults
    /// is the CLI's business, and a client indexes the vault it runs in.
    pub root: String,
    #[serde(default = "default_quantization")]
    pub quantization: Quantization,
    #[serde(default = "default_compression")]
    pub compression: Compression,
    /// Documents per shard. The CLI's own default, so an artifact built either
    /// way shards identically.
    #[serde(default = "default_shard_docs")]
    pub shard_docs: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_overlap")]
    pub overlap: usize,
}

fn default_quantization() -> Quantization {
    Quantization::I8
}

fn default_compression() -> Compression {
    Compression::Deflate
}

fn default_shard_docs() -> usize {
    256
}

fn default_max_tokens() -> usize {
    chunker::ChunkConfig::default().max_tokens
}

fn default_overlap() -> usize {
    chunker::ChunkConfig::default().overlap
}

/// What [`ArtifactBuilder::add_document`] tells the caller to do next.
#[derive(Debug, Clone)]
pub struct Added {
    /// True when the document was carried over from the previous artifact
    /// unchanged, so `texts` is empty and nothing needs embedding.
    pub reused: bool,
    /// The exact strings to embed, in chunk order, document prefix and
    /// heading trail included — because that is what the native indexer
    /// embeds. A caller that embedded
    /// the body alone would produce vectors from different inputs.
    pub texts: Vec<String>,
}

/// A finished artifact, ready to write.
#[derive(Debug)]
pub struct Built {
    pub manifest: Manifest,
    /// Shard name and bytes, excluding any whose bytes are unchanged.
    pub shards: Vec<(String, Vec<u8>)>,
    /// Shards identical to the previous artifact's, by name. Rewriting a file
    /// a syncer already has is the churn sharding exists to avoid.
    pub unchanged: Vec<String>,
    pub documents: usize,
    pub chunks: usize,
}

/// The content hash a document's entry records: blake3 of its text, hex.
///
/// Public so a client checking staleness compares against exactly what the
/// builder stored, rather than a second implementation of it.
pub fn content_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

/// One document on its way into the artifact.
struct Pending {
    entry: DocumentEntry,
    chunks: Vec<ArtifactChunk>,
    /// Set until the caller supplies vectors. A document that reaches
    /// `finish` still waiting is an error rather than an index with a hole.
    needs_vectors: bool,
}

/// Accumulates documents, then packs them into shards.
pub struct ArtifactBuilder {
    spec: BuildSpec,
    /// Insertion order is irrelevant — `finish` sorts — but the map makes
    /// `set_vectors` a lookup rather than a scan.
    documents: HashMap<String, Pending>,
    /// The artifact being replaced, if any: the source of both the hashes that
    /// decide what can be skipped and the vectors that are then carried over.
    ///
    /// Owned rather than borrowed. A builder that borrowed it would need a
    /// lifetime, and a `wasm_bindgen` class cannot have one — the alternative
    /// is a self-referential struct and the unsafe code that implies, for no
    /// gain: a build replaces the artifact it reused.
    previous: Option<Artifact>,
}

impl ArtifactBuilder {
    /// A builder with nothing to reuse: every document will be embedded.
    pub fn new(spec: BuildSpec) -> Self {
        Self {
            spec,
            documents: HashMap::new(),
            previous: None,
        }
    }

    /// A builder that reuses an existing artifact.
    ///
    /// This is what makes a rebuild cheap: a document whose content hash
    /// matches keeps its chunks *and its vectors*, so an ordinary day's
    /// editing re-embeds a handful of chunks rather than a vault.
    ///
    /// An artifact built by a different model cannot be reused and is refused
    /// rather than mixed: a mixture of two vector spaces ranks confidently and
    /// wrongly, which is the one failure mode with no symptom.
    pub fn reusing(spec: BuildSpec, previous: Artifact) -> Result<Self> {
        match previous.manifest().space(Space::Text) {
            Some(model)
                if model.id == spec.model
                    && model.dim == spec.dim
                    && model.document_prefix == spec.document_prefix => {}
            Some(model) if model.id == spec.model && model.dim == spec.dim => bail!(
                "the existing index embedded documents with prefix {:?} and this build uses \
                 {:?} — rebuild from scratch instead of reusing it",
                model.document_prefix,
                spec.document_prefix
            ),
            Some(model) => bail!(
                "the existing index was built by '{}' at {}d and this build uses '{}' at {}d — \
                 rebuild from scratch instead of reusing it",
                model.id,
                model.dim,
                spec.model,
                spec.dim
            ),
            None => bail!("the existing index has no text space to reuse"),
        }
        Ok(Self {
            spec,
            documents: HashMap::new(),
            previous: Some(previous),
        })
    }

    /// Offer one document to the index.
    ///
    /// `mtime` is in seconds, as the artifact records it.
    pub fn add_document(&mut self, path: &str, text: &str, mtime: i64) -> Added {
        let hash = content_hash(text);

        if let Some(pending) = self.reuse(path, &hash) {
            self.documents.insert(path.to_string(), pending);
            return Added {
                reused: true,
                texts: Vec::new(),
            };
        }

        let parsed = parse::parse_markdown(std::path::Path::new(path), text);
        let frontmatter = parsed
            .frontmatter
            .as_deref()
            .map(parse::frontmatter_text)
            .unwrap_or_default();
        let chunks = chunker::chunk_document(
            &parsed.title,
            &frontmatter,
            &parsed.body,
            self.spec.max_tokens,
            self.spec.overlap,
        );
        let texts: Vec<String> = chunks
            .iter()
            .map(|c| {
                format!(
                    "{}{}",
                    self.spec.document_prefix,
                    crate::join_lines(&c.heading_path, &c.text)
                )
            })
            .collect();

        self.documents.insert(
            path.to_string(),
            Pending {
                entry: DocumentEntry {
                    // Rewritten by `finish`, which assigns the roots.
                    root: 0,
                    path: path.to_string(),
                    title: parsed.title,
                    kind: "markdown".to_string(),
                    content_hash: hash,
                    mtime,
                    links: parsed.links,
                    tags: parsed.tags,
                },
                chunks: chunks
                    .into_iter()
                    .map(|c| ArtifactChunk {
                        // Rewritten by `finish`, which places the documents.
                        doc_index: 0,
                        ord: c.ord as u32,
                        space: Space::Text,
                        modality: "text".to_string(),
                        heading_path: c.heading_path,
                        text: c.text,
                        vector: Vec::new(),
                    })
                    .collect(),
                needs_vectors: true,
            },
        );

        Added {
            reused: false,
            texts,
        }
    }

    /// Supply one document's vectors, flat and in chunk order.
    pub fn set_vectors(&mut self, path: &str, vectors: &[f32]) -> Result<()> {
        let dim = self.spec.dim;
        let Some(pending) = self.documents.get_mut(path) else {
            bail!("'{path}' was never added");
        };
        let expected = pending.chunks.len() * dim;
        if vectors.len() != expected {
            bail!(
                "'{path}' has {} chunks at {dim}d, so it needs {expected} floats, not {}",
                pending.chunks.len(),
                vectors.len()
            );
        }
        for (i, chunk) in pending.chunks.iter_mut().enumerate() {
            chunk.vector = vectors[i * dim..(i + 1) * dim].to_vec();
        }
        pending.needs_vectors = false;
        Ok(())
    }

    /// How many documents are waiting for vectors, for a progress report.
    pub fn pending(&self) -> usize {
        self.documents.values().filter(|p| p.needs_vectors).count()
    }

    /// How many documents have been offered.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Pack everything added into shards.
    ///
    /// `created_unix` comes from the caller because wasm has no clock. It is
    /// the one field that makes two otherwise identical artifacts differ.
    pub fn finish(self, created_unix: i64) -> Result<Built> {
        if self.documents.is_empty() {
            bail!("no documents were added — nothing to build");
        }
        if let Some(path) = self
            .documents
            .iter()
            .find(|(_, p)| p.needs_vectors)
            .map(|(path, _)| path)
        {
            bail!("'{path}' is still waiting for its vectors");
        }

        // Sorted by path, which is the order `gnosis export` reads documents
        // out of the database in (`ORDER BY r.path, d.rel_path`). Matching it
        // is what makes an artifact built here byte-identical to one built
        // there rather than merely equivalent.
        let mut paths: Vec<&String> = self.documents.keys().collect();
        paths.sort();

        let shard_count = paths.len().div_ceil(self.spec.shard_docs).max(1);
        let mut shard_documents: Vec<Vec<DocumentEntry>> = vec![Vec::new(); shard_count];
        let mut shard_chunks: Vec<Vec<ArtifactChunk>> = vec![Vec::new(); shard_count];

        for path in &paths {
            let pending = &self.documents[*path];
            // The same assignment the CLI uses: hashing the path means adding
            // a note cannot move any other note between shards.
            let shard = assign_shard(0, path, shard_count);
            let local = shard_documents[shard].len();
            shard_documents[shard].push(pending.entry.clone());
            for chunk in &pending.chunks {
                let mut chunk = chunk.clone();
                chunk.doc_index = local;
                shard_chunks[shard].push(chunk);
            }
        }

        let specs = vec![SpaceSpec {
            space: Space::Text,
            dim: self.spec.dim,
        }];
        let previous: HashMap<&str, &str> = self
            .previous
            .as_ref()
            .map(|a| {
                a.manifest()
                    .shards
                    .iter()
                    .map(|s| (s.name.as_str(), s.hash.as_str()))
                    .collect()
            })
            .unwrap_or_default();

        let mut shards = Vec::new();
        let mut unchanged = Vec::new();
        let mut infos = Vec::with_capacity(shard_count);
        let mut chunks_total = 0usize;
        for i in 0..shard_count {
            let body = write_shard(
                &shard_documents[i],
                &shard_chunks[i],
                &specs,
                self.spec.quantization,
                self.spec.compression,
            )?;
            let name = shard_name(i);
            let hash = shard_hash(&body);
            if previous.get(name.as_str()) == Some(&hash.as_str()) {
                unchanged.push(name.clone());
            } else {
                shards.push((name.clone(), body));
            }
            chunks_total += shard_chunks[i].len();
            infos.push(ShardInfo {
                name,
                hash,
                documents: shard_documents[i].len(),
                chunks: shard_chunks[i].len(),
            });
        }

        Ok(Built {
            manifest: Manifest {
                format_version: ARTIFACT_FORMAT_VERSION,
                created_unix,
                spaces: vec![SpaceInfo {
                    space: Space::Text,
                    model: ModelInfo {
                        id: self.spec.model.clone(),
                        pooling: self.spec.pooling.clone(),
                        dim: self.spec.dim,
                        query_id: None,
                        query_prefix: self.spec.query_prefix.clone(),
                        document_prefix: self.spec.document_prefix.clone(),
                    },
                }],
                quantization: self.spec.quantization,
                compression: self.spec.compression,
                roots: vec![self.spec.root.clone()],
                shard_docs: self.spec.shard_docs,
                shards: infos,
            },
            documents: paths.len(),
            chunks: chunks_total,
            shards,
            unchanged,
        })
    }

    /// The previous artifact's entry for `path`, if its content is unchanged.
    fn reuse(&self, path: &str, hash: &str) -> Option<Pending> {
        let artifact = self.previous.as_ref()?;
        let (index, entry) = artifact
            .documents()
            .iter()
            .enumerate()
            .find(|(_, d)| d.path == path)?;
        if entry.content_hash != hash {
            return None;
        }
        let chunks: Vec<ArtifactChunk> = artifact
            .chunks()
            .iter()
            .filter(|c| c.doc_index == index && c.space == Space::Text)
            .cloned()
            .collect();
        // A document with no chunks cannot be reused: there would be nothing
        // to retrieve it by, and re-chunking it is what fixes that.
        if chunks.is_empty() {
            return None;
        }
        Some(Pending {
            entry: entry.clone(),
            chunks,
            needs_vectors: false,
        })
    }
}
