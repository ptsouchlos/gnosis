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
    /// Build an image space beside the text one: image files become documents,
    /// and every document gets a title proxy, as `gnosis index` writes them
    /// with `[embed.image]` on. `None` builds the text space alone.
    #[serde(default)]
    pub image: Option<ImageSpec>,
}

/// The image space a build records, which is the CLI's CLIP pair.
///
/// Two models, as in `gnosis export`: the vectors of real images come from
/// CLIP's vision tower (`model`) and the title proxies from its text tower,
/// which is also how a client's text query reaches the space (`query_id`).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageSpec {
    /// The vision model's id, recorded as the space's model id.
    pub model: String,
    /// The text tower's id, which queries the space and embeds title proxies.
    pub query_id: String,
    pub dim: usize,
}

/// How CLIP's text tower pools, as `gnosis export` records it for the space.
const CLIP_TEXT_POOLING: &str = "mean";

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
    /// The title to embed with CLIP's text tower for the document's title
    /// proxy, when the build has an image space and the document still needs
    /// one — including a reused document whose previous index had none.
    pub title: Option<String>,
}

/// What [`ArtifactBuilder::add_image`] tells the caller to do next.
#[derive(Debug, Clone)]
pub struct AddedImage {
    /// True when the image was carried over unchanged: its vision vector and
    /// title proxy are already in hand.
    pub reused: bool,
    /// The title to embed for the proxy (the file's stem), if one is needed.
    pub title: Option<String>,
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

/// The content hash an image's entry records: blake3 of its bytes, hex.
pub fn content_hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// One document on its way into the artifact.
struct Pending {
    entry: DocumentEntry,
    chunks: Vec<ArtifactChunk>,
    /// Set until the caller supplies vectors. A document that reaches
    /// `finish` still waiting is an error rather than an index with a hole.
    needs_vectors: bool,
    /// An image file, whose one vector comes from the vision tower.
    is_image: bool,
    /// The title proxy's text while its vector is still owed; the chunk joins
    /// `chunks` when `set_title_vector` supplies it.
    awaiting_title: Option<String>,
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
    /// Whether the previous artifact's image space is the one this build
    /// records, so its image vectors and title proxies may be carried over.
    previous_image_ok: bool,
}

impl ArtifactBuilder {
    /// A builder with nothing to reuse: every document will be embedded.
    pub fn new(spec: BuildSpec) -> Self {
        Self {
            spec,
            documents: HashMap::new(),
            previous: None,
            previous_image_ok: false,
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
        let previous_image_ok = match (&spec.image, previous.manifest().space(Space::Image)) {
            (Some(want), Some(have)) => {
                have.id == want.model
                    && have.dim == want.dim
                    && have.query_id.as_deref() == Some(want.query_id.as_str())
            }
            _ => false,
        };
        Ok(Self {
            spec,
            documents: HashMap::new(),
            previous: Some(previous),
            previous_image_ok,
        })
    }

    /// Offer one document to the index.
    ///
    /// `mtime` is in seconds, as the artifact records it.
    pub fn add_document(&mut self, path: &str, text: &str, mtime: i64) -> Added {
        let hash = content_hash(text);

        if let Some(pending) = self.reuse(path, &hash, "markdown") {
            let title = pending.awaiting_title.clone();
            self.documents.insert(path.to_string(), pending);
            return Added {
                reused: true,
                texts: Vec::new(),
                title,
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
        let awaiting_title = self.spec.image.is_some().then(|| parsed.title.clone());
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
                    created_unix: parsed
                        .frontmatter
                        .as_deref()
                        .and_then(parse::extract_created_unix),
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
                is_image: false,
                awaiting_title: awaiting_title.clone(),
            },
        );

        Added {
            reused: false,
            texts,
            title: awaiting_title,
        }
    }

    /// Offer one image file to the index.
    ///
    /// `hash` is [`content_hash_bytes`] of the file; `mtime` is in seconds. An
    /// unchanged image keeps its vectors. A new one is a document of kind
    /// `image` with one chunk, which `set_image_vector` fills, and a title
    /// proxy, which `set_title_vector` does — both owed before `finish`.
    pub fn add_image(&mut self, path: &str, hash: &str, mtime: i64) -> Result<AddedImage> {
        if self.spec.image.is_none() {
            bail!("this build has no image space; set `image` in the spec to index images");
        }
        if let Some(pending) = self.reuse(path, hash, "image") {
            let title = pending.awaiting_title.clone();
            self.documents.insert(path.to_string(), pending);
            return Ok(AddedImage {
                reused: true,
                title,
            });
        }
        // The file's stem, as `gnosis index` titles an image.
        let title = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(path)
            .to_string();
        self.documents.insert(
            path.to_string(),
            Pending {
                entry: DocumentEntry {
                    root: 0,
                    path: path.to_string(),
                    title: title.clone(),
                    kind: "image".to_string(),
                    content_hash: hash.to_string(),
                    mtime,
                    created_unix: None,
                    links: Vec::new(),
                    tags: Vec::new(),
                },
                chunks: vec![ArtifactChunk {
                    doc_index: 0,
                    ord: 0,
                    space: Space::Image,
                    modality: "image".to_string(),
                    heading_path: String::new(),
                    text: String::new(),
                    vector: Vec::new(),
                }],
                needs_vectors: true,
                is_image: true,
                awaiting_title: Some(title.clone()),
            },
        );
        Ok(AddedImage {
            reused: false,
            title: Some(title),
        })
    }

    /// Supply an image's vision-tower vector.
    pub fn set_image_vector(&mut self, path: &str, vector: &[f32]) -> Result<()> {
        let dim = self.image_dim()?;
        let Some(pending) = self.documents.get_mut(path) else {
            bail!("'{path}' was never added");
        };
        if !pending.is_image {
            bail!("'{path}' is not an image; use set_vectors");
        }
        if vector.len() != dim {
            bail!(
                "'{path}' needs {dim} floats for its image, not {}",
                vector.len()
            );
        }
        pending.chunks[0].vector = vector.to_vec();
        pending.needs_vectors = false;
        Ok(())
    }

    /// Supply a document's title-proxy vector, from CLIP's text tower.
    pub fn set_title_vector(&mut self, path: &str, vector: &[f32]) -> Result<()> {
        let dim = self.image_dim()?;
        let Some(pending) = self.documents.get_mut(path) else {
            bail!("'{path}' was never added");
        };
        let Some(title) = pending.awaiting_title.take() else {
            bail!("'{path}' is not waiting for a title vector");
        };
        if vector.len() != dim {
            bail!(
                "'{path}' needs {dim} floats for its title, not {}",
                vector.len()
            );
        }
        pending.chunks.push(ArtifactChunk {
            doc_index: 0,
            ord: pending.chunks.len() as u32,
            space: Space::Image,
            modality: "text_title".to_string(),
            heading_path: String::new(),
            text: title,
            vector: vector.to_vec(),
        });
        Ok(())
    }

    fn image_dim(&self) -> Result<usize> {
        match &self.spec.image {
            Some(image) => Ok(image.dim),
            None => bail!("this build has no image space"),
        }
    }

    /// Supply one document's vectors, flat and in chunk order.
    pub fn set_vectors(&mut self, path: &str, vectors: &[f32]) -> Result<()> {
        let dim = self.spec.dim;
        let Some(pending) = self.documents.get_mut(path) else {
            bail!("'{path}' was never added");
        };
        if pending.is_image {
            bail!("'{path}' is an image; use set_image_vector");
        }
        // The text chunks only: a title proxy may already have joined them.
        let text_chunks = pending
            .chunks
            .iter()
            .filter(|c| c.space == Space::Text)
            .count();
        let expected = text_chunks * dim;
        if vectors.len() != expected {
            bail!(
                "'{path}' has {text_chunks} chunks at {dim}d, so it needs {expected} floats, not {}",
                vectors.len()
            );
        }
        for (i, chunk) in pending
            .chunks
            .iter_mut()
            .filter(|c| c.space == Space::Text)
            .enumerate()
        {
            chunk.vector = vectors[i * dim..(i + 1) * dim].to_vec();
        }
        pending.needs_vectors = false;
        Ok(())
    }

    /// How many documents are waiting for vectors, for a progress report.
    pub fn pending(&self) -> usize {
        self.documents
            .values()
            .filter(|p| p.needs_vectors || p.awaiting_title.is_some())
            .count()
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
            .find(|(_, p)| p.needs_vectors || p.awaiting_title.is_some())
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

        let mut specs = vec![SpaceSpec {
            space: Space::Text,
            dim: self.spec.dim,
        }];
        if let Some(image) = &self.spec.image {
            specs.push(SpaceSpec {
                space: Space::Image,
                dim: image.dim,
            });
        }
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
                spaces: {
                    let mut spaces = vec![SpaceInfo {
                        space: Space::Text,
                        model: ModelInfo {
                            id: self.spec.model.clone(),
                            pooling: self.spec.pooling.clone(),
                            dim: self.spec.dim,
                            query_id: None,
                            query_prefix: self.spec.query_prefix.clone(),
                            document_prefix: self.spec.document_prefix.clone(),
                        },
                    }];
                    if let Some(image) = &self.spec.image {
                        spaces.push(SpaceInfo {
                            space: Space::Image,
                            model: ModelInfo {
                                id: image.model.clone(),
                                pooling: CLIP_TEXT_POOLING.to_string(),
                                dim: image.dim,
                                query_id: Some(image.query_id.clone()),
                                query_prefix: String::new(),
                                document_prefix: String::new(),
                            },
                        });
                    }
                    spaces
                },
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

    /// The previous artifact's entry for `path`, if its content is unchanged
    /// and it is of `kind`.
    ///
    /// Text chunks are carried when this build has a text space to fill. The
    /// image space's chunks — an image's vector, a title proxy — are carried
    /// only if the previous artifact's image space is the one this build
    /// records; a document missing its proxy is reused and asks for it.
    fn reuse(&self, path: &str, hash: &str, kind: &str) -> Option<Pending> {
        let artifact = self.previous.as_ref()?;
        let (index, entry) = artifact
            .documents()
            .iter()
            .enumerate()
            .find(|(_, d)| d.path == path)?;
        if entry.content_hash != hash || entry.kind != kind {
            return None;
        }
        let image_ok = self.previous_image_ok;
        let chunks: Vec<ArtifactChunk> = artifact
            .chunks()
            .iter()
            .filter(|c| {
                c.doc_index == index
                    && match c.space {
                        Space::Text => kind == "markdown",
                        Space::Image => image_ok,
                    }
            })
            .cloned()
            .collect();
        let is_image = kind == "image";
        // A document with no chunks cannot be reused: there would be nothing
        // to retrieve it by, and re-chunking it is what fixes that. An image
        // whose own vector is missing is in the same state.
        let has_body = chunks.iter().any(|c| {
            if is_image {
                c.modality == "image"
            } else {
                c.space == Space::Text
            }
        });
        if !has_body {
            return None;
        }
        let has_title = chunks.iter().any(|c| c.modality == "text_title");
        let awaiting_title = (self.spec.image.is_some() && !has_title).then(|| entry.title.clone());
        Some(Pending {
            entry: entry.clone(),
            chunks,
            needs_vectors: false,
            is_image,
            awaiting_title,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(image: bool) -> BuildSpec {
        BuildSpec {
            model: "bge-small-en-v1.5".to_string(),
            pooling: "cls".to_string(),
            dim: 2,
            query_prefix: String::new(),
            document_prefix: String::new(),
            root: "vault".to_string(),
            quantization: Quantization::F32,
            compression: Compression::None,
            shard_docs: 256,
            max_tokens: default_max_tokens(),
            overlap: default_overlap(),
            image: image.then(|| ImageSpec {
                model: "clip-vit-b-32".to_string(),
                query_id: "clip-vit-b-32".to_string(),
                dim: 3,
            }),
        }
    }

    fn open(built: &Built) -> Artifact {
        let json = built.manifest.to_json().unwrap();
        let mut shards = built.shards.clone();
        // Shards a rebuild left untouched are not in `built.shards`; these
        // tests build from nothing, so every shard is.
        shards.sort();
        Artifact::open(&json, &shards).unwrap()
    }

    /// A note and an image, each with its vectors supplied.
    fn note_and_image(builder: &mut ArtifactBuilder) {
        let note = builder.add_document("a.md", "# Alpha\n\nbody words", 10);
        assert_eq!(note.texts.len(), 1);
        assert_eq!(note.title.as_deref(), Some("Alpha"));
        builder.set_vectors("a.md", &[1.0, 0.0]).unwrap();
        builder.set_title_vector("a.md", &[0.0, 1.0, 0.0]).unwrap();

        let image = builder.add_image("img/pic.png", "deadbeef", 20).unwrap();
        assert!(!image.reused);
        assert_eq!(image.title.as_deref(), Some("pic"));
        builder
            .set_image_vector("img/pic.png", &[0.0, 0.0, 1.0])
            .unwrap();
        builder
            .set_title_vector("img/pic.png", &[1.0, 0.0, 0.0])
            .unwrap();
    }

    #[test]
    fn an_image_space_records_both_clip_models_and_carries_images_and_title_proxies() {
        let mut builder = ArtifactBuilder::new(spec(true));
        note_and_image(&mut builder);
        let built = builder.finish(1).unwrap();
        let artifact = open(&built);

        let image = artifact
            .manifest()
            .space(Space::Image)
            .expect("an image space");
        assert_eq!(image.id, "clip-vit-b-32");
        assert_eq!(image.dim, 3);
        assert_eq!(image.pooling, "mean");
        assert_eq!(image.query_id.as_deref(), Some("clip-vit-b-32"));

        let kinds: Vec<(&str, &str)> = artifact
            .documents()
            .iter()
            .map(|d| (d.path.as_str(), d.kind.as_str()))
            .collect();
        assert_eq!(kinds, [("a.md", "markdown"), ("img/pic.png", "image")]);
        assert_eq!(artifact.documents()[1].title, "pic");
        assert_eq!(artifact.documents()[1].content_hash, "deadbeef");

        let mut chunks: Vec<(usize, u32, Space, &str)> = artifact
            .chunks()
            .iter()
            .map(|c| (c.doc_index, c.ord, c.space, c.modality.as_str()))
            .collect();
        chunks.sort_by_key(|c| (c.0, c.1));
        assert_eq!(
            chunks,
            [
                (0, 0, Space::Text, "text"),
                (0, 1, Space::Image, "text_title"),
                (1, 0, Space::Image, "image"),
                (1, 1, Space::Image, "text_title"),
            ]
        );
        let vision = artifact
            .chunks()
            .iter()
            .find(|c| c.modality == "image")
            .unwrap();
        assert_eq!(vision.vector, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_build_without_an_image_space_is_what_it_was() {
        let mut builder = ArtifactBuilder::new(spec(false));
        let note = builder.add_document("a.md", "# Alpha\n\nbody", 10);
        assert_eq!(note.title, None, "no title proxy to embed");
        builder.set_vectors("a.md", &[1.0, 0.0]).unwrap();
        let built = builder.finish(1).unwrap();
        let artifact = open(&built);
        assert!(artifact.manifest().space(Space::Image).is_none());
        assert!(artifact.chunks().iter().all(|c| c.space == Space::Text));

        let mut builder = ArtifactBuilder::new(spec(false));
        let err = builder
            .add_image("pic.png", "00", 1)
            .expect_err("no image space");
        assert!(err.to_string().contains("no image space"), "{err}");
    }

    #[test]
    fn a_document_still_owed_its_title_proxy_cannot_finish() {
        let mut builder = ArtifactBuilder::new(spec(true));
        builder.add_document("a.md", "# Alpha\n\nbody", 10);
        builder.set_vectors("a.md", &[1.0, 0.0]).unwrap();
        assert_eq!(builder.pending(), 1);
        let err = builder.finish(1).expect_err("no title vector");
        assert!(err.to_string().contains("still waiting"), "{err}");
    }

    #[test]
    fn vectors_of_the_wrong_size_or_for_the_wrong_kind_are_refused() {
        let mut builder = ArtifactBuilder::new(spec(true));
        builder.add_document("a.md", "# Alpha\n\nbody", 10);
        builder.add_image("pic.png", "00", 1).unwrap();
        assert!(
            builder.set_title_vector("a.md", &[1.0, 0.0]).is_err(),
            "2d for a 3d space"
        );
        assert!(builder.set_image_vector("pic.png", &[1.0]).is_err());
        assert!(
            builder.set_image_vector("a.md", &[1.0, 0.0, 0.0]).is_err(),
            "a note is not an image"
        );
        assert!(
            builder.set_vectors("pic.png", &[1.0, 0.0]).is_err(),
            "an image has no text chunks"
        );
    }

    #[test]
    fn a_rebuild_reuses_unchanged_images_and_title_proxies() {
        let mut first = ArtifactBuilder::new(spec(true));
        note_and_image(&mut first);
        let built = first.finish(1).unwrap();

        let mut second = ArtifactBuilder::reusing(spec(true), open(&built)).unwrap();
        let note = second.add_document("a.md", "# Alpha\n\nbody words", 10);
        assert!(note.reused);
        assert_eq!(note.title, None, "its proxy is already in the index");
        let image = second.add_image("img/pic.png", "deadbeef", 20).unwrap();
        assert!(image.reused);
        assert_eq!(image.title, None);
        assert_eq!(second.pending(), 0);
        let rebuilt = second.finish(2).unwrap();
        assert_eq!(rebuilt.chunks, 4, "both documents and both title proxies");
        assert!(rebuilt.shards.is_empty(), "nothing to rewrite");
        assert_eq!(rebuilt.unchanged.len(), built.shards.len());

        // A changed image is embedded again.
        let mut third = ArtifactBuilder::reusing(spec(true), open(&built)).unwrap();
        let image = third.add_image("img/pic.png", "cafe", 30).unwrap();
        assert!(!image.reused);
    }

    #[test]
    fn a_text_only_index_is_reused_and_asks_only_for_the_title_proxies_it_lacks() {
        let mut text = ArtifactBuilder::new(spec(false));
        text.add_document("a.md", "# Alpha\n\nbody words", 10);
        text.set_vectors("a.md", &[1.0, 0.0]).unwrap();
        let built = text.finish(1).unwrap();

        let mut with_images = ArtifactBuilder::reusing(spec(true), open(&built)).unwrap();
        let note = with_images.add_document("a.md", "# Alpha\n\nbody words", 10);
        assert!(note.reused, "its text vectors are not embedded again");
        assert!(note.texts.is_empty());
        assert_eq!(note.title.as_deref(), Some("Alpha"));
        assert_eq!(with_images.pending(), 1);
        with_images
            .set_title_vector("a.md", &[0.0, 1.0, 0.0])
            .unwrap();
        assert_eq!(with_images.pending(), 0);
        assert!(with_images.finish(2).is_ok());
    }

    #[test]
    fn an_image_space_from_another_model_is_not_reused() {
        let mut first = ArtifactBuilder::new(spec(true));
        note_and_image(&mut first);
        let built = first.finish(1).unwrap();

        let mut other = spec(true);
        other.image.as_mut().unwrap().dim = 4;
        let mut rebuild = ArtifactBuilder::reusing(other, open(&built)).unwrap();
        let image = rebuild.add_image("img/pic.png", "deadbeef", 20).unwrap();
        assert!(
            !image.reused,
            "vectors from another space are never mixed in"
        );
        let note = rebuild.add_document("a.md", "# Alpha\n\nbody words", 10);
        assert!(note.reused, "the text space is unchanged");
        assert_eq!(note.title.as_deref(), Some("Alpha"));
    }

    #[test]
    fn the_hash_of_bytes_is_the_hash_of_the_same_text() {
        assert_eq!(content_hash_bytes(b"abc"), content_hash("abc"));
    }
}
