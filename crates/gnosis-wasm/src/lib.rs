//! WebAssembly bindings for querying a portable gnosis artifact.
//!
//! This is the whole of what leaves Rust for the Obsidian plugin. Everything
//! that decides a ranking — the two channels, their normalizations, the
//! fusion weight, the title bonus, the per-space merge, BM25+'s parameters —
//! stays behind this boundary, in [`query`] and [`search`], where it is
//! measured. The plugin supplies bytes, a query vector and a limit.
//!
//! It lives in the gnosis workspace rather than in the plugin repo for the
//! same reason: an API defined next to the code it wraps cannot drift from it
//! without a failing test here.
//!
//! **The embedder stays outside.** [`Index::search`] takes a vector, not a
//! string. That is what keeps ONNX Runtime out of this build — the client
//! embeds with transformers.js, using the model id and *pooling* that
//! [`Index::info`] reports, since the index and the query must agree about
//! both or the similarity numbers are meaningless rather than merely worse.
//!
//! **JSON is the boundary's data format.** Arguments and results cross as
//! plain JS objects, converted through `JSON.stringify`/`JSON.parse` and
//! `serde_json`. It costs a serialization pass per call on objects that are
//! kilobytes at most, and it buys one conversion path rather than a
//! hand-written one per type.

pub mod builder;

use std::collections::HashMap;

use js_sys::{Float32Array, JSON, Map, Uint8Array};
use serde::{Deserialize, Serialize};
use store::artifact::{Artifact, Compression, Quantization};
use store::artifact_store::ArtifactStore;
use store::{Space, TextQuery};
use wasm_bindgen::prelude::*;

/// A loaded artifact, queried through gnosis's own retrieval pipeline.
#[wasm_bindgen]
pub struct Index {
    store: ArtifactStore,
    cfg: search::bm25::SearchConfig,
}

/// The `--from` / `--tag` filter, as a client sends it.
///
/// `from` matches a root *label* rather than a path: an artifact records what
/// its vaults were called, not where they lived on the machine that built it.
#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Filter {
    from: Vec<String>,
    tags: Vec<String>,
}

impl Filter {
    fn as_query(&self) -> TextQuery<'_> {
        TextQuery {
            from: (!self.from.is_empty()).then_some(self.from.as_slice()),
            tags: (!self.tags.is_empty()).then_some(self.tags.as_slice()),
        }
    }
}

/// What a space's model is, as a client needs it to embed a query.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SpaceReport {
    space: String,
    /// The model that embedded the stored vectors.
    model: String,
    /// The model a *query* goes through. The same as `model` for text; for the
    /// image space it is CLIP's text tower, which is a different model than
    /// the vision tower that embedded the images.
    query_model: String,
    /// How token vectors were pooled into one. A client that guesses this
    /// produces healthy-looking vectors with much of their usable range
    /// thrown away, so it is reported rather than assumed.
    pooling: String,
    dim: usize,
    chunks: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InfoReport {
    format_version: u32,
    /// When the artifact was built. An artifact is never fresher than this.
    created_unix: i64,
    documents: usize,
    chunks: usize,
    quantization: Quantization,
    compression: Compression,
    roots: Vec<String>,
    spaces: Vec<SpaceReport>,
}

/// One entry of a client's own view of its vault, for [`Index::staleness`].
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VaultEntry {
    /// Vault-relative, matching what the artifact stores.
    path: String,
    /// blake3 of the file's content, hex, if the client computes one.
    #[serde(default)]
    hash: Option<String>,
    /// Modification time in seconds, used only when `hash` is absent.
    #[serde(default)]
    mtime: Option<i64>,
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct StalenessReport {
    created_unix: i64,
    /// In the vault and in the artifact, but the content differs.
    changed: Vec<String>,
    /// In the vault, absent from the artifact.
    added: Vec<String>,
    /// In the artifact, absent from the vault.
    removed: Vec<String>,
    /// Documents the artifact holds and the vault agrees with.
    unchanged: usize,
    /// Entries whose freshness could not be determined because the client
    /// supplied neither a hash nor an mtime. Reported rather than counted as
    /// unchanged: a client cannot be told "nothing has changed" on the
    /// strength of a comparison that never happened.
    undetermined: Vec<String>,
}

#[wasm_bindgen]
impl Index {
    /// Assemble an index from bytes the client has already read.
    ///
    /// `shards` maps each shard's name, as the manifest lists it, to its
    /// bytes. Shards are verified against the manifest's hashes, so a
    /// partially synced artifact fails to open rather than answering queries
    /// from a mixture of two indexes.
    pub fn open(manifest_json: &str, shards: &Map) -> Result<Index, JsError> {
        let artifact = Artifact::open(manifest_json, &shard_bytes(shards)?).map_err(anyhow_err)?;
        Ok(Index {
            store: ArtifactStore::new(artifact),
            cfg: search::bm25::SearchConfig::default(),
        })
    }

    /// Override the retrieval configuration.
    ///
    /// The artifact does not carry one — it is the indexing machine's local
    /// setting, and the defaults here are the measured ones. A client that
    /// changes it is choosing to diverge from what this project's numbers
    /// describe, which is why it is a separate, explicit call.
    #[wasm_bindgen(js_name = setSearchConfig)]
    pub fn set_search_config(&mut self, config: JsValue) -> Result<(), JsError> {
        self.cfg = from_js(&config)?;
        Ok(())
    }

    /// The configuration in force, so a client can report it.
    #[wasm_bindgen(js_name = searchConfig)]
    pub fn search_config(&self) -> Result<JsValue, JsError> {
        to_js(&self.cfg)
    }

    /// Run a hybrid query: one vector per space, fused with the lexical
    /// channel in the text space.
    ///
    /// `vectors` maps a space name (`"text"`, `"image"`) to a `Float32Array`.
    /// The spaces searched are exactly the ones a vector was supplied for —
    /// a space without an embedder is not silently skipped, it is simply not
    /// asked about.
    pub fn search(
        &self,
        query: &str,
        vectors: &Map,
        limit: usize,
        filter: JsValue,
    ) -> Result<JsValue, JsError> {
        let mut embedder = Supplied::from_js(vectors, &self.store)?;
        let spaces = embedder.spaces();
        let filter: Filter = optional_from_js(&filter)?;
        // The query string is not redundant with the vector: the lexical
        // channel and the title bonus read the text, the dense channel the
        // vector. Passing `""` is a legitimate dense-only query, which is what
        // an image query is.
        let hits = query::run_query(
            &self.store,
            &mut embedder,
            query,
            &spaces,
            limit,
            &filter.as_query(),
            &self.cfg,
        )
        .map_err(anyhow_err)?;
        to_js(&hits)
    }

    /// Search the lexical channel alone, with no model at all.
    ///
    /// This is the tier a client can offer before it has downloaded anything:
    /// BM25+ over the artifact's own text, scored and ranked by the same code
    /// the hybrid path uses, with the title bonus applied. On scifact it
    /// scores 0.666 nDCG@10 against the hybrid path's 0.722 — worse, and far
    /// from nothing.
    #[wasm_bindgen(js_name = lexicalSearch)]
    pub fn lexical_search(
        &self,
        query: &str,
        limit: usize,
        filter: JsValue,
    ) -> Result<JsValue, JsError> {
        let filter: Filter = optional_from_js(&filter)?;
        // An empty dense side rather than a separate code path: fusion with
        // one channel absent is the same convex combination with one term at
        // zero, so the scores are comparable with a hybrid query's and the
        // ranking comes from the same function.
        let hits = query::fuse_text_space(
            &self.store,
            query,
            Vec::new(),
            limit,
            &filter.as_query(),
            &self.cfg,
        )
        .map_err(anyhow_err)?;
        to_js(&hits)
    }

    /// Documents most similar to `path`'s own chunks — the other model-free
    /// tier, at full quality, since both sides of the comparison are vectors
    /// the artifact already carries.
    ///
    /// Unless `include_linked` is set, the source and everything it links to
    /// are excluded: the point is unlinked mentions. Link resolution here
    /// matches filename stems only — an artifact carries no frontmatter, so a
    /// host with its own alias index (Obsidian has one) should resolve aliases
    /// there and pass the results in a filter rather than expect this to.
    pub fn related(
        &self,
        path: &str,
        limit: usize,
        include_linked: bool,
        filter: JsValue,
    ) -> Result<JsValue, JsError> {
        let filter: Filter = optional_from_js(&filter)?;
        let spaces = self.available_spaces();
        let hits = query::run_related(
            &self.store,
            path,
            &spaces,
            limit,
            include_linked,
            &filter.as_query(),
            query::no_aliases,
        )
        .map_err(anyhow_err)?;
        to_js(&hits)
    }

    /// What this artifact is: its models and their pooling, its dimensions,
    /// how much it holds and when it was built.
    pub fn info(&self) -> Result<JsValue, JsError> {
        let manifest = self.store.artifact().manifest();
        let chunks = |space: Space| {
            self.store
                .artifact()
                .chunks()
                .iter()
                .filter(|c| c.space == space)
                .count()
        };
        let spaces: Vec<SpaceReport> = manifest
            .spaces
            .iter()
            .map(|s| SpaceReport {
                space: s.space.as_str().to_string(),
                model: s.model.id.clone(),
                query_model: s
                    .model
                    .query_id
                    .clone()
                    .unwrap_or_else(|| s.model.id.clone()),
                pooling: s.model.pooling.clone(),
                dim: s.model.dim,
                chunks: chunks(s.space),
            })
            .collect();
        to_js(&InfoReport {
            format_version: manifest.format_version,
            created_unix: manifest.created_unix,
            documents: self.store.artifact().documents().len(),
            chunks: self.store.artifact().chunks().len(),
            quantization: manifest.quantization,
            compression: manifest.compression,
            roots: manifest.roots.clone(),
            spaces,
        })
    }

    /// Compare a client's own view of its vault against what the artifact
    /// holds.
    ///
    /// The client does the walk — only it can — and this does the comparison.
    /// Nothing here can *fix* staleness: only the machine with the models can
    /// re-index. So the requirement is an accurate report, and an entry whose
    /// freshness cannot be determined is reported as such rather than counted
    /// as fresh.
    pub fn staleness(&self, entries: JsValue) -> Result<JsValue, JsError> {
        let entries: Vec<VaultEntry> = from_js(&entries)?;
        let indexed: HashMap<&str, (&str, i64)> = self
            .store
            .artifact()
            .documents()
            .iter()
            .map(|d| (d.path.as_str(), (d.content_hash.as_str(), d.mtime)))
            .collect();

        let mut report = StalenessReport {
            created_unix: self.store.artifact().manifest().created_unix,
            ..Default::default()
        };
        let mut seen: Vec<&str> = Vec::with_capacity(entries.len());
        for entry in &entries {
            let Some((hash, mtime)) = indexed.get(entry.path.as_str()).copied() else {
                report.added.push(entry.path.clone());
                continue;
            };
            seen.push(entry.path.as_str());
            match (&entry.hash, entry.mtime) {
                (Some(theirs), _) => {
                    if theirs.eq_ignore_ascii_case(hash) {
                        report.unchanged += 1;
                    } else {
                        report.changed.push(entry.path.clone());
                    }
                }
                // An mtime is weaker evidence than a hash — it moves when
                // content does not — so it is the fallback, not the default.
                (None, Some(theirs)) => {
                    if theirs > mtime {
                        report.changed.push(entry.path.clone());
                    } else {
                        report.unchanged += 1;
                    }
                }
                (None, None) => report.undetermined.push(entry.path.clone()),
            }
        }
        for doc in self.store.artifact().documents() {
            if !seen.contains(&doc.path.as_str()) {
                report.removed.push(doc.path.clone());
            }
        }
        to_js(&report)
    }
}

impl Index {
    /// Every space the artifact actually holds, in gnosis's own order.
    fn available_spaces(&self) -> Vec<Space> {
        Space::ALL
            .iter()
            .copied()
            .filter(|s| self.store.artifact().manifest().space(*s).is_some())
            .collect()
    }
}

/// A [`query::QueryEmbedder`] over vectors the client already computed.
///
/// The pipeline asks for a vector per space; here they all arrive up front,
/// because the model that produces them lives in JS.
struct Supplied {
    vectors: Vec<(Space, Vec<f32>)>,
}

impl Supplied {
    /// Reads the `Map` and checks each vector's length against the manifest.
    ///
    /// A wrong dimension is caught here rather than producing a confidently
    /// wrong ranking: a query vector of the wrong width is not a worse query,
    /// it is a meaningless one.
    fn from_js(vectors: &Map, store: &ArtifactStore) -> Result<Self, JsError> {
        let mut out: Vec<(Space, Vec<f32>)> = Vec::new();
        let entries = vectors.entries();
        loop {
            let next = entries.next().map_err(js_err)?;
            if next.done() {
                break;
            }
            let pair = js_sys::Array::from(&next.value());
            let key = pair
                .get(0)
                .as_string()
                .ok_or_else(|| JsError::new("every vector key must be a space name"))?;
            let space: Space = key.parse().map_err(anyhow_err)?;
            let array = pair.get(1);
            let array = array.dyn_ref::<Float32Array>().ok_or_else(|| {
                JsError::new(&format!("the vector for '{key}' must be a Float32Array"))
            })?;
            let vector = array.to_vec();
            let expected = store
                .artifact()
                .manifest()
                .space(space)
                .ok_or_else(|| JsError::new(&format!("this artifact holds no '{key}' space")))?
                .dim;
            if vector.len() != expected {
                return Err(JsError::new(&format!(
                    "the '{key}' query vector has {} dimensions but the index has {expected}",
                    vector.len()
                )));
            }
            out.push((space, vector));
        }
        if out.is_empty() {
            return Err(JsError::new(
                "supply at least one space's query vector, e.g. map.set('text', vector)",
            ));
        }
        out.sort_by_key(|(space, _)| Space::ALL.iter().position(|s| s == space));
        Ok(Self { vectors: out })
    }

    fn spaces(&self) -> Vec<Space> {
        self.vectors.iter().map(|(s, _)| *s).collect()
    }
}

impl query::QueryEmbedder for Supplied {
    fn embed_query(&mut self, space: Space, _query: &str) -> anyhow::Result<Vec<f32>> {
        self.vectors
            .iter()
            .find(|(s, _)| *s == space)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| anyhow::anyhow!("no query vector was supplied for '{space}'"))
    }
}

/// Read a `Map` of shard name to bytes, as both `Index::open` and the builder
/// take them.
pub(crate) fn shard_bytes(shards: &Map) -> Result<Vec<(String, Vec<u8>)>, JsError> {
    let mut named: Vec<(String, Vec<u8>)> = Vec::with_capacity(shards.size() as usize);
    let entries = shards.entries();
    loop {
        let next = entries.next().map_err(js_err)?;
        if next.done() {
            break;
        }
        let pair = js_sys::Array::from(&next.value());
        let name = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsError::new("every shard key must be a string"))?;
        let bytes = pair.get(1);
        let bytes = bytes
            .dyn_ref::<Uint8Array>()
            .ok_or_else(|| JsError::new(&format!("shard '{name}' must be a Uint8Array")))?;
        named.push((name, bytes.to_vec()));
    }
    Ok(named)
}

/// `anyhow`'s full context chain, not just the outermost message: these errors
/// cross into a JS console where nobody can add the context back.
pub(crate) fn anyhow_err(err: anyhow::Error) -> JsError {
    JsError::new(&format!("{err:#}"))
}

pub(crate) fn js_err(err: JsValue) -> JsError {
    JsError::new(&JSON::stringify(&err).map_or_else(|_| "unknown error".into(), String::from))
}

pub(crate) fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    let json = serde_json::to_string(value).map_err(|e| JsError::new(&e.to_string()))?;
    JSON::parse(&json).map_err(js_err)
}

pub(crate) fn from_js<T: for<'de> Deserialize<'de>>(value: &JsValue) -> Result<T, JsError> {
    let json = JSON::stringify(value).map_err(js_err)?;
    serde_json::from_str(&String::from(json)).map_err(|e| JsError::new(&e.to_string()))
}

/// Like [`from_js`], but `undefined` and `null` mean "the default", so a client
/// can leave an optional filter out entirely.
fn optional_from_js<T: Default + for<'de> Deserialize<'de>>(value: &JsValue) -> Result<T, JsError> {
    if value.is_undefined() || value.is_null() {
        return Ok(T::default());
    }
    from_js(value)
}
