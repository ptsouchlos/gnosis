//! Building an index from the client's side of the boundary.
//!
//! The counterpart to [`crate::Index`], and the same division of labor: the
//! client walks the vault, reads files and runs the embedding model, because
//! only it can; everything that decides what the index *contains* happens in
//! [`index::builder`], which is the code the CLI runs too.
//!
//! This file is the marshalling, and nothing else — `Map`s of bytes in,
//! `Float32Array`s of vectors in, JSON out.

use index::builder::{ArtifactBuilder, Built};
use js_sys::{Float32Array, Map, Uint8Array};
use serde::Serialize;
use store::artifact::Artifact;
use wasm_bindgen::prelude::*;

use crate::{anyhow_err, from_js, js_err, to_js};

/// The content hash `addDocument` records for `text`, for `Index.staleness`.
///
/// A client that sends a hash gets an exact answer, where an mtime moves
/// whenever a syncer or another plugin rewrites a file unchanged.
#[wasm_bindgen(js_name = contentHash)]
pub fn content_hash(text: &str) -> String {
    index::builder::content_hash(text)
}

/// The content hash `addImage` expects for an image file's bytes.
#[wasm_bindgen(js_name = contentHashBytes)]
pub fn content_hash_bytes(bytes: &[u8]) -> String {
    index::builder::content_hash_bytes(bytes)
}

/// What `addDocument` tells the client to do next.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Added {
    /// True when the document was carried over from the previous index
    /// unchanged, so `texts` is empty and nothing needs embedding.
    reused: bool,
    /// The exact strings to embed, in chunk order.
    texts: Vec<String>,
    /// The title to embed with CLIP's text tower for the document's title
    /// proxy, when the spec has an image space and one is still owed.
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
}

/// What `addImage` tells the client to do next.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AddedImage {
    /// True when the image was carried over unchanged; nothing is owed for it
    /// unless `title` says its title proxy is.
    reused: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
}

/// What `finish` hands back, beside the shards themselves.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    /// The manifest, ready to write as `manifest.json`.
    manifest: String,
    documents: usize,
    chunks: usize,
    /// Shards whose bytes are unchanged, by name. They are not returned for
    /// writing — see [`index::builder::Built::unchanged`].
    unchanged: Vec<String>,
}

/// Accumulates documents, then packs them into an artifact.
#[wasm_bindgen]
pub struct Builder {
    /// `None` once `finish` has consumed it, so a second `finish` says so
    /// rather than building an empty artifact.
    inner: Option<ArtifactBuilder>,
}

#[wasm_bindgen]
impl Builder {
    /// A builder with nothing to reuse: every document will be embedded.
    pub fn create(spec: JsValue) -> Result<Builder, JsError> {
        Ok(Builder {
            inner: Some(ArtifactBuilder::new(from_js(&spec)?)),
        })
    }

    /// A builder that reuses an existing artifact: a document whose content
    /// hash matches keeps its chunks *and its vectors*, so an ordinary day's
    /// editing re-embeds a handful of chunks rather than a vault.
    #[wasm_bindgen(js_name = fromArtifact)]
    pub fn from_artifact(
        spec: JsValue,
        manifest_json: &str,
        shards: &Map,
    ) -> Result<Builder, JsError> {
        let artifact =
            Artifact::open(manifest_json, &crate::shard_bytes(shards)?).map_err(anyhow_err)?;
        Ok(Builder {
            inner: Some(ArtifactBuilder::reusing(from_js(&spec)?, artifact).map_err(anyhow_err)?),
        })
    }

    /// Offer one document to the index.
    ///
    /// `mtime_ms` is a JavaScript timestamp in milliseconds; the artifact
    /// records seconds, as the CLI does from the filesystem.
    #[wasm_bindgen(js_name = addDocument)]
    pub fn add_document(
        &mut self,
        path: &str,
        text: &str,
        mtime_ms: f64,
    ) -> Result<JsValue, JsError> {
        let added = self
            .inner_mut()?
            .add_document(path, text, (mtime_ms / 1000.0).floor() as i64);
        to_js(&Added {
            reused: added.reused,
            texts: added.texts,
            title: added.title,
        })
    }

    /// Offer one image file to the index; needs `image` in the spec.
    ///
    /// `hash` is `contentHashBytes` of the file's bytes. A new image owes a
    /// vision-tower vector (`setImageVector`) and a title proxy
    /// (`setTitleVector`, with the `title` this returns).
    #[wasm_bindgen(js_name = addImage)]
    pub fn add_image(&mut self, path: &str, hash: &str, mtime_ms: f64) -> Result<JsValue, JsError> {
        let added = self
            .inner_mut()?
            .add_image(path, hash, (mtime_ms / 1000.0).floor() as i64)
            .map_err(anyhow_err)?;
        to_js(&AddedImage {
            reused: added.reused,
            title: added.title,
        })
    }

    /// Drop a document that was offered but cannot be completed, such as an
    /// image that will not decode; `gnosis index` skips those too. Returns
    /// whether it was there.
    #[wasm_bindgen(js_name = removeDocument)]
    pub fn remove_document(&mut self, path: &str) -> Result<bool, JsError> {
        Ok(self.inner_mut()?.remove(path))
    }

    /// Supply an image's vector, from CLIP's vision tower.
    #[wasm_bindgen(js_name = setImageVector)]
    pub fn set_image_vector(&mut self, path: &str, vector: &Float32Array) -> Result<(), JsError> {
        self.inner_mut()?
            .set_image_vector(path, &vector.to_vec())
            .map_err(anyhow_err)
    }

    /// Supply a document's title-proxy vector, from CLIP's text tower.
    #[wasm_bindgen(js_name = setTitleVector)]
    pub fn set_title_vector(&mut self, path: &str, vector: &Float32Array) -> Result<(), JsError> {
        self.inner_mut()?
            .set_title_vector(path, &vector.to_vec())
            .map_err(anyhow_err)
    }

    /// Supply one document's vectors, flat and in chunk order.
    ///
    /// Flat rather than an array of arrays: this is the one call that moves
    /// real volume across the boundary, and a `Float32Array` of
    /// `chunks * dim` copies once.
    #[wasm_bindgen(js_name = setVectors)]
    pub fn set_vectors(&mut self, path: &str, vectors: &Float32Array) -> Result<(), JsError> {
        let flat = vectors.to_vec();
        self.inner_mut()?
            .set_vectors(path, &flat)
            .map_err(anyhow_err)
    }

    /// How many documents are waiting for vectors, for a progress report.
    #[wasm_bindgen(js_name = pendingCount)]
    pub fn pending_count(&self) -> Result<usize, JsError> {
        Ok(self.inner()?.pending())
    }

    /// How many documents have been offered.
    #[wasm_bindgen(js_name = documentCount)]
    pub fn document_count(&self) -> Result<usize, JsError> {
        Ok(self.inner()?.len())
    }

    /// Pack everything added into shards.
    ///
    /// `created_unix` comes from the client because wasm has no clock. It is
    /// the one field that makes two otherwise identical artifacts differ.
    pub fn finish(&mut self, created_unix: f64) -> Result<JsValue, JsError> {
        let built: Built = self
            .inner
            .take()
            .ok_or_else(|| JsError::new("this build has already finished"))?
            .finish(created_unix as i64)
            .map_err(anyhow_err)?;

        let shards = Map::new();
        for (name, body) in &built.shards {
            shards.set(&JsValue::from_str(name), &Uint8Array::from(body.as_slice()));
        }
        let report = to_js(&Report {
            manifest: built.manifest.to_json().map_err(anyhow_err)?,
            documents: built.documents,
            chunks: built.chunks,
            unchanged: built.unchanged,
        })?;
        // The shards go back as bytes in a `Map` rather than through JSON,
        // which would mean base64 and a copy of a copy.
        js_sys::Reflect::set(&report, &JsValue::from_str("shards"), &shards).map_err(js_err)?;
        Ok(report)
    }
}

impl Builder {
    fn inner(&self) -> Result<&ArtifactBuilder, JsError> {
        self.inner
            .as_ref()
            .ok_or_else(|| JsError::new("this build has already finished"))
    }

    fn inner_mut(&mut self) -> Result<&mut ArtifactBuilder, JsError> {
        self.inner
            .as_mut()
            .ok_or_else(|| JsError::new("this build has already finished"))
    }
}
