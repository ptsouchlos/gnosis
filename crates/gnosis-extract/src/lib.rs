//! WebAssembly bindings that extract per-page text from PDF bytes.
//!
//! This is the half of the PDF pipeline that is allowed to crash a wasm
//! instance. [`parse::parse_pdf`] runs `pdf-extract` inside `catch_unwind`,
//! which in a native build turns a malformed PDF into a per-file error — but
//! this crate is built with the `wasm-release` profile, whose `panic =
//! "abort"` makes a panic a trap that `catch_unwind` cannot catch. Here a
//! malformed PDF does not return an error: it traps the call and kills the
//! whole instance. A malformed PDF is a hostile input, and the host should
//! not be able to lose its query client because one document was broken.
//!
//! So the extraction runs in its own disposable instance: this crate, loaded in
//! a worker the host respawns per file. The host feeds the bytes, gets back
//! the title and pages, and never holds the instance across files it cares
//! about. This crate is deliberately a separate bundle from the query client
//! rather than a second export on the same `cdylib`: a Rust crate links as one
//! `.wasm`, and the query client's linker would then carry `pdf-extract` — and
//! its `lopdf` and the font-parsing crates it drags in — into every query, for
//! the one feature (reading a PDF) that most queries do not touch.

use js_sys::JSON;
use wasm_bindgen::prelude::*;

use serde::Serialize;

/// One page of extracted text, as the host sees it.
#[derive(Serialize)]
struct Page {
    /// 1-indexed page number, as `parse_pdf` reports it.
    page: u32,
    /// The page's text, or empty for a page that yielded none.
    text: String,
}

/// The extracted document: a title and its pages.
#[derive(Serialize)]
struct Extracted {
    /// The document's metadata `Title`, or empty when it carries none — the
    /// host substitutes the file stem, the only fallback that does not require
    /// guessing from body text.
    title: String,
    /// `(page number, text)` per page, 1-indexed, in document order.
    pages: Vec<Page>,
}

/// Extract the title and per-page text from PDF bytes.
///
/// `bytes` is the whole file; it is read as a `Uint8Array`. Returns `{ title,
/// pages }` on success, where `pages` is an array of `{ page, text }` in
/// document order. Rejects bytes that are not a readable PDF and documents that
/// yield no text anywhere (scanned, image-only PDFs) — the same two failures
/// the indexing pipeline treats as a per-file skip, so the host learns the
/// difference between "wrong file type" and "no text layer" from one call.
#[wasm_bindgen(js_name = extractPdf)]
pub fn extract_pdf(bytes: &[u8]) -> Result<JsValue, JsError> {
    let parsed = parse::parse_pdf(bytes).map_err(anyhow_err)?;
    to_js(&Extracted {
        title: parsed.title,
        pages: parsed
            .pages
            .into_iter()
            .map(|(page, text)| Page { page, text })
            .collect(),
    })
}

/// Turn an `anyhow::Error` into a `JsError` that carries the full context chain.
///
/// `anyhow`'s `{:#}` form prints every layer of context, which is what makes a
/// failure readable from the JS console rather than a bare message.
fn anyhow_err(err: anyhow::Error) -> JsError {
    JsError::new(&format!("{err:#}"))
}

/// Serialize to a JS object across the boundary: `serde_json` to a string,
/// then `JSON.parse` into a native object. JSON is the boundary's data format,
/// so a struct and its JS form cannot drift.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    let json = serde_json::to_string(value).map_err(|e| JsError::new(&e.to_string()))?;
    JSON::parse(&json).map_err(js_err)
}

/// Turn a `JsValue` failure into a `JsError` with the value's string form.
fn js_err(err: JsValue) -> JsError {
    JsError::new(&JSON::stringify(&err).map_or_else(|_| "unknown error".into(), String::from))
}
