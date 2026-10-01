//! Digital (text-layer) PDF parsing.
//!
//! Scanned PDFs are out of scope: with no text layer they extract to nothing
//! and are rejected like any other unreadable file. OCR is a later milestone.

use anyhow::{Result, bail};
use pdf_extract::{Document, Object};

/// A parsed PDF: a title and its per-page text.
#[derive(Debug, Clone)]
pub struct ParsedPdf {
    /// Metadata `Title` when the document carries a usable one, else empty —
    /// the caller substitutes the file stem, which is the only fallback that
    /// doesn't require guessing from body text.
    pub title: String,
    /// `(1-indexed page number, extracted text)` per page.
    pub pages: Vec<(u32, String)>,
}

/// Extract per-page text and the metadata title from PDF bytes.
///
/// Fails when the bytes aren't a readable PDF, or when nothing anywhere in the
/// document yields text. The latter is what a scanned, image-only PDF looks
/// like, and failing is the honest outcome: indexing it would store a document
/// with no searchable content. The indexing pipeline treats this as a
/// per-file soft failure, so one such PDF costs only itself.
pub fn parse_pdf(bytes: &[u8]) -> Result<ParsedPdf> {
    let pages: Vec<String> = extract_pages_without_panicking(bytes)?;

    let pages: Vec<(u32, String)> = pages
        .into_iter()
        .enumerate()
        .map(|(i, text)| (i as u32 + 1, text))
        .collect();

    if pages.iter().all(|(_, text)| text.trim().is_empty()) {
        bail!("no extractable text (an image-only or scanned PDF has no text layer)");
    }

    Ok(ParsedPdf {
        title: metadata_title(bytes).unwrap_or_default(),
        pages,
    })
}

/// Run `pdf_extract`'s page extraction, turning a panic into an error.
///
/// `pdf_extract` indexes content-stream operands without checking their length
/// in around fifty places, so a malformed operator — a `lineto` with no
/// coordinates, say — panics instead of returning `Err`. Left uncaught that
/// unwinds straight through the indexing pipeline's per-file error handling and
/// aborts the whole run, so a single bad PDF anywhere in a vault would cost
/// every other file. Containing it here keeps the "one bad file costs only
/// itself" contract that the rest of the pipeline is built around.
///
/// Only panics raised inside `pdf_extract` are silenced; anything else still
/// reaches the previous hook, so a genuine bug in gnosis is never hidden.
fn extract_pages_without_panicking(bytes: &[u8]) -> Result<Vec<String>> {
    install_quiet_pdf_panic_hook();

    // `&[u8]` holds no interior mutability, so there is no broken invariant to
    // observe after an unwind; the closure borrows it and returns owned data.
    let extracted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text_from_mem_by_pages(bytes)
    }));

    match extracted {
        Ok(Ok(pages)) => Ok(pages),
        Ok(Err(e)) => bail!("pdf text extraction failed: {e}"),
        Err(_) => bail!(
            "pdf text extraction panicked (malformed content stream); \
             the file was skipped"
        ),
    }
}

/// Install, once, a panic hook that stays quiet for panics originating inside
/// `pdf_extract` and delegates everything else to the hook that was already in
/// place.
///
/// Without this, every malformed PDF prints a `panicked at ...` backtrace
/// notice to stderr mid-index, which reads like a crash even though the file is
/// being skipped cleanly. Installed once rather than swapped around each call,
/// so there is no window in which another thread's panic message is lost.
fn install_quiet_pdf_panic_hook() {
    use std::sync::Once;
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let from_pdf_extract = info
                .location()
                .is_some_and(|l| l.file().contains("pdf-extract"));
            if !from_pdf_extract {
                previous(info);
            }
        }));
    });
}

/// The document's `Info` dictionary `Title`, if it has a non-empty one.
///
/// `pdf_extract` re-exports `lopdf`, but keeps its own metadata helpers
/// private, so the `Info` dictionary is read directly here.
fn metadata_title(bytes: &[u8]) -> Option<String> {
    let doc = Document::load_mem(bytes).ok()?;
    let info = match doc.trailer.get(b"Info").ok()? {
        Object::Reference(id) => doc.get_object(*id).ok()?.as_dict().ok()?,
        Object::Dictionary(dict) => dict,
        _ => return None,
    };
    let raw = info.get(b"Title").ok()?.as_str().ok()?;
    let title = decode_pdf_string(raw);
    let title = title.trim();
    (!title.is_empty()).then(|| title.to_string())
}

/// Decode a PDF text string.
///
/// Two encodings are permitted: UTF-16BE, flagged by a byte-order mark, and
/// PDFDocEncoding otherwise. PDFDocEncoding agrees with Latin-1 across the
/// range titles actually use, and treating it as such avoids carrying the full
/// table for a field that only ever supplies a display name.
fn decode_pdf_string(raw: &[u8]) -> String {
    if raw.starts_with(&[0xFE, 0xFF]) {
        let units: Vec<u16> = raw[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_be_bytes(*pair))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    raw.iter().map(|&b| b as char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_plain_latin1_string() {
        assert_eq!(decode_pdf_string(b"Hello"), "Hello");
    }

    #[test]
    fn decodes_a_utf16be_string_with_a_bom() {
        // BOM + "Hi" in UTF-16BE.
        let raw = [0xFE, 0xFF, 0x00, 0x48, 0x00, 0x69];
        assert_eq!(decode_pdf_string(&raw), "Hi");
    }

    #[test]
    fn decodes_a_latin1_accent() {
        // 0xE9 is 'é' in both Latin-1 and PDFDocEncoding.
        assert_eq!(decode_pdf_string(&[0x52, 0xE9, 0x73]), "Rés");
    }

    #[test]
    fn an_empty_string_decodes_to_empty() {
        assert_eq!(decode_pdf_string(b""), "");
    }
}
