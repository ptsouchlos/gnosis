//! Each submodule implements one `gnosis` subcommand: it defines the command's
//! `*Args` struct (the CLI input) and an `execute` function that runs it.

use anyhow::Result;
use embed::{EmbedConfig, Embedder};

use crate::embedder::{build_clip_text_embedder, build_image_embedder, build_text_embedder};

pub mod forget;
pub mod index;
pub mod init;
pub mod related;
pub mod rebuild;
pub mod search;
pub mod status;

/// Print an indexing run's collected errors (files skipped rather than
/// aborting the run — currently only possible for images), shared by
/// `index`/`rebuild`. No-op when there are none.
pub(crate) fn print_index_errors(errors: &[::index::IndexError]) {
    if errors.is_empty() {
        return;
    }
    println!("{} file(s) failed to index:", errors.len());
    for err in errors {
        println!("  {} — {}", err.path, err.message);
    }
}

/// Owned embedder trio built from config, from which callers assemble an
/// `index::EmbedderSet` (whose `text` field borrows rather than owns).
pub(crate) struct Embedders {
    pub text: Box<dyn Embedder>,
    pub image: Option<Box<dyn Embedder>>,
    pub image_text: Option<Box<dyn Embedder>>,
}

/// Build the text embedder, and — when `[embed.image] enabled = true` — the
/// image and CLIP-text embedders image ingestion needs, from `config`.
/// Shared by `index`/`rebuild`, which both assemble the same trio into an
/// `index::EmbedderSet` before running.
pub(crate) fn build_embedders(config: &EmbedConfig) -> Result<Embedders> {
    let text = build_text_embedder(&config.text.model)?;
    let (image, image_text) = if config.image.enabled {
        (
            Some(build_image_embedder(&config.image.model)?),
            Some(build_clip_text_embedder(&config.image.model)?),
        )
    } else {
        (None, None)
    };
    Ok(Embedders { text, image, image_text })
}
