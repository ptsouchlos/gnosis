//! Filesystem-backed [`Walker`] implementation. Native (depends on the
//! `ignore` crate), so it lives in the CLI binary crate rather than the
//! `walk` interface crate — mirrors how `store.rs`'s `SqliteStore` stays out
//! of the `store` interface crate.
use std::path::Path;

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use walk::{DocKind, Found};
pub use walk::Walker;

/// Filesystem-backed [`Walker`] using the `ignore` crate.
pub struct FsWalker;

impl Walker for FsWalker {
    fn discover(&self, root: &Path, ignore_globs: &[String]) -> Result<Vec<Found>> {
        let mut overrides = OverrideBuilder::new(root);
        // An entry prefixed with `!` is an ignore glob. With no whitelist globs
        // present, everything else is included by default.
        for glob in ignore_globs {
            overrides
                .add(&format!("!{glob}"))
                .with_context(|| format!("invalid ignore glob '{glob}'"))?;
        }
        let overrides = overrides.build().context("building ignore overrides")?;

        let mut found = Vec::new();
        for result in WalkBuilder::new(root).overrides(overrides).build() {
            let entry = result.context("walking vault")?;
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            if let Some(kind) = DocKind::from_path(entry.path()) {
                found.push(Found {
                    path: entry.path().to_path_buf(),
                    kind,
                });
            }
        }
        Ok(found)
    }
}
