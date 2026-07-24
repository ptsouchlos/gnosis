use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;

/// File kinds gnosis knows how to index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocKind {
    Markdown,
}

impl DocKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DocKind::Markdown => "markdown",
        }
    }

    /// Classify a path by extension, or `None` if gnosis doesn't index it.
    fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("md") => Some(DocKind::Markdown),
            _ => None,
        }
    }
}

/// A discovered file to (potentially) index.
#[derive(Debug)]
pub struct Found {
    pub path: PathBuf,
    pub kind: DocKind,
}

/// Walk `root` for indexable files, honoring `.gitignore` and the configured
/// ignore globs. Hidden files/dirs are skipped by default.
pub fn discover(root: &Path, ignore_globs: &[String]) -> Result<Vec<Found>> {
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
