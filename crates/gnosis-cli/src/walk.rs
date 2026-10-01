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
pub struct FsWalker {
    /// Whether `.pdf` files are discovered. `[pdf] enabled` is opt-*out*
    /// (default true), so disabling it means "don't index PDFs" rather than
    /// "this vault unexpectedly contains PDFs" — hence a silent skip here,
    /// like an ignore glob, rather than an error deeper in the pipeline.
    /// Contrast `[embed.image] enabled`, which is opt-*in* and errors when an
    /// image turns up with it off, forcing an explicit choice.
    index_pdf: bool,
}

impl FsWalker {
    pub fn new(index_pdf: bool) -> Self {
        Self { index_pdf }
    }
}

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
                if kind == DocKind::Pdf && !self.index_pdf {
                    continue;
                }
                found.push(Found {
                    path: entry.path().to_path_buf(),
                    kind,
                });
            }
        }
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_vault(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gnosis-walk-test-{}-{}",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("note.md"), "# n").unwrap();
        std::fs::write(dir.join("paper.pdf"), "%PDF-1.4").unwrap();
        std::fs::write(dir.join("photo.png"), "x").unwrap();
        dir
    }

    fn kinds(found: &[Found]) -> Vec<DocKind> {
        let mut k: Vec<DocKind> = found.iter().map(|f| f.kind).collect();
        k.sort_by_key(|k| k.as_str());
        k
    }

    #[test]
    fn discovers_pdfs_when_enabled() {
        let dir = temp_vault("pdf-on");
        let found = FsWalker::new(true).discover(&dir, &[]).unwrap();
        assert_eq!(
            kinds(&found),
            vec![DocKind::Image, DocKind::Markdown, DocKind::Pdf]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Disabling PDFs reads as "don't index PDFs", not "this vault
    /// unexpectedly has PDFs" — so they are skipped silently, the way an
    /// ignore glob skips a file, rather than raising an error.
    #[test]
    fn skips_pdfs_when_disabled() {
        let dir = temp_vault("pdf-off");
        let found = FsWalker::new(false).discover(&dir, &[]).unwrap();
        assert_eq!(
            kinds(&found),
            vec![DocKind::Image, DocKind::Markdown],
            "a .pdf must not be discovered when pdf indexing is off"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabling_pdfs_does_not_affect_other_kinds() {
        let dir = temp_vault("pdf-off-others");
        let on = FsWalker::new(true).discover(&dir, &[]).unwrap();
        let off = FsWalker::new(false).discover(&dir, &[]).unwrap();
        let non_pdf = |f: &Vec<Found>| {
            let mut v: Vec<String> = f
                .iter()
                .filter(|x| x.kind != DocKind::Pdf)
                .map(|x| x.path.to_string_lossy().to_string())
                .collect();
            v.sort();
            v
        };
        assert_eq!(non_pdf(&on), non_pdf(&off));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
