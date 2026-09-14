//! `std::fs`-backed [`FileReader`] implementation. Native, so it lives in the
//! CLI binary crate rather than the `fs` interface crate — mirrors how
//! `store.rs`'s `SqliteStore` and `walk.rs`'s `FsWalker` stay out of their
//! interface crates.
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
pub use fs::FileReader;

/// `std::fs`-backed [`FileReader`].
pub struct StdFs;

impl FileReader for StdFs {
    fn canonicalize(&self, path: &Path) -> Result<PathBuf> {
        std::fs::canonicalize(path).with_context(|| format!("resolving {}", path.display()))
    }

    fn read(&self, path: &Path) -> Result<Vec<u8>> {
        std::fs::read(path).with_context(|| format!("reading {}", path.display()))
    }

    fn mtime(&self, path: &Path) -> i64 {
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    fn image_dimensions(&self, path: &Path) -> Option<(u32, u32)> {
        image::image_dimensions(path).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_dimensions_reads_real_png() {
        // 2x1 minimal PNG, embedded so the test has no fixture file to manage.
        let png: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
            0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x7B, 0x40, 0xE8, 0xDD, 0x00, 0x00, 0x00,
            0x0F, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xC0,
            0xC0, 0xF0, 0x1F, 0x00, 0x07, 0x00, 0x01, 0xFF, 0x7E, 0x08, 0xB1, 0xD0,
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let path = std::env::temp_dir().join(format!("gnosis-fs-test-{}.png", std::process::id()));
        std::fs::write(&path, png).unwrap();

        let dims = StdFs.image_dimensions(&path);

        std::fs::remove_file(&path).ok();
        assert_eq!(dims, Some((2, 1)));
    }

    #[test]
    fn image_dimensions_none_for_missing_file() {
        let path = std::env::temp_dir().join("gnosis-fs-test-does-not-exist.png");
        assert_eq!(StdFs.image_dimensions(&path), None);
    }
}
