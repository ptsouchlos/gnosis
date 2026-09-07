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
}
