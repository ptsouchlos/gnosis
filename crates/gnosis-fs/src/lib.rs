//! Interface for reading files during indexing. Kept free of any concrete
//! backend's dependencies (e.g. `std::fs`) so it can be depended on by
//! non-native targets (e.g. a future wasm build) that need only the shape of
//! file access, not a native filesystem.
use std::path::{Path, PathBuf};

use anyhow::Result;

/// Reads and inspects files for the indexing pipeline.
pub trait FileReader {
    /// Resolve a path to its canonical form (symlinks resolved, normalized).
    fn canonicalize(&self, path: &Path) -> Result<PathBuf>;

    /// Read a file's full contents.
    fn read(&self, path: &Path) -> Result<Vec<u8>>;

    /// Last-modified time as unix seconds, or 0 if unavailable.
    fn mtime(&self, path: &Path) -> i64;
}
