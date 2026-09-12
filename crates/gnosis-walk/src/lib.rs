//! Interface for discovering indexable files under a vault root. Kept free of
//! any concrete backend's dependencies (e.g. the `ignore` crate) so it can be
//! depended on by non-native targets (e.g. a future wasm build) that need
//! only the shape of a walker, not a filesystem implementation.
use std::path::{Path, PathBuf};

use anyhow::Result;

/// File kinds gnosis knows how to index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocKind {
    Markdown,
    Image,
}

impl DocKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DocKind::Markdown => "markdown",
            DocKind::Image => "image",
        }
    }

    /// Classify a path by extension, or `None` if gnosis doesn't index it.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("md") => Some(DocKind::Markdown),
            Some(ext)
                if ext.eq_ignore_ascii_case("png")
                    || ext.eq_ignore_ascii_case("jpg")
                    || ext.eq_ignore_ascii_case("jpeg")
                    || ext.eq_ignore_ascii_case("webp") =>
            {
                Some(DocKind::Image)
            }
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

/// Discovers indexable files under a vault root.
pub trait Walker {
    /// Walk `root` for indexable files, honoring `.gitignore` and the
    /// configured ignore globs. Hidden files/dirs are skipped by default.
    fn discover(&self, root: &Path, ignore_globs: &[String]) -> Result<Vec<Found>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn classifies_markdown_and_image_extensions() {
        assert_eq!(DocKind::from_path(Path::new("note.md")), Some(DocKind::Markdown));
        assert_eq!(DocKind::from_path(Path::new("photo.PNG")), Some(DocKind::Image));
        assert_eq!(DocKind::from_path(Path::new("photo.jpg")), Some(DocKind::Image));
        assert_eq!(DocKind::from_path(Path::new("photo.JPEG")), Some(DocKind::Image));
        assert_eq!(DocKind::from_path(Path::new("photo.webp")), Some(DocKind::Image));
        assert_eq!(DocKind::from_path(Path::new("photo.gif")), None);
        assert_eq!(DocKind::from_path(Path::new("readme.txt")), None);
        assert_eq!(DocKind::Image.as_str(), "image");
    }
}
