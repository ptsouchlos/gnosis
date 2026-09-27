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

    /// Identifies the format from the file's **contents**, not its extension.
    ///
    /// `image::image_dimensions` would guess from the path, which silently
    /// fails on the misnamed files vaults accumulate — a WebP or JPEG saved
    /// as `.png` by a browser or screenshot tool. Those are formats gnosis
    /// supports; only the name is wrong, so the bytes are what to trust.
    /// Still a header-only probe: no pixel data is decoded.
    fn image_dimensions(&self, path: &Path) -> Option<(u32, u32)> {
        image::ImageReader::open(path)
            .ok()?
            .with_guessed_format()
            .ok()?
            .into_dimensions()
            .ok()
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

    /// A 3x2 lossless WebP. Vaults accumulate these under a `.png` name —
    /// browsers and screenshot tools save WebP bytes with whatever extension
    /// the source URL had. WebP is a format gnosis supports, so the only thing
    /// wrong with such a file is its name.
    const WEBP_3X2: &[u8] = &[
        0x52, 0x49, 0x46, 0x46, 0x1E, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x4C, 0x11, 0x00, 0x00, 0x00, 0x2F, 0x02, 0x40, 0x00, 0x00, 0x07, 0x50, 0x8F, 0x22, 0x17,
        0xA5, 0xFF, 0x81, 0x88, 0xE8, 0x7F, 0x00, 0x00
    ];

    /// Encode a 3x2 JPEG at test time rather than embedding ~630 bytes of
    /// baseline JPEG (fixed Huffman/quantization tables make a hand-embedded
    /// one an order of magnitude larger than the WebP above).
    fn jpeg_3x2() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(3, 2, image::Rgb([200, 30, 40]));
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut std::io::Cursor::new(&mut bytes))
            .encode_image(&image::DynamicImage::ImageRgb8(img))
            .unwrap();
        bytes
    }

    fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "gnosis-fs-fmt-{}-{}",
            std::process::id(),
            name
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn image_dimensions_reads_webp_misnamed_as_png() {
        let path = write_temp("webp-as.png", WEBP_3X2);
        let dims = StdFs.image_dimensions(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(
            dims,
            Some((3, 2)),
            "a WebP named .png must still be read: format comes from the bytes, not the name"
        );
    }

    #[test]
    fn image_dimensions_reads_jpeg_misnamed_as_png() {
        let path = write_temp("jpeg-as.png", &jpeg_3x2());
        let dims = StdFs.image_dimensions(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(dims, Some((3, 2)), "a JPEG named .png must still be read");
    }

    #[test]
    fn image_dimensions_none_for_empty_file() {
        let path = write_temp("empty.png", b"");
        let dims = StdFs.image_dimensions(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(dims, None, "a 0-byte file has nothing to sniff");
    }

    #[test]
    fn image_dimensions_none_for_garbage() {
        let path = write_temp("garbage.png", b"this is not an image at all");
        let dims = StdFs.image_dimensions(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(dims, None, "unrecognizable bytes must not be guessed at");
    }
}
