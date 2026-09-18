use std::path::PathBuf;

use anyhow::{Result, bail};
use fastembed::{EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions, TextEmbedding};

pub use embed::Embedder;

/// Text embedder backed by a local fastembed (ONNX) model.
pub struct TextEmbedder {
    model: TextEmbedding,
    model_id: String,
    dim: usize,
}

impl TextEmbedder {
    /// Construct from a config model name. Downloads/caches the model on first use.
    pub fn new(model_name: &str) -> Result<Self> {
        let (model, dim) = resolve_text_model(model_name)?;
        let mut opts = InitOptions::new(model);
        if let Some(dir) = model_cache_dir() {
            std::fs::create_dir_all(&dir).ok();
            opts = opts.with_cache_dir(dir);
        }
        let embedding = TextEmbedding::try_new(opts)?;
        Ok(Self {
            model: embedding,
            model_id: model_name.to_string(),
            dim,
        })
    }
}

/// Build the configured text embedder as a trait object, so callers (e.g. the
/// `index` crate) stay agnostic to which concrete backend is in use.
pub fn build_text_embedder(model_name: &str) -> Result<Box<dyn Embedder>> {
    Ok(Box::new(TextEmbedder::new(model_name)?))
}

/// Stable per-user cache directory for downloaded models, so fastembed doesn't
/// litter a `.fastembed_cache` in the current working directory. Falls back to
/// fastembed's default when no cache dir can be determined.
fn model_cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("gnosis").join("models"))
}

impl Embedder for TextEmbedder {
    fn space(&self) -> &str {
        "text"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.model.embed(inputs.to_vec(), None)?;
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Loads the real model (downloads on first run), so it's network-gated and
    /// excluded from the default `cargo test` run. Run with:
    ///   cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore = "downloads model and runs inference"]
    fn embeds_text_sanely() {
        let mut embedder = TextEmbedder::new("bge-small-en-v1.5").expect("load model");
        assert_eq!(embedder.dim(), 384);

        let inputs = vec![
            "the king ruled the kingdom".to_string(),
            "the queen ruled the kingdom".to_string(),
            "I replaced the carburetor in my car".to_string(),
        ];
        let v = embedder.embed(&inputs).expect("embed");

        assert_eq!(v.len(), 3);
        assert_eq!(v[0].len(), 384);

        // fastembed returns L2-normalized vectors.
        let norm: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");

        // king/queen should be closer than king/carburetor.
        let kq = cosine(&v[0], &v[1]);
        let kc = cosine(&v[0], &v[2]);
        println!("cos(king,queen)={kq:.4}  cos(king,carburetor)={kc:.4}");
        assert!(kq > kc, "semantic ordering wrong: {kq} !> {kc}");
    }

    /// Loads real CLIP models (downloads on first run) — network-gated,
    /// excluded from the default `cargo test` run, same as
    /// `embeds_text_sanely`. Run with:
    ///   cargo test --release -- --ignored --nocapture clip
    #[test]
    #[ignore = "downloads models and runs inference"]
    fn clip_text_and_image_embeddings_share_a_space() {
        let mut image_embedder = ImageEmbedder::new("clip-vit-b-32").expect("load image model");
        let mut text_embedder = ClipTextEmbedder::new("clip-vit-b-32").expect("load text model");
        assert_eq!(image_embedder.dim(), text_embedder.dim());
        assert_eq!(image_embedder.space(), "image");
        assert_eq!(text_embedder.space(), "image");

        // A verified-valid 2x1 red/blue PNG, embedded as a temp file so the
        // test needs no external fixture.
        let png: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
            0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x7B, 0x40, 0xE8, 0xDD, 0x00, 0x00, 0x00,
            0x0F, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xC0,
            0xC0, 0xF0, 0x1F, 0x00, 0x07, 0x00, 0x01, 0xFF, 0x7E, 0x08, 0xB1, 0xD0,
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let path = std::env::temp_dir().join(format!("gnosis-embedder-test-{}.png", std::process::id()));
        std::fs::write(&path, png).unwrap();

        let image_vec = image_embedder
            .embed(&[path.to_string_lossy().to_string()])
            .expect("embed image")
            .into_iter()
            .next()
            .unwrap();
        std::fs::remove_file(&path).ok();

        let matching_vec = text_embedder
            .embed(&["a small red and blue image".to_string()])
            .expect("embed matching caption")
            .into_iter()
            .next()
            .unwrap();
        let unrelated_vec = text_embedder
            .embed(&["a recipe for chocolate cake".to_string()])
            .expect("embed unrelated caption")
            .into_iter()
            .next()
            .unwrap();

        let cosine = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let matching_score = cosine(&image_vec, &matching_vec);
        let unrelated_score = cosine(&image_vec, &unrelated_vec);
        println!("matching={matching_score:.4} unrelated={unrelated_score:.4}");
        assert!(matching_score > unrelated_score);
    }
}

/// Map a config model name to a fastembed model enum and its dimensionality.
fn resolve_text_model(name: &str) -> Result<(EmbeddingModel, usize)> {
    let m = match name {
        "bge-small-en-v1.5" => (EmbeddingModel::BGESmallENV15, 384),
        "bge-base-en-v1.5" => (EmbeddingModel::BGEBaseENV15, 768),
        "all-MiniLM-L6-v2" => (EmbeddingModel::AllMiniLML6V2, 384),
        "nomic-embed-text-v1.5" => (EmbeddingModel::NomicEmbedTextV15, 768),
        other => bail!(
            "unknown text model '{other}' (try: bge-small-en-v1.5, bge-base-en-v1.5, \
             all-MiniLM-L6-v2, nomic-embed-text-v1.5)"
        ),
    };
    Ok(m)
}

/// Image embedder backed by a local fastembed (ONNX) CLIP vision model.
/// `embed()`'s inputs are file paths, not text — matches `TextEmbedder`'s
/// convention of treating each input string as "the thing to embed" for its
/// modality.
pub struct ImageEmbedder {
    model: ImageEmbedding,
    model_id: String,
    dim: usize,
}

impl ImageEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let (model, dim) = resolve_image_model(model_name)?;
        let opts = ImageInitOptions::new(model);
        let embedding = ImageEmbedding::try_new(opts)?;
        Ok(Self {
            model: embedding,
            model_id: model_name.to_string(),
            dim,
        })
    }
}

pub fn build_image_embedder(model_name: &str) -> Result<Box<dyn Embedder>> {
    Ok(Box::new(ImageEmbedder::new(model_name)?))
}

impl Embedder for ImageEmbedder {
    fn space(&self) -> &str {
        "image"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.model.embed(inputs.to_vec(), None)?;
        Ok(vectors)
    }
}

fn resolve_image_model(name: &str) -> Result<(ImageEmbeddingModel, usize)> {
    match name {
        "clip-vit-b-32" => Ok((ImageEmbeddingModel::ClipVitB32, 512)),
        other => bail!("unknown image model '{other}' (try: clip-vit-b-32)"),
    }
}

/// Text embedder backed by the CLIP *text* encoder — produces vectors in
/// the same space as `ImageEmbedder`'s CLIP *vision* encoder, so a text
/// query (or a document title, for the title-proxy mechanism) can be
/// compared against real image embeddings.
pub struct ClipTextEmbedder {
    model: TextEmbedding,
    model_id: String,
    dim: usize,
}

impl ClipTextEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let (model, dim) = resolve_clip_text_model(model_name)?;
        let opts = InitOptions::new(model);
        let embedding = TextEmbedding::try_new(opts)?;
        Ok(Self {
            model: embedding,
            model_id: model_name.to_string(),
            dim,
        })
    }
}

pub fn build_clip_text_embedder(model_name: &str) -> Result<Box<dyn Embedder>> {
    Ok(Box::new(ClipTextEmbedder::new(model_name)?))
}

impl Embedder for ClipTextEmbedder {
    fn space(&self) -> &str {
        "image"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.model.embed(inputs.to_vec(), None)?;
        Ok(vectors)
    }
}

fn resolve_clip_text_model(name: &str) -> Result<(EmbeddingModel, usize)> {
    match name {
        "clip-vit-b-32" => Ok((EmbeddingModel::ClipVitB32, 512)),
        other => bail!("unknown image model '{other}' (try: clip-vit-b-32)"),
    }
}
