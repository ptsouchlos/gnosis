use std::path::PathBuf;

use anyhow::{Result, bail};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// An embedder produces L2-normalized vectors for a single vector space,
/// so cosine similarity reduces to a dot product.
pub trait Embedder {
    /// The vector space this embedder feeds (e.g. "text", "image").
    fn space(&self) -> &str;
    /// Dimensionality of the produced vectors.
    fn dim(&self) -> usize;
    /// Identifier of the underlying model, stored in `meta` to guard against
    /// mixing vectors from different models in one index.
    fn model_id(&self) -> &str;
    /// Embed a batch of inputs, preserving order.
    fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>>;
}

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
