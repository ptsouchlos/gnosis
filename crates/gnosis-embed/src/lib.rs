use anyhow::Result;
use serde::{Deserialize, Serialize};

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

/// Text/image embedding configuration, shared by every `Embedder` implementor
/// (native fastembed today, a JS-backed embedder in a future WASM plugin).
/// `Default` is derived: both fields carry their own non-trivial `Default`
/// impls below, and a derive delegates to those.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedConfig {
    pub text: TextEmbedConfig,
    pub image: ImageEmbedConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TextEmbedConfig {
    pub model: String,
    /// How many chunks to embed per model call.
    ///
    /// Bounds peak memory: ONNX activation memory scales with
    /// batch size × sequence length, and a single large document can produce
    /// hundreds of near-max-length chunks. Left unbounded (fastembed's default
    /// of 256) a 112-page PDF peaked at 3.5 GB, and the worst file in a real
    /// corpus reached 6.1 GB — enough to get the process OOM-killed. Smaller
    /// is safer but slower; raise it if you have memory to spare.
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageEmbedConfig {
    pub enabled: bool,
    pub model: String,
    /// How many pending image/title embeds to accumulate before issuing one
    /// batched `Embedder::embed` call, instead of one call per file.
    pub batch_size: usize,
}

impl Default for TextEmbedConfig {
    fn default() -> Self {
        Self {
            model: "bge-small-en-v1.5".to_string(),
            batch_size: 16,
        }
    }
}

impl Default for ImageEmbedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "clip-vit-b-32".to_string(),
            batch_size: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_config_default_matches_previous_values() {
        let cfg = EmbedConfig::default();
        assert_eq!(cfg.text.model, "bge-small-en-v1.5");
        assert!(!cfg.image.enabled);
        assert_eq!(cfg.image.model, "clip-vit-b-32");
    }

    #[test]
    fn image_embed_config_default_batch_size_is_32() {
        let cfg = EmbedConfig::default();
        assert_eq!(cfg.image.batch_size, 32);
    }

    #[test]
    fn text_batch_size_is_bounded_by_default() {
        let cfg = TextEmbedConfig::default();
        assert!(
            cfg.batch_size > 0 && cfg.batch_size <= 32,
            "the default must bound ONNX activation memory, got {}",
            cfg.batch_size
        );
    }

    #[test]
    fn text_batch_size_round_trips_through_toml() {
        let cfg: TextEmbedConfig = toml::from_str("batch_size = 4\n").unwrap();
        assert_eq!(cfg.batch_size, 4);
        assert_eq!(cfg.model, TextEmbedConfig::default().model, "other fields keep their defaults");
    }

    #[test]
    fn omitting_text_batch_size_keeps_the_default() {
        let cfg: TextEmbedConfig = toml::from_str("model = \"all-MiniLM-L6-v2\"\n").unwrap();
        assert_eq!(cfg.batch_size, TextEmbedConfig::default().batch_size);
    }
}
