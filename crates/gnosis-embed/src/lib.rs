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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedConfig {
    pub text: TextEmbedConfig,
    pub image: ImageEmbedConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TextEmbedConfig {
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageEmbedConfig {
    pub enabled: bool,
    pub model: String,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            text: TextEmbedConfig::default(),
            image: ImageEmbedConfig::default(),
        }
    }
}

impl Default for TextEmbedConfig {
    fn default() -> Self {
        Self {
            model: "bge-small-en-v1.5".to_string(),
        }
    }
}

impl Default for ImageEmbedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "clip-vit-b-32".to_string(),
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
}
