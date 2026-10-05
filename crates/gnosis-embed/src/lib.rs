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

    /// Embed a *search query*, as opposed to indexed content.
    ///
    /// Asymmetric retrieval models are trained with an instruction on the query
    /// side only — `bge-small-en-v1.5` wants
    /// `"Represent this sentence for searching relevant passages: "` — so a
    /// query embedded exactly like a document is embedded in a way the model
    /// was not trained for. Only the query side differs; stored chunk vectors
    /// are unaffected, which is why `related` (document-to-document) keeps
    /// using `embed`.
    ///
    /// Defaults to `embed`, so a symmetric model implements nothing.
    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        self.embed(std::slice::from_ref(&query.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("embedding produced no vector"))
    }
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
    /// Prepend the model's own query instruction when embedding a search query.
    ///
    /// On by default, because a model that wants one was trained that way. Set
    /// `false` to reproduce pre-instruction measurements, or for a model whose
    /// instruction gnosis guesses wrong.
    pub query_instruction: bool,
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
            query_instruction: true,
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

    #[test]
    fn query_instruction_defaults_on_and_round_trips_through_toml() {
        assert!(TextEmbedConfig::default().query_instruction);
        let cfg: TextEmbedConfig = toml::from_str("query_instruction = false\n").unwrap();
        assert!(!cfg.query_instruction);
    }

    /// A symmetric embedder implements nothing, so the default must pass the
    /// query through untouched — otherwise turning the instruction on for one
    /// model would quietly alter every other backend's queries too.
    #[test]
    fn default_embed_query_passes_the_query_through_unchanged() {
        struct Spy {
            seen: Vec<String>,
        }
        impl Embedder for Spy {
            fn space(&self) -> &str {
                "text"
            }
            fn dim(&self) -> usize {
                1
            }
            fn model_id(&self) -> &str {
                "spy"
            }
            fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
                self.seen.extend_from_slice(inputs);
                Ok(inputs.iter().map(|_| vec![1.0]).collect())
            }
        }

        let mut spy = Spy { seen: Vec::new() };
        let v = spy.embed_query("tent repair").expect("embed query");
        assert_eq!(v, vec![1.0]);
        assert_eq!(spy.seen, vec!["tent repair".to_string()]);
    }
}
