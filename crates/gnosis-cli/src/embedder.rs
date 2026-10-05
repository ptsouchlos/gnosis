use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use fastembed::{EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions, TextEmbedding};

pub use embed::Embedder;

/// Text embedder backed by a local fastembed (ONNX) model.
pub struct TextEmbedder {
    model: TextEmbedding,
    model_id: String,
    dim: usize,
    /// Instruction prepended to a search query, empty when the model wants
    /// none or the user turned it off. See `Embedder::embed_query`.
    query_prefix: &'static str,
    /// Longest input the model accepts, in its own tokens.
    max_input_tokens: usize,
    /// Chunks per model call. Bounds ONNX activation memory, which scales with
    /// batch size × sequence length; see `TextEmbedConfig::batch_size`.
    batch_size: usize,
}

impl TextEmbedder {
    /// Construct from a config model name. Downloads/caches the model on first use.
    pub fn new(model_name: &str, batch_size: usize, query_instruction: bool) -> Result<Self> {
        let (model, dim, _, max_input_tokens) = resolve_text_model(model_name)?;
        let prefix = resolve_query_prefix(model_name, query_instruction)?;
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
            query_prefix: prefix,
            max_input_tokens,
            batch_size: batch_size.max(1),
        })
    }
}

/// Build the configured text embedder as a trait object, so callers (e.g. the
/// `index` crate) stay agnostic to which concrete backend is in use.
pub fn build_text_embedder(cfg: &embed::TextEmbedConfig) -> Result<Box<dyn Embedder>> {
    Ok(Box::new(TextEmbedder::new(
        &cfg.model,
        cfg.batch_size,
        cfg.query_instruction,
    )?))
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
        let vectors = self.model.embed(inputs, Some(self.batch_size))?;
        Ok(vectors)
    }

    fn max_input_tokens(&self) -> Option<usize> {
        Some(self.max_input_tokens)
    }

    fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{query}", self.query_prefix);
        self.embed(std::slice::from_ref(&prefixed))?
            .into_iter()
            .next()
            .context("embedding produced no vector")
    }
}

/// The query-side instruction each model was trained with, or `""` for a
/// symmetric model. fastembed applies none of these itself — it embeds queries
/// and documents identically — so gnosis has to carry them.
///
/// BAAI's for the bge English v1.5 family; nomic's `search_query:`/
/// `search_document:` pair, where the document side is *also* required, so it
/// stays empty until indexing can prefix documents too. `all-MiniLM-L6-v2` is
/// symmetric and wants nothing.
const BGE_EN_QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";

/// A resolved text model: fastembed enum, dimensionality, query instruction,
/// and the longest input it accepts in its own tokens.
type TextModel = (EmbeddingModel, usize, &'static str, usize);

/// Map a config model name to its fastembed enum, dimensionality, query
/// instruction, and input limit.
fn resolve_text_model(name: &str) -> Result<TextModel> {
    let m = match name {
        "bge-small-en-v1.5" => (EmbeddingModel::BGESmallENV15, 384, BGE_EN_QUERY_INSTRUCTION, 512),
        "bge-small-en-v1.5-q" => (EmbeddingModel::BGESmallENV15Q, 384, BGE_EN_QUERY_INSTRUCTION, 512),
        "bge-base-en-v1.5" => (EmbeddingModel::BGEBaseENV15, 768, BGE_EN_QUERY_INSTRUCTION, 512),
        "all-MiniLM-L6-v2" => (EmbeddingModel::AllMiniLML6V2, 384, "", 256),
        "nomic-embed-text-v1.5" => (EmbeddingModel::NomicEmbedTextV15, 768, "", 8192),
        other => bail!(
            "unknown text model '{other}' (try: bge-small-en-v1.5, bge-small-en-v1.5-q, \
             bge-base-en-v1.5, all-MiniLM-L6-v2, nomic-embed-text-v1.5)"
        ),
    };
    Ok(m)
}

/// The query instruction to prepend for `model_name`, or `""` when the model is
/// symmetric or the user disabled the behavior.
///
/// Reads the instruction out of `resolve_text_model`'s table rather than keeping
/// a second list of model names, so a model added there cannot silently lose
/// its instruction.
fn resolve_query_prefix(model_name: &str, enabled: bool) -> Result<&'static str> {
    if !enabled {
        return Ok("");
    }
    Ok(resolve_text_model(model_name)?.2)
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

    /// Each input is a file path. The bytes are read here and handed to
    /// `embed_bytes` rather than letting fastembed open the paths itself:
    /// its path-taking `embed` picks a decoder from the file extension, which
    /// fails on the misnamed images vaults accumulate (a WebP or JPEG saved
    /// under a `.png` name). `embed_bytes` guesses the format from the content,
    /// so a supported image is read regardless of what it is called.
    fn embed(&mut self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let bytes: Vec<Vec<u8>> = inputs
            .iter()
            .map(|path| std::fs::read(path).with_context(|| format!("reading image {path}")))
            .collect::<Result<_>>()?;
        let slices: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
        let vectors = self.model.embed_bytes(&slices, None)?;
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

/// Titles are short, so a larger batch is safe here than for body chunks —
/// but still bounded, for the same reason.
const CLIP_TEXT_BATCH_SIZE: usize = 32;

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
        let vectors = self.model.embed(inputs, Some(CLIP_TEXT_BATCH_SIZE))?;
        Ok(vectors)
    }
}

fn resolve_clip_text_model(name: &str) -> Result<(EmbeddingModel, usize)> {
    match name {
        "clip-vit-b-32" => Ok((EmbeddingModel::ClipVitB32, 512)),
        other => bail!("unknown image model '{other}' (try: clip-vit-b-32)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn resolve_text_model_recognizes_quantized_bge_small() {
        let (model, dim, _, _) = resolve_text_model("bge-small-en-v1.5-q").expect("known model name");
        assert_eq!(model, EmbeddingModel::BGESmallENV15Q);
        assert_eq!(dim, 384);
    }

    #[test]
    fn bge_models_carry_a_query_instruction_and_symmetric_ones_do_not() {
        for bge in ["bge-small-en-v1.5", "bge-small-en-v1.5-q", "bge-base-en-v1.5"] {
            assert_eq!(
                resolve_query_prefix(bge, true).expect("known model"),
                BGE_EN_QUERY_INSTRUCTION,
                "{bge} is asymmetric and wants its instruction"
            );
        }
        assert_eq!(
            resolve_query_prefix("all-MiniLM-L6-v2", true).expect("known model"),
            "",
            "a symmetric model must not get an instruction"
        );
    }

    #[test]
    fn disabling_query_instruction_clears_it_even_for_a_model_that_wants_one() {
        assert_eq!(
            resolve_query_prefix("bge-small-en-v1.5", false).expect("known model"),
            "",
            "query_instruction = false must reproduce pre-instruction behavior exactly"
        );
    }

    /// The instruction has to be prepended to the query and *only* the query:
    /// prefixing indexed content as well would shift every stored vector and
    /// silently invalidate the index.
    ///
    /// Loads the real model (downloads on first run), so it's network-gated.
    /// Run with: cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore = "downloads model and runs inference"]
    fn query_instruction_changes_the_query_vector_but_not_the_document_vector() {
        let text = "ergonomic keyboard layouts".to_string();

        let mut with = TextEmbedder::new("bge-small-en-v1.5", 16, true).expect("load model");
        let mut without = TextEmbedder::new("bge-small-en-v1.5", 16, false).expect("load model");

        // Documents go through `embed`, which must be identical either way.
        let doc_with = with.embed(std::slice::from_ref(&text)).expect("embed");
        let doc_without = without.embed(std::slice::from_ref(&text)).expect("embed");
        assert_eq!(
            doc_with[0], doc_without[0],
            "query_instruction must not touch indexed content"
        );

        // Queries go through `embed_query`, where it must take effect.
        let q_with = with.embed_query(&text).expect("embed query");
        let q_without = without.embed_query(&text).expect("embed query");
        assert_ne!(
            q_with, q_without,
            "the instruction should change the query vector"
        );
        assert_eq!(
            q_without, doc_without[0],
            "with the instruction off, a query embeds exactly like a document"
        );

        let norm: f32 = q_with.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
    }

    /// Loads the real model (downloads on first run), so it's network-gated and
    /// excluded from the default `cargo test` run. Run with:
    ///   cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore = "downloads model and runs inference"]
    fn embeds_text_sanely() {
        let mut embedder = TextEmbedder::new("bge-small-en-v1.5", 16, true).expect("load model");
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

    /// A WebP or JPEG saved under a `.png` name is still a supported image, so
    /// it must embed. fastembed's path-taking `embed` picks its decoder from
    /// the file extension and fails on these; `embed_bytes` sniffs the content
    /// instead. Network-gated like the tests above. Run with:
    ///   cargo test --release -- --ignored --nocapture misnamed
    #[test]
    #[ignore = "downloads models and runs inference"]
    fn embeds_images_whose_extension_lies_about_the_format() {
        // 3x2 lossless WebP, deliberately written under a .png name.
        let webp: &[u8] = &[
            0x52, 0x49, 0x46, 0x46, 0x1E, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
            0x4C, 0x11, 0x00, 0x00, 0x00, 0x2F, 0x02, 0x40, 0x00, 0x00, 0x07, 0x50, 0x8F, 0x22, 0x17,
            0xA5, 0xFF, 0x81, 0x88, 0xE8, 0x7F, 0x00, 0x00
        ];
        let mut embedder = ImageEmbedder::new("clip-vit-b-32").expect("load image model");

        let path = std::env::temp_dir()
            .join(format!("gnosis-embedder-misnamed-{}.png", std::process::id()));
        std::fs::write(&path, webp).unwrap();

        let result = embedder.embed(&[path.to_string_lossy().to_string()]);
        std::fs::remove_file(&path).ok();

        let vectors = result.expect("a WebP named .png must embed: format comes from the bytes");
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].len(), 512);
    }

}
