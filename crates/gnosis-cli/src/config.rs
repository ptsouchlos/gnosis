use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chunker::ChunkConfig;
use embed::EmbedConfig;
use serde::{Deserialize, Serialize};

/// Default config file name, looked up relative to the current directory.
pub const CONFIG_FILE: &str = "gnosis.toml";

/// Top-level gnosis configuration, deserialized from `gnosis.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Knowledge-base roots to index. In local mode this is a single entry
    /// (defaulting to "."); in global mode it may list several vaults.
    pub vaults: Vec<PathBuf>,
    /// Directory (relative to the config location) holding the SQLite db and
    /// ANN indexes.
    pub db_dir: PathBuf,
    pub embed: EmbedConfig,
    pub chunk: ChunkConfig,
    pub pdf: PdfConfig,
    pub ignore: IgnoreConfig,
    pub ann: AnnConfig,
    #[serde(default)]
    pub search: search::bm25::SearchConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PdfConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AnnConfig {
    /// On-disk ANN index vector precision: "f16" (the default), "f32" (full
    /// precision), or "i8" (8-bit, smallest).
    ///
    /// Measured nDCG@10 and index size, three corpora:
    ///
    /// | corpus | f32 | f16 | i8 | f32 size | i8 size |
    /// | --- | --- | --- | --- | --- | --- |
    /// | scifact | 0.722 | 0.722 | 0.725 | 10.3 MB | 3.3 MB |
    /// | nfcorpus | 0.347 | 0.346 | 0.342 | 7.6 MB | 2.4 MB |
    /// | mr-tydi korean | 0.841 | 0.840 | 0.835 | 33.0 MB | 10.4 MB |
    ///
    /// f16 is free — within 0.001 everywhere, at 1.84x smaller — which is why it
    /// is the default. i8 costs at most 0.005 for 3.16x, worth taking when disk
    /// matters; the korean row is the honest one for both, since it is scored
    /// with the lexical channel off and so nothing masks the dense error.
    ///
    /// SQLite always stores full f32 vectors regardless, so `gnosis rebuild`
    /// can always safely regenerate the index at a new quantization.
    pub quantization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IgnoreConfig {
    pub globs: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vaults: vec![PathBuf::from(".")],
            db_dir: PathBuf::from(".gnosis"),
            embed: EmbedConfig::default(),
            chunk: ChunkConfig::default(),
            pdf: PdfConfig::default(),
            ignore: IgnoreConfig::default(),
            ann: AnnConfig::default(),
            search: search::bm25::SearchConfig::default(),
        }
    }
}

impl Default for PdfConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl Default for AnnConfig {
    fn default() -> Self {
        Self { quantization: "f16".to_string() }
    }
}

impl Default for IgnoreConfig {
    fn default() -> Self {
        Self {
            globs: vec![
                "node_modules/**".to_string(),
                ".git/**".to_string(),
                ".gnosis/**".to_string(),
                ".obsidian/**".to_string(),
            ],
        }
    }
}

impl Config {
    /// Load config from an explicit path, or fall back to `./gnosis.toml`,
    /// or defaults if no file exists.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let path = explicit.map(Path::to_path_buf).or_else(|| {
            let default = PathBuf::from(CONFIG_FILE);
            default.exists().then_some(default)
        });

        match path {
            Some(p) => {
                let text = std::fs::read_to_string(&p)
                    .with_context(|| format!("reading config {}", p.display()))?;
                let cfg: Config = toml::from_str(&text)
                    .with_context(|| format!("parsing config {}", p.display()))?;
                Ok(cfg)
            }
            None => Ok(Config::default()),
        }
    }

    /// Serialize the config to TOML.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).context("serializing config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f16 is the default because it measured free — within 0.001 nDCG@10 of f32
    /// on three corpora, at 1.84x smaller on disk. If this ever goes back to
    /// f32, the measurement in `AnnConfig`'s docs is the thing to re-check.
    #[test]
    fn ann_quantization_defaults_to_f16() {
        assert_eq!(AnnConfig::default().quantization, "f16");
    }

    /// Every supported value has to resolve, since the default is now one of
    /// them and `gnosis index` resolves it on every run.
    #[test]
    fn every_documented_quantization_resolves() {
        for name in ["f32", "f16", "i8"] {
            assert!(
                crate::store::resolve_quantization(name).is_ok(),
                "{name} is documented in AnnConfig but does not resolve"
            );
        }
        assert!(crate::store::resolve_quantization("f8").is_err());
    }
}
