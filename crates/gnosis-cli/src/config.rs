use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Default config file name, looked up relative to the current directory.
pub const CONFIG_FILE: &str = "gnosis.toml";

/// Top-level gnosis configuration, deserialized from `gnosis.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Root directory of the knowledge base to index.
    pub vault: PathBuf,
    /// Directory (relative to `vault`) holding the SQLite db and ANN indexes.
    pub db_dir: PathBuf,
    pub embed: EmbedConfig,
    pub chunk: ChunkConfig,
    pub pdf: PdfConfig,
    pub ignore: IgnoreConfig,
    pub obsidian: ObsidianConfig,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkConfig {
    pub max_tokens: usize,
    pub overlap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PdfConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IgnoreConfig {
    pub globs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObsidianConfig {
    /// Auto-detect Obsidian features (wikilinks, frontmatter) per file.
    pub auto: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vault: PathBuf::from("."),
            db_dir: PathBuf::from(".gnosis"),
            embed: EmbedConfig::default(),
            chunk: ChunkConfig::default(),
            pdf: PdfConfig::default(),
            ignore: IgnoreConfig::default(),
            obsidian: ObsidianConfig::default(),
        }
    }
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

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 384,
            overlap: 64,
        }
    }
}

impl Default for PdfConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl Default for IgnoreConfig {
    fn default() -> Self {
        Self {
            globs: vec![
                "node_modules/**".to_string(),
                ".git/**".to_string(),
                ".gnosis/**".to_string(),
            ],
        }
    }
}

impl Default for ObsidianConfig {
    fn default() -> Self {
        Self { auto: true }
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

    /// Absolute path to the database/index directory.
    pub fn db_dir(&self) -> PathBuf {
        self.vault.join(&self.db_dir)
    }

    /// Path to the SQLite database file.
    pub fn db_path(&self) -> PathBuf {
        self.db_dir().join("gnosis.db")
    }
}
