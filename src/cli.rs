use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "gnosis",
    version,
    about,
    long_about = "Local semantic search and related-notes over a markdown knowledge base."
)]
pub struct Cli {
    /// Path to the config file (defaults to ./gnosis.toml).
    #[arg(long, short, global = true)]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Write a default gnosis.toml in the current directory.
    Init {
        /// Overwrite an existing config file.
        #[arg(long)]
        force: bool,
    },

    /// Build or incrementally update the index for the vault.
    Index {
        /// Vault path to index (overrides the config `vault`).
        path: Option<PathBuf>,
    },

    /// Semantic search over the indexed content.
    Search {
        /// The natural-language query.
        query: String,
        /// Restrict to these vector spaces (e.g. text,image). Defaults to all.
        #[arg(long, value_delimiter = ',')]
        r#in: Vec<String>,
        /// Maximum number of results.
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Print matching chunk text, not just paths.
        #[arg(long)]
        full: bool,
        /// Emit results as JSON.
        #[arg(long)]
        json: bool,
    },

    /// List notes related to a given file.
    Related {
        /// The file to find related items for.
        file: PathBuf,
        /// Maximum number of results.
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Include items already linked from the source (Obsidian vaults).
        #[arg(long)]
        include_linked: bool,
        /// Emit results as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Show index statistics.
    Status,

    /// Force a full re-embed and rebuild of the index.
    Rebuild,
}
