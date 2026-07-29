use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::commands;

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

    /// Operate on the global index (~/.gnosis) instead of the local vault.
    #[arg(long, short, global = true, conflicts_with = "config")]
    pub global: bool,

    #[command(subcommand)]
    pub command: Command,
}

/// Each variant wraps the `Args` struct defined in the matching `commands` module.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Write a default gnosis.toml in the current directory.
    Init(commands::init::InitArgs),
    /// Build or incrementally update the index for the vault.
    Index(commands::index::IndexArgs),
    /// Semantic search over the indexed content.
    Search(commands::search::SearchArgs),
    /// List notes related to a given file.
    Related(commands::related::RelatedArgs),
    /// Remove a vault from the index and delete its documents.
    Forget(commands::forget::ForgetArgs),
    /// Show index statistics.
    Status(commands::status::StatusArgs),
    /// Force a full re-embed and rebuild of the index.
    Rebuild(commands::rebuild::RebuildArgs),
}
