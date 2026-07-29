use std::path::PathBuf;

use anyhow::{Result, bail};

use crate::workspace::Workspace;

/// List notes related to a given file.
#[derive(Debug, clap::Args)]
pub struct RelatedArgs {
    /// The file to find related items for.
    pub file: PathBuf,
    /// Maximum number of results.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,
    /// Include items already linked from the source (Obsidian vaults).
    #[arg(long)]
    pub include_linked: bool,
    /// Emit results as JSON.
    #[arg(long)]
    pub json: bool,
}

pub fn execute(_ws: &Workspace, _args: RelatedArgs) -> Result<()> {
    bail!("`related` is not implemented yet — coming in a later milestone");
}
