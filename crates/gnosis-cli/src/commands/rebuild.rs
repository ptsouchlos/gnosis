use anyhow::{Context, Result, bail};

use crate::indexer;
use crate::workspace::Workspace;

/// Force a full re-embed and rebuild of the index.
#[derive(Debug, clap::Args)]
pub struct RebuildArgs {}

pub fn execute(ws: &Workspace, _args: RebuildArgs) -> Result<()> {
    if ws.db_path.exists() {
        std::fs::remove_file(&ws.db_path)
            .with_context(|| format!("removing {}", ws.db_path.display()))?;
    }

    let roots = ws.roots();
    if roots.is_empty() {
        bail!(
            "no vaults registered to rebuild; add them to {}",
            ws.config_path.display()
        );
    }

    println!("Rebuilding index from scratch…");
    let report = indexer::run(ws, &roots, true)?;
    println!("Done: {} indexed, {} chunks.", report.indexed, report.chunks);
    Ok(())
}
