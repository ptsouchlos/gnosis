use anyhow::{Context, Result, bail};

use crate::embedder::build_text_embedder;
use crate::fs::StdFs;
use crate::progress::IndicatifProgress;
use crate::store::SqliteStore;
use crate::walk::FsWalker;
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
    let mut store = SqliteStore::open(&ws.db_path)?;
    let mut embedder = build_text_embedder(&ws.config.embed.text.model)?;
    let progress = IndicatifProgress::new();
    let mut indexer_args = index::IndexerArgs {
        store: &mut store,
        walker: &FsWalker,
        fs_reader: &StdFs,
        embedder: embedder.as_mut(),
        progress: &progress,
    };
    let report = index::run(
        &mut indexer_args,
        &roots,
        &ws.config.ignore.globs,
        &ws.config.chunk,
        true,
    )?;
    println!("Done: {} indexed, {} chunks.", report.indexed, report.chunks);

    store.rebuild_index("text")?;
    println!("Updated search index.");
    Ok(())
}
