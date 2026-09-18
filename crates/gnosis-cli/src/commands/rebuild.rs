use anyhow::{Context, Result, bail};

use crate::commands::build_embedders;
use crate::fs::StdFs;
use crate::progress::IndicatifProgress;
use crate::store::{Space, SqliteStore};
use crate::walk::FsWalker;
use crate::workspace::Workspace;

/// Force a full re-embed and rebuild of the index.
#[derive(Debug, clap::Args)]
pub struct RebuildArgs {
    /// Stop immediately on the first file that fails to index (e.g. a
    /// corrupt image), instead of skipping it and reporting it at the end.
    #[arg(long)]
    pub fail_fast: bool,
}

pub fn execute(ws: &Workspace, args: RebuildArgs) -> Result<()> {
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
    let image_enabled = ws.config.embed.image.enabled;
    let mut embedders = build_embedders(&ws.config.embed)?;
    let progress = IndicatifProgress::new();
    let mut indexer_args = index::IndexerArgs {
        store: &mut store,
        walker: &FsWalker,
        fs_reader: &StdFs,
        embedders: index::EmbedderSet {
            text: embedders.text.as_mut(),
            image: embedders.image,
            image_text: embedders.image_text,
        },
        progress: &progress,
    };
    let report = index::run(
        &mut indexer_args,
        &roots,
        &ws.config.ignore.globs,
        &ws.config.chunk,
        true,
        args.fail_fast,
    )?;
    println!("Done: {} indexed, {} chunks.", report.indexed, report.chunks);
    crate::commands::print_index_errors(&report.errors);

    store.rebuild_index(Space::Text)?;
    if image_enabled {
        store.rebuild_index(Space::Image)?;
    }
    println!("Updated search index.");
    Ok(())
}
