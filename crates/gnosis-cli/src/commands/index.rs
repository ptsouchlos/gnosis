use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::commands::build_embedders;
use crate::fs::StdFs;
use crate::progress::IndicatifProgress;
use crate::store::{Space, SqliteStore};
use crate::walk::FsWalker;
use crate::workspace::{Workspace, expand_tilde};

/// Build or incrementally update the index.
#[derive(Debug, clap::Args)]
pub struct IndexArgs {
    /// Vault path to index. Locally this overrides the configured vaults for
    /// this run; globally it registers (and indexes) the vault.
    pub path: Option<PathBuf>,
    /// Stop immediately on the first file that fails to index (e.g. a
    /// corrupt image), instead of skipping it and reporting it at the end.
    #[arg(long)]
    pub fail_fast: bool,
}

pub fn execute(mut ws: Workspace, args: IndexArgs) -> Result<()> {
    let roots = resolve_roots(&mut ws, args.path)?;

    println!("Indexing {} vault(s)…", roots.len());
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
        false,
        args.fail_fast,
        ws.config.embed.image.batch_size,
    )?;
    println!(
        "Done: {} scanned, {} (re)indexed, {} unchanged, {} removed, {} chunks.",
        report.scanned, report.indexed, report.skipped, report.deleted, report.chunks
    );
    crate::commands::print_index_errors(&report.errors);

    let quantization = crate::store::resolve_quantization(&ws.config.ann.quantization)?;
    store.rebuild_index(Space::Text, quantization)?;
    if image_enabled {
        store.rebuild_index(Space::Image, quantization)?;
    }
    println!("Updated search index.");
    Ok(())
}

/// Determine which vault roots to index, registering a new one in global mode.
fn resolve_roots(ws: &mut Workspace, path: Option<PathBuf>) -> Result<Vec<PathBuf>> {
    if !ws.global {
        // Local: a positional path overrides the config vaults for this run.
        return Ok(match path {
            Some(p) => vec![p],
            None => ws.roots(),
        });
    }

    match path {
        Some(path) => {
            let canon = std::fs::canonicalize(expand_tilde(&path))
                .with_context(|| format!("resolving {}", path.display()))?;
            let known = ws.config.vaults.iter().any(|v| {
                std::fs::canonicalize(expand_tilde(v))
                    .map(|p| p == canon)
                    .unwrap_or(false)
            });
            if !known {
                ws.config.vaults.push(canon.clone());
                ws.save_config()?;
            }
            Ok(vec![canon])
        }
        None => {
            let roots = ws.roots();
            if roots.is_empty() {
                bail!(
                    "no vaults registered; run `gnosis -g index <path>` or add them to {}",
                    ws.config_path.display()
                );
            }
            Ok(roots)
        }
    }
}
