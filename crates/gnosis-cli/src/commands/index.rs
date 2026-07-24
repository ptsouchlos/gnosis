use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::indexer;
use crate::workspace::{Workspace, expand_tilde};

/// Build or incrementally update the index.
#[derive(Debug, clap::Args)]
pub struct IndexArgs {
    /// Vault path to index. Locally this overrides the configured vaults for
    /// this run; globally it registers (and indexes) the vault.
    pub path: Option<PathBuf>,
}

pub fn execute(mut ws: Workspace, args: IndexArgs) -> Result<()> {
    let roots = resolve_roots(&mut ws, args.path)?;

    println!("Indexing {} vault(s)…", roots.len());
    let report = indexer::run(&ws, &roots, false)?;
    println!(
        "Done: {} scanned, {} (re)indexed, {} unchanged, {} removed, {} chunks.",
        report.scanned, report.indexed, report.skipped, report.deleted, report.chunks
    );
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
