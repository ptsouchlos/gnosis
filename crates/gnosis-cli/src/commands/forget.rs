use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::store::Store;
use crate::workspace::{Workspace, expand_tilde};

/// Remove a vault from the index and delete its documents.
#[derive(Debug, clap::Args)]
pub struct ForgetArgs {
    /// The vault path to forget.
    pub path: PathBuf,
}

pub fn execute(mut ws: Workspace, args: ForgetArgs) -> Result<()> {
    let canon = std::fs::canonicalize(expand_tilde(&args.path))
        .with_context(|| format!("resolving {}", args.path.display()))?;
    let canon_str = canon.to_string_lossy().to_string();

    // Drop it from the registered vault list (comparing canonical paths).
    let before = ws.config.vaults.len();
    ws.config.vaults.retain(|v| {
        std::fs::canonicalize(expand_tilde(v))
            .map(|p| p != canon)
            .unwrap_or(true)
    });
    let unregistered = before - ws.config.vaults.len();
    if unregistered > 0 {
        ws.save_config()?;
    }

    // Delete its documents from the database, if one exists.
    let mut removed_docs = 0;
    if ws.db_path.exists() {
        let store = Store::open(&ws.db_path)?;
        removed_docs = store.delete_by_root(&canon_str)?;
    }

    println!(
        "Forgot {}: removed {removed_docs} document(s){}.",
        canon.display(),
        if unregistered > 0 {
            " and unregistered the vault"
        } else {
            ""
        }
    );
    Ok(())
}
