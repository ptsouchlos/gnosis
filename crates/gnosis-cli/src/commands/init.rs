use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::config::{CONFIG_FILE, Config};
use crate::workspace::global_dir;

/// Write a default config file.
#[derive(Debug, clap::Args)]
pub struct InitArgs {
    /// Overwrite an existing config file.
    #[arg(long)]
    pub force: bool,
}

pub fn execute(global: bool, args: InitArgs) -> Result<()> {
    // Global config starts with no vaults registered; local defaults to ".".
    let (path, config) = if global {
        let cfg = Config {
            vaults: Vec::new(),
            ..Config::default()
        };
        (global_dir()?.join("config.toml"), cfg)
    } else {
        (PathBuf::from(CONFIG_FILE), Config::default())
    };

    if path.exists() && !args.force {
        bail!("{} already exists; pass --force to overwrite", path.display());
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let toml = config.to_toml()?;
    std::fs::write(&path, toml).with_context(|| format!("writing {}", path.display()))?;
    println!("Wrote {}", path.display());
    Ok(())
}
