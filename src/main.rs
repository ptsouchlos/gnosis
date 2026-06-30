mod cli;
mod config;
mod store;

use std::path::Path;

use anyhow::{Context, Result, bail};
use clap::Parser;

use cli::{Cli, Command};
use config::{CONFIG_FILE, Config};
use store::Store;

fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        Command::Init { force } => cmd_init(force),
        Command::Status => cmd_status(args.config.as_deref()),
        Command::Index { .. }
        | Command::Search { .. }
        | Command::Related { .. }
        | Command::Rebuild => {
            bail!("not implemented yet — coming in a later milestone");
        }
    }
}

/// Write a default `gnosis.toml` in the current directory.
fn cmd_init(force: bool) -> Result<()> {
    let path = Path::new(CONFIG_FILE);
    if path.exists() && !force {
        bail!("{CONFIG_FILE} already exists; pass --force to overwrite");
    }

    let toml = Config::default().to_toml()?;
    std::fs::write(path, toml).with_context(|| format!("writing {CONFIG_FILE}"))?;
    println!("Wrote {CONFIG_FILE}");
    Ok(())
}

/// Print index statistics.
fn cmd_status(config_path: Option<&Path>) -> Result<()> {
    let cfg = Config::load(config_path)?;
    let db_path = cfg.db_path();

    if !db_path.exists() {
        println!(
            "No index found at {} — run `gnosis index`.",
            db_path.display()
        );
        return Ok(());
    }

    let store = Store::open(&db_path)?;
    let stats = store.stats()?;
    let text_model = store
        .get_meta("model.text")?
        .unwrap_or_else(|| cfg.embed.text.model.clone());

    println!("vault:      {}", cfg.vault.display());
    println!("database:   {}", db_path.display());
    println!("text model: {text_model}");
    println!("documents:  {}", stats.documents);
    println!(
        "chunks:     {} text, {} image",
        stats.chunks_text, stats.chunks_image
    );
    match stats.indexed_at {
        Some(ts) => println!("last index: {ts} (unix)"),
        None => println!("last index: never"),
    }
    Ok(())
}
