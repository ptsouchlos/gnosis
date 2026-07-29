use anyhow::Result;

use crate::store::Store;
use crate::workspace::Workspace;

/// Show index statistics.
#[derive(Debug, clap::Args)]
pub struct StatusArgs {}

pub fn execute(ws: &Workspace, _args: StatusArgs) -> Result<()> {
    if !ws.db_path.exists() {
        println!(
            "No index found at {} — run `gnosis index`.",
            ws.db_path.display()
        );
        return Ok(());
    }

    let store = Store::open(&ws.db_path)?;
    let stats = store.stats()?;
    let text_model = store
        .get_meta("model.text")?
        .unwrap_or_else(|| ws.config.embed.text.model.clone());

    println!("mode:       {}", if ws.global { "global" } else { "local" });
    println!("config:     {}", ws.config_path.display());
    println!("database:   {}", ws.db_path.display());
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

    let counts = store.counts_by_root()?;
    if counts.len() > 1 {
        println!("vaults:");
        for (root, n) in counts {
            println!("  {n:>6}  {root}");
        }
    }
    Ok(())
}
