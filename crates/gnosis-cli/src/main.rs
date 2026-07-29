mod chunk;
mod cli;
mod commands;
mod config;
mod embed;
mod indexer;
mod parse;
mod store;
mod walk;
mod workspace;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command};
use workspace::Workspace;

fn main() -> Result<()> {
    let Cli {
        config,
        global,
        command,
    } = Cli::parse();
    let config_path = config.as_deref();

    // `init` writes a config file, so it resolves paths itself rather than
    // loading an existing workspace.
    if let Command::Init(args) = command {
        return commands::init::execute(global, args);
    }

    let ws = Workspace::resolve(global, config_path)?;
    match command {
        Command::Init(_) => unreachable!("handled above"),
        Command::Index(args) => commands::index::execute(ws, args),
        Command::Search(args) => commands::search::execute(&ws, args),
        Command::Related(args) => commands::related::execute(&ws, args),
        Command::Forget(args) => commands::forget::execute(ws, args),
        Command::Status(args) => commands::status::execute(&ws, args),
        Command::Rebuild(args) => commands::rebuild::execute(&ws, args),
    }
}
