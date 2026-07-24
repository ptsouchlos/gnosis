use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::{CONFIG_FILE, Config};

/// Directory name for the global index under the user's home directory.
const GLOBAL_DIR: &str = ".gnosis";

/// Resolves *where* the index lives (config + database) and how to persist
/// config edits. Local mode ties the database to the config's directory; global
/// mode uses a single central location under the home directory.
pub struct Workspace {
    pub config: Config,
    /// Where the active config is (or would be) written.
    pub config_path: PathBuf,
    /// SQLite database backing this workspace.
    pub db_path: PathBuf,
    pub global: bool,
}

impl Workspace {
    /// Build a workspace from the `--global` flag and an optional `--config` path.
    pub fn resolve(global: bool, explicit_config: Option<&Path>) -> Result<Workspace> {
        if global {
            Self::resolve_global()
        } else {
            Self::resolve_local(explicit_config)
        }
    }

    fn resolve_local(explicit_config: Option<&Path>) -> Result<Workspace> {
        let config_path = explicit_config
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(CONFIG_FILE));
        let config = Config::load(explicit_config)?;

        // The database lives relative to the config's directory (cwd for the
        // default ./gnosis.toml), so `index` and `search` from the same place
        // share one database.
        let config_dir = config_path.parent().filter(|p| !p.as_os_str().is_empty());
        let db_path = match config_dir {
            Some(dir) => dir.join(&config.db_dir).join("gnosis.db"),
            None => config.db_dir.join("gnosis.db"),
        };

        Ok(Workspace {
            config,
            config_path,
            db_path,
            global: false,
        })
    }

    fn resolve_global() -> Result<Workspace> {
        let dir = global_dir()?;
        let config_path = dir.join("config.toml");

        // A missing global config starts empty (no vaults registered yet).
        let config = if config_path.exists() {
            Config::load(Some(&config_path))?
        } else {
            Config {
                vaults: Vec::new(),
                ..Config::default()
            }
        };

        Ok(Workspace {
            config,
            config_path,
            db_path: dir.join("global.db"),
            global: true,
        })
    }

    /// The vault roots for this workspace, tilde-expanded (canonicalization is
    /// left to the indexer so missing roots can be reported rather than fail).
    pub fn roots(&self) -> Vec<PathBuf> {
        self.config.vaults.iter().map(|p| expand_tilde(p)).collect()
    }

    /// Persist the current config to `config_path`, creating parent dirs.
    pub fn save_config(&self) -> Result<()> {
        if let Some(parent) = self.config_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let toml = self.config.to_toml()?;
        std::fs::write(&self.config_path, toml)
            .with_context(|| format!("writing {}", self.config_path.display()))?;
        Ok(())
    }
}

/// Absolute path to the global gnosis directory (`~/.gnosis`).
pub fn global_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine home directory")?;
    Ok(home.join(GLOBAL_DIR))
}

/// Expand a leading `~` to the user's home directory.
pub fn expand_tilde(path: &Path) -> PathBuf {
    if let Ok(rest) = path.strip_prefix("~")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    path.to_path_buf()
}
