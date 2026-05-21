//! First-run setup: create directories and write default config files
//! if they don't yet exist.

use anyhow::{Context, Result};
use std::fs;

use crate::paths;

const DEFAULT_MODELS_TOML: &str = include_str!("../share/models.toml");

const DEFAULT_PROXY_TOML: &str = r#"# loca proxy configuration. All keys are optional; defaults are shown.

[proxy]
# listen_host = "127.0.0.1"
# listen_port = 11431
# backend_host = "127.0.0.1"
# backend_port_start = 11432      # first port allocated to a backend
# backend_port_count = 16         # how many ports to reserve
# idle_timeout_sec = 600          # default per-instance idle timeout (10 min)
# spawn_timeout_sec = 240         # max seconds to wait for a backend to come up
# max_tokens = 16384              # max tokens for `mlx_lm.server --max-tokens`
# ram_budget_gb = 24              # advisory; the TUI shows red when exceeded
"#;

/// Returns true if anything was created.
pub fn ensure_layout() -> Result<bool> {
    let mut created = false;

    let config_dir = paths::config_dir()?;
    if !config_dir.exists() {
        fs::create_dir_all(&config_dir)
            .with_context(|| format!("creating {}", config_dir.display()))?;
        created = true;
    }

    let state_dir = paths::state_dir()?;
    if !state_dir.exists() {
        fs::create_dir_all(&state_dir)
            .with_context(|| format!("creating {}", state_dir.display()))?;
        created = true;
    }

    // mlx_lm.server crashes scanning this directory if it doesn't exist.
    let hf_cache = paths::hf_cache_root()?;
    if !hf_cache.exists() {
        fs::create_dir_all(&hf_cache)
            .with_context(|| format!("creating {}", hf_cache.display()))?;
        created = true;
    }

    let models_path = paths::models_toml()?;
    if !models_path.exists() {
        fs::write(&models_path, DEFAULT_MODELS_TOML)
            .with_context(|| format!("writing {}", models_path.display()))?;
        eprintln!("loca: wrote default {}", models_path.display());
        created = true;
    }

    let proxy_path = paths::proxy_toml()?;
    if !proxy_path.exists() {
        fs::write(&proxy_path, DEFAULT_PROXY_TOML)
            .with_context(|| format!("writing {}", proxy_path.display()))?;
        eprintln!("loca: wrote template {}", proxy_path.display());
        created = true;
    }

    Ok(created)
}
