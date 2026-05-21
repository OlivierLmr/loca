//! Configuration: models.toml (alias map) + proxy.toml (defaults overrides).

use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde::Deserialize;
use std::fs;

use crate::paths;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub hf: String,
    #[serde(default)]
    pub ram_gb: Option<u32>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ModelsFile {
    /// IndexMap preserves the order from models.toml so the table is stable.
    #[serde(default)]
    pub models: IndexMap<String, ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ProxyCfg {
    pub listen_host: String,
    pub listen_port: u16,
    pub backend_host: String,
    pub backend_port_start: u16,
    pub backend_port_count: u16,
    pub idle_timeout_sec: u64,
    pub spawn_timeout_sec: u64,
    pub max_tokens: u32,
    pub ram_budget_gb: u32,
}

impl Default for ProxyCfg {
    fn default() -> Self {
        Self {
            listen_host: "127.0.0.1".into(),
            listen_port: 11431,
            backend_host: "127.0.0.1".into(),
            backend_port_start: 11432,
            backend_port_count: 16,
            idle_timeout_sec: 600,
            spawn_timeout_sec: 240,
            max_tokens: 16384,
            ram_budget_gb: 24,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ProxyFile {
    proxy: ProxyCfg,
}

/// Parse a models.toml text. Pure function — useful in tests.
pub fn parse_models(text: &str) -> Result<IndexMap<String, ModelEntry>> {
    let parsed: ModelsFile = toml::from_str(text).context("parsing models.toml")?;
    Ok(parsed.models)
}

/// Parse a proxy.toml text. Pure function — useful in tests.
pub fn parse_proxy_cfg(text: &str) -> Result<ProxyCfg> {
    let parsed: ProxyFile = toml::from_str(text).context("parsing proxy.toml")?;
    Ok(parsed.proxy)
}

pub fn load_models() -> Result<IndexMap<String, ModelEntry>> {
    let path = paths::models_toml()?;
    if !path.exists() {
        return Ok(IndexMap::new());
    }
    let txt = fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    parse_models(&txt).with_context(|| format!("in {}", path.display()))
}

pub fn load_proxy_cfg() -> Result<ProxyCfg> {
    let path = paths::proxy_toml()?;
    if !path.exists() {
        return Ok(ProxyCfg::default());
    }
    let txt = fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    parse_proxy_cfg(&txt).with_context(|| format!("in {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_model_entry() {
        let txt = r#"
[models.foo]
hf = "mlx-community/Foo-4bit"
"#;
        let m = parse_models(txt).unwrap();
        assert_eq!(m.len(), 1);
        let entry = &m["foo"];
        assert_eq!(entry.hf, "mlx-community/Foo-4bit");
        assert_eq!(entry.ram_gb, None);
        assert_eq!(entry.description, None);
    }

    #[test]
    fn parses_full_model_entry_with_quoted_alias() {
        let txt = r#"
[models."qwen3.5-9b"]
hf = "mlx-community/Qwen3.5-9B-MLX-4bit"
ram_gb = 6
description = "Qwen 3.5 9B"
"#;
        let m = parse_models(txt).unwrap();
        let entry = &m["qwen3.5-9b"];
        assert_eq!(entry.hf, "mlx-community/Qwen3.5-9B-MLX-4bit");
        assert_eq!(entry.ram_gb, Some(6));
        assert_eq!(entry.description.as_deref(), Some("Qwen 3.5 9B"));
    }

    #[test]
    fn empty_file_yields_empty_map() {
        let m = parse_models("").unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn preserves_insertion_order() {
        let txt = r#"
[models.zeta]
hf = "x/z"
[models.alpha]
hf = "x/a"
[models.mu]
hf = "x/m"
"#;
        let m = parse_models(txt).unwrap();
        let order: Vec<_> = m.keys().collect();
        assert_eq!(order, vec!["zeta", "alpha", "mu"]);
    }

    #[test]
    fn proxy_cfg_defaults_when_empty() {
        let cfg = parse_proxy_cfg("").unwrap();
        assert_eq!(cfg.listen_port, 11431);
        assert_eq!(cfg.backend_port_start, 11432);
        assert_eq!(cfg.backend_port_count, 16);
        assert_eq!(cfg.idle_timeout_sec, 600);
        assert_eq!(cfg.max_tokens, 16384);
        assert_eq!(cfg.ram_budget_gb, 24);
    }

    #[test]
    fn proxy_cfg_partial_override() {
        let cfg = parse_proxy_cfg(
            r#"
[proxy]
listen_port = 9000
idle_timeout_sec = 1800
ram_budget_gb = 16
"#,
        )
        .unwrap();
        assert_eq!(cfg.listen_port, 9000);
        assert_eq!(cfg.idle_timeout_sec, 1800);
        assert_eq!(cfg.ram_budget_gb, 16);
        // unchanged
        assert_eq!(cfg.backend_port_start, 11432);
    }

    #[test]
    fn bundled_default_models_toml_parses() {
        let txt = include_str!("../share/models.toml");
        let m = parse_models(txt).unwrap();
        assert!(m.contains_key("qwen3.5-9b"));
        assert!(m.contains_key("phi4-14b"));
        assert!(m.values().all(|e| e.hf.starts_with("mlx-community/")));
    }
}
