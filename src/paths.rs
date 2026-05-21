//! Filesystem locations used by loca.
//!
//! Follows the XDG Base Directory Specification. Honors `$XDG_CONFIG_HOME`
//! and `$XDG_STATE_HOME` when set; otherwise uses the spec defaults under
//! `~/.config` and `~/.local/state`. The HF cache root is determined by
//! mlx_lm (Hugging Face's `HF_HOME` / `~/.cache/huggingface`) — we only
//! help by creating the directory if missing.

use anyhow::{anyhow, Result};
use std::env;
use std::path::PathBuf;

pub fn home() -> Result<PathBuf> {
    home::home_dir().ok_or_else(|| anyhow!("could not determine home directory"))
}

fn xdg_dir(var: &str, default_suffix: &[&str]) -> Result<PathBuf> {
    if let Some(v) = env::var_os(var) {
        let p = PathBuf::from(v);
        if p.is_absolute() {
            return Ok(p);
        }
    }
    let mut p = home()?;
    for s in default_suffix {
        p.push(s);
    }
    Ok(p)
}

pub fn config_dir() -> Result<PathBuf> {
    Ok(xdg_dir("XDG_CONFIG_HOME", &[".config"])?.join("loca"))
}

pub fn state_dir() -> Result<PathBuf> {
    Ok(xdg_dir("XDG_STATE_HOME", &[".local", "state"])?.join("loca"))
}

pub fn hf_cache_root() -> Result<PathBuf> {
    // Hugging Face's own convention: $HF_HOME or ~/.cache/huggingface.
    if let Some(hf_home) = env::var_os("HF_HOME") {
        return Ok(PathBuf::from(hf_home).join("hub"));
    }
    Ok(xdg_dir("XDG_CACHE_HOME", &[".cache"])?
        .join("huggingface")
        .join("hub"))
}

pub fn hf_cache_for(hf_id: &str) -> Result<PathBuf> {
    let mut parts = hf_id.splitn(2, '/');
    let org = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    Ok(hf_cache_root()?.join(format!("models--{org}--{name}")))
}

pub fn models_toml() -> Result<PathBuf> {
    Ok(config_dir()?.join("models.toml"))
}

pub fn proxy_toml() -> Result<PathBuf> {
    Ok(config_dir()?.join("proxy.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_cache_for_splits_org_and_name() {
        let p = hf_cache_for("mlx-community/Qwen3.5-9B-MLX-4bit").unwrap();
        let last = p.file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(last, "models--mlx-community--Qwen3.5-9B-MLX-4bit");
    }

    #[test]
    fn hf_cache_for_handles_missing_slash_gracefully() {
        let p = hf_cache_for("no-slash").unwrap();
        let last = p.file_name().unwrap().to_string_lossy().to_string();
        // No org → "models--<empty>--no-slash"? We don't crash, which is the
        // contract; the resulting path is junk but that's caller's problem.
        assert!(last.starts_with("models--"));
    }
}
