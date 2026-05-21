//! loca — wake-on-request local LLM proxy with a live TUI.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use loca::{config, init, log_bus, manager, paths, proxy, tui};

#[derive(Parser)]
#[command(name = "loca", version, about = "Wake-on-request local LLM proxy with a live TUI.")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// List aliases defined in models.toml.
    Ls,
    /// One-shot status from the running loca (requires loca running).
    Status,
    /// Warm a model now (requires loca running).
    Preload {
        /// Alias from models.toml.
        alias: String,
    },
    /// Force-create config dirs and default files if missing.
    Init,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => run_tui().await,
        Some(Command::Ls) => cmd_ls(),
        Some(Command::Status) => cmd_status().await,
        Some(Command::Preload { alias }) => cmd_preload(&alias).await,
        Some(Command::Init) => cmd_init(),
    }
}

async fn run_tui() -> Result<()> {
    init::ensure_layout()?;
    let cfg = config::load_proxy_cfg()?;
    let models = config::load_models()?;
    if models.is_empty() {
        anyhow::bail!(
            "no models in {}. Run `loca init` to write the default.",
            paths::models_toml()?.display()
        );
    }

    let bus = log_bus::LogBus::new();
    let mgr = manager::BackendManager::new(cfg.clone(), models, bus.clone());

    let listener = proxy::bind_listener(&cfg).await?;
    let router = proxy::make_router(mgr.clone());

    bus.emit("loca", &format!("listening on http://{}:{}", cfg.listen_host, cfg.listen_port));
    bus.emit("loca", &format!("{} aliases loaded from models.toml", mgr.models.len()));
    bus.emit("loca", &format!("default idle timeout: {}s", cfg.idle_timeout_sec));
    bus.emit("loca", &format!("RAM budget: {} GB (advisory, not enforced)", cfg.ram_budget_gb));

    // Background tasks.
    let server_mgr = mgr.clone();
    let server_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
        let _ = server_mgr;
    });
    let idle_mgr = mgr.clone();
    let idle_handle = tokio::spawn(async move { idle_mgr.idle_watch().await });

    // Run TUI in foreground; when it returns, tear down.
    let result = tui::run(mgr.clone(), bus.clone(), cfg).await;

    // Cleanup.
    mgr.stop_all().await;
    idle_handle.abort();
    server_handle.abort();
    result
}

fn cmd_ls() -> Result<()> {
    init::ensure_layout()?;
    let models = config::load_models()?;
    if models.is_empty() {
        println!("loca: no models in {}", paths::models_toml()?.display());
        return Ok(());
    }
    println!("{:<22} {:>7}  cached  hf id", "alias", "ram");
    println!("{:-<22} {:->7}  ------  -----", "", "");
    for (alias, m) in &models {
        let cached = paths::hf_cache_for(&m.hf)
            .map(|p| p.exists())
            .unwrap_or(false);
        let marker = if cached { "  yes " } else { "  no  " };
        let ram = match m.ram_gb {
            Some(g) => format!("~{g} GB"),
            None => "—".into(),
        };
        println!("{alias:<22} {ram:>7}  {marker}  {}", m.hf);
    }
    Ok(())
}

async fn cmd_status() -> Result<()> {
    let cfg = config::load_proxy_cfg()?;
    let url = format!("http://{}:{}/admin/status", cfg.listen_host, cfg.listen_port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let v: serde_json::Value = resp.json().await.context("parsing status JSON")?;
            println!("{}", serde_json::to_string_pretty(&v)?);
            Ok(())
        }
        Ok(resp) => {
            anyhow::bail!(
                "loca: {}:{} responded {} — something other than loca is listening on that port",
                cfg.listen_host, cfg.listen_port, resp.status()
            );
        }
        Err(e) => {
            anyhow::bail!(
                "loca: not reachable at {}:{} ({e}). Start it with: loca",
                cfg.listen_host, cfg.listen_port
            );
        }
    }
}

async fn cmd_preload(alias: &str) -> Result<()> {
    let cfg = config::load_proxy_cfg()?;
    let url = format!(
        "http://{}:{}/admin/preload?model={}",
        cfg.listen_host, cfg.listen_port, alias
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let resp = client.post(&url).send().await.with_context(|| {
        format!("not reachable at {}:{} — start loca first", cfg.listen_host, cfg.listen_port)
    })?;
    let v: serde_json::Value = resp.json().await?;
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}

fn cmd_init() -> Result<()> {
    let created = init::ensure_layout()?;
    if !created {
        println!("loca: config layout already present; nothing to do.");
    }
    Ok(())
}

