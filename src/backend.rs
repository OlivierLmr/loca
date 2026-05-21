//! One mlx_lm.server subprocess wrapped as a BackendInstance.

use anyhow::{anyhow, Result};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::sleep;

use crate::config::ProxyCfg;
use crate::log_bus::LogBus;
use crate::paths;

pub struct BackendInstance {
    pub alias: String,
    pub hf_id: String,
    pub port: u16,
    cfg: ProxyCfg,
    bus: Arc<LogBus>,
    inner: Mutex<Inner>,
    pub request_count: AtomicU64,
    pub error_count: AtomicU64,
    // Stored as seconds-since-epoch; 0 means "never".
    pub last_request_unix: AtomicU64,
    pub idle_timeout_sec: parking_lot::Mutex<u64>,
}

struct Inner {
    child: Option<Child>,
    spawn_started_at: Option<Instant>,
    spawn_completed_at: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendState {
    Stopped,
    Loading,
    Ready,
}

impl BackendInstance {
    pub fn new(alias: String, hf_id: String, port: u16, cfg: ProxyCfg, bus: Arc<LogBus>) -> Self {
        let idle = cfg.idle_timeout_sec;
        Self {
            alias,
            hf_id,
            port,
            cfg,
            bus,
            inner: Mutex::new(Inner {
                child: None,
                spawn_started_at: None,
                spawn_completed_at: None,
            }),
            request_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            last_request_unix: AtomicU64::new(0),
            idle_timeout_sec: parking_lot::Mutex::new(idle),
        }
    }

    fn log(&self, msg: &str) {
        self.bus.emit(&self.alias, msg);
    }

    pub async fn state(&self) -> BackendState {
        let g = self.inner.lock().await;
        if g.child.is_none() {
            BackendState::Stopped
        } else if g.spawn_completed_at.is_none() {
            BackendState::Loading
        } else {
            BackendState::Ready
        }
    }

    pub async fn pid(&self) -> Option<u32> {
        self.inner.lock().await.child.as_ref().and_then(|c| c.id())
    }

    pub async fn spawn(&self) -> Result<()> {
        let cache_dir = paths::hf_cache_for(&self.hf_id)?;
        if !cache_dir.exists() {
            self.log("first use: weights will download from HF (multi-GB, one-time)");
        }
        self.log(&format!("spawning mlx_lm.server on port {}", self.port));

        let mut child = Command::new("mlx_lm.server")
            .arg("--model").arg(&self.hf_id)
            .arg("--host").arg(&self.cfg.backend_host)
            .arg("--port").arg(self.port.to_string())
            .arg("--log-level").arg("INFO")
            .arg("--max-tokens").arg(self.cfg.max_tokens.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| anyhow!("failed to spawn mlx_lm.server: {e} (is mlx-lm installed? try `uv tool install mlx-lm`)"))?;

        let started = Instant::now();
        {
            let mut g = self.inner.lock().await;
            g.spawn_started_at = Some(started);
            g.spawn_completed_at = None;
            // Take stdout / stderr to pipe into LogBus before storing the child.
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            g.child = Some(child);
            if let Some(out) = stdout {
                let bus = self.bus.clone();
                let alias = self.alias.clone();
                tokio::spawn(async move {
                    let mut lines = BufReader::new(out).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        bus.emit(&alias, &line);
                    }
                });
            }
            if let Some(err) = stderr {
                let bus = self.bus.clone();
                let alias = self.alias.clone();
                tokio::spawn(async move {
                    let mut lines = BufReader::new(err).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        bus.emit(&alias, &line);
                    }
                });
            }
        }

        // Wait for readiness by polling the backend's /v1/models.
        let url = format!("http://{}:{}/v1/models", self.cfg.backend_host, self.port);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let deadline = started + Duration::from_secs(self.cfg.spawn_timeout_sec);
        loop {
            if Instant::now() > deadline {
                self.terminate_internal().await;
                return Err(anyhow!("backend failed to become ready within timeout"));
            }
            // Check the process didn't exit prematurely.
            {
                let mut g = self.inner.lock().await;
                if let Some(child) = g.child.as_mut() {
                    if let Ok(Some(status)) = child.try_wait() {
                        g.child = None;
                        return Err(anyhow!("mlx_lm.server exited (code {status}) before becoming ready"));
                    }
                }
            }
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    let now = Instant::now();
                    let mut g = self.inner.lock().await;
                    g.spawn_completed_at = Some(now);
                    let elapsed = now.duration_since(started).as_secs_f64();
                    drop(g);
                    self.log(&format!("ready in {:.1}s", elapsed));
                    return Ok(());
                }
            }
            sleep(Duration::from_millis(500)).await;
        }
    }

    pub async fn stop(&self) {
        self.terminate_internal().await;
    }

    async fn terminate_internal(&self) {
        let mut g = self.inner.lock().await;
        if let Some(mut child) = g.child.take() {
            self.log("stopping");
            let _ = child.start_kill();
            // Wait up to 10s for graceful exit; kill_on_drop will SIGKILL anyway.
            let _ = tokio::time::timeout(Duration::from_secs(10), child.wait()).await;
        }
        g.spawn_started_at = None;
        g.spawn_completed_at = None;
    }

    pub fn idle_timeout(&self) -> u64 {
        *self.idle_timeout_sec.lock()
    }

    pub fn set_idle_timeout(&self, v: u64) {
        *self.idle_timeout_sec.lock() = v.max(60);
    }

    pub fn note_request(&self) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0);
        self.last_request_unix.store(now, Ordering::Relaxed);
    }

    pub fn note_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn last_request_age(&self) -> Option<u64> {
        let last = self.last_request_unix.load(Ordering::Relaxed);
        if last == 0 { return None; }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0);
        Some(now.saturating_sub(last))
    }
}
