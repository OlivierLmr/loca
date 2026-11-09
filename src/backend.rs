//! One mlx_lm.server subprocess wrapped as a BackendInstance.

use anyhow::{anyhow, Result};
use std::process::Stdio;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::sleep;

use crate::config::ProxyCfg;
use crate::log_bus::LogBus;
use crate::paths;
use crate::stats::RequestStats;

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
    /// Currently in-flight (streaming) requests.
    pub in_flight: AtomicU32,
    /// Start time (unix seconds) of the most recent request still in flight,
    /// or 0 if none.
    pub current_request_started_unix: AtomicU64,
    /// Bytes streamed for the most recent in-flight request (reset on each
    /// `enter_request`).
    pub current_request_bytes: AtomicU64,
    /// Speed stats for in-flight requests, plus the last finished one.
    requests: parking_lot::Mutex<RequestSlots>,
    next_request_id: AtomicU64,
}

#[derive(Default)]
struct RequestSlots {
    /// In arrival order; the newest is last.
    active: Vec<(u64, RequestStats)>,
    last: Option<RequestStats>,
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
            in_flight: AtomicU32::new(0),
            current_request_started_unix: AtomicU64::new(0),
            current_request_bytes: AtomicU64::new(0),
            requests: parking_lot::Mutex::new(RequestSlots::default()),
            next_request_id: AtomicU64::new(0),
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
            .args(prompt_cache_args(&self.cfg))
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
        self.last_request_unix.store(now_unix(), Ordering::Relaxed);
    }

    pub fn note_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn last_request_age(&self) -> Option<u64> {
        let last = self.last_request_unix.load(Ordering::Relaxed);
        if last == 0 { return None; }
        Some(now_unix().saturating_sub(last))
    }

    /// Called when an HTTP request starts streaming through to this backend.
    /// Increments in_flight and resets the per-request stats. Returns a
    /// guard that decrements on drop (whether the stream completes naturally
    /// or the client disconnects).
    pub fn enter_request(self: &Arc<Self>, bus: Arc<crate::log_bus::LogBus>) -> RequestGuard {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let now = now_unix();
        self.current_request_started_unix.store(now, Ordering::Relaxed);
        self.current_request_bytes.store(0, Ordering::Relaxed);
        self.last_request_unix.store(now, Ordering::Relaxed);
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.requests
            .lock()
            .active
            .push((id, RequestStats::new(Instant::now())));
        RequestGuard {
            inst: self.clone(),
            bus,
            started: now,
            id,
        }
    }

    /// Runs `f` on the stats of in-flight request `id`, if it's still active.
    pub fn with_request_stats<R>(&self, id: u64, f: impl FnOnce(&mut RequestStats) -> R) -> Option<R> {
        let mut slots = self.requests.lock();
        slots.active.iter_mut().find(|(i, _)| *i == id).map(|(_, s)| f(s))
    }

    /// Stats of the newest in-flight request, how many are in flight, and
    /// the last finished request.
    pub fn request_stats(&self) -> (Option<RequestStats>, usize, Option<RequestStats>) {
        let slots = self.requests.lock();
        (
            slots.active.last().map(|(_, s)| s.clone()),
            slots.active.len(),
            slots.last.clone(),
        )
    }

    pub fn add_bytes(&self, n: u64) {
        self.current_request_bytes.fetch_add(n, Ordering::Relaxed);
        self.last_request_unix.store(now_unix(), Ordering::Relaxed);
    }

    pub fn is_busy(&self) -> bool {
        self.in_flight.load(Ordering::Relaxed) > 0
    }

    pub fn current_request_elapsed_sec(&self) -> Option<u64> {
        let started = self.current_request_started_unix.load(Ordering::Relaxed);
        if started == 0 {
            return None;
        }
        Some(now_unix().saturating_sub(started))
    }

    pub fn current_request_bytes_count(&self) -> u64 {
        self.current_request_bytes.load(Ordering::Relaxed)
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Returned by `BackendInstance::enter_request`. Drop fires when the
/// streaming response is fully sent (or aborted) — decrements in_flight
/// and logs the final stats. Cancel-safe: if the client disconnects or
/// the response stream is dropped early, this still runs.
pub struct RequestGuard {
    inst: Arc<BackendInstance>,
    bus: Arc<crate::log_bus::LogBus>,
    started: u64,
    pub id: u64,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.inst.in_flight.fetch_sub(1, Ordering::Relaxed);
        let bytes = self.inst.current_request_bytes.load(Ordering::Relaxed);
        let elapsed = now_unix().saturating_sub(self.started);
        // Only clear started-at if this was the last in-flight request,
        // so concurrent requests don't lose their elapsed counter.
        if self.inst.in_flight.load(Ordering::Relaxed) == 0 {
            self.inst.current_request_started_unix.store(0, Ordering::Relaxed);
        }
        let finished = {
            let mut slots = self.inst.requests.lock();
            let idx = slots.active.iter().position(|(i, _)| *i == self.id);
            idx.map(|idx| {
                let (_, mut s) = slots.active.remove(idx);
                s.ended = Some(Instant::now());
                slots.last = Some(s.clone());
                s
            })
        };
        let msg = match finished {
            // Nothing came back: an error response or a non-streaming request.
            Some(s) if s.first_token.is_some() || s.prompt_started.is_some() => {
                format!("{}, {} streamed", s.summary(Instant::now()), fmt_bytes(bytes))
            }
            _ => format!("request done in {elapsed}s, {} streamed", fmt_bytes(bytes)),
        };
        self.bus.emit(&self.inst.alias, &msg);
    }
}

/// `--prompt-cache-bytes` for mlx_lm.server, unless the cap is disabled (0).
/// Without it mlx keeps up to 10 KV caches with no size limit.
fn prompt_cache_args(cfg: &ProxyCfg) -> Vec<String> {
    if cfg.prompt_cache_gb > 0.0 {
        vec!["--prompt-cache-bytes".into(), format!("{}G", cfg.prompt_cache_gb)]
    } else {
        Vec::new()
    }
}

fn fmt_bytes(n: u64) -> String {
    if n < 1024 {
        format!("{n} B")
    } else if n < 1024 * 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
    }
}
