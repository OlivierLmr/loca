//! BackendManager: dictionary of alias → BackendInstance, with a port pool.

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::backend::{BackendInstance, BackendState};
use crate::config::{ModelEntry, ProxyCfg};
use crate::log_bus::LogBus;

pub struct BackendManager {
    pub cfg: ProxyCfg,
    pub models: IndexMap<String, ModelEntry>,
    pub bus: Arc<LogBus>,
    inner: Mutex<Inner>,
    // Per-alias lock to serialize concurrent spawn requests for the same model.
    per_alias_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

struct Inner {
    instances: HashMap<String, Arc<BackendInstance>>,
    ports_free: VecDeque<u16>,
}

impl BackendManager {
    pub fn new(cfg: ProxyCfg, models: IndexMap<String, ModelEntry>, bus: Arc<LogBus>) -> Arc<Self> {
        let ports = (cfg.backend_port_start
            ..cfg.backend_port_start + cfg.backend_port_count)
            .collect();
        Arc::new(Self {
            cfg,
            models,
            bus,
            inner: Mutex::new(Inner {
                instances: HashMap::new(),
                ports_free: ports,
            }),
            per_alias_locks: Mutex::new(HashMap::new()),
        })
    }

    fn log(&self, msg: &str) {
        self.bus.emit("loca", msg);
    }

    /// Resolve a user-supplied model string to (alias, hf_id).
    /// Accepts either the alias itself or the HF id.
    pub fn resolve(&self, requested: &str) -> Option<(String, String)> {
        if let Some(m) = self.models.get(requested) {
            return Some((requested.to_string(), m.hf.clone()));
        }
        for (alias, m) in &self.models {
            if m.hf == requested {
                return Some((alias.clone(), m.hf.clone()));
            }
        }
        None
    }

    /// Ensure the given model is loaded and ready. If already loaded, returns
    /// the existing instance. Otherwise spawns a new mlx_lm.server.
    pub async fn ensure(&self, alias: &str, hf_id: &str) -> Result<Arc<BackendInstance>> {
        let lock = self.alias_lock(alias).await;
        let _guard = lock.lock().await;

        // Fast path: already running.
        {
            let g = self.inner.lock().await;
            if let Some(inst) = g.instances.get(alias) {
                if inst.state().await == BackendState::Ready
                    || inst.state().await == BackendState::Loading
                {
                    return Ok(inst.clone());
                }
            }
        }

        // Otherwise: allocate a port and spawn.
        let port = {
            let mut g = self.inner.lock().await;
            // Clean up any stale entry.
            if let Some(stale) = g.instances.remove(alias) {
                stale.stop().await;
                g.ports_free.push_back(stale.port);
            }
            g.ports_free
                .pop_front()
                .ok_or_else(|| anyhow!("no free backend ports"))?
        };

        let inst = Arc::new(BackendInstance::new(
            alias.to_string(),
            hf_id.to_string(),
            port,
            self.cfg.clone(),
            self.bus.clone(),
        ));

        // Insert before spawn so the TUI shows "LOADING".
        {
            let mut g = self.inner.lock().await;
            g.instances.insert(alias.to_string(), inst.clone());
        }
        self.log(&format!("loading {alias}"));

        match inst.spawn().await {
            Ok(()) => {
                self.log(&format!(
                    "loaded {alias} (RAM use: {} GB)",
                    self.ram_used_gb().await
                ));
                Ok(inst)
            }
            Err(e) => {
                // Roll back.
                let mut g = self.inner.lock().await;
                g.instances.remove(alias);
                g.ports_free.push_back(port);
                Err(e)
            }
        }
    }

    pub async fn stop(&self, alias: &str) {
        let inst = {
            let mut g = self.inner.lock().await;
            let inst = g.instances.remove(alias);
            if let Some(ref i) = inst {
                g.ports_free.push_back(i.port);
            }
            inst
        };
        if let Some(inst) = inst {
            inst.stop().await;
            self.log(&format!(
                "stopped {alias} (RAM use: {} GB)",
                self.ram_used_gb().await
            ));
        }
    }

    pub async fn stop_all(&self) {
        let aliases: Vec<String> = self.inner.lock().await.instances.keys().cloned().collect();
        for a in aliases {
            self.stop(&a).await;
        }
    }

    pub async fn instances(&self) -> Vec<Arc<BackendInstance>> {
        self.inner.lock().await.instances.values().cloned().collect()
    }

    pub async fn get(&self, alias: &str) -> Option<Arc<BackendInstance>> {
        self.inner.lock().await.instances.get(alias).cloned()
    }

    pub async fn ram_used_gb(&self) -> u32 {
        let g = self.inner.lock().await;
        let mut total = 0;
        for (alias, _) in &g.instances {
            if let Some(m) = self.models.get(alias) {
                total += m.ram_gb.unwrap_or(0);
            }
        }
        total
    }

    async fn alias_lock(&self, alias: &str) -> Arc<Mutex<()>> {
        let mut g = self.per_alias_locks.lock().await;
        g.entry(alias.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Background idle eviction. Runs forever; cancel by dropping the JoinHandle.
    pub async fn idle_watch(self: Arc<Self>) {
        loop {
            tokio::time::sleep(Duration::from_secs(15)).await;
            let instances = self.instances().await;
            for inst in instances {
                if inst.state().await != BackendState::Ready {
                    continue;
                }
                // Never evict a backend that's actively streaming a response.
                if inst.is_busy() {
                    continue;
                }
                let Some(age) = inst.last_request_age() else { continue };
                if age > inst.idle_timeout() {
                    self.bus.emit(&inst.alias, &format!("idle eviction after {age}s"));
                    self.stop(&inst.alias).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! These tests intentionally never call `ensure()`, so no `mlx_lm.server`
    //! subprocess is ever spawned and no model weights are downloaded.

    use super::*;
    use crate::config::ModelEntry;
    use crate::log_bus::LogBus;
    use indexmap::IndexMap;

    fn fixture() -> Arc<BackendManager> {
        let mut models = IndexMap::new();
        models.insert(
            "alpha".to_string(),
            ModelEntry {
                hf: "mlx-community/Alpha-4bit".into(),
                ram_gb: Some(5),
                description: Some("alpha test model".into()),
            },
        );
        models.insert(
            "beta".to_string(),
            ModelEntry {
                hf: "mlx-community/Beta-4bit".into(),
                ram_gb: Some(8),
                description: None,
            },
        );
        let cfg = ProxyCfg::default();
        let bus = LogBus::new();
        BackendManager::new(cfg, models, bus)
    }

    #[test]
    fn resolve_finds_alias() {
        let mgr = fixture();
        let (alias, hf) = mgr.resolve("alpha").expect("found");
        assert_eq!(alias, "alpha");
        assert_eq!(hf, "mlx-community/Alpha-4bit");
    }

    #[test]
    fn resolve_finds_hf_id_as_fallback() {
        let mgr = fixture();
        let (alias, hf) = mgr.resolve("mlx-community/Beta-4bit").expect("found");
        assert_eq!(alias, "beta");
        assert_eq!(hf, "mlx-community/Beta-4bit");
    }

    #[test]
    fn resolve_returns_none_for_unknown() {
        let mgr = fixture();
        assert!(mgr.resolve("zzz").is_none());
    }

    #[tokio::test]
    async fn ram_used_is_zero_with_no_loaded_instances() {
        let mgr = fixture();
        assert_eq!(mgr.ram_used_gb().await, 0);
    }

    #[tokio::test]
    async fn instances_empty_at_start() {
        let mgr = fixture();
        assert!(mgr.instances().await.is_empty());
    }

    #[tokio::test]
    async fn stop_unknown_alias_is_no_op() {
        let mgr = fixture();
        // Should not panic or hang.
        mgr.stop("never-loaded").await;
        assert!(mgr.instances().await.is_empty());
    }
}
