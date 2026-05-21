//! axum HTTP proxy. OpenAI-compatible endpoints + admin endpoints.

use anyhow::Result;
use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

use crate::manager::BackendManager;

#[derive(Clone)]
pub struct ProxyState {
    pub mgr: Arc<BackendManager>,
    pub http: reqwest::Client,
}

pub fn make_router(mgr: Arc<BackendManager>) -> Router {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(60 * 60))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client");
    let state = ProxyState { mgr, http };

    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(forward))
        .route("/v1/completions", post(forward))
        .route("/admin/status", get(admin_status))
        .route("/admin/stop", post(admin_stop))
        .route("/admin/preload", post(admin_preload))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true }))
}

async fn list_models(State(s): State<ProxyState>) -> Json<Value> {
    let mut data = Vec::new();
    let loaded: std::collections::HashSet<String> = s
        .mgr
        .instances()
        .await
        .iter()
        .map(|i| i.alias.clone())
        .collect();
    for (alias, m) in &s.mgr.models {
        data.push(json!({
            "id": alias,
            "object": "model",
            "owned_by": "loca",
            "hf_id": m.hf,
            "loaded": loaded.contains(alias),
        }));
    }
    Json(json!({ "object": "list", "data": data }))
}

#[derive(Deserialize)]
struct ModelQuery {
    model: Option<String>,
}

#[derive(Deserialize)]
struct ModelBody {
    model: Option<String>,
}

async fn admin_status(State(s): State<ProxyState>) -> Json<Value> {
    let mut loaded = Vec::new();
    for inst in s.mgr.instances().await {
        let state = match inst.state().await {
            crate::backend::BackendState::Stopped => "stopped",
            crate::backend::BackendState::Loading => "loading",
            crate::backend::BackendState::Ready => "ready",
        };
        loaded.push(json!({
            "alias": inst.alias,
            "hf_id": inst.hf_id,
            "port": inst.port,
            "pid": inst.pid().await,
            "state": state,
            "last_request_age_sec": inst.last_request_age(),
            "idle_timeout_sec": inst.idle_timeout(),
            "request_count": inst.request_count.load(std::sync::atomic::Ordering::Relaxed),
            "error_count": inst.error_count.load(std::sync::atomic::Ordering::Relaxed),
            "ram_gb": s.mgr.models.get(&inst.alias).and_then(|m| m.ram_gb),
        }));
    }
    Json(json!({
        "loaded": loaded,
        "ram_used_gb": s.mgr.ram_used_gb().await,
        "ram_budget_gb": s.mgr.cfg.ram_budget_gb,
    }))
}

async fn admin_stop(
    State(s): State<ProxyState>,
    Query(q): Query<ModelQuery>,
    body: Option<Json<ModelBody>>,
) -> Json<Value> {
    let model = q.model.or_else(|| body.and_then(|b| b.0.model));
    match model {
        Some(alias) => {
            s.mgr.stop(&alias).await;
            Json(json!({ "ok": true, "stopped": alias }))
        }
        None => {
            s.mgr.stop_all().await;
            Json(json!({ "ok": true, "stopped": "all" }))
        }
    }
}

async fn admin_preload(
    State(s): State<ProxyState>,
    Query(q): Query<ModelQuery>,
    body: Option<Json<ModelBody>>,
) -> (StatusCode, Json<Value>) {
    let model = match q.model.or_else(|| body.and_then(|b| b.0.model)) {
        Some(m) => m,
        None => return (StatusCode::BAD_REQUEST, Json(json!({"error": "model required"}))),
    };
    let resolved = s.mgr.resolve(&model);
    let (alias, hf) = match resolved {
        Some(x) => x,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": format!("unknown model: {model}")})),
            )
        }
    };
    match s.mgr.ensure(&alias, &hf).await {
        Ok(_) => (StatusCode::OK, Json(json!({ "ok": true, "loaded": alias }))),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": e.to_string()})),
        ),
    }
}

/// Forward POSTs to /v1/chat/completions or /v1/completions to the right backend.
async fn forward(
    State(s): State<ProxyState>,
    method: Method,
    req_headers: HeaderMap,
    uri: axum::http::Uri,
    body: axum::body::Bytes,
) -> Response {
    let _ = (method, req_headers);
    // Parse body as JSON to find the model.
    let mut payload: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": "invalid JSON body"}})),
            )
                .into_response()
        }
    };
    let requested = payload
        .get("model")
        .and_then(|v| v.as_str())
        .map(String::from);
    let requested = match requested {
        Some(r) => r,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": "model required"}})),
            )
                .into_response()
        }
    };
    let resolved = s.mgr.resolve(&requested);
    let (alias, hf) = match resolved {
        Some(x) => x,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": {"message": format!("unknown model: {requested}")}})),
            )
                .into_response()
        }
    };

    let path = uri.path().to_string();
    let is_stream = payload.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    s.mgr
        .bus
        .emit(&alias, &format!("POST {path} stream={is_stream}"));

    let inst = match s.mgr.ensure(&alias, &hf).await {
        Ok(i) => i,
        Err(e) => {
            s.mgr.bus.emit(&alias, &format!("load failed: {e}"));
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": {"message": e.to_string()}})),
            )
                .into_response();
        }
    };
    inst.note_request();

    // Rewrite the model field to the HF id for the upstream call.
    if let Some(m) = payload.get_mut("model") {
        *m = Value::String(hf.clone());
    }

    let upstream_url = format!(
        "http://{}:{}{}",
        s.mgr.cfg.backend_host, inst.port, path
    );

    let upstream_resp = match s
        .http
        .post(&upstream_url)
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            inst.note_error();
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": format!("upstream error: {e}")}})),
            )
                .into_response();
        }
    };

    let status = upstream_resp.status();
    let content_type = upstream_resp
        .headers()
        .get("content-type")
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));

    // Stream the body through. reqwest's bytes_stream gives Result<Bytes, reqwest::Error>;
    // axum's Body::from_stream wants Result<Bytes, BoxError>.
    use futures::stream::StreamExt;
    let alias_for_stream = alias.clone();
    let bus = s.mgr.bus.clone();
    let stream = upstream_resp.bytes_stream().map(move |chunk| {
        match chunk {
            Ok(b) => Ok::<_, std::io::Error>(b),
            Err(e) => {
                bus.emit(
                    &alias_for_stream,
                    &format!("upstream stream error: {e}"),
                );
                Err(std::io::Error::new(std::io::ErrorKind::Other, e))
            }
        }
    });

    let body = Body::from_stream(stream);
    let mut response = Response::new(body);
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(axum::http::header::CONTENT_TYPE, content_type);
    response
}

/// Bind the proxy port. Fails fast with EADDRINUSE if another loca is running.
pub async fn bind_listener(cfg: &crate::config::ProxyCfg) -> Result<tokio::net::TcpListener> {
    let addr = format!("{}:{}", cfg.listen_host, cfg.listen_port);
    tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            anyhow::anyhow!(
                "port {} is already in use — another loca is already running.\n      check with:  loca status",
                cfg.listen_port
            )
        } else {
            anyhow::anyhow!("failed to bind {addr}: {e}")
        }
    })
}

#[derive(Serialize)]
#[allow(dead_code)]
struct StatusResponse {
    loaded: Vec<Value>,
    ram_used_gb: u32,
    ram_budget_gb: u32,
}
