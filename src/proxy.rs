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
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

use crate::manager::BackendManager;
use crate::assemble::ChatAssembler;
use crate::stats::SseTap;

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
    let now = std::time::Instant::now();
    for inst in s.mgr.instances().await {
        let (active, in_flight, last) = inst.request_stats();
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
            "in_flight": in_flight,
            "current_request": active.map(|r| r.to_json(now)),
            "last_request": last.map(|r| r.to_json(now)),
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
    let guard = inst.enter_request(s.mgr.bus.clone());

    // Rewrite the model field to the HF id for the upstream call.
    if let Some(m) = payload.get_mut("model") {
        *m = Value::String(hf.clone());
    }
    // Non-streaming chat requests are streamed upstream anyway so their
    // stats are live, then reassembled into one JSON reply (ChatAssembler).
    // Logprobs aren't carried over by the reassembly, so those stay as-is.
    let rebuild = !is_stream
        && path == "/v1/chat/completions"
        && !payload["logprobs"].as_bool().unwrap_or(false);
    if rebuild {
        payload["stream"] = Value::Bool(true);
    }
    // Ask for the final usage chunk (exact token counts, cached prompt
    // tokens) on every stream; SseTap drops it if the client didn't ask.
    // A reassembled reply always carries usage, like mlx's own JSON reply.
    let client_wants_usage = payload["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    let strip_usage = !client_wants_usage && !rebuild;
    if (is_stream || rebuild) && !client_wants_usage {
        if let Some(obj) = payload.as_object_mut() {
            let opts = obj.entry("stream_options").or_insert_with(|| json!({}));
            if let Some(opts) = opts.as_object_mut() {
                opts.insert("include_usage".into(), Value::Bool(true));
            }
        }
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

    // Stream the body through, counting bytes and emitting periodic
    // heartbeats so the log pane doesn't look frozen during long token
    // generation. `_guard` is held inside the closure and drops when the
    // stream ends (or the client disconnects), decrementing in_flight.
    use futures::stream::StreamExt;
    let alias_for_stream = alias.clone();
    let bus = s.mgr.bus.clone();
    let inst_for_stream = inst.clone();
    let mut last_heartbeat = std::time::Instant::now();
    let heartbeat_every = std::time::Duration::from_secs(5);
    let tap = std::sync::Arc::new(parking_lot::Mutex::new(SseTap::new(strip_usage)));
    let tap_for_end = tap.clone();
    let tap_events = (is_stream || rebuild) && status.is_success();
    let stream = upstream_resp.bytes_stream().map(move |chunk| {
        // Borrow keeps the guard alive for the closure's lifetime.
        let g = &guard;
        match chunk {
            Ok(b) => {
                inst_for_stream.add_bytes(b.len() as u64);
                let b = if tap_events {
                    let now = std::time::Instant::now();
                    inst_for_stream
                        .with_request_stats(g.id, |s| tap.lock().feed(&b, s, now))
                        .unwrap_or(b)
                } else {
                    b
                };
                if last_heartbeat.elapsed() >= heartbeat_every {
                    last_heartbeat = std::time::Instant::now();
                    let elapsed = inst_for_stream
                        .current_request_elapsed_sec()
                        .unwrap_or(0);
                    let bytes = inst_for_stream.current_request_bytes_count();
                    bus.emit(
                        &alias_for_stream,
                        &format!("…streaming ({} elapsed, {})", fmt_elapsed(elapsed), fmt_bytes(bytes)),
                    );
                }
                Ok::<_, std::io::Error>(b)
            }
            Err(e) => {
                bus.emit(
                    &alias_for_stream,
                    &format!("upstream stream error: {e}"),
                );
                Err(std::io::Error::new(std::io::ErrorKind::Other, e))
            }
        }
    });

    // Forward any trailing partial event once upstream closes.
    let tail = futures::stream::once(async move {
        Ok::<_, std::io::Error>(tap_for_end.lock().finish())
    });
    let stream = stream.chain(tail);

    if rebuild && status.is_success() {
        let mut stream = Box::pin(stream);
        let mut assembler = ChatAssembler::default();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(b) => assembler.feed(&b),
                Err(e) => {
                    inst.note_error();
                    return (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({"error": {"message": format!("upstream error: {e}")}})),
                    )
                        .into_response();
                }
            }
        }
        return Json(assembler.finish()).into_response();
    }

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

fn fmt_bytes(n: u64) -> String {
    if n < 1024 {
        format!("{n} B")
    } else if n < 1024 * 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
    }
}

fn fmt_elapsed(sec: u64) -> String {
    if sec < 60 {
        format!("{sec}s")
    } else {
        format!("{}m {}s", sec / 60, sec % 60)
    }
}
