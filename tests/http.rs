//! HTTP integration tests for the proxy router.
//!
//! These tests build the axum Router with a `BackendManager` that has known
//! aliases but no loaded instances. They exercise:
//!   - /health
//!   - /v1/models   (list aliases, none loaded)
//!   - /admin/status
//!   - /admin/stop (no-op when nothing loaded)
//!   - /admin/preload validation (missing model → 400, unknown → 404)
//!   - /v1/chat/completions validation (missing model → 400, unknown → 404)
//!
//! Critically, **no test triggers BackendManager::ensure()**, so
//! `mlx_lm.server` is never spawned and no model weights are downloaded.

use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use indexmap::IndexMap;
use loca::config::{ModelEntry, ProxyCfg};
use loca::log_bus::LogBus;
use loca::manager::BackendManager;
use loca::proxy::make_router;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt; // for `oneshot`

fn fixture() -> Arc<BackendManager> {
    let mut models = IndexMap::new();
    models.insert(
        "alpha".to_string(),
        ModelEntry {
            hf: "mlx-community/Alpha-4bit".into(),
            ram_gb: Some(5),
            description: Some("alpha".into()),
        },
    );
    models.insert(
        "beta".to_string(),
        ModelEntry {
            hf: "mlx-community/Beta-4bit".into(),
            ram_gb: Some(9),
            description: None,
        },
    );
    let cfg = ProxyCfg::default();
    let bus = LogBus::new();
    BackendManager::new(cfg, models, bus)
}

async fn json_response(router: axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = router.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
    (status, v)
}

#[tokio::test]
async fn health_endpoint() {
    let router = make_router(fixture());
    let (status, body) = json_response(
        router,
        Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"ok": true}));
}

#[tokio::test]
async fn list_models_returns_known_aliases_unloaded() {
    let router = make_router(fixture());
    let (status, body) = json_response(
        router,
        Request::builder()
            .method(Method::GET)
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let data = body["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"alpha"));
    assert!(ids.contains(&"beta"));
    for m in data {
        assert_eq!(m["loaded"], json!(false));
        assert!(m["hf_id"].as_str().unwrap().starts_with("mlx-community/"));
    }
}

#[tokio::test]
async fn admin_status_reports_empty_state() {
    let router = make_router(fixture());
    let (status, body) = json_response(
        router,
        Request::builder()
            .method(Method::GET)
            .uri("/admin/status")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["loaded"], json!([]));
    assert_eq!(body["ram_used_gb"], json!(0));
    assert_eq!(body["ram_budget_gb"], json!(24));
}

#[tokio::test]
async fn admin_stop_with_no_model_is_ok() {
    let router = make_router(fixture());
    let (status, body) = json_response(
        router,
        Request::builder()
            .method(Method::POST)
            .uri("/admin/stop")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"ok": true, "stopped": "all"}));
}

#[tokio::test]
async fn admin_stop_for_unknown_alias_is_ok() {
    let router = make_router(fixture());
    let (status, body) = json_response(
        router,
        Request::builder()
            .method(Method::POST)
            .uri("/admin/stop?model=neverexisted")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["stopped"], json!("neverexisted"));
}

#[tokio::test]
async fn admin_preload_missing_model_400() {
    let router = make_router(fixture());
    let (status, body) = json_response(
        router,
        Request::builder()
            .method(Method::POST)
            .uri("/admin/preload")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn admin_preload_unknown_model_404() {
    let router = make_router(fixture());
    let (status, body) = json_response(
        router,
        Request::builder()
            .method(Method::POST)
            .uri("/admin/preload?model=does-not-exist")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let err = body["error"].as_str().unwrap();
    assert!(err.contains("does-not-exist"));
}

#[tokio::test]
async fn forward_rejects_invalid_json() {
    let router = make_router(fixture());
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("this is not json"))
        .unwrap();
    let (status, body) = json_response(router, req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("invalid JSON"));
}

#[tokio::test]
async fn forward_rejects_missing_model_field() {
    let router = make_router(fixture());
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"messages":[]}"#))
        .unwrap();
    let (status, body) = json_response(router, req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("model required"));
}

#[tokio::test]
async fn forward_rejects_unknown_model() {
    let router = make_router(fixture());
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"not-a-real-alias","messages":[]}"#))
        .unwrap();
    let (status, body) = json_response(router, req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.contains("unknown model"));
    assert!(msg.contains("not-a-real-alias"));
}

#[tokio::test]
async fn forward_completions_endpoint_also_validates() {
    let router = make_router(fixture());
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{}"#))
        .unwrap();
    let (status, _) = json_response(router, req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
