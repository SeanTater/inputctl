//! Embedded web server for the agent debugger.
//!
//! This module implements a lightweight Axum-based server that exposes the
//! current state of an [`Agent`] over HTTP and WebSockets.
//!
//! # Endpoints
//! - `GET /api/state`: Returns the full current [`AgentState`] as JSON.
//! - `GET /ws`: A WebSocket endpoint that streams [`AgentState`] updates in real-time.

use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::{State, ws::{WebSocketUpgrade, WebSocket}},
    response::{IntoResponse, Response},
    http::{header, StatusCode, Uri},
};
use std::sync::Arc;
use crate::debugger::{StateStore, AgentState};
use crate::llm::Message;
use tower_http::cors::CorsLayer;
use tracing::info;
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "ui/dist/"]
struct Assets;

pub struct DebugServer {
    state_store: Arc<StateStore>,
}

impl DebugServer {
    pub fn new(state_store: Arc<StateStore>) -> Self {
        Self { state_store }
    }

    pub async fn run(&self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let app_state = Arc::new(AppState {
            state_store: self.state_store.clone(),
        });

        let app = Router::new()
            .route("/api/state", get(get_state))
            .route("/api/pause", post(pause_agent))
            .route("/api/resume", post(resume_agent))
            .route("/api/inject", post(inject_message))
            .route("/ws", get(ws_handler))
            .fallback(static_handler)
            .layer(CorsLayer::permissive())
            .with_state(app_state);

        let addr = format!("0.0.0.0:{}", port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        info!("Debugger server running on http://{}", addr);
        axum::serve(listener, app).await?;
        Ok(())
    }
}
async fn static_handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');

    if path.is_empty() || path == "index.html" {
        return index_html().await;
    }

    match Assets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            Response::builder()
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(axum::body::Body::from(content.data))
                .unwrap()
        }
        None => index_html().await,
    }
}

async fn index_html() -> Response {
    match Assets::get("index.html") {
        Some(content) => Response::builder()
            .header(header::CONTENT_TYPE, "text/html")
            .body(axum::body::Body::from(content.data))
            .unwrap(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

struct AppState {
    state_store: Arc<StateStore>,
}

async fn get_state(State(state): State<Arc<AppState>>) -> Json<AgentState> {
    Json(state.state_store.get_state())
}

async fn pause_agent(State(state): State<Arc<AppState>>) -> StatusCode {
    state.state_store.set_paused(true);
    StatusCode::OK
}

async fn resume_agent(State(state): State<Arc<AppState>>) -> StatusCode {
    state.state_store.set_paused(false);
    StatusCode::OK
}

async fn inject_message(
    State(state): State<Arc<AppState>>,
    Json(msg): Json<Message>,
) -> StatusCode {
    state.state_store.inject_message(msg);
    StatusCode::OK
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.state_store.subscribe();
    
    // Send initial state
    let initial = state.state_store.get_state();
    if let Ok(msg) = serde_json::to_string(&initial) {
        if socket.send(axum::extract::ws::Message::Text(msg.into())).await.is_err() {
            return;
        }
    }

    while let Ok(update) = rx.recv().await {
        if let Ok(msg) = serde_json::to_string(&update) {
            if socket.send(axum::extract::ws::Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    }
}
