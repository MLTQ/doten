//! HTTP surface for Gruve / remote viewers.
//!
//! The Tauri webview talks to the backend over invoke(), but friends opening
//! Doten from their Gruve lobby get the *frontend* over HTTP — Tauri IPC does
//! not exist for them. This module serves the built frontend (dist/) plus a
//! read-only JSON API mirroring the library commands. Parsing/deleting/tagging
//! stay invoke-only on purpose: remote viewers browse, they don't mutate.

use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;

use tauri::{AppHandle, Manager};
use tiny_http::{Header, Method, Response, Server};

use crate::library::{self, AggregateFilter};

pub const DEFAULT_PORT: u16 = 9171;

pub fn port() -> u16 {
    std::env::var("DOTEN_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

/// Locate the built frontend. Bundled builds carry dist/ as a Tauri resource;
/// `tauri dev` runs with cwd = src-tauri, where the last `pnpm build` output
/// sits at ../dist.
fn dist_dir(app: &AppHandle) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(res) = app.path().resource_dir() {
        candidates.push(res.join("dist"));
        candidates.push(res.join("_up_").join("dist")); // resources declared as "../dist"
    }
    // raw cargo binary (target/{debug,release}/app): walk up to the app dir,
    // independent of launch cwd
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(PathBuf::from);
        for _ in 0..4 {
            if let Some(d) = dir {
                candidates.push(d.join("dist"));
                dir = d.parent().map(PathBuf::from);
            } else {
                break;
            }
        }
    }
    candidates.push(PathBuf::from("../dist"));
    candidates.push(PathBuf::from("dist"));
    candidates.into_iter().find(|p| p.join("index.html").is_file())
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript",
        "css" => "text/css",
        "json" => "application/json",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn json_response(body: String, status: u32) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string(body)
        .with_status_code(status as u16)
        .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
}

fn handle_api(
    app: &AppHandle,
    method: &Method,
    path: &str,
    body: &mut dyn Read,
) -> Response<std::io::Cursor<Vec<u8>>> {
    let err = |e: String| json_response(format!("{{\"error\":{:?}}}", e), 500);
    match (method, path) {
        (Method::Get, "/api/games") => match library::read_index(app) {
            Ok(games) => json_response(serde_json::to_string(&games).unwrap_or_default(), 200),
            Err(e) => err(e.to_string()),
        },
        (Method::Get, p) if p.starts_with("/api/games/") => {
            let Some(id) = p["/api/games/".len()..].parse::<u64>().ok() else {
                return json_response("{\"error\":\"bad match id\"}".into(), 400);
            };
            match library::load_game(app, id) {
                Ok(game) => json_response(serde_json::to_string(&game).unwrap_or_default(), 200),
                Err(_) => json_response("{\"error\":\"not found\"}".into(), 404),
            }
        }
        (Method::Post, "/api/aggregate") => {
            let mut buf = String::new();
            let _ = body.take(1 << 20).read_to_string(&mut buf);
            let Ok(filter) = serde_json::from_str::<AggregateFilter>(&buf) else {
                return json_response("{\"error\":\"bad filter\"}".into(), 400);
            };
            match library::aggregate_events(app, &filter) {
                Ok(res) => json_response(serde_json::to_string(&res).unwrap_or_default(), 200),
                Err(e) => err(e.to_string()),
            }
        }
        _ => json_response("{\"error\":\"not found\"}".into(), 404),
    }
}

/// Start the server thread + gruve announce. Returns the bound port, or None
/// if the port is taken (e.g. a second instance) — the app works fine without.
pub fn start(app: AppHandle) -> Option<u16> {
    let port = port();
    let server = match Server::http(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("doten http: failed to bind 127.0.0.1:{port}: {e}");
            return None;
        }
    };
    let server = Arc::new(server);

    let dist = dist_dir(&app);
    if dist.is_none() {
        eprintln!("doten http: no dist/ found — API only, no remote UI (run `pnpm build`)");
    }

    std::thread::Builder::new()
        .name("doten-http".into())
        .spawn(move || {
            for mut request in server.incoming_requests() {
                let path = request.url().split('?').next().unwrap_or("/").to_string();
                let method = request.method().clone();

                if path.starts_with("/api/") {
                    let mut body_reader = request.as_reader();
                    let mut body_buf = Vec::new();
                    let _ = std::io::Read::by_ref(&mut body_reader)
                        .take(1 << 20)
                        .read_to_end(&mut body_buf);
                    let resp =
                        handle_api(&app, &method, &path, &mut std::io::Cursor::new(body_buf));
                    let _ = request.respond(resp);
                    continue;
                }

                // static frontend, with SPA fallback to index.html
                let Some(dist) = &dist else {
                    let _ = request.respond(
                        Response::from_string("doten: frontend not built").with_status_code(503),
                    );
                    continue;
                };
                let rel = path.trim_start_matches('/');
                let mut file = dist.join(rel);
                // no traversal, no surprises: resolve and require the dist prefix
                if !file
                    .canonicalize()
                    .map(|c| c.starts_with(dist.canonicalize().unwrap_or_default()))
                    .unwrap_or(false)
                    || rel.is_empty()
                    || !file.is_file()
                {
                    file = dist.join("index.html");
                }
                match std::fs::read(&file) {
                    Ok(bytes) => {
                        let ct = content_type(&file.to_string_lossy());
                        let _ = request.respond(
                            Response::from_data(bytes).with_header(
                                Header::from_bytes(&b"Content-Type"[..], ct.as_bytes()).unwrap(),
                            ),
                        );
                    }
                    Err(_) => {
                        let _ = request
                            .respond(Response::from_string("not found").with_status_code(404));
                    }
                }
            }
        })
        .ok()?;

    Some(port)
}
