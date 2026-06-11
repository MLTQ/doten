//! gruve-sdk — make a Rust app discoverable on a Gruve mesh.
//!
//! Implements the Gruve Adapter Protocol, levels 1 (announce) and 2 (dispatch). Zero
//! dependencies: the agent lives on localhost, so `std::net` HTTP/1.1 is all we need.
//!
//! ```no_run
//! // after your HTTP server is LISTENING (the agent probes the port):
//! let _gruve = gruve_sdk::Announce::app("hunger", "Hunger", 9700)
//!     .blurb("novelty-driven crawler")
//!     .hue(95)
//!     .upstream("api", 9701)
//!     .start();
//! // heartbeats from a background thread; agent absent = silently retried (your app
//! // works identically without Gruve). Withdraws on drop/stop.
//! ```
//!
//! A capability instead of a lobby app (`/svc/<id>/`, no tile):
//! ```no_run
//! let _svc = gruve_sdk::Announce::service("osint-feed", 9800).start();
//! let llm = gruve_sdk::service_base("inference"); // consume mesh capabilities by name
//! ```

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Agent address: `GRUVE_AGENT` env (host:port) or the default local agent.
pub fn agent_addr() -> String {
    std::env::var("GRUVE_AGENT").unwrap_or_else(|_| "127.0.0.1:8088".to_string())
}

/// Base URL for a named mesh capability, resolved by the local agent (protocol L2).
/// `service_base("inference")` → `http://127.0.0.1:8088/svc/inference`
pub fn service_base(name: &str) -> String {
    format!("http://{}/svc/{}", agent_addr(), name)
}

/// Builder for an announcement (protocol L1).
pub struct Announce {
    id: String,
    name: String,
    port: u16,
    ttl: u32,
    hue: u32,
    blurb: String,
    upstreams: Vec<(String, u16)>,
    service: bool,
}

impl Announce {
    /// A lobby app: a tile friends can open. `port` must serve your HTTP UI.
    pub fn app(id: &str, name: &str, port: u16) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            port,
            ttl: 60,
            hue: 250,
            blurb: String::new(),
            upstreams: Vec::new(),
            service: false,
        }
    }

    /// A mesh capability at `/svc/<id>/` — no tile, consumable by name from any node.
    pub fn service(id: &str, port: u16) -> Self {
        let mut a = Self::app(id, id, port);
        a.service = true;
        a
    }

    pub fn blurb(mut self, b: &str) -> Self {
        self.blurb = b.into();
        self
    }
    pub fn hue(mut self, h: u32) -> Self {
        self.hue = h;
        self
    }
    pub fn ttl(mut self, secs: u32) -> Self {
        self.ttl = secs.clamp(5, 300);
        self
    }
    /// Declare a named backend port, reachable from your frontend via apiBase(name).
    pub fn upstream(mut self, name: &str, port: u16) -> Self {
        self.upstreams.push((name.into(), port));
        self
    }

    /// Start heartbeating from a background thread. Never blocks, never panics into the host.
    pub fn start(self) -> AnnounceHandle {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let beat_every = Duration::from_secs((self.ttl / 3).max(2) as u64);
        let body = self.json();
        let (id, service) = (self.id.clone(), self.service);

        thread::Builder::new()
            .name("gruve-announce".into())
            .spawn(move || {
                loop {
                    // resilience (1.2): errors are expected when no agent runs — stay quiet
                    let _ = http_request("POST", "/gruve/announce", Some(&body));
                    // sleep in slices so stop() withdraws promptly (1.5)
                    let mut slept = Duration::ZERO;
                    while slept < beat_every {
                        if flag.load(Ordering::Relaxed) {
                            let q = if service { "&service=1" } else { "" };
                            let _ = http_request(
                                "DELETE",
                                &format!("/gruve/announce?id={}{}", id, q),
                                None,
                            );
                            return;
                        }
                        thread::sleep(Duration::from_millis(250));
                        slept += Duration::from_millis(250);
                    }
                }
            })
            .ok();

        AnnounceHandle { stop }
    }

    fn json(&self) -> String {
        let ups = self
            .upstreams
            .iter()
            .map(|(n, p)| format!("{}:{}", json_str(n), p))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"id\":{},\"name\":{},\"port\":{},\"ttl\":{},\"hue\":{},\"blurb\":{},\"upstreams\":{{{}}},\"service\":{}}}",
            json_str(&self.id),
            json_str(&self.name),
            self.port,
            self.ttl,
            self.hue,
            json_str(&self.blurb),
            ups,
            self.service
        )
    }
}

/// Stops the heartbeat and withdraws the announcement (also on drop).
pub struct AnnounceHandle {
    stop: Arc<AtomicBool>,
}

impl AnnounceHandle {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for AnnounceHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Minimal HTTP/1.1 over localhost — returns the status code.
fn http_request(method: &str, path: &str, body: Option<&str>) -> std::io::Result<u16> {
    let addr = agent_addr();
    let mut stream = TcpStream::connect(&addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let b = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    );
    stream.write_all(req.as_bytes())?;
    let mut buf = [0u8; 64];
    let n = stream.read(&mut buf)?;
    let line = String::from_utf8_lossy(&buf[..n]);
    let code = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    Ok(code)
}
