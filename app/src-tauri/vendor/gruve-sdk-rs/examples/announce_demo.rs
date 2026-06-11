//! Conformance demo: a tiny std-only HTTP server, announced to the local agent.
//! Run a local agent (`./gruve`), then: cargo run --example announce_demo

use std::io::Write;
use std::net::TcpListener;

fn main() {
    // 1.4: listen BEFORE announcing — the agent probes the port.
    let listener = TcpListener::bind("127.0.0.1:9700").expect("bind 9700");
    println!("rust demo app on :9700");

    let _gruve = gruve_sdk::Announce::app("rustdemo", "Rust Demo", 9700)
        .blurb("announced by gruve-sdk (rust)")
        .hue(25)
        .start();
    println!("announced; check the lobby. ctrl-c to exit (withdraws).");

    for stream in listener.incoming().flatten() {
        let mut s = stream;
        let _ = s.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n\
              <body style='background:#141821;color:#f0b27a;font-family:monospace;display:grid;place-items:center;height:100vh'>\
              <h1>hello from rust &#129408;</h1></body>",
        );
    }
}
