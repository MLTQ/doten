// Quick CLI harness for testing the replay extractor without launching the app:
//   cargo run --release --example parse_cli -- ../../replays/8846649222.dem

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: parse_cli <demofile>");
    let start = std::time::Instant::now();
    let data = app_lib::demparse::parse_file(&path, |t| eprintln!("  ...parsed {:.0}s of game time", t))?;
    eprintln!("parsed in {:?}", start.elapsed());
    eprintln!(
        "match {} winner {} duration {:.0}s",
        data.match_id, data.winner, data.duration_s
    );
    for p in &data.players {
        eprintln!(
            "  slot {} team {} hero {} ({}) player '{}'",
            p.slot, p.team, p.hero_id, p.hero_name, p.name
        );
    }
    let mut counts = std::collections::BTreeMap::new();
    for e in &data.events {
        *counts.entry(e.kind.clone()).or_insert(0) += 1;
    }
    eprintln!("event counts: {:?}", counts);
    eprintln!(
        "track lengths: {:?}",
        data.tracks.iter().map(|t| t.len()).collect::<Vec<_>>()
    );
    if let Some(s) = data.tracks[0].get(600) {
        eprintln!("slot0 @ sample 600: {:?}", s);
    }
    for e in data.events.iter().filter(|e| e.kind == "tower").take(3) {
        eprintln!("tower: {:?}", e);
    }
    for e in data.events.iter().filter(|e| e.kind == "obs").take(3) {
        eprintln!("obs: {:?}", e);
    }
    let json = serde_json::to_string(&data)?;
    eprintln!("json size: {:.1} MB", json.len() as f32 / 1e6);
    std::fs::write("/tmp/gamedata.json", json)?;
    Ok(())
}
