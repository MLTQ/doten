pub mod extractor;
pub mod game_time;
pub mod types;
pub mod wards;

use std::io::BufReader;

use anyhow::Result;
use source2_demo::prelude::*;

pub use types::{GameData, MatchEvent, GameSummary, PlayerMeta, TrackSample};

/// Parse a .dem replay into GameData. `progress` receives seconds of
/// game time processed (roughly every 30s of game time).
pub fn parse_file(path: &str, progress: impl Fn(f32) + 'static) -> Result<GameData> {
    let input = BufReader::new(std::fs::File::open(path)?);
    let mut parser = Parser::from_reader(input)?;

    let game_time = parser.register_observer::<game_time::GameTime>();
    let wards = parser.register_observer::<wards::Wards>();
    let app = parser.register_observer::<extractor::Extractor>();

    app.borrow_mut().game_time = game_time.clone();
    app.borrow_mut().progress = Some(Box::new(progress));

    game_time.borrow_mut().register_observer(app.clone());
    wards.borrow_mut().register_observer(app.clone());

    parser.run_to_end()?;

    let data = app.borrow_mut().finalize();
    Ok(data)
}
