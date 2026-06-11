use serde::{Deserialize, Serialize};

/// One position/economy sample per second per player:
/// (t, x, y, alive, networth, xp, lastHits, level)
/// Serialized as a flat JSON array for compactness.
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct TrackSample(
    pub f32, // t (seconds, 0 = horn)
    pub f32, // world x
    pub f32, // world y
    pub u8,  // alive (1/0)
    pub u32, // net worth
    pub u32, // total xp
    pub u16, // last hits
    pub u8,  // level
);

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PlayerMeta {
    pub slot: u8,        // 0..9 parse order
    pub player_slot: u8, // 0-4 radiant, 128-132 dire (dota convention)
    pub team: u8,        // 2 radiant, 3 dire
    pub hero_id: i32,
    pub hero_name: String, // npc_dota_hero_*
    pub name: String,      // player display name
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct MatchEvent {
    pub t: f32,
    pub kind: String, // kill, death, lastHit, deny, obs, sen, obsLeft, senLeft,
    // rune, smoke, tower, rax, fort, roshan, aegis, buyback, purchase
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slot: Option<i8>, // acting player slot 0..9
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_slot: Option<i8>, // victim (kill) / killer (death)
    pub x: f32,
    pub y: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team: Option<u8>, // owning team for buildings / wards
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>, // item name, rune type, building name, ward killer...
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GameData {
    pub match_id: u64,
    pub duration_s: f32,
    pub winner: u8, // 2 radiant, 3 dire, 0 unknown
    /// Dota client version (steam.inf ClientVersion) the game was recorded on,
    /// parsed from the demo header's game_directory ("dota_v6802"). 0 = unknown.
    #[serde(default)]
    pub game_build: u32,
    pub players: Vec<PlayerMeta>,
    /// tracks[slot] = 1 Hz samples
    pub tracks: Vec<Vec<TrackSample>>,
    pub events: Vec<MatchEvent>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GameSummary {
    pub match_id: u64,
    pub duration_s: f32,
    pub winner: u8,
    #[serde(default)]
    pub game_build: u32,
    pub heroes_radiant: Vec<i32>,
    pub heroes_dire: Vec<i32>,
    pub parsed_at: u64, // unix seconds
    #[serde(default)]
    pub tag: String, // user-assigned bucket, e.g. "3k", "7k", "mine"
}

impl GameData {
    pub fn summary(&self, parsed_at: u64) -> GameSummary {
        GameSummary {
            match_id: self.match_id,
            duration_s: self.duration_s,
            winner: self.winner,
            game_build: self.game_build,
            heroes_radiant: self
                .players
                .iter()
                .filter(|p| p.team == 2)
                .map(|p| p.hero_id)
                .collect(),
            heroes_dire: self
                .players
                .iter()
                .filter(|p| p.team == 3)
                .map(|p| p.hero_id)
                .collect(),
            parsed_at,
            tag: String::new(),
        }
    }
}
