use std::fs;
use std::path::PathBuf;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::demparse::{GameData, GameSummary};

fn library_dir(app: &AppHandle) -> Result<PathBuf> {
    let dir = app
        .path()
        .app_data_dir()
        .context("no app data dir")?
        .join("library");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn index_path(app: &AppHandle) -> Result<PathBuf> {
    Ok(library_dir(app)?.join("index.json"))
}

pub fn read_index(app: &AppHandle) -> Result<Vec<GameSummary>> {
    let path = index_path(app)?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn write_index(app: &AppHandle, index: &[GameSummary]) -> Result<()> {
    fs::write(index_path(app)?, serde_json::to_string(index)?)?;
    Ok(())
}

pub fn save_game(app: &AppHandle, data: &GameData, tag: &str) -> Result<()> {
    let dir = library_dir(app)?;
    fs::write(
        dir.join(format!("{}.json", data.match_id)),
        serde_json::to_string(data)?,
    )?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let mut summary = data.summary(now);
    let mut index = read_index(app)?;
    if let Some(existing) = index.iter().position(|s| s.match_id == data.match_id) {
        // keep an existing tag on re-parse unless a new one was given
        summary.tag = if tag.is_empty() {
            index[existing].tag.clone()
        } else {
            tag.to_string()
        };
        index[existing] = summary;
    } else {
        summary.tag = tag.to_string();
        index.push(summary);
    }
    write_index(app, &index)
}

pub fn load_game(app: &AppHandle, match_id: u64) -> Result<GameData> {
    let path = library_dir(app)?.join(format!("{match_id}.json"));
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

pub fn delete_game(app: &AppHandle, match_id: u64) -> Result<()> {
    let path = library_dir(app)?.join(format!("{match_id}.json"));
    if path.exists() {
        fs::remove_file(path)?;
    }
    let mut index = read_index(app)?;
    index.retain(|s| s.match_id != match_id);
    write_index(app, &index)
}

pub fn set_tag(app: &AppHandle, match_id: u64, tag: &str) -> Result<()> {
    let mut index = read_index(app)?;
    if let Some(s) = index.iter_mut().find(|s| s.match_id == match_id) {
        s.tag = tag.to_string();
    }
    write_index(app, &index)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AggregateFilter {
    pub kinds: Vec<String>,
    /// 2 radiant, 3 dire; None = both
    pub team: Option<u8>,
    /// true = only games the filtered team won; None = all
    pub win: Option<bool>,
    /// restrict to a tag bucket; None/empty = all
    pub tag: Option<String>,
}

/// Aggregated event points across the library: [t, x, y] triples,
/// normalized so t is fraction of game duration when `normalize_time`.
#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AggregateResult {
    pub points: Vec<[f32; 3]>,
    pub games: u32,
}

/// A cross-game *player selection*: pick players out of each game by identity
/// (hero / account / name) and facet (team, win), then pool the events those
/// selected players performed. This generalizes `aggregate_events` from
/// team-scoped to player-scoped — the basis for "study my Slark games",
/// "just my wards across every game", etc.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SelectionFilter {
    pub kinds: Vec<String>,
    /// Classifier — a player is selected when it matches every *non-empty*
    /// criterion. All empty ⇒ every player in the game qualifies.
    #[serde(default)]
    pub heroes: Vec<i32>, // hero_id whitelist
    #[serde(default)]
    pub accounts: Vec<u32>, // 32-bit account_id whitelist
    #[serde(default)]
    pub name_query: Option<String>, // case-insensitive substring on player name
    /// Facets, evaluated per selected player (not the whole game):
    pub team: Option<u8>, // 2 radiant / 3 dire; None = both
    pub win: Option<bool>, // did the selected player's team win? None = both
    pub tag: Option<String>,
}

/// Pool the events of players selected by `filter` across the whole library.
/// Classification runs against the full per-game player list (so hero/name/team
/// facets work even on games parsed before account ids existed; the account
/// facet only matches games re-parsed since).
pub fn aggregate_selection(app: &AppHandle, filter: &SelectionFilter) -> Result<AggregateResult> {
    let index = read_index(app)?;
    let name_q = filter
        .name_query
        .as_deref()
        .map(str::to_lowercase)
        .filter(|s| !s.is_empty());
    let mut points = Vec::new();
    let mut games = 0u32;
    for summary in &index {
        if let Some(tag) = &filter.tag {
            if !tag.is_empty() && &summary.tag != tag {
                continue;
            }
        }
        let Ok(data) = load_game(app, summary.match_id) else {
            continue;
        };
        // Which slots in this game does the classifier pick?
        let selected: Vec<i8> = data
            .players
            .iter()
            .filter(|p| filter.heroes.is_empty() || filter.heroes.contains(&p.hero_id))
            .filter(|p| filter.accounts.is_empty() || filter.accounts.contains(&p.account_id))
            .filter(|p| match &name_q {
                Some(q) => p.name.to_lowercase().contains(q),
                None => true,
            })
            .filter(|p| filter.team.is_none_or(|t| p.team == t))
            .filter(|p| filter.win.is_none_or(|w| (data.winner == p.team) == w))
            .map(|p| p.slot as i8)
            .collect();
        if selected.is_empty() {
            continue;
        }
        games += 1;
        for e in &data.events {
            if !filter.kinds.iter().any(|k| k == &e.kind) {
                continue;
            }
            if e.slot.is_some_and(|s| selected.contains(&s)) {
                points.push([e.t, e.x, e.y]);
            }
        }
    }
    Ok(AggregateResult { points, games })
}

pub fn aggregate_events(app: &AppHandle, filter: &AggregateFilter) -> Result<AggregateResult> {
    let index = read_index(app)?;
    let mut points = Vec::new();
    let mut games = 0u32;
    for summary in &index {
        if let Some(tag) = &filter.tag {
            if !tag.is_empty() && &summary.tag != tag {
                continue;
            }
        }
        if let (Some(team), Some(win)) = (filter.team, filter.win) {
            let team_won = summary.winner == team;
            if team_won != win {
                continue;
            }
        }
        let Ok(data) = load_game(app, summary.match_id) else {
            continue;
        };
        games += 1;
        for e in &data.events {
            if !filter.kinds.iter().any(|k| k == &e.kind) {
                continue;
            }
            if let Some(team) = filter.team {
                if e.team != Some(team) {
                    continue;
                }
            }
            points.push([e.t, e.x, e.y]);
        }
    }
    Ok(AggregateResult { points, games })
}
