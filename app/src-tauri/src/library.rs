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
