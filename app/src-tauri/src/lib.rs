pub mod demparse;
pub mod http_api;
pub mod library;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ParseProgress {
    seconds: f32,
}

#[tauri::command]
async fn parse_replay(
    app: AppHandle,
    path: String,
    tag: String,
) -> Result<demparse::GameData, String> {
    let progress_app = app.clone();
    let data = tauri::async_runtime::spawn_blocking(move || {
        demparse::parse_file(&path, move |seconds| {
            let _ = progress_app.emit("parse:progress", ParseProgress { seconds });
        })
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    library::save_game(&app, &data, &tag).map_err(|e| e.to_string())?;
    Ok(data)
}

#[tauri::command]
fn list_games(app: AppHandle) -> Result<Vec<demparse::GameSummary>, String> {
    library::read_index(&app).map_err(|e| e.to_string())
}

#[tauri::command]
fn load_game(app: AppHandle, match_id: u64) -> Result<demparse::GameData, String> {
    library::load_game(&app, match_id).map_err(|e| e.to_string())
}

#[tauri::command]
fn delete_game(app: AppHandle, match_id: u64) -> Result<(), String> {
    library::delete_game(&app, match_id).map_err(|e| e.to_string())
}

#[tauri::command]
fn set_tag(app: AppHandle, match_id: u64, tag: String) -> Result<(), String> {
    library::set_tag(&app, match_id, &tag).map_err(|e| e.to_string())
}

#[tauri::command]
async fn aggregate_events(
    app: AppHandle,
    filter: library::AggregateFilter,
) -> Result<library::AggregateResult, String> {
    tauri::async_runtime::spawn_blocking(move || library::aggregate_events(&app, &filter))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // HTTP surface + mesh announce (gruve). The app works identically
            // when the port is taken or no gruve agent is running.
            if let Some(port) = http_api::start(app.handle().clone()) {
                let announce = gruve_sdk::Announce::app("doten", "Doten", port)
                    .blurb("Dota replays in space-time")
                    .hue(265)
                    .upstream("api", port)
                    .start();
                // keep heartbeating for the app's lifetime; withdraws on exit
                app.manage(announce);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            parse_replay,
            list_games,
            load_game,
            delete_game,
            set_tag,
            aggregate_events
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
