use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use source2_demo::prelude::*;
use source2_demo::proto::*;

use super::game_time::{GameTime, GameTimeObserver};
use super::types::*;
use super::wards::{WardClass, WardEvent, WardsObserver};

const CELL_SIZE: f32 = 128.0;
const MAX_COORD: f32 = 16384.0;

fn world_coord(cell: u16, vec: f32) -> f32 {
    cell as f32 * CELL_SIZE + vec - MAX_COORD
}

/// Extracts a compact GameData from a Dota 2 replay.
/// Times are kept on the raw game clock during parsing; `finalize()`
/// rebases everything so 0 = horn (start_time).
pub struct Extractor {
    pub game_time: Rc<RefCell<GameTime>>,
    pub progress: Option<Box<dyn Fn(f32)>>, // called with seconds of game time parsed

    match_id: u64,
    winner: u8,
    game_build: u32,
    start_time: f32,
    players: Vec<PlayerMeta>,
    tracks: Vec<Vec<TrackSample>>,
    events: Vec<MatchEvent>,

    init: bool,
    post_game: bool,
    next_interval: i32,
    valid_indices: [i32; 10],
    cur_pos: [[f32; 2]; 10],
    cur_alive: [bool; 10],
    slot_team: [u8; 10],
    // combat-log name (npc_dota_hero_*) -> slot
    name_to_slot: HashMap<String, usize>,
    class_to_log_name: HashMap<String, String>,
    // building entities never expose names in the replay stream, so we track
    // life-state transitions with positions here and pair them with the named
    // combat-log death (by kind + team + time) when it arrives
    building_state: HashMap<u32, (u8, f32, f32, u8)>, // ent idx -> (life, x, y, team)
    recent_building_deaths: Vec<(&'static str, u8, f32, f32, f32)>, // kind, team, x, y, t
    pending_building_cl: Vec<(f32, &'static str, u8, Option<i8>, String)>, // t, kind, team, killer slot, name
    file_info_players: Vec<(String, String)>, // (hero_name, player_name)
}

impl Default for Extractor {
    fn default() -> Self {
        Extractor {
            game_time: Default::default(),
            progress: None,
            match_id: 0,
            winner: 0,
            game_build: 0,
            start_time: 0.0,
            players: (0..10)
                .map(|i| PlayerMeta {
                    slot: i,
                    player_slot: 0,
                    team: 0,
                    hero_id: 0,
                    hero_name: String::new(),
                    name: String::new(),
                })
                .collect(),
            tracks: vec![Vec::new(); 10],
            events: Vec::new(),
            init: false,
            post_game: false,
            next_interval: 0,
            valid_indices: [0; 10],
            cur_pos: [[0.0; 2]; 10],
            cur_alive: [false; 10],
            slot_team: [0; 10],
            name_to_slot: HashMap::new(),
            class_to_log_name: HashMap::new(),
            building_state: HashMap::new(),
            recent_building_deaths: Vec::new(),
            pending_building_cl: Vec::new(),
            file_info_players: Vec::new(),
        }
    }
}

impl Extractor {
    fn time(&self, ctx: &Context) -> anyhow::Result<f32> {
        Ok(self.game_time.borrow().tick(ctx)? as f32 / 30.0)
    }

    fn slot_of(&self, combat_log_name: &str) -> Option<usize> {
        self.name_to_slot.get(combat_log_name).copied()
    }

    fn push_event(&mut self, ev: MatchEvent) {
        self.events.push(ev);
    }

    fn event_at_slot(&self, t: f32, kind: &str, slot: usize) -> MatchEvent {
        MatchEvent {
            t,
            kind: kind.to_string(),
            slot: Some(slot as i8),
            target_slot: None,
            x: self.cur_pos[slot][0],
            y: self.cur_pos[slot][1],
            team: Some(self.slot_team[slot]),
            key: None,
        }
    }

    /// Rebase times, trim pre-game noise, fill metadata. Called after run_to_end.
    pub fn finalize(&mut self) -> GameData {
        // pair building combat-log deaths (which carry names) with entity
        // life-state deaths (which carry positions)
        let pending = std::mem::take(&mut self.pending_building_cl);
        for (t, kind, cl_team, slot, name) in pending {
            let matched = self
                .recent_building_deaths
                .iter()
                .position(|&(k, bteam, _, _, bt)| {
                    k == kind && (cl_team == 0 || bteam == cl_team) && (bt - t).abs() < 10.0
                });
            let (x, y) = match matched {
                Some(i) => {
                    let (_, _, x, y, _) = self.recent_building_deaths.remove(i);
                    (x, y)
                }
                None => (0.0, 0.0),
            };
            self.events.push(MatchEvent {
                t,
                kind: kind.to_string(),
                slot,
                target_slot: None,
                x,
                y,
                team: Some(cl_team),
                key: Some(name),
            });
        }
        self.events
            .sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        let t0 = self.start_time;
        for track in &mut self.tracks {
            for s in track.iter_mut() {
                s.0 -= t0;
            }
            track.retain(|s| s.0 >= -90.0);
        }
        for e in &mut self.events {
            e.t -= t0;
        }
        self.events.retain(|e| e.t >= -90.0);
        // pair player names from file info (matched by hero name)
        for (hero_name, player_name) in &self.file_info_players {
            if let Some(&slot) = self.name_to_slot.get(hero_name.as_str()) {
                self.players[slot].name = player_name.clone();
            }
        }
        let duration = self
            .tracks
            .iter()
            .flat_map(|t| t.last())
            .map(|s| s.0)
            .fold(0.0f32, f32::max);
        GameData {
            match_id: self.match_id,
            duration_s: duration,
            winner: self.winner,
            game_build: self.game_build,
            players: std::mem::take(&mut self.players),
            tracks: std::mem::take(&mut self.tracks),
            events: std::mem::take(&mut self.events),
        }
    }
}

#[observer]
#[uses_entities]
#[uses_combat_log]
impl Extractor {
    #[on_message]
    fn handle_file_header(&mut self, _ctx: &Context, header: CDemoFileHeader) -> ObserverResult {
        // game_directory looks like "/opt/srcds/dota/dota_v6802/dota"; the
        // v-suffix is the steam.inf ClientVersion, our patch fingerprint
        if let Some(i) = header.game_directory().find("dota_v") {
            self.game_build = header.game_directory()[i + 6..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0);
        }
        Ok(())
    }

    #[on_message]
    fn handle_file_info(&mut self, _ctx: &Context, file_info: CDemoFileInfo) -> ObserverResult {
        if let Some(game_info) = &file_info.game_info {
            if let Some(dota) = &game_info.dota {
                self.match_id = dota.match_id() as u64;
                self.winner = match dota.game_winner() {
                    2 => 2,
                    3 => 3,
                    _ => 0,
                };
                for pi in &dota.player_info {
                    self.file_info_players
                        .push((pi.hero_name().to_string(), String::from_utf8_lossy(pi.player_name()).to_string()));
                }
            }
        }
        Ok(())
    }

    #[on_tick_start]
    fn tick_start(&mut self, ctx: &Context) -> ObserverResult {
        let Ok(pr) = ctx.entities().get_by_class_name("CDOTA_PlayerResource") else {
            return Ok(());
        };

        if !self.init {
            let mut added = 0usize;
            let mut i = 0i32;
            let mut waiting_for_draft = false;
            while added < 10 && i < 30 {
                let player_team: i32 = property!(pr, "m_vecPlayerData.{i:04}.m_iPlayerTeam");
                let team_slot: i32 = property!(pr, "m_vecPlayerTeamData.{i:04}.m_iTeamSlot");
                if player_team == 2 || player_team == 3 {
                    let p = &mut self.players[added];
                    p.team = player_team as u8;
                    p.player_slot = (if player_team == 2 { 0 } else { 128 } + team_slot) as u8;
                    if let Some(name) =
                        try_property!(pr, "m_vecPlayerData.{i:04}.m_iszPlayerName")
                    {
                        let name: String = name;
                        p.name = name;
                    }
                    self.slot_team[added] = player_team as u8;
                    self.valid_indices[added] = i;
                    added += 1;
                }
                if player_team == 14 {
                    waiting_for_draft = true;
                    break;
                }
                i += 1;
            }
            if !waiting_for_draft && added == 10 {
                self.init = true;
            }
        }

        if !self.init || self.post_game {
            return Ok(());
        }

        let Ok(t) = self.time(ctx) else {
            return Ok(());
        };

        if self.next_interval == 0 {
            self.next_interval = t as i32;
        }
        if (t as i32) < self.next_interval {
            return Ok(());
        }
        self.next_interval = t as i32 + 1;

        for slot in 0..10usize {
            let i = self.valid_indices[slot];
            let hero_id: i32 = property!(pr, "m_vecPlayerTeamData.{i:04}.m_nSelectedHeroID");
            let hero_handle: usize = property!(pr, "m_vecPlayerTeamData.{i:04}.m_hSelectedHero");
            let team_slot: i32 = property!(pr, "m_vecPlayerTeamData.{i:04}.m_iTeamSlot");
            let level: u8 =
                try_property!(pr, "m_vecPlayerTeamData.{i:04}.m_iLevel").unwrap_or(0);

            if hero_id > 0 && self.players[slot].hero_id == 0 {
                self.players[slot].hero_id = hero_id;
            }

            let data_team = if self.slot_team[slot] == 2 {
                ctx.entities().get_by_class_name("CDOTA_DataRadiant")
            } else {
                ctx.entities().get_by_class_name("CDOTA_DataDire")
            };

            let (mut networth, mut xp, mut lh) = (0u32, 0u32, 0u16);
            if let Ok(dt) = data_team {
                if team_slot >= 0 {
                    networth =
                        try_property!(dt, "m_vecDataTeam.{team_slot:04}.m_iNetWorth").unwrap_or(0);
                    xp = try_property!(dt, "m_vecDataTeam.{team_slot:04}.m_iTotalEarnedXP")
                        .unwrap_or(0);
                    lh = try_property!(dt, "m_vecDataTeam.{team_slot:04}.m_iLastHitCount")
                        .unwrap_or(0);
                }
            }

            if let Ok(hero) = ctx.entities().get_by_handle(hero_handle) {
                let cell_x: u16 = try_property!(hero, "CBodyComponent.m_cellX").unwrap_or(0);
                let cell_y: u16 = try_property!(hero, "CBodyComponent.m_cellY").unwrap_or(0);
                let vec_x: f32 = try_property!(hero, "CBodyComponent.m_vecX").unwrap_or(0.0);
                let vec_y: f32 = try_property!(hero, "CBodyComponent.m_vecY").unwrap_or(0.0);
                let life_state: u8 = try_property!(hero, "m_lifeState").unwrap_or(2);
                let x = world_coord(cell_x, vec_x);
                let y = world_coord(cell_y, vec_y);
                self.cur_pos[slot] = [x, y];
                self.cur_alive[slot] = life_state == 0;

                // build combat-log name mapping once per hero class
                let class = hero.class().name();
                if self.players[slot].hero_name.is_empty() && class.len() > 16 {
                    let base = &class["CDOTA_Unit_Hero_".len()..];
                    let name1 = format!("npc_dota_hero_{}", base.to_lowercase());
                    let name2 = format!(
                        "npc_dota_hero{}",
                        base.chars()
                            .map(|c| {
                                if c.is_ascii_uppercase() {
                                    format!("_{}", c.to_ascii_lowercase())
                                } else {
                                    c.to_string()
                                }
                            })
                            .collect::<String>()
                    );
                    self.name_to_slot.insert(name1.clone(), slot);
                    self.name_to_slot.insert(name2, slot);
                    self.class_to_log_name.insert(class.to_string(), name1.clone());
                    self.players[slot].hero_name = name1;
                }

                self.tracks[slot].push(TrackSample(
                    t,
                    x,
                    y,
                    if life_state == 0 { 1 } else { 0 },
                    networth,
                    xp,
                    lh,
                    level,
                ));
            }
        }

        if let Some(cb) = &self.progress {
            if (t as i32) % 30 == 0 {
                cb(t - self.start_time);
            }
        }

        Ok(())
    }

    #[on_entity]
    fn track_units(&mut self, ctx: &Context, event: EntityEvents, entity: &Entity) -> ObserverResult {
        let class = entity.class().name();
        let kind = match class {
            "CDOTA_BaseNPC_Tower" => "tower",
            "CDOTA_BaseNPC_Barracks" => "rax",
            "CDOTA_BaseNPC_Fort" => "fort",
            "CDOTA_Unit_Roshan" => "roshan",
            _ => return Ok(()),
        };
        if event == EntityEvents::Deleted {
            self.building_state.remove(&entity.index());
            return Ok(());
        }
        let life: u8 = try_property!(entity, "m_lifeState").unwrap_or(0);
        let cell_x: u16 = try_property!(entity, "CBodyComponent.m_cellX").unwrap_or(0);
        let cell_y: u16 = try_property!(entity, "CBodyComponent.m_cellY").unwrap_or(0);
        let vec_x: f32 = try_property!(entity, "CBodyComponent.m_vecX").unwrap_or(0.0);
        let vec_y: f32 = try_property!(entity, "CBodyComponent.m_vecY").unwrap_or(0.0);
        let team: u8 = try_property!(entity, "m_iTeamNum").unwrap_or(0);
        let x = world_coord(cell_x, vec_x);
        let y = world_coord(cell_y, vec_y);

        let prev = self.building_state.insert(entity.index(), (life, x, y, team));
        let was_alive = prev.map(|p| p.0 == 0).unwrap_or(event == EntityEvents::Updated);
        if was_alive && life != 0 {
            let t = self.time(ctx).unwrap_or(0.0);
            self.recent_building_deaths.push((kind, team, x, y, t));
        }
        Ok(())
    }

    #[on_combat_log]
    fn handle_cle(&mut self, cle: &CombatLogEntry) -> ObserverResult {
        let Ok(t) = cle.timestamp() else {
            return Ok(());
        };

        match cle.r#type() {
            DotaCombatlogTypes::DotaCombatlogGameState => {
                if cle.value().unwrap_or(0) == 6 {
                    self.post_game = true;
                }
            }
            DotaCombatlogTypes::DotaCombatlogDeath => {
                let Ok(target) = cle.target_name() else {
                    return Ok(());
                };
                let target = target.to_string();
                let target_is_hero = cle.is_target_hero().unwrap_or(false);
                let target_is_illusion = cle.is_target_illusion().unwrap_or(false);
                let attacker = cle.attacker_name().unwrap_or("").to_string();
                let attacker_is_hero = cle.is_attacker_hero().unwrap_or(false);
                let attacker_is_illusion = cle.is_attacker_illusion().unwrap_or(false);
                let attacker_slot = self.slot_of(&attacker);

                if target_is_hero && !target_is_illusion {
                    let Some(victim_slot) = self.slot_of(&target) else {
                        return Ok(());
                    };
                    // death at victim position
                    let mut death = self.event_at_slot(t, "death", victim_slot);
                    death.target_slot = attacker_slot.map(|s| s as i8).or(Some(-1));
                    death.key = Some(attacker.clone());
                    self.push_event(death);
                    // kill credited to attacking hero
                    if attacker_is_hero && !attacker_is_illusion {
                        if let Some(killer_slot) = attacker_slot {
                            let mut kill = self.event_at_slot(t, "kill", killer_slot);
                            kill.target_slot = Some(victim_slot as i8);
                            kill.key = Some(target.clone());
                            self.push_event(kill);
                        }
                    }
                } else if target.contains("_tower")
                    || target.contains("_rax_")
                    || target.contains("_fort")
                    || target == "npc_dota_roshan"
                {
                    let kind = if target.contains("_tower") {
                        "tower"
                    } else if target.contains("_rax_") {
                        "rax"
                    } else if target.contains("_fort") {
                        "fort"
                    } else {
                        "roshan"
                    };
                    let cl_team: u8 = if target.contains("goodguys") {
                        2
                    } else if target.contains("badguys") {
                        3
                    } else {
                        0 // roshan
                    };
                    // position is paired with an entity life-state death in finalize()
                    self.pending_building_cl.push((
                        t,
                        kind,
                        cl_team,
                        attacker_slot.map(|s| s as i8),
                        target,
                    ));
                } else if attacker_is_hero && !attacker_is_illusion && !target_is_hero {
                    // creep / neutral last hit or deny, at the farming hero's position
                    if let Some(slot) = attacker_slot {
                        // skip wards; the ward observer handles those
                        if target.contains("_wards") {
                            return Ok(());
                        }
                        let attacker_team = self.slot_team[slot];
                        let is_deny = (attacker_team == 2 && target.contains("goodguys"))
                            || (attacker_team == 3 && target.contains("badguys"));
                        let mut ev =
                            self.event_at_slot(t, if is_deny { "deny" } else { "lastHit" }, slot);
                        ev.key = Some(target);
                        self.push_event(ev);
                    }
                }
            }
            DotaCombatlogTypes::DotaCombatlogItem => {
                if let (Ok(attacker), Ok(inflictor)) = (cle.attacker_name(), cle.inflictor_name()) {
                    if inflictor == "item_smoke_of_deceit" {
                        if let Some(slot) = self.slot_of(attacker) {
                            let ev = self.event_at_slot(t, "smoke", slot);
                            self.push_event(ev);
                        }
                    }
                }
            }
            DotaCombatlogTypes::DotaCombatlogPurchase => {
                if let (Ok(target), Ok(item)) = (cle.target_name(), cle.value_name()) {
                    if let Some(slot) = self.slot_of(target) {
                        let mut ev = self.event_at_slot(t, "purchase", slot);
                        ev.key = Some(item.to_string());
                        self.push_event(ev);
                    }
                }
            }
            DotaCombatlogTypes::DotaCombatlogBuyback => {
                let slot = cle.value().unwrap_or(99) as usize;
                if slot < 10 {
                    let ev = self.event_at_slot(t, "buyback", slot);
                    self.push_event(ev);
                }
            }
            _ => {}
        }
        Ok(())
    }

    #[on_message]
    fn on_chat_event(&mut self, ctx: &Context, event: CDotaUserMsgChatEvent) -> ObserverResult {
        let Ok(t) = self.time(ctx) else {
            return Ok(());
        };
        let type_name = format!("{:?}", event.r#type());
        let slot = event.playerid_1() as i32;
        if !(0..10).contains(&slot) {
            return Ok(());
        }
        let slot = slot as usize;
        match type_name.as_str() {
            "ChatMessageRunePickup" => {
                let mut ev = self.event_at_slot(t, "rune", slot);
                ev.key = Some(event.value().to_string());
                self.push_event(ev);
            }
            "ChatMessageRuneBottle" => {
                let mut ev = self.event_at_slot(t, "rune", slot);
                ev.key = Some(format!("bottled:{}", event.value()));
                self.push_event(ev);
            }
            "ChatMessageAegis" => {
                let ev = self.event_at_slot(t, "aegis", slot);
                self.push_event(ev);
            }
            _ => {}
        }
        Ok(())
    }
}

impl GameTimeObserver for Extractor {
    fn on_game_started(&mut self, _ctx: &Context, start_time: f32) -> ObserverResult {
        self.start_time = start_time;
        Ok(())
    }
}

impl WardsObserver for Extractor {
    fn on_ward(
        &mut self,
        ctx: &Context,
        ward_class: WardClass,
        event: WardEvent,
        ward: &Entity,
    ) -> ObserverResult {
        let Ok(t) = self.time(ctx) else {
            return Ok(());
        };
        let cell_x: u16 = try_property!(ward, "CBodyComponent.m_cellX").unwrap_or(0);
        let cell_y: u16 = try_property!(ward, "CBodyComponent.m_cellY").unwrap_or(0);
        let vec_x: f32 = try_property!(ward, "CBodyComponent.m_vecX").unwrap_or(0.0);
        let vec_y: f32 = try_property!(ward, "CBodyComponent.m_vecY").unwrap_or(0.0);
        let team: u8 = try_property!(ward, "m_iTeamNum").unwrap_or(0);

        let is_obs = ward_class == WardClass::Observer;
        let placed = event == WardEvent::Placed;
        let kind = match (is_obs, placed) {
            (true, true) => "obs",
            (true, false) => "obsLeft",
            (false, true) => "sen",
            (false, false) => "senLeft",
        };

        let mut slot: Option<i8> = None;
        let owner_handle: usize = try_property!(ward, "m_hOwnerEntity").unwrap_or(0xFFFFFF);
        if let Ok(owner) = ctx.entities().get_by_handle(owner_handle) {
            let pid: Option<i32> = try_property!(owner, "m_iPlayerID")
                .or_else(|| try_property!(owner, "m_nPlayerID"));
            if let Some(pid) = pid {
                let s = pid >> 1;
                if (0..10).contains(&s) {
                    slot = Some(s as i8);
                }
            }
        }

        self.push_event(MatchEvent {
            t,
            kind: kind.to_string(),
            slot,
            target_slot: None,
            x: world_coord(cell_x, vec_x),
            y: world_coord(cell_y, vec_y),
            team: Some(team),
            key: match event {
                WardEvent::Killed(killer) => Some(killer.to_string()),
                _ => None,
            },
        });
        Ok(())
    }
}
