// Mirrors the Rust types in src-tauri/src/demparse/types.rs

/** [t, x, y, alive, networth, xp, lastHits, level] — 1 Hz */
export type TrackSample = [number, number, number, number, number, number, number, number];

export interface PlayerMeta {
  slot: number; // 0..9
  playerSlot: number; // 0-4 radiant, 128-132 dire
  team: number; // 2 radiant, 3 dire
  heroId: number;
  heroName: string; // npc_dota_hero_*
  name: string;
  accountId: number; // 32-bit Steam account id; 0 = unknown/bot
}

export type EventKind =
  | 'kill'
  | 'death'
  | 'lastHit'
  | 'deny'
  | 'obs'
  | 'sen'
  | 'obsLeft'
  | 'senLeft'
  | 'rune'
  | 'smoke'
  | 'tower'
  | 'rax'
  | 'fort'
  | 'roshan'
  | 'aegis'
  | 'buyback'
  | 'purchase';

export interface MatchEvent {
  t: number;
  kind: EventKind;
  slot?: number;
  targetSlot?: number;
  x: number;
  y: number;
  team?: number;
  key?: string;
}

export interface GameData {
  matchId: number;
  durationS: number;
  winner: number; // 2 radiant, 3 dire, 0 unknown
  gameBuild?: number; // dota ClientVersion at record time (0/undefined = unknown)
  players: PlayerMeta[];
  tracks: TrackSample[][];
  events: MatchEvent[];
}

/** Per-player identity carried in the library index for cross-game classifiers. */
export interface PlayerRef {
  slot: number; // indexes into GameData.tracks / .players
  accountId: number; // 0 = unknown/bot
  team: number; // 2 radiant, 3 dire
  heroId: number;
  name: string;
}

export interface GameSummary {
  matchId: number;
  durationS: number;
  winner: number;
  gameBuild?: number;
  heroesRadiant: number[];
  heroesDire: number[];
  players?: PlayerRef[]; // absent on pre-analytics index entries
  parsedAt: number;
  tag: string;
}

export interface AggregateFilter {
  kinds: string[];
  team?: number;
  win?: boolean;
  tag?: string;
}

/**
 * Cross-game player selection: pick players out of each game by identity
 * (hero / account / name) and facet (team, win), then pool their events.
 * Empty classifier arrays / undefined facets mean "no constraint".
 */
export interface SelectionFilter {
  kinds: string[];
  heroes?: number[]; // hero_id whitelist
  accounts?: number[]; // 32-bit account_id whitelist
  nameQuery?: string; // case-insensitive substring on player name
  team?: number; // 2 radiant / 3 dire
  win?: boolean; // selected player's team won
  tag?: string;
}

export interface AggregateResult {
  points: [number, number, number][]; // [t, x, y]
  games: number;
}
