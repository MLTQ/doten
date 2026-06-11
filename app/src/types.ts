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

export interface GameSummary {
  matchId: number;
  durationS: number;
  winner: number;
  gameBuild?: number;
  heroesRadiant: number[];
  heroesDire: number[];
  parsedAt: number;
  tag: string;
}

export interface AggregateFilter {
  kinds: string[];
  team?: number;
  win?: boolean;
  tag?: string;
}

export interface AggregateResult {
  points: [number, number, number][]; // [t, x, y]
  games: number;
}
