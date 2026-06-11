import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { apiBase } from 'gruve-sdk';
import type { AggregateFilter, AggregateResult, GameData, GameSummary } from './types';

export const inTauri = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

/**
 * Remote viewers (gruve mesh / plain browser) can browse but not mutate:
 * parsing needs a local .dem path and Tauri IPC, so those stay invoke-only.
 */
export const readOnly = !inTauri;

interface Api {
  parseReplay(path: string, tag?: string): Promise<GameData>;
  listGames(): Promise<GameSummary[]>;
  loadGame(matchId: number): Promise<GameData>;
  deleteGame(matchId: number): Promise<void>;
  setTag(matchId: number, tag: string): Promise<void>;
  aggregateEvents(filter: AggregateFilter): Promise<AggregateResult>;
}

const realApi: Api = {
  parseReplay: (path, tag = '') => invoke<GameData>('parse_replay', { path, tag }),
  listGames: () => invoke<GameSummary[]>('list_games'),
  loadGame: (matchId) => invoke<GameData>('load_game', { matchId }),
  deleteGame: (matchId) => invoke<void>('delete_game', { matchId }),
  setTag: (matchId, tag) => invoke<void>('set_tag', { matchId, tag }),
  aggregateEvents: (filter) => invoke<AggregateResult>('aggregate_events', { filter }),
};

// ---------------------------------------------------------------------------
// Browser mode: talk to the Rust backend's HTTP surface. Served through gruve
// this resolves to <prefix>/__gruve/api (works from a friend's machine too);
// served by our own backend the fallback '' makes it same-origin.
// ---------------------------------------------------------------------------

const API = apiBase('api', { fallback: '' });

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + path, init);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

const noWrite = async (): Promise<never> => {
  throw new Error('Read-only viewer — parsing/editing happens in the host app');
};

const httpApi: Api = {
  parseReplay: noWrite,
  deleteGame: noWrite,
  setTag: noWrite,
  listGames: () => http<GameSummary[]>('/api/games'),
  loadGame: (matchId) => http<GameData>(`/api/games/${matchId}`),
  aggregateEvents: (filter) =>
    http<AggregateResult>('/api/aggregate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(filter),
    }),
};

// Fixture fallback so the 3D scene is still developable on a bare vite dev
// server (no backend): public/gamedata.json acts as a one-game library.
let fixtureGame: GameData | null = null;
async function fetchFixture(): Promise<GameData> {
  if (!fixtureGame) {
    const res = await fetch('gamedata.json');
    if (!res.ok) throw new Error('no gamedata.json fixture');
    fixtureGame = (await res.json()) as GameData;
  }
  return fixtureGame;
}

const browserApi: Api = {
  ...httpApi,
  listGames: async () => {
    try {
      return await httpApi.listGames();
    } catch {
      // no backend reachable — fall back to the dev fixture (if present)
      try {
        const g = await fetchFixture();
        return [
          {
            matchId: g.matchId,
            durationS: g.durationS,
            winner: g.winner,
            gameBuild: g.gameBuild,
            heroesRadiant: g.players.filter((p) => p.team === 2).map((p) => p.heroId),
            heroesDire: g.players.filter((p) => p.team === 3).map((p) => p.heroId),
            parsedAt: Date.now() / 1000,
            tag: 'fixture',
          },
        ];
      } catch {
        return [];
      }
    }
  },
  loadGame: async (matchId) => {
    try {
      return await httpApi.loadGame(matchId);
    } catch {
      return fetchFixture();
    }
  },
  aggregateEvents: async (filter) => {
    try {
      return await httpApi.aggregateEvents(filter);
    } catch {
      const g = await fetchFixture();
      const points = g.events
        .filter((e) => filter.kinds.includes(e.kind))
        .filter((e) => !filter.team || e.team === filter.team)
        .map((e) => [e.t, e.x, e.y] as [number, number, number]);
      return { points, games: 1 };
    }
  },
};

export const api: Api = inTauri ? realApi : browserApi;

export function onParseProgress(cb: (seconds: number) => void): Promise<UnlistenFn> {
  if (!inTauri) return Promise.resolve(() => {});
  return listen<{ seconds: number }>('parse:progress', (e) => cb(e.payload.seconds));
}
