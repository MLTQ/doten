import { create } from 'zustand';
import type { AggregateResult, EventKind, GameData, GameSummary } from './types';
import { ALL_KINDS, KIND_STYLES } from './lib/meta';
import {
  clearBounds,
  loadBounds,
  type MapBounds,
  mapForBuild,
  MAP_VERSIONS,
  saveBounds,
} from './lib/coords';

export type View = 'library' | 'game' | 'aggregate';
export type TrailMode = 'off' | 'tail' | 'full';

interface DotenState {
  view: View;
  library: GameSummary[];
  game: GameData | null;
  aggregate: AggregateResult | null;
  aggregateLabel: string;

  // playback
  t: number;
  playing: boolean;
  speed: number; // game seconds per real second

  // layers
  iconKinds: Set<EventKind>;
  cloudKinds: Set<EventKind>;
  cloudEnabled: boolean;
  cloudOpacity: number; // 0..1 intensity for the additive cloud points
  trailMode: TrailMode;
  visibleSlots: Set<number>;
  showEcon: boolean;

  // active minimap (per detected patch) + calibration
  mapKey: string;
  mapLabel: string;
  mapImage: string;
  bounds: MapBounds;
  boundsRev: number; // bumped on calibration change so memos recompute

  // parse status
  parsing: boolean;
  parseSeconds: number;
  parseError: string | null;

  // gruve session
  peerCount: number;
  setPeerCount: (n: number) => void;

  setView: (v: View) => void;
  setLibrary: (l: GameSummary[]) => void;
  openGame: (g: GameData) => void;
  openAggregate: (a: AggregateResult, label: string) => void;
  setT: (t: number) => void;
  advance: (dt: number) => void;
  setPlaying: (p: boolean) => void;
  setSpeed: (s: number) => void;
  toggleIconKind: (k: EventKind) => void;
  toggleCloudKind: (k: EventKind) => void;
  setCloudEnabled: (b: boolean) => void;
  setCloudOpacity: (v: number) => void;
  setTrailMode: (m: TrailMode) => void;
  toggleSlot: (s: number) => void;
  setShowEcon: (b: boolean) => void;
  setParsing: (parsing: boolean, seconds?: number, error?: string | null) => void;
  setBounds: (b: Partial<MapBounds>) => void;
  resetBounds: () => void;
}

// Session-sync hooks (set by lib/session.ts): called on USER-initiated
// playback/view changes so they can be broadcast to the room. The continuous
// per-frame advance() never notifies — only transitions do.
type Notifier = (() => void) | null;
export const sessionNotify: { playback: Notifier; view: Notifier } = {
  playback: null,
  view: null,
};

function mapStateForBuild(build: number) {
  const map = mapForBuild(build);
  return {
    mapKey: map.key,
    mapLabel: map.label,
    mapImage: map.image,
    bounds: loadBounds(map),
  };
}

export const useStore = create<DotenState>((set, get) => ({
  view: 'library',
  library: [],
  game: null,
  aggregate: null,
  aggregateLabel: '',

  t: 0,
  playing: false,
  speed: 60,

  iconKinds: new Set(ALL_KINDS.filter((k) => KIND_STYLES[k].defaultOn)),
  cloudKinds: new Set(ALL_KINDS.filter((k) => KIND_STYLES[k].defaultCloud)),
  cloudEnabled: false,
  cloudOpacity: Number(localStorage.getItem('doten.cloudOpacity') ?? 0.35),
  trailMode: 'tail',
  visibleSlots: new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
  showEcon: true,

  ...mapStateForBuild(0),
  boundsRev: 0,

  parsing: false,
  parseSeconds: 0,
  parseError: null,

  peerCount: 0,
  setPeerCount: (peerCount) => set({ peerCount }),

  setView: (view) => {
    set({ view });
    sessionNotify.view?.();
  },
  setLibrary: (library) => set({ library }),
  openGame: (game) => {
    set((s) => ({
      game,
      view: 'game',
      t: 0,
      playing: false,
      visibleSlots: new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
      ...mapStateForBuild(game.gameBuild ?? 0),
      boundsRev: s.boundsRev + 1,
    }));
    sessionNotify.view?.();
  },
  openAggregate: (aggregate, aggregateLabel) => {
    set((s) => ({
      aggregate,
      aggregateLabel,
      view: 'aggregate',
      t: 0,
      playing: false,
      // aggregates render on the newest map
      ...mapStateForBuild(Number.MAX_SAFE_INTEGER),
      boundsRev: s.boundsRev + 1,
    }));
    sessionNotify.view?.();
  },
  setT: (t) => {
    set({ t });
    sessionNotify.playback?.();
  },
  advance: (dt) => {
    const { t, playing, speed, game } = get();
    if (!playing) return;
    const max = game ? game.durationS : 3600;
    const nt = t + dt * speed;
    if (nt >= max) set({ t: max, playing: false });
    else set({ t: nt });
  },
  setPlaying: (playing) => {
    set({ playing });
    sessionNotify.playback?.();
  },
  setSpeed: (speed) => {
    set({ speed });
    sessionNotify.playback?.();
  },
  toggleIconKind: (k) =>
    set((s) => {
      const next = new Set(s.iconKinds);
      next.has(k) ? next.delete(k) : next.add(k);
      return { iconKinds: next };
    }),
  toggleCloudKind: (k) =>
    set((s) => {
      const next = new Set(s.cloudKinds);
      next.has(k) ? next.delete(k) : next.add(k);
      return { cloudKinds: next };
    }),
  setCloudEnabled: (cloudEnabled) => set({ cloudEnabled }),
  setCloudOpacity: (cloudOpacity) => {
    localStorage.setItem('doten.cloudOpacity', String(cloudOpacity));
    set({ cloudOpacity });
  },
  setTrailMode: (trailMode) => set({ trailMode }),
  toggleSlot: (slot) =>
    set((s) => {
      const next = new Set(s.visibleSlots);
      next.has(slot) ? next.delete(slot) : next.add(slot);
      return { visibleSlots: next };
    }),
  setShowEcon: (showEcon) => set({ showEcon }),
  setParsing: (parsing, seconds = 0, error = null) =>
    set({ parsing, parseSeconds: seconds, parseError: error }),
  setBounds: (b) =>
    set((s) => {
      const bounds = { ...s.bounds, ...b };
      saveBounds(s.mapKey, bounds);
      return { bounds, boundsRev: s.boundsRev + 1 };
    }),
  resetBounds: () =>
    set((s) => {
      clearBounds(s.mapKey);
      const map = MAP_VERSIONS.find((m) => m.key === s.mapKey)!;
      return { bounds: { ...map.defaultBounds }, boundsRev: s.boundsRev + 1 };
    }),
}));
