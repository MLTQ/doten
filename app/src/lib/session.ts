// Gruve shared-viewing session: everyone in the room sees the same match,
// the same scan-plane moment, the same playback state.
//
// Contract notes (DESIGN-FOR-GRUVE.md rule 6 / protocol 3.5):
// - Appliers are idempotent renderers: remote state drives the zustand store
//   directly; we never simulate input events or re-trigger the producing action.
// - The `applying` flag stops applied state from re-notifying the session
//   (the SDK already swallows own-echo and unchanged values; this guard stops
//   the synchronous loop before it starts).
// - Playback is synced as a transition {t, playing, speed, at} rather than a
//   per-frame t stream: each client extrapolates from the same anchor, so a
//   playing room costs ~zero messages.

import { joinSession, type SessionHandle } from 'gruve-sdk';
import { api } from '../api';
import { sessionNotify, useStore } from '../store';

interface PlaybackState {
  t: number;
  playing: boolean;
  speed: number;
  at: number; // wallclock ms when t was sampled
}

interface ViewState {
  kind: 'library' | 'game' | 'aggregate';
  matchId?: number;
  aggregate?: { filter: unknown; label: string };
}

let session: SessionHandle | null = null;
let applying = false;
let debounceTimer: ReturnType<typeof setTimeout> | null = null;

function currentPlayback(): PlaybackState {
  const s = useStore.getState();
  return { t: s.t, playing: s.playing, speed: s.speed, at: Date.now() };
}

function pushPlayback() {
  if (applying || !session) return;
  // debounce: scrub drags emit setT continuously
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    session?.state.set('playback', currentPlayback());
  }, 120);
}

function pushView() {
  if (applying || !session) return;
  const s = useStore.getState();
  const view: ViewState = { kind: s.view };
  if (s.view === 'game' && s.game) view.matchId = s.game.matchId;
  if (s.view === 'aggregate') view.aggregate = { filter: null, label: s.aggregateLabel };
  session.state.set('view', view);
}

function applyPlayback(p: PlaybackState) {
  const { setT, setPlaying, setSpeed } = useStore.getState();
  if (debounceTimer) clearTimeout(debounceTimer); // drop stale local pushes
  applying = true;
  try {
    const t = p.playing ? p.t + ((Date.now() - p.at) / 1000) * p.speed : p.t;
    setSpeed(p.speed);
    setT(Math.max(0, t));
    setPlaying(p.playing);
  } finally {
    applying = false;
  }
}

async function applyView(v: ViewState) {
  const s = useStore.getState();
  if (v.kind === 'game' && v.matchId) {
    if (s.view === 'game' && s.game?.matchId === v.matchId) return; // already there
    try {
      const game = await api.loadGame(v.matchId);
      applying = true;
      try {
        useStore.getState().openGame(game);
      } finally {
        applying = false;
      }
    } catch (e) {
      console.warn('session: could not follow to match', v.matchId, e);
    }
  } else if (v.kind === 'library' && s.view !== 'library') {
    applying = true;
    try {
      useStore.getState().setView('library');
    } finally {
      applying = false;
    }
  }
  // aggregate views are label-only for now (filters are host-side state);
  // viewers keep their own aggregate until they build one
}

export function initSession() {
  if (session) return;
  session = joinSession({
    onPeers: (n) => useStore.getState().setPeerCount(n),
  });

  session.state.subscribe((key, value) => {
    // remote truth only (the SDK guarantees no own-echo / no unchanged values)
    if (key === 'playback' && value && typeof value === 'object') {
      applyPlayback(value as PlaybackState);
    } else if (key === 'view' && value && typeof value === 'object') {
      void applyView(value as ViewState);
    }
  });

  sessionNotify.playback = pushPlayback;
  sessionNotify.view = pushView;
}
