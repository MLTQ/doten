import type { TrackSample } from '../types';

export interface TrackPoint {
  x: number;
  y: number;
  alive: boolean;
  networth: number;
  xp: number;
  lh: number;
  level: number;
}

/** Interpolated sample at game time t (tracks are ~1 Hz, sorted). */
export function sampleTrack(track: TrackSample[], t: number): TrackPoint | null {
  if (track.length === 0) return null;
  if (t <= track[0][0]) {
    const s = track[0];
    return { x: s[1], y: s[2], alive: s[3] === 1, networth: s[4], xp: s[5], lh: s[6], level: s[7] };
  }
  const last = track[track.length - 1];
  if (t >= last[0]) {
    return { x: last[1], y: last[2], alive: last[3] === 1, networth: last[4], xp: last[5], lh: last[6], level: last[7] };
  }
  // 1 Hz samples: estimate index then walk
  let i = Math.min(track.length - 2, Math.max(0, Math.floor(t - track[0][0])));
  while (i > 0 && track[i][0] > t) i--;
  while (i < track.length - 2 && track[i + 1][0] <= t) i++;
  const a = track[i];
  const b = track[i + 1];
  const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
  return {
    x: a[1] + (b[1] - a[1]) * f,
    y: a[2] + (b[2] - a[2]) * f,
    alive: a[3] === 1,
    networth: a[4],
    xp: a[5],
    lh: a[6],
    level: a[7],
  };
}
