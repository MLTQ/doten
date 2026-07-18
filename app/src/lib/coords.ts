// World <-> scene coordinate mapping, per map version.
//
// The scene puts the minimap as a MAP_SIZE x MAP_SIZE plane on y=0, centered
// at the origin, with TIME rising along +Y ("Z-up" in spirit: the vertical
// axis is the time axis).
//
// Patch detection: the demo header's game_directory carries the Dota client
// version (steam.inf ClientVersion, e.g. dota_v6802). Thresholds below were
// derived from SteamDatabase/GameTracking-Dota2 steam.inf history around each
// patch release date (midpoint of last-pre-patch and first-post-patch builds).
//
// Bounds = world coordinates at the edges of each bundled minimap image.
// They differ per image because each render crops differently. Defaults were
// calibrated against building/fountain positions from a real replay and can
// be fine-tuned in-app (Map calibration panel; persisted in localStorage).

import { useStore } from '../store';

export const MAP_SIZE = 512;
export const UNITS_PER_MINUTE = 12; // vertical scene units per game minute

export interface MapBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface MapVersion {
  key: string;
  label: string;
  image: string;
  minBuild: number; // first ClientVersion of this map era
  defaultBounds: MapBounds;
}

// newest first; mapForBuild picks the first entry with minBuild <= build
export const MAP_VERSIONS: MapVersion[] = [
  {
    key: '7_40',
    label: '7.40+',
    image: 'minimaps/7_40.jpg',
    minBuild: 6648, // 7.40 released 2025-12-16
    defaultBounds: { minX: -8760, maxX: 10420, minY: -8580, maxY: 8525 },
  },
  {
    key: '7_39',
    label: '7.39',
    image: 'minimaps/7_39.jpg',
    minBuild: 6417, // 7.39 released 2025-05-22
    defaultBounds: { minX: -8900, maxX: 9100, minY: -8900, maxY: 9100 },
  },
  {
    key: '7_38',
    label: '7.38',
    image: 'minimaps/7_38.jpg',
    minBuild: 6320, // 7.38 released 2025-02-19
    defaultBounds: { minX: -8900, maxX: 9100, minY: -8900, maxY: 9100 },
  },
  {
    key: '7_31',
    label: '≤7.37',
    image: 'minimaps/7_31.jpg',
    minBuild: 0,
    defaultBounds: { minX: -8200, maxX: 8200, minY: -8200, maxY: 8200 },
  },
];

export function mapForBuild(build: number): MapVersion {
  return MAP_VERSIONS.find((m) => build >= m.minBuild) ?? MAP_VERSIONS[MAP_VERSIONS.length - 1];
}

const LS_PREFIX = 'doten.mapBounds.';

export function loadBounds(map: MapVersion): MapBounds {
  try {
    const raw = localStorage.getItem(LS_PREFIX + map.key);
    if (raw) return { ...map.defaultBounds, ...JSON.parse(raw) };
  } catch {
    /* fall through */
  }
  return { ...map.defaultBounds };
}

export function saveBounds(key: string, bounds: MapBounds) {
  localStorage.setItem(LS_PREFIX + key, JSON.stringify(bounds));
}

export function clearBounds(key: string) {
  localStorage.removeItem(LS_PREFIX + key);
}

export function worldToScene(x: number, y: number): [number, number] {
  const b = useStore.getState().bounds;
  const nx = (x - b.minX) / (b.maxX - b.minX);
  const ny = (y - b.minY) / (b.maxY - b.minY);
  return [nx * MAP_SIZE - MAP_SIZE / 2, ny * MAP_SIZE - MAP_SIZE / 2];
}

/**
 * Full scene position: x/z from world coords, y (up) from time.
 * The minimap plane's north (+world y) faces -z after rotation, hence -sy.
 */
export function scenePos(x: number, y: number, t: number): [number, number, number] {
  const [sx, sy] = worldToScene(x, y);
  return [sx, timeToHeight(t), -sy];
}

export function timeToHeight(t: number): number {
  return Math.max(0, t / 60) * UNITS_PER_MINUTE;
}

export function fmtClock(t: number): string {
  const neg = t < 0;
  const a = Math.abs(Math.round(t));
  const m = Math.floor(a / 60);
  const s = a % 60;
  return `${neg ? '-' : ''}${m}:${s.toString().padStart(2, '0')}`;
}
