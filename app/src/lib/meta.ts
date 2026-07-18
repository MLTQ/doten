import heroesJson from '../data/heroes.json';
import type { EventKind } from '../types';

interface HeroInfo {
  name: string;
  short: string;
  localized: string;
}

const heroes = heroesJson as Record<string, HeroInfo>;

export function heroById(id: number): HeroInfo | undefined {
  return heroes[String(id)];
}

export function heroIconUrl(id: number): string | undefined {
  const h = heroById(id);
  if (!h) return undefined;
  // relative (no leading slash): must resolve under a sub-path when served
  // through gruve at /apps/doten/
  return `heroes/${h.short}.png`;
}

// Classic Dota player colors by slot 0..9
export const PLAYER_COLORS = [
  '#3375FF', '#66FFC0', '#BF00BF', '#F3F00B', '#FF6B00',
  '#FE86C2', '#A1B447', '#65D9F7', '#008321', '#A46900',
];

export interface KindStyle {
  label: string;
  glyph: string; // rendered into a sprite texture
  color: string;
  size: number; // sprite scale in scene units
  defaultOn: boolean; // shown as scan-plane icons by default
  defaultCloud: boolean; // included in activity cloud by default
}

export const KIND_STYLES: Record<EventKind, KindStyle> = {
  kill:     { label: 'Kills',      glyph: '⚔️', color: '#ff453a', size: 10, defaultOn: true,  defaultCloud: false },
  death:    { label: 'Deaths',     glyph: '💀', color: '#e0e0e0', size: 10, defaultOn: true,  defaultCloud: true  },
  lastHit:  { label: 'Farm (LH)',  glyph: '🪙', color: '#ffd60a', size: 5,  defaultOn: false, defaultCloud: true  },
  deny:     { label: 'Denies',     glyph: '🚫', color: '#8e8e93', size: 5,  defaultOn: false, defaultCloud: false },
  obs:      { label: 'Obs wards',  glyph: '👁️', color: '#ffcc00', size: 9,  defaultOn: true,  defaultCloud: false },
  sen:      { label: 'Sentries',   glyph: '🔮', color: '#5e9be6', size: 8,  defaultOn: true,  defaultCloud: false },
  obsLeft:  { label: 'Obs killed', glyph: '🙈', color: '#b08900', size: 7,  defaultOn: false, defaultCloud: false },
  senLeft:  { label: 'Sen killed', glyph: '🕳️', color: '#3a5f8a', size: 7,  defaultOn: false, defaultCloud: false },
  rune:     { label: 'Runes',      glyph: '⚡', color: '#bf5af2', size: 8,  defaultOn: true,  defaultCloud: false },
  smoke:    { label: 'Smokes',     glyph: '💨', color: '#98989d', size: 10, defaultOn: true,  defaultCloud: false },
  tower:    { label: 'Towers',     glyph: '🗼', color: '#ff9f0a', size: 12, defaultOn: true,  defaultCloud: false },
  rax:      { label: 'Barracks',   glyph: '🏛️', color: '#ff6961', size: 11, defaultOn: true,  defaultCloud: false },
  fort:     { label: 'Ancients',   glyph: '👑', color: '#ffd700', size: 14, defaultOn: true,  defaultCloud: false },
  roshan:   { label: 'Roshan',     glyph: '🐉', color: '#d4a373', size: 12, defaultOn: true,  defaultCloud: false },
  aegis:    { label: 'Aegis',      glyph: '🛡️', color: '#ffd700', size: 10, defaultOn: true,  defaultCloud: false },
  buyback:  { label: 'Buybacks',   glyph: '💸', color: '#34c759', size: 9,  defaultOn: false, defaultCloud: false },
  purchase: { label: 'Purchases',  glyph: '🛒', color: '#64d2ff', size: 6,  defaultOn: false, defaultCloud: false },
};

export const ALL_KINDS = Object.keys(KIND_STYLES) as EventKind[];
