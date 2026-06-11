import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useStore } from '../store';
import { scenePos } from '../lib/coords';
import { sampleTrack } from '../lib/track';
import { heroIconUrl, PLAYER_COLORS } from '../lib/meta';
import { heroTexture } from '../lib/textures';

const HERO_SIZE = 17;

/** One billboard sprite per hero, riding the scan plane. */
export function HeroMarkers() {
  const game = useStore((s) => s.game)!;
  const visibleSlots = useStore((s) => s.visibleSlots);

  const sprites = useMemo(
    () =>
      game.players.map((p) => {
        const short = p.heroName.replace('npc_dota_hero_', '');
        const initials = short.slice(0, 2).toUpperCase();
        return heroTexture(heroIconUrl(p.heroId), PLAYER_COLORS[p.slot], initials);
      }),
    [game],
  );

  const refs = useRef<(THREE.Sprite | null)[]>([]);

  useFrame(() => {
    const t = useStore.getState().t;
    for (let slot = 0; slot < 10; slot++) {
      const spr = refs.current[slot];
      if (!spr) continue;
      const pt = sampleTrack(game.tracks[slot] ?? [], t);
      if (!pt) {
        spr.visible = false;
        continue;
      }
      spr.visible = visibleSlots.has(slot);
      const [x, h, z] = scenePos(pt.x, pt.y, t);
      spr.position.set(x, h + 4, z);
      const mat = spr.material as THREE.SpriteMaterial;
      mat.opacity = pt.alive ? 1 : 0.25;
      const s = pt.alive ? HERO_SIZE : HERO_SIZE * 0.7;
      spr.scale.set(s, s, 1);
    }
  });

  return (
    <group>
      {game.players.map((p, i) => (
        <sprite key={p.slot} ref={(el) => (refs.current[i] = el)}>
          <spriteMaterial map={sprites[i]} depthWrite={false} transparent />
        </sprite>
      ))}
    </group>
  );
}
