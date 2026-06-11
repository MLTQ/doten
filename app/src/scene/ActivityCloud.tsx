import { useMemo } from 'react';
import * as THREE from 'three';
import { useStore } from '../store';
import { scenePos } from '../lib/coords';
import { KIND_STYLES, ALL_KINDS } from '../lib/meta';
import { blobTexture } from '../lib/textures';
import type { EventKind } from '../types';

/**
 * The superimposed activity cloud: every selected event across the whole
 * game (or library aggregate) as soft additive points in the time column.
 */
export function ActivityCloud({ aggregate }: { aggregate: boolean }) {
  const game = useStore((s) => s.game);
  const agg = useStore((s) => s.aggregate);
  const cloudEnabled = useStore((s) => s.cloudEnabled);
  const cloudKinds = useStore((s) => s.cloudKinds);
  const cloudOpacity = useStore((s) => s.cloudOpacity);
  const boundsRev = useStore((s) => s.boundsRev);

  const clouds = useMemo(() => {
    const out: { kind: EventKind; positions: Float32Array }[] = [];
    if (aggregate) {
      if (!agg) return out;
      const positions = new Float32Array(agg.points.length * 3);
      agg.points.forEach(([t, x, y], i) => {
        const [sx, h, sz] = scenePos(x, y, t);
        positions[i * 3] = sx;
        positions[i * 3 + 1] = h;
        positions[i * 3 + 2] = sz;
      });
      out.push({ kind: 'kill', positions }); // styled below by aggregate flag
      return out;
    }
    if (!game) return out;
    for (const kind of ALL_KINDS) {
      if (!cloudKinds.has(kind)) continue;
      const evs = game.events.filter((e) => e.kind === kind);
      if (evs.length === 0) continue;
      const positions = new Float32Array(evs.length * 3);
      evs.forEach((e, i) => {
        const [sx, h, sz] = scenePos(e.x, e.y, e.t);
        positions[i * 3] = sx;
        positions[i * 3 + 1] = h;
        positions[i * 3 + 2] = sz;
      });
      out.push({ kind, positions });
    }
    return out;
  }, [aggregate, game, agg, cloudKinds, boundsRev]);

  if (!aggregate && !cloudEnabled) return null;

  return (
    <group>
      {clouds.map(({ kind, positions }) => (
        <points key={kind + positions.length}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[positions, 3]}
            />
          </bufferGeometry>
          <pointsMaterial
            map={blobTexture()}
            color={aggregate ? '#7fd4c1' : KIND_STYLES[kind].color}
            size={aggregate ? 26 : 20}
            transparent
            // intensity slider maps to per-point alpha; additive blending
            // stacks it, so even small values read where density is high
            opacity={0.01 + cloudOpacity * (aggregate ? 0.25 : 0.3)}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            sizeAttenuation
          />
        </points>
      ))}
    </group>
  );
}
