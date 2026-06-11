import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import { useStore } from '../store';
import { scenePos, timeToHeight } from '../lib/coords';
import { KIND_STYLES } from '../lib/meta';
import { glyphTexture } from '../lib/textures';
import type { MatchEvent } from '../types';

const WINDOW_BEFORE = 8; // seconds an icon lingers after its event
const WINDOW_AHEAD = 0.5;

/** Event icons that pop onto the scan plane as it passes them. */
export function EventMarkers() {
  const game = useStore((s) => s.game)!;
  const iconKinds = useStore((s) => s.iconKinds);
  const visibleSlots = useStore((s) => s.visibleSlots);
  const tSec = useStore((s) => Math.floor(s.t * 2) / 2); // 2 Hz refresh
  const boundsRev = useStore((s) => s.boundsRev);

  const active = useMemo(() => {
    const out: MatchEvent[] = [];
    for (const e of game.events) {
      if (e.t < tSec - WINDOW_BEFORE || e.t > tSec + WINDOW_AHEAD) continue;
      if (!iconKinds.has(e.kind)) continue;
      if (e.slot !== undefined && e.slot >= 0 && !visibleSlots.has(e.slot)) continue;
      out.push(e);
      if (out.length > 200) break;
    }
    return out;
  }, [game, tSec, iconKinds, visibleSlots, boundsRev]);

  // killer -> victim connector lines for kills in window (the old kd_lines)
  const killLines = useMemo(() => {
    const lines: { from: [number, number, number]; to: [number, number, number] }[] = [];
    if (!iconKinds.has('kill')) return lines;
    const deaths = active.filter((e) => e.kind === 'death');
    for (const k of active) {
      if (k.kind !== 'kill') continue;
      const d = deaths.find((d) => d.t === k.t && d.slot === k.targetSlot);
      if (d) {
        lines.push({ from: scenePos(k.x, k.y, k.t), to: scenePos(d.x, d.y, d.t) });
      }
    }
    return lines;
  }, [active, iconKinds]);

  const planeH = timeToHeight(tSec);

  return (
    <group>
      {active.map((e, i) => {
        const st = KIND_STYLES[e.kind];
        const [x, , z] = scenePos(e.x, e.y, e.t);
        // age fade
        const age = (tSec - e.t) / WINDOW_BEFORE;
        const opacity = Math.max(0.15, 1 - age * 0.8);
        return (
          <sprite key={`${e.t}-${e.kind}-${i}`} position={[x, planeH + 3, z]} scale={[st.size, st.size, 1]}>
            <spriteMaterial
              map={glyphTexture(st.glyph)}
              transparent
              opacity={opacity}
              depthWrite={false}
            />
          </sprite>
        );
      })}
      {killLines.map((l, i) => (
        <Line
          key={i}
          points={[l.from, l.to]}
          color={KIND_STYLES.kill.color}
          lineWidth={2}
          transparent
          opacity={0.8}
        />
      ))}
    </group>
  );
}
