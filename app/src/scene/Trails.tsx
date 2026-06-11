import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import { useStore } from '../store';
import { scenePos } from '../lib/coords';
import { PLAYER_COLORS } from '../lib/meta';

const TAIL_SECONDS = 90;

/** Hero path lines rising through the time column. */
export function Trails() {
  const game = useStore((s) => s.game)!;
  const trailMode = useStore((s) => s.trailMode);
  const visibleSlots = useStore((s) => s.visibleSlots);
  // re-render at 1 Hz during playback, not every frame
  const tSec = useStore((s) => Math.floor(s.t));
  const boundsRev = useStore((s) => s.boundsRev);

  const lines = useMemo(() => {
    if (trailMode === 'off') return [];
    return game.players.map((p) => {
      const track = game.tracks[p.slot] ?? [];
      const from = trailMode === 'full' ? -Infinity : tSec - TAIL_SECONDS;
      const to = trailMode === 'full' ? Infinity : tSec;
      const pts: [number, number, number][] = [];
      for (let i = 0; i < track.length; i += 2) {
        const s = track[i];
        if (s[0] < from || s[0] > to) continue;
        pts.push(scenePos(s[1], s[2], s[0]));
      }
      return { slot: p.slot, pts };
    });
  }, [game, trailMode, trailMode === 'tail' ? tSec : 0, boundsRev]);

  if (trailMode === 'off') return null;
  return (
    <group>
      {lines.map(
        ({ slot, pts }) =>
          pts.length >= 2 &&
          visibleSlots.has(slot) && (
            <Line
              key={slot}
              points={pts}
              color={PLAYER_COLORS[slot]}
              lineWidth={1.5}
              transparent
              opacity={trailMode === 'full' ? 0.35 : 0.8}
            />
          ),
      )}
    </group>
  );
}
