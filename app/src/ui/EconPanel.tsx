import { useMemo, useRef } from 'react';
import { useStore } from '../store';
import { fmtClock } from '../lib/coords';
import { PLAYER_COLORS, heroById } from '../lib/meta';
import { sampleTrack } from '../lib/track';

const W = 360;
const H = 110;

/** Net-worth advantage graph synced to the scan time; click/drag to scrub. */
export function EconPanel() {
  const game = useStore((s) => s.game)!;
  const t = useStore((s) => s.t);
  const setT = useStore((s) => s.setT);
  const svgRef = useRef<SVGSVGElement>(null);

  const { path, maxAbs } = useMemo(() => {
    // radiant net worth minus dire net worth, per second
    const duration = Math.max(1, game.durationS);
    const diffs: number[] = [];
    let maxAbs = 1000;
    for (let s = 0; s <= duration; s += 5) {
      let rad = 0;
      let dire = 0;
      for (const p of game.players) {
        const pt = sampleTrack(game.tracks[p.slot] ?? [], s);
        if (!pt) continue;
        if (p.team === 2) rad += pt.networth;
        else dire += pt.networth;
      }
      const d = rad - dire;
      diffs.push(d);
      maxAbs = Math.max(maxAbs, Math.abs(d));
    }
    const pts = diffs.map((d, i) => {
      const x = (i * 5 / duration) * W;
      const y = H / 2 - (d / maxAbs) * (H / 2 - 6);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return { path: `M${pts.join(' L')}`, maxAbs };
  }, [game]);

  const scrub = (e: React.MouseEvent) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const f = (e.clientX - rect.left) / rect.width;
    setT(Math.max(0, Math.min(1, f)) * game.durationS);
  };

  const cursorX = (t / Math.max(1, game.durationS)) * W;

  // per-player networth at t
  const rows = useMemo(
    () =>
      game.players.map((p) => ({
        slot: p.slot,
        team: p.team,
        name: heroById(p.heroId)?.localized ?? '?',
        nw: sampleTrack(game.tracks[p.slot] ?? [], t)?.networth ?? 0,
      })),
    [game, Math.floor(t)],
  );
  const maxNw = Math.max(1, ...rows.map((r) => r.nw));

  return (
    <div className="econ-panel">
      <div className="econ-title">
        Net worth — <span className="rad">Radiant</span> vs <span className="dire">Dire</span>{' '}
        <span className="dim">(±{(maxAbs / 1000).toFixed(0)}k)</span>
      </div>
      <svg
        ref={svgRef}
        width={W}
        height={H}
        onMouseDown={scrub}
        onMouseMove={(e) => e.buttons === 1 && scrub(e)}
      >
        <rect x={0} y={0} width={W} height={H / 2} fill="#1f6f43" opacity={0.12} />
        <rect x={0} y={H / 2} width={W} height={H / 2} fill="#8a2b2b" opacity={0.12} />
        <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="#555" strokeDasharray="3 3" />
        <path d={path} fill="none" stroke="#e8c468" strokeWidth={1.5} />
        <line x1={cursorX} y1={0} x2={cursorX} y2={H} stroke="#3a86ff" strokeWidth={1.5} />
        <text x={Math.min(cursorX + 4, W - 40)} y={12} fill="#9aa4b8" fontSize={11}>
          {fmtClock(t)}
        </text>
      </svg>
      <div className="nw-bars">
        {rows.map((r) => (
          <div key={r.slot} className="nw-row" title={`${r.name}: ${r.nw.toLocaleString()}g`}>
            <span className="nw-label" style={{ color: PLAYER_COLORS[r.slot] }}>
              {r.name.slice(0, 10)}
            </span>
            <div className="nw-bar-track">
              <div
                className="nw-bar"
                style={{
                  width: `${(r.nw / maxNw) * 100}%`,
                  background: PLAYER_COLORS[r.slot],
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
