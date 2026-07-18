import { useEffect, useMemo, useRef } from 'react';
import { useStore } from '../store';
import { fmtClock } from '../lib/coords';
import { heroById, PLAYER_COLORS, KIND_STYLES } from '../lib/meta';
import type { EventKind, MatchEvent, PlayerMeta } from '../types';

// "Announcement"-worthy events — combat + objectives, no farm/ward spam. The
// log further intersects this with the user's enabled icon kinds, so the feed
// matches what's drawn in the scene.
const LOG_KINDS: EventKind[] = [
  'kill',
  'tower',
  'rax',
  'fort',
  'roshan',
  'aegis',
  'buyback',
  'smoke',
  'rune',
];
const LOG_SET = new Set<string>(LOG_KINDS);

/** In-game-style event feed under the net-worth chart, synced to the scan time. */
export function EventLog() {
  const game = useStore((s) => s.game)!;
  const t = useStore((s) => s.t);
  const setT = useStore((s) => s.setT);
  const iconKinds = useStore((s) => s.iconKinds);
  const listRef = useRef<HTMLDivElement>(null);

  const bySlot = useMemo(() => {
    const m = new Map<number, PlayerMeta>();
    for (const p of game.players) m.set(p.slot, p);
    return m;
  }, [game]);

  // notable events for this game, sorted by time (events already come sorted)
  const notable = useMemo(() => game.events.filter((e) => LOG_SET.has(e.kind)), [game]);

  // …up to the current scan time, respecting the user's icon toggles
  const shown = useMemo(
    () => notable.filter((e) => e.t <= t && iconKinds.has(e.kind as EventKind)),
    [notable, Math.floor(t), iconKinds],
  );

  // follow the feed as playback advances (only when a new event lands)
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown.length]);

  const hero = (slot?: number) => {
    if (slot == null || slot < 0) return <span className="ev-hero">?</span>;
    const p = bySlot.get(slot);
    return (
      <span className="ev-hero" style={{ color: PLAYER_COLORS[slot] }}>
        {heroById(p?.heroId ?? -1)?.localized ?? '?'}
      </span>
    );
  };

  const team = (n?: number) => (
    <span className={n === 2 ? 'rad' : 'dire'}>{n === 2 ? 'Radiant' : 'Dire'}</span>
  );

  const describe = (e: MatchEvent) => {
    switch (e.kind) {
      case 'kill':
        return (
          <>
            {hero(e.slot)} killed {hero(e.targetSlot)}
          </>
        );
      case 'tower':
        return (
          <>
            {team(e.team)} tower destroyed
            {e.slot != null && e.slot >= 0 ? <> by {hero(e.slot)}</> : null}
          </>
        );
      case 'rax':
        return <>{team(e.team)} barracks destroyed</>;
      case 'fort':
        return <>{team(e.team)} ancient destroyed</>;
      case 'roshan':
        return (
          <>
            Roshan slain
            {e.slot != null && e.slot >= 0 ? <> by {hero(e.slot)}</> : null}
          </>
        );
      case 'aegis':
        return <>{hero(e.slot)} took the Aegis</>;
      case 'buyback':
        return <>{hero(e.slot)} bought back</>;
      case 'smoke':
        return <>{hero(e.slot)} used Smoke of Deceit</>;
      case 'rune':
        return <>{hero(e.slot)} grabbed a rune</>;
      default:
        return <>{KIND_STYLES[e.kind as EventKind]?.label ?? e.kind}</>;
    }
  };

  return (
    <div className="event-log">
      <div className="ev-title">Event log</div>
      <div className="ev-list" ref={listRef}>
        {shown.length === 0 && <div className="ev-empty">No events yet.</div>}
        {shown.map((e, i) => (
          <button
            key={`${e.t}-${e.kind}-${i}`}
            className={`ev-row${t - e.t < 6 ? ' fresh' : ''}`}
            title="Jump to this moment"
            onClick={() => setT(e.t)}
          >
            <span className="ev-time">{fmtClock(e.t)}</span>
            <span className="ev-glyph">{KIND_STYLES[e.kind as EventKind]?.glyph ?? '•'}</span>
            <span className="ev-desc">{describe(e)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
