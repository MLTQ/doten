import { useStore } from '../store';
import { fmtClock } from '../lib/coords';

const SPEEDS = [10, 30, 60, 120, 300];

export function PlaybackBar() {
  const game = useStore((s) => s.game);
  const t = useStore((s) => s.t);
  const playing = useStore((s) => s.playing);
  const speed = useStore((s) => s.speed);
  const { setT, setPlaying, setSpeed } = useStore.getState();

  const duration = game?.durationS ?? 3600;

  return (
    <div className="playback-bar">
      <button className="play-btn" onClick={() => setPlaying(!playing)}>
        {playing ? '⏸' : '▶'}
      </button>
      <span className="clock">{fmtClock(t)}</span>
      <input
        type="range"
        min={0}
        max={duration}
        step={1}
        value={t}
        onChange={(e) => setT(Number(e.target.value))}
        className="scrubber"
      />
      <span className="clock dim">{fmtClock(duration)}</span>
      <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
    </div>
  );
}
