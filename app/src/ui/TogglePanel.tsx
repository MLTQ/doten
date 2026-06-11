import { useState } from 'react';
import { useStore } from '../store';
import { ALL_KINDS, KIND_STYLES, PLAYER_COLORS, heroIconUrl, heroById } from '../lib/meta';
import type { MapBounds } from '../lib/coords';

/**
 * Fine-tune where world coordinates land on the minimap image. Defaults are
 * calibrated per map version; tweaks persist in localStorage per version.
 */
function MapCalibration() {
  const [open, setOpen] = useState(false);
  const bounds = useStore((s) => s.bounds);
  const mapLabel = useStore((s) => s.mapLabel);
  const { setBounds, resetBounds } = useStore.getState();

  const fields: { key: keyof MapBounds; label: string }[] = [
    { key: 'minX', label: 'West (min X)' },
    { key: 'maxX', label: 'East (max X)' },
    { key: 'minY', label: 'South (min Y)' },
    { key: 'maxY', label: 'North (max Y)' },
  ];

  return (
    <>
      <h4>
        <button className="link-btn" onClick={() => setOpen(!open)}>
          {open ? '▾' : '▸'} Map calibration ({mapLabel})
        </button>
      </h4>
      {open && (
        <div className="calib">
          <p className="hint">
            Drag until trails sit on the lanes. Negative = grow that edge.
          </p>
          {fields.map(({ key, label }) => (
            <label key={key} className="calib-row">
              <span>{label}</span>
              <input
                type="range"
                min={key.startsWith('min') ? -12000 : 4000}
                max={key.startsWith('min') ? -4000 : 12000}
                step={25}
                value={bounds[key]}
                onChange={(e) => setBounds({ [key]: Number(e.target.value) })}
              />
              <input
                type="number"
                step={25}
                value={bounds[key]}
                onChange={(e) => setBounds({ [key]: Number(e.target.value) })}
              />
            </label>
          ))}
          <button onClick={resetBounds}>Reset to defaults</button>
        </div>
      )}
    </>
  );
}

export function TogglePanel() {
  const game = useStore((s) => s.game)!;
  const iconKinds = useStore((s) => s.iconKinds);
  const cloudKinds = useStore((s) => s.cloudKinds);
  const cloudEnabled = useStore((s) => s.cloudEnabled);
  const cloudOpacity = useStore((s) => s.cloudOpacity);
  const trailMode = useStore((s) => s.trailMode);
  const visibleSlots = useStore((s) => s.visibleSlots);
  const {
    toggleIconKind,
    toggleCloudKind,
    setCloudEnabled,
    setCloudOpacity,
    setTrailMode,
    toggleSlot,
  } = useStore.getState();

  return (
    <div className="toggle-panel">
      <h4>Players</h4>
      <div className="player-list">
        {game.players.map((p) => {
          const url = heroIconUrl(p.heroId);
          const on = visibleSlots.has(p.slot);
          return (
            <button
              key={p.slot}
              className={`player-toggle ${on ? 'on' : 'off'}`}
              style={{ borderColor: PLAYER_COLORS[p.slot] }}
              onClick={() => toggleSlot(p.slot)}
              title={`${heroById(p.heroId)?.localized ?? p.heroName} — ${p.name}`}
            >
              {url ? <img src={url} alt="" /> : <span className="hero-dot" />}
              <span className="player-name">{heroById(p.heroId)?.localized ?? '?'}</span>
            </button>
          );
        })}
      </div>

      <h4>Event icons</h4>
      <div className="kind-list">
        {ALL_KINDS.map((k) => (
          <label key={k} className="check">
            <input type="checkbox" checked={iconKinds.has(k)} onChange={() => toggleIconKind(k)} />
            <span>{KIND_STYLES[k].glyph} {KIND_STYLES[k].label}</span>
          </label>
        ))}
      </div>

      <h4>
        <label className="check head-check">
          <input
            type="checkbox"
            checked={cloudEnabled}
            onChange={(e) => setCloudEnabled(e.target.checked)}
          />
          Activity cloud
        </label>
      </h4>
      {cloudEnabled && (
        <>
          <label className="opacity-row" title="Cloud intensity">
            <span>◌</span>
            <input
              type="range"
              min={0.02}
              max={1}
              step={0.02}
              value={cloudOpacity}
              onChange={(e) => setCloudOpacity(Number(e.target.value))}
            />
            <span>●</span>
          </label>
          <div className="kind-list">
            {ALL_KINDS.map((k) => (
              <label key={k} className="check">
                <input
                  type="checkbox"
                  checked={cloudKinds.has(k)}
                  onChange={() => toggleCloudKind(k)}
                />
                <span style={{ color: KIND_STYLES[k].color }}>● {KIND_STYLES[k].label}</span>
              </label>
            ))}
          </div>
        </>
      )}

      <h4>Trails</h4>
      <div className="trail-modes">
        {(['off', 'tail', 'full'] as const).map((m) => (
          <button
            key={m}
            className={trailMode === m ? 'on' : ''}
            onClick={() => setTrailMode(m)}
          >
            {m}
          </button>
        ))}
      </div>

      <MapCalibration />
    </div>
  );
}
