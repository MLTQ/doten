import { useEffect, useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { api, readOnly } from '../api';
import { useStore } from '../store';
import { heroIconUrl } from '../lib/meta';
import { fmtClock, mapForBuild } from '../lib/coords';
import { ALL_KINDS, KIND_STYLES } from '../lib/meta';
import type { EventKind, GameSummary } from '../types';

export function parseFiles(paths: string[]) {
  if (readOnly) return;
  const { setParsing, openGame, setLibrary } = useStore.getState();
  const dems = paths.filter((p) => p.toLowerCase().endsWith('.dem'));
  if (dems.length === 0) return;
  (async () => {
    setParsing(true);
    try {
      let last = null;
      for (const p of dems) {
        last = await api.parseReplay(p, '');
      }
      setLibrary(await api.listGames());
      setParsing(false);
      if (last && dems.length === 1) openGame(last);
    } catch (e) {
      setParsing(false, 0, String(e));
    }
  })();
}

function HeroIcons({ ids }: { ids: number[] }) {
  return (
    <span className="hero-icons">
      {ids.map((id, i) => {
        const url = heroIconUrl(id);
        return url ? <img key={i} src={url} alt="" /> : <span key={i} className="hero-dot" />;
      })}
    </span>
  );
}

function GameRow({ g }: { g: GameSummary }) {
  const [tag, setTag] = useState(g.tag);
  const setLibrary = useStore((s) => s.setLibrary);

  const openIt = async () => {
    const data = await api.loadGame(g.matchId);
    useStore.getState().openGame(data);
  };

  return (
    <div className="game-row">
      <button className="game-main" onClick={openIt}>
        <span className="match-id">{g.matchId}</span>
        <span className={g.winner === 2 ? 'team-win' : ''}>
          <HeroIcons ids={g.heroesRadiant} />
        </span>
        <span className="vs">vs</span>
        <span className={g.winner === 3 ? 'team-win' : ''}>
          <HeroIcons ids={g.heroesDire} />
        </span>
        <span className="patch-chip">{mapForBuild(g.gameBuild ?? 0).label}</span>
        <span className="duration">{fmtClock(g.durationS)}</span>
      </button>
      {!readOnly && (
        <>
          <input
            className="tag-input"
            placeholder="tag"
            value={tag}
            onChange={(e) => setTag(e.target.value)}
            onBlur={async () => {
              await api.setTag(g.matchId, tag);
              setLibrary(await api.listGames());
            }}
          />
          <button
            className="delete-btn"
            title="Remove from library"
            onClick={async () => {
              await api.deleteGame(g.matchId);
              setLibrary(await api.listGames());
            }}
          >
            ✕
          </button>
        </>
      )}
    </div>
  );
}

function AggregateBuilder() {
  const library = useStore((s) => s.library);
  const [kinds, setKinds] = useState<Set<EventKind>>(new Set(['lastHit']));
  const [team, setTeam] = useState<string>('');
  const [win, setWin] = useState<string>('');
  const [tag, setTag] = useState('');
  const [busy, setBusy] = useState(false);

  const tags = [...new Set(library.map((g) => g.tag).filter(Boolean))];

  const build = async () => {
    setBusy(true);
    try {
      const filter = {
        kinds: [...kinds],
        team: team ? Number(team) : undefined,
        win: win ? win === 'win' : undefined,
        tag: tag || undefined,
      };
      const res = await api.aggregateEvents(filter);
      const teamLabel = team === '2' ? 'Radiant' : team === '3' ? 'Dire' : 'Both teams';
      const label = `${[...kinds].map((k) => KIND_STYLES[k].label).join(' + ')} — ${teamLabel}${
        win ? (win === 'win' ? ', victories' : ', defeats') : ''
      }${tag ? `, tag "${tag}"` : ''} (${res.games} games)`;
      useStore.getState().openAggregate(res, label);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="agg-builder">
      <h3>Library cloud</h3>
      <p className="hint">Aggregate events from all parsed games into one density cloud — the classic Doten heatmaps, interactive.</p>
      <div className="agg-kinds">
        {ALL_KINDS.map((k) => (
          <label key={k} className="check">
            <input
              type="checkbox"
              checked={kinds.has(k)}
              onChange={() => {
                const next = new Set(kinds);
                next.has(k) ? next.delete(k) : next.add(k);
                setKinds(next);
              }}
            />
            {KIND_STYLES[k].glyph} {KIND_STYLES[k].label}
          </label>
        ))}
      </div>
      <div className="agg-row">
        <select value={team} onChange={(e) => setTeam(e.target.value)}>
          <option value="">Both teams</option>
          <option value="2">Radiant</option>
          <option value="3">Dire</option>
        </select>
        <select value={win} onChange={(e) => setWin(e.target.value)} disabled={!team}>
          <option value="">Win + loss</option>
          <option value="win">Victories</option>
          <option value="loss">Defeats</option>
        </select>
        <select value={tag} onChange={(e) => setTag(e.target.value)}>
          <option value="">All tags</option>
          {tags.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <button onClick={build} disabled={busy || kinds.size === 0 || library.length === 0}>
          {busy ? 'Building…' : 'Build cloud'}
        </button>
      </div>
    </div>
  );
}

export function Library() {
  const library = useStore((s) => s.library);
  const parsing = useStore((s) => s.parsing);
  const parseSeconds = useStore((s) => s.parseSeconds);
  const parseError = useStore((s) => s.parseError);
  const setLibrary = useStore((s) => s.setLibrary);

  useEffect(() => {
    api.listGames().then(setLibrary).catch(console.error);
  }, [setLibrary]);

  const pickFile = async () => {
    const sel = await open({
      multiple: true,
      filters: [{ name: 'Dota 2 replay', extensions: ['dem'] }],
    });
    if (sel) parseFiles(Array.isArray(sel) ? sel : [sel]);
  };

  return (
    <div className="library">
      <div className="library-head">
        <h2>Game library</h2>
        {!readOnly && (
          <button className="primary" onClick={pickFile} disabled={parsing}>
            {parsing ? `Parsing… ${fmtClock(parseSeconds)} of game time` : 'Open replay (.dem)…'}
          </button>
        )}
      </div>
      {parseError && <div className="error">Parse failed: {parseError}</div>}
      {readOnly ? (
        <p className="hint">Viewing the host's library — open a game to watch together.</p>
      ) : (
        <p className="hint">
          Drop .dem files anywhere in this window. Replays live in{' '}
          <code>…/Steam/steamapps/common/dota 2 beta/game/dota/replays</code>.
        </p>
      )}
      <div className="game-list">
        {library.length === 0 && <p className="empty">No games parsed yet.</p>}
        {[...library]
          .sort((a, b) => b.parsedAt - a.parsedAt)
          .map((g) => (
            <GameRow key={g.matchId} g={g} />
          ))}
      </div>
      <AggregateBuilder />
    </div>
  );
}
