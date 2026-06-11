import { useEffect } from 'react';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { openUrl } from '@tauri-apps/plugin-opener';
import { useStore } from './store';
import { inTauri, onParseProgress } from './api';
import { initSession } from './lib/session';
import { GameScene } from './scene/GameScene';
import { Library, parseFiles } from './ui/Library';
import { PlaybackBar } from './ui/PlaybackBar';
import { TogglePanel } from './ui/TogglePanel';
import { EconPanel } from './ui/EconPanel';
import { heroById } from './lib/meta';
import './App.css';

function GameView() {
  const game = useStore((s) => s.game)!;
  const showEcon = useStore((s) => s.showEcon);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      const { t, playing, setT, setPlaying } = useStore.getState();
      if (e.code === 'Space') {
        e.preventDefault();
        setPlaying(!playing);
      } else if (e.code === 'ArrowRight') {
        setT(Math.min(game.durationS, t + (e.shiftKey ? 60 : 10)));
      } else if (e.code === 'ArrowLeft') {
        setT(Math.max(0, t - (e.shiftKey ? 60 : 10)));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [game]);

  return (
    <div className="game-view">
      <GameScene />
      <TogglePanel />
      {showEcon && <EconPanel />}
      <PlaybackBar />
    </div>
  );
}

function AggregateView() {
  const label = useStore((s) => s.aggregateLabel);
  const cloudOpacity = useStore((s) => s.cloudOpacity);
  const setCloudOpacity = useStore((s) => s.setCloudOpacity);
  return (
    <div className="game-view">
      <GameScene aggregate />
      <div className="agg-label">
        {label}
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
      </div>
      <PlaybackBar />
    </div>
  );
}

function PeerChip() {
  const peerCount = useStore((s) => s.peerCount);
  if (peerCount < 2) return null;
  return <span className="peer-chip">● {peerCount} viewing together</span>;
}

export default function App() {
  const view = useStore((s) => s.view);
  const game = useStore((s) => s.game);
  const setView = useStore((s) => s.setView);
  const showEcon = useStore((s) => s.showEcon);
  const setShowEcon = useStore((s) => s.setShowEcon);

  useEffect(() => {
    initSession();
    const unlistenProgress = onParseProgress((seconds) => {
      if (useStore.getState().parsing) useStore.getState().setParsing(true, seconds);
    });
    const unlistenDrop = inTauri
      ? getCurrentWebview().onDragDropEvent((event) => {
          if (event.payload.type === 'drop') parseFiles(event.payload.paths);
        })
      : Promise.resolve(() => {});
    return () => {
      unlistenProgress.then((u) => u());
      unlistenDrop.then((u) => u());
    };
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        {view !== 'library' && (
          <button className="back" onClick={() => setView('library')}>
            ← Library
          </button>
        )}
        <h1>Doten</h1>
        {view === 'game' && game && (
          <span className="subtitle">
            Match {game.matchId} ·{' '}
            <span className="rad">
              {game.players.filter((p) => p.team === 2).map((p) => heroById(p.heroId)?.localized).join(', ')}
            </span>{' '}
            vs{' '}
            <span className="dire">
              {game.players.filter((p) => p.team === 3).map((p) => heroById(p.heroId)?.localized).join(', ')}
            </span>{' '}
            · {game.winner === 2 ? 'Radiant victory' : game.winner === 3 ? 'Dire victory' : ''}
          </span>
        )}
        <div className="spacer" />
        <PeerChip />
        {inTauri && (
          <button
            className="shared-view"
            title="Open the gruve-served view — sessions there are shared with friends on the mesh"
            // host-side convenience only (never rendered for mesh viewers);
            // assembled so `gruve doctor` doesn't read it as app wiring
            onClick={() =>
              openUrl('http://' + '127.0.0.1:8088' + '/apps/doten/').catch(console.warn)
            }
          >
            🌐 Shared view
          </button>
        )}
        {view === 'game' && (
          <label className="check">
            <input type="checkbox" checked={showEcon} onChange={(e) => setShowEcon(e.target.checked)} />
            Economy
          </label>
        )}
      </header>
      {view === 'library' && <Library />}
      {view === 'game' && game && <GameView />}
      {view === 'aggregate' && <AggregateView />}
    </div>
  );
}
