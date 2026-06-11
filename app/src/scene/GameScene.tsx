import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { Suspense } from 'react';
import { useStore } from '../store';
import { MAP_SIZE, timeToHeight } from '../lib/coords';
import { MinimapPlane } from './MinimapPlane';
import { ScanPlane } from './ScanPlane';
import { HeroMarkers } from './HeroMarkers';
import { Trails } from './Trails';
import { EventMarkers } from './EventMarkers';
import { ActivityCloud } from './ActivityCloud';
import { TimeAxis } from './TimeAxis';

/** Advances playback time every frame. */
function Clock() {
  const advance = useStore((s) => s.advance);
  useFrame((_, dt) => advance(Math.min(dt, 0.1)));
  return null;
}

export function GameScene({ aggregate = false }: { aggregate?: boolean }) {
  const game = useStore((s) => s.game);
  const durationS = aggregate
    ? 3600
    : game?.durationS ?? 1800;
  const maxHeight = timeToHeight(durationS);

  return (
    <Canvas
      camera={{ position: [MAP_SIZE * 0.75, MAP_SIZE * 0.6, MAP_SIZE * 0.75], fov: 50, near: 1, far: 8000 }}
      dpr={[1, 2]}
      style={{ background: '#0b0d12' }}
    >
      <Suspense fallback={null}>
        <Clock />
        <ambientLight intensity={1} />
        <MinimapPlane />
        <TimeAxis durationS={durationS} />
        <ScanPlane />
        {!aggregate && game && (
          <>
            <Trails />
            <HeroMarkers />
            <EventMarkers />
          </>
        )}
        <ActivityCloud aggregate={aggregate} />
        <OrbitControls
          target={[0, Math.min(60, maxHeight / 4), 0]}
          maxDistance={4000}
          minDistance={40}
        />
      </Suspense>
    </Canvas>
  );
}
