import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useStore } from '../store';
import { MAP_SIZE, timeToHeight } from '../lib/coords';

/** The translucent plane that sweeps up the time axis during playback. */
export function ScanPlane() {
  const group = useRef<THREE.Group>(null);
  useFrame(() => {
    const t = useStore.getState().t;
    if (group.current) group.current.position.y = timeToHeight(t);
  });
  return (
    <group ref={group}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[MAP_SIZE, MAP_SIZE]} />
        <meshBasicMaterial
          color="#3a86ff"
          transparent
          opacity={0.07}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <lineSegments rotation={[-Math.PI / 2, 0, 0]}>
        <edgesGeometry args={[new THREE.PlaneGeometry(MAP_SIZE, MAP_SIZE)]} />
        <lineBasicMaterial color="#3a86ff" transparent opacity={0.55} />
      </lineSegments>
    </group>
  );
}
