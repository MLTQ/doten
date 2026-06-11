import { useTexture } from '@react-three/drei';
import * as THREE from 'three';
import { useStore } from '../store';
import { MAP_SIZE } from '../lib/coords';

export function MinimapPlane() {
  const mapImage = useStore((s) => s.mapImage);
  const tex = useTexture(mapImage);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]}>
      <planeGeometry args={[MAP_SIZE, MAP_SIZE]} />
      <meshBasicMaterial map={tex} side={THREE.DoubleSide} toneMapped={false} />
    </mesh>
  );
}
