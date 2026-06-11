import { useMemo } from 'react';
import { Line, Text } from '@react-three/drei';
import { MAP_SIZE, timeToHeight } from '../lib/coords';

/** Corner pillars + minute labels so the time axis reads at a glance. */
export function TimeAxis({ durationS }: { durationS: number }) {
  const h = timeToHeight(durationS);
  const c = MAP_SIZE / 2;

  const ticks = useMemo(() => {
    const out: { y: number; label: string }[] = [];
    for (let m = 0; m <= durationS / 60; m += 5) {
      out.push({ y: timeToHeight(m * 60), label: `${m}m` });
    }
    return out;
  }, [durationS]);

  const corners: [number, number][] = [
    [-c, -c],
    [-c, c],
    [c, -c],
    [c, c],
  ];

  return (
    <group>
      {corners.map(([x, z], i) => (
        <Line
          key={i}
          points={[
            [x, 0, z],
            [x, h, z],
          ]}
          color="#2a3142"
          lineWidth={1}
          transparent
          opacity={0.8}
        />
      ))}
      {ticks.map((tk) => (
        <group key={tk.label}>
          <Text
            position={[-c - 14, tk.y, -c]}
            fontSize={10}
            color="#7a8499"
            anchorX="right"
          >
            {tk.label}
          </Text>
          <Line
            points={[
              [-c, tk.y, -c],
              [-c + 10, tk.y, -c],
            ]}
            color="#7a8499"
            lineWidth={1}
          />
        </group>
      ))}
    </group>
  );
}
