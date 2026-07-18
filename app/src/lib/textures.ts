import * as THREE from 'three';

const cache = new Map<string, THREE.Texture>();

/** Render a glyph (emoji) onto a transparent canvas texture. */
export function glyphTexture(glyph: string): THREE.Texture {
  const key = `glyph:${glyph}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d')!;
  ctx.font = '96px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(glyph, 64, 70);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  cache.set(key, tex);
  return tex;
}

/** Soft radial blob for cloud points. */
export function blobTexture(): THREE.Texture {
  const key = 'blob';
  const hit = cache.get(key);
  if (hit) return hit;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d')!;
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.4, 'rgba(255,255,255,0.45)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  cache.set(key, tex);
  return tex;
}

/** Colored disc with a darker ring + initials, used as hero icon fallback. */
function discCanvas(color: string, initials: string): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d')!;
  ctx.beginPath();
  ctx.arc(64, 64, 58, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 8;
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.stroke();
  ctx.font = 'bold 52px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#fff';
  ctx.fillText(initials, 64, 66);
  return c;
}

/**
 * Hero icon texture: starts as a colored disc, swaps to the CDN minimap
 * icon if it loads (composited over a colored ring so team/slot reads).
 */
export function heroTexture(url: string | undefined, color: string, initials: string): THREE.CanvasTexture {
  const key = `hero:${url}:${color}:${initials}`;
  const hit = cache.get(key);
  if (hit) return hit as THREE.CanvasTexture;

  const canvas = discCanvas(color, initials);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  cache.set(key, tex);

  if (url) {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const ctx = canvas.getContext('2d')!;
      ctx.clearRect(0, 0, 128, 128);
      // colored ring
      ctx.beginPath();
      ctx.arc(64, 64, 62, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      // icon clipped to inner circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(64, 64, 53, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(img, 6, 6, 116, 116);
      ctx.restore();
      tex.needsUpdate = true;
    };
    img.src = url;
  }
  return tex;
}
