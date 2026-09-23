/**
 * Hash-based value noise. Terrain height and colour are functions of WORLD coordinates, not of
 * the shared game Rng, so a recycled ground tile always rebuilds to exactly the same surface
 * and the world stays replayable without consuming random numbers the spawner relies on.
 */

/** Deterministic 2-D integer hash → [0, 1). */
export function hash2(ix: number, iy: number, seed = 0): number {
  let h = (ix | 0) * 374761393 + (iy | 0) * 668265263 + (seed | 0) * 1274126177;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Bilinear value noise in [0, 1). */
export function valueNoise(x: number, y: number, seed = 0): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = smooth(x - ix);
  const fy = smooth(y - iy);
  const a = hash2(ix, iy, seed);
  const b = hash2(ix + 1, iy, seed);
  const c = hash2(ix, iy + 1, seed);
  const d = hash2(ix + 1, iy + 1, seed);
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

/** Fractal Brownian motion of value noise, normalised to [0, 1). */
export function fbm(x: number, y: number, octaves = 4, seed = 0): number {
  let amp = 0.5;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * valueNoise(x * freq, y * freq, seed + i * 17);
    norm += amp;
    amp *= 0.5;
    freq *= 2.05;
  }
  return sum / norm;
}
