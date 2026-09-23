import type { Rng } from './types';

/** Seeded, deterministic PRNG (mulberry32) so a run can be replayed and engines compared on identical worlds. */
export function createRng(seed: number): Rng {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    range: (lo, hi) => lo + (hi - lo) * next(),
    pick: (xs) => xs[Math.floor(next() * xs.length)] as (typeof xs)[number],
  };
}
