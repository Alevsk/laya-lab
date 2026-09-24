/**
 * Headless check of the spawner invariants: no renderer, a real THREE.Scene, a seeded rng and
 * a scripted drone flying 30 simulated seconds at a fixed 60 Hz step. Not a unit-test
 * framework (the web package has none); it throws on the first violated invariant, which
 * makes node exit non-zero.
 *
 * Run from `web/`:
 *   npx esbuild src/obstacles/spawner.check.ts --bundle --platform=node --format=esm \
 *     --outfile=/tmp/spawner.check.mjs && node /tmp/spawner.check.mjs
 */
import * as THREE from 'three';
import { createRng } from '../core/rng';
import type { Obstacle, ObstacleKind, WorldContext } from '../core/types';
import { EXPIRE_BEHIND_M } from './base';
import { DEFAULT_CORRIDOR } from './corridor';
import { defaultFactories, disposeFactories } from './registry';
import { createSpawner } from './spawner';

const SEED = 20260922;
const SECONDS = 30;
const DT = 1 / 60;
const HALF_WIDTH = 20;
const BIRD_BAND: readonly [number, number] = [3, 25];
const CAPS: Record<string, number> = { tree: 150, rock: 40, bird: 12 };

interface RunResult {
  signature: string;
  spawnedTotal: Record<string, number>;
  peak: Record<string, number>;
  final: Record<string, number>;
  rowsSeen: number;
  minGapClearance: number;
  maxAbsAcross: Record<string, number>;
  birdAltitude: [number, number];
  difficulty: number;
  updateMsP95: number;
}

function fail(message: string): never {
  throw new Error(`FAIL: ${message}`);
}

function run(seed: number): RunResult {
  const c = DEFAULT_CORRIDOR;
  const rng = createRng(seed);
  const factories = defaultFactories();
  const spawner = createSpawner(rng, factories);
  const ctx: WorldContext = {
    time: 0,
    dt: DT,
    scene: new THREE.Scene(),
    rng,
    bounds: { altitude_min: 1, altitude_max: 30, speed_max: 18, half_width: HALF_WIDTH, corridor_length: 200 },
    drone: { position: new THREE.Vector3(0, 4, 0), velocity: new THREE.Vector3(), heading_deg: 0 },
    difficulty: 3,
    emit: (o) => {
      obstacles.push(o);
    },
  };
  const obstacles: Obstacle[] = [];
  const spawnedTotal: Record<string, number> = {};
  const peak: Record<string, number> = {};
  const maxAbsAcross: Record<string, number> = {};
  let rowsSeen = 0;
  let minGapClearance = Infinity;
  let birdAltitude: [number, number] = [Infinity, -Infinity];
  const signatureParts: string[] = [];
  const updateMs: number[] = [];
  const seenRows = new Set<number>();

  const steps = Math.round(SECONDS / DT);
  for (let step = 0; step < steps; step++) {
    ctx.time = step * DT;
    // Scripted flight: 14 m/s along the corridor, gentle lateral weave, level at 4 m.
    const speed = 14;
    const weave = Math.sin(ctx.time * 0.5) * 6;
    c.compose(speed * ctx.time, weave, 4, ctx.drone.position);
    c.compose(speed, Math.cos(ctx.time * 0.5) * 3, 0, ctx.drone.velocity);

    const t0 = performance.now();
    const fresh = spawner.update(ctx, obstacles);
    updateMs.push(performance.now() - t0);
    for (const o of fresh) {
      obstacles.push(o);
      spawnedTotal[o.kind] = (spawnedTotal[o.kind] ?? 0) + 1;
      signatureParts.push(`${o.kind}:${o.position.x.toFixed(3)},${o.position.y.toFixed(3)},${o.position.z.toFixed(3)}`);
      if (o.object3d.parent !== ctx.scene) fail(`${o.kind}#${o.id} was not added to the scene`);
    }
    for (const o of obstacles) o.update(ctx);

    // --- invariants, every step ---
    const counts = new Map<ObstacleKind, number>();
    const droneAlong = c.along(ctx.drone.position);
    for (const o of obstacles) {
      counts.set(o.kind, (counts.get(o.kind) ?? 0) + 1);
      const across = Math.abs(c.across(o.position));
      maxAbsAcross[o.kind] = Math.max(maxAbsAcross[o.kind] ?? 0, across);
      if (across > HALF_WIDTH) fail(`${o.kind}#${o.id} left the corridor: |across|=${across.toFixed(2)}`);
      // Expiry is applied at the start of the next update; a bird may move up to ~0.13 m past it in between.
      if (c.along(o.position) < droneAlong - EXPIRE_BEHIND_M - 0.5) fail(`${o.kind}#${o.id} survived past expiry`);
      if (o.kind === 'bird') {
        const h = o.position.y;
        birdAltitude = [Math.min(birdAltitude[0], h), Math.max(birdAltitude[1], h)];
        if (h < BIRD_BAND[0] - 0.3 || h > BIRD_BAND[1] + 0.3) fail(`bird#${o.id} out of altitude band: ${h.toFixed(2)}`);
      }
      if (o.object3d.position.distanceTo(o.position) > 1e-9) fail(`${o.kind}#${o.id} object3d out of sync`);
    }
    for (const [kind, n] of counts) {
      peak[kind] = Math.max(peak[kind] ?? 0, n);
      if (n > (CAPS[kind] ?? Infinity)) fail(`${kind} count ${n} exceeds cap ${CAPS[kind]}`);
    }
    // Every live row must keep its clear gap free of gap-respecting obstacles (trees, rocks).
    for (const row of spawner.rows()) {
      if (!seenRows.has(row.along)) {
        seenRows.add(row.along);
        rowsSeen++;
      }
      for (const o of obstacles) {
        if (o.kind === 'bird' || o.kind === 'projectile') continue; // moving / thrown things are not layout
        if (Math.abs(c.along(o.position) - row.along) > 3.5) continue; // rowSpacing/2 + jitter
        const clearance = Math.abs(c.across(o.position) - row.gapCentre) - o.radius - row.gapHalfWidth;
        minGapClearance = Math.min(minGapClearance, clearance);
        if (clearance < -1e-6) fail(`row @${row.along} gap blocked by ${o.kind}#${o.id} (clearance ${clearance.toFixed(2)})`);
      }
    }
  }

  const sorted = [...updateMs].sort((a, b) => a - b);
  const final: Record<string, number> = {};
  for (const o of obstacles) final[o.kind] = (final[o.kind] ?? 0) + 1;
  for (const o of obstacles) o.dispose();
  if (ctx.scene.children.length !== 0) fail(`scene still holds ${ctx.scene.children.length} objects after dispose`);
  disposeFactories(factories);

  return {
    signature: signatureParts.join('|'),
    spawnedTotal,
    peak,
    final,
    rowsSeen,
    minGapClearance,
    maxAbsAcross,
    birdAltitude,
    difficulty: spawner.difficulty(),
    updateMsP95: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
  };
}

const a = run(SEED);
const b = run(SEED);
if (a.signature !== b.signature) fail('same seed produced a different forest');
const other = run(SEED + 1);
if (other.signature === a.signature) fail('different seed produced the same forest');

const { signature: _sig, ...report } = a;
console.log(JSON.stringify({ seconds: SECONDS, dt: DT, halfWidth: HALF_WIDTH, ...report, spawnEvents: a.signature.split('|').length }, null, 2));
console.log('OK: counts bounded, corridor respected, every row has a clear gap, deterministic per seed');
